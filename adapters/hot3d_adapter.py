"""Adapter for HOT3D fisheye-to-pinhole undistortion.

This is the SOLE location in GloPose that imports hand_tracking_toolkit camera
internals. All other modules use the undistorted images and pinhole parameters
produced here.

HOT3D uses FISHEYE624 cameras (Meta Aria — arctan projection + OVR624 distortion,
12 distortion coefficients). COLMAP, RoMa, DINOv2 all expect pinhole images, so
we undistort here before images enter the pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from kornia.geometry import Se3, PinholeCamera

from hand_tracking_toolkit.camera import (
    CameraModel,
    PinholePlaneCameraModel,
    from_json as httk_from_json,
)
from hand_tracking_toolkit.math_utils import quat_trans_to_matrix

logger = logging.getLogger(__name__)


def parse_hot3d_scene_camera(cam_model_dict: dict) -> CameraModel:
    """Parse a HOT3D BOP scenewise cam_model dict into a hand_tracking_toolkit CameraModel.

    The BOP scenewise ``scene_camera_rgb.json`` stores ``cam_model`` as a raw
    calibration dict with keys: projection_params, projection_model_type,
    image_width, image_height, label, serial_number.  Extrinsics (cam_R_w2c,
    cam_t_w2c) are stored separately and are NOT needed for the undistortion
    remap — we set T_world_from_eye to identity.

    Parameters
    ----------
    cam_model_dict : dict
        The ``cam_model`` sub-dict from a single frame in scene_camera_rgb.json.

    Returns
    -------
    CameraModel
        Typically an OVR624CameraModel for Aria FISHEYE624 cameras.
    """
    # Reconstruct the JSON structure expected by hand_tracking_toolkit.camera.from_json
    # from_json expects: {"calibration": {...}, "T_world_from_camera": {"quaternion_wxyz": [...], "translation_xyz": [...]}}
    # We use identity extrinsics for the remap (only projection model matters).
    calibration_json = {
        "calibration": cam_model_dict,
        "T_world_from_camera": {
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "translation_xyz": [0.0, 0.0, 0.0],
        },
    }
    return httk_from_json(calibration_json)


def make_pinhole_camera(fisheye_cam: CameraModel, focal_scale: float = 1.0) -> PinholePlaneCameraModel:
    """Create a pinhole camera with the same image size and scaled focal length.

    Equivalent to ``clip_util.convert_to_pinhole_camera()`` from the HOT3D
    clips tooling.

    Parameters
    ----------
    fisheye_cam : CameraModel
        Source fisheye camera (e.g. OVR624CameraModel).
    focal_scale : float
        Multiplier for the focal length.  1.0 keeps the fisheye's central
        focal length, <1.0 gives a wider FOV (more of the fisheye visible).
    """
    fx, fy = fisheye_cam.f
    cx, cy = fisheye_cam.c
    return PinholePlaneCameraModel(
        fisheye_cam.width,
        fisheye_cam.height,
        (fx * focal_scale, fy * focal_scale),
        (cx, cy),
        (),  # no distortion coefficients for pinhole
        fisheye_cam.T_world_from_eye,
        serial=fisheye_cam.serial,
        label=fisheye_cam.label,
    )


def compute_undistortion_maps(
    fisheye_cam: CameraModel,
    pinhole_cam: PinholePlaneCameraModel,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cv2.remap lookup tables for fisheye → pinhole undistortion.

    The maps are reusable across all frames in a sequence (Aria RGB camera
    intrinsics are constant within a sequence).

    Parameters
    ----------
    fisheye_cam : CameraModel
        Source fisheye camera.
    pinhole_cam : PinholePlaneCameraModel
        Target pinhole camera (same image size, possibly different focal length).

    Returns
    -------
    map_x, map_y : np.ndarray
        Float32 remap arrays of shape (H, W), for use with ``cv2.remap()``.
    """
    W, H = pinhole_cam.width, pinhole_cam.height
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    dst_win_pts = np.column_stack((px.flatten(), py.flatten())).astype(np.float64)

    # Unproject pinhole pixels to eye-space rays
    dst_eye_pts = pinhole_cam.window_to_eye(dst_win_pts)

    # Project through fisheye model (both cameras share identity extrinsics,
    # so eye_to_world / world_to_eye are identity — skip for efficiency)
    src_win_pts = fisheye_cam.eye_to_window(dst_eye_pts)

    # Mask points behind the camera
    mask = dst_eye_pts[:, 2] < 0
    src_win_pts[mask] = -1

    src_win_pts = src_win_pts.astype(np.float32)
    map_x = src_win_pts[:, 0].reshape((H, W))
    map_y = src_win_pts[:, 1].reshape((H, W))

    return map_x, map_y


def _pinhole_params_from_camera(
    pinhole_cam: PinholePlaneCameraModel,
    scale: float,
    device: str,
) -> PinholeCamera:
    """Convert a hand_tracking_toolkit PinholePlaneCameraModel to a Kornia PinholeCamera."""
    fx, fy = pinhole_cam.f
    cx, cy = pinhole_cam.c
    cam_K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=torch.float32, device=device)
    cam_w2c = Se3.identity(device=device).matrix()
    width = torch.tensor(pinhole_cam.width, dtype=torch.float32, device=device)
    height = torch.tensor(pinhole_cam.height, dtype=torch.float32, device=device)

    pinhole = PinholeCamera(cam_K.unsqueeze(0), cam_w2c.unsqueeze(0),
                            height.unsqueeze(0), width.unsqueeze(0))
    pinhole = pinhole.scale(torch.tensor(scale, device=device).unsqueeze(0))
    return pinhole


def undistort_hot3d_sequence(
    scene_camera_json: Path,
    gt_images: Dict[int, Path],
    gt_segs: Dict[int, Path],
    cache_dir: Path,
    focal_scale: float = 1.0,
    scale: float = 1.0,
    device: str = 'cpu',
) -> Tuple[Dict[int, Path], Dict[int, Path], Dict[int, PinholeCamera]]:
    """Undistort all HOT3D images and masks in a sequence from fisheye to pinhole.

    Computes the remap once from frame 0's camera model (all frames in a HOT3D
    stream share the same intrinsics), then applies to all images and masks.
    Results are cached on disk — re-running skips existing files.

    Parameters
    ----------
    scene_camera_json : Path
        Path to ``scene_camera_rgb.json`` for this sequence.
    gt_images : dict[int, Path]
        Frame-id → original (distorted) image path.
    gt_segs : dict[int, Path]
        Frame-id → original (distorted) segmentation mask path.
    cache_dir : Path
        Directory for undistorted outputs.  Structure: ``cache_dir/rgb/``,
        ``cache_dir/mask_visib_rgb/``.
    focal_scale : float
        Focal length multiplier for the target pinhole camera.
    scale : float
        Image downscale factor (applied to Kornia PinholeCamera params).
    device : str
        Torch device for the returned PinholeCamera objects.

    Returns
    -------
    undist_images : dict[int, Path]
        Frame-id → undistorted image path.
    undist_segs : dict[int, Path]
        Frame-id → undistorted mask path.
    pinhole_params : dict[int, PinholeCamera]
        Frame-id → Kornia PinholeCamera with pinhole intrinsics.
    """
    with open(scene_camera_json, 'r') as f:
        scene_camera_data = json.load(f)

    # Parse camera model from first available frame (all frames share intrinsics)
    first_frame_key = next(iter(scene_camera_data))
    first_frame_data = scene_camera_data[first_frame_key]
    cam_model_dict = first_frame_data['cam_model']

    fisheye_cam = parse_hot3d_scene_camera(cam_model_dict)
    pinhole_cam = make_pinhole_camera(fisheye_cam, focal_scale=focal_scale)

    logger.info(
        f"HOT3D undistortion: {type(fisheye_cam).__name__} "
        f"({fisheye_cam.width}x{fisheye_cam.height}, f={fisheye_cam.f}) → "
        f"PinholePlane (f={pinhole_cam.f}, focal_scale={focal_scale})"
    )

    # Compute remap tables once
    map_x, map_y = compute_undistortion_maps(fisheye_cam, pinhole_cam)

    # Prepare output directories
    rgb_cache = cache_dir / 'rgb'
    mask_cache = cache_dir / 'mask_visib_rgb'
    rgb_cache.mkdir(parents=True, exist_ok=True)
    mask_cache.mkdir(parents=True, exist_ok=True)

    # Undistort all images and masks
    undist_images: Dict[int, Path] = {}
    undist_segs: Dict[int, Path] = {}

    for frame_id in sorted(gt_images.keys()):
        # Undistort image
        src_img_path = gt_images[frame_id]
        dst_img_path = rgb_cache / src_img_path.name
        undist_images[frame_id] = dst_img_path

        if not dst_img_path.exists():
            img = cv2.imread(str(src_img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.warning(f"Could not read image: {src_img_path}")
                continue
            undist_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
            # Convert grayscale to BGR for downstream pipeline (DINOv2, SAM2, matchers expect RGB)
            if undist_img.ndim == 2:
                undist_img = cv2.cvtColor(undist_img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(dst_img_path), undist_img)

        # Undistort segmentation mask (if available)
        if frame_id in gt_segs:
            src_seg_path = gt_segs[frame_id]
            dst_seg_path = mask_cache / src_seg_path.name
            undist_segs[frame_id] = dst_seg_path

            if not dst_seg_path.exists():
                seg = cv2.imread(str(src_seg_path), cv2.IMREAD_UNCHANGED)
                if seg is None:
                    logger.warning(f"Could not read mask: {src_seg_path}")
                    continue
                # INTER_NEAREST preserves binary mask values
                undist_seg = cv2.remap(seg, map_x, map_y, cv2.INTER_NEAREST)
                cv2.imwrite(str(dst_seg_path), undist_seg)

    logger.info(f"Undistorted {len(undist_images)} images, {len(undist_segs)} masks → {cache_dir}")

    # Build Kornia PinholeCamera params for all frames
    kornia_pinhole = _pinhole_params_from_camera(pinhole_cam, scale, device)
    pinhole_params = {frame_id: kornia_pinhole for frame_id in gt_images}

    return undist_images, undist_segs, pinhole_params
