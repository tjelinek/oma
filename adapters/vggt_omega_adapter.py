"""Adapter for the VGGT-Omega external repository.

This is the SOLE location in GloPose that imports VGGT-Omega internals.
All other modules import from here.

VGGT-Omega (facebookresearch/vggt-omega) is VGGT retrained for dynamic ("4D")
scenes: a feed-forward transformer whose camera head predicts per-frame
camera-from-world extrinsics + intrinsics (from a pose encoding) and whose dense
head predicts per-frame depth + confidence. There is no track/point head — the
global point cloud is obtained by unprojecting the predicted depth maps through
the predicted cameras, exactly like VGGT. The geometry lives in an arbitrary,
scale-free world frame; alignment to GT (Kabsch / depths) happens later in the
pipeline.

The released checkpoints are gated on HuggingFace (manual approval) and are NOT
downloadable at runtime — the checkpoint path must point to a local copy
(see OnboardingConfig.vggt_omega_weights_path).
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pycolmap
from PIL import Image

from onboarding.colmap_utils import add_posed_image_to_reconstruction, make_point2d_list

OMEGA_REPO = Path(__file__).resolve().parent.parent / 'repositories' / 'vggt-omega'


def _ensure_omega_on_path():
    omega_str = str(OMEGA_REPO)
    if omega_str not in sys.path:
        sys.path.insert(0, omega_str)


def load_vggt_omega_model(weights_path: str, device: str = 'cuda'):
    """Load VGGT-Omega (camera + dense heads) from a local checkpoint."""
    _ensure_omega_on_path()
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega(enable_camera=True, enable_depth=True, enable_alignment=False).eval()
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model.to(device)


def _unproject_depth(depth_map: np.ndarray, extrinsic: np.ndarray,
                     intrinsic: np.ndarray) -> np.ndarray:
    """Unproject (N, H, W, 1) depth through cam-from-world extrinsics to world points.

    Same math as vggt-omega's demo_gradio.unproject_depth_map_to_point_map (vendored
    here because the demo module imports gradio).
    """
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    pts_cam = np.stack([(x - cx) * depth / fx, (y - cy) * depth / fy, depth], axis=-1)

    R = extrinsic[:, :3, :3]  # cam-from-world
    t = extrinsic[:, :3, 3]
    # world = R^T @ (cam - t)
    pts_world = np.einsum('nji,nhwj->nhwi', R, pts_cam - t[:, None, None, :])
    return pts_world


def reconstruct_with_vggt_omega(
    image_paths: list[Path],
    image_names: list[str],
    weights_path: str,
    device: str = 'cuda',
    camera_K: Optional[torch.Tensor] = None,
    conf_percentile: float = 20.0,
    image_resolution: int = 512,
    max_points: int = 100_000,
    segmentation_paths: Optional[list[Path]] = None,
    model=None,
) -> Optional[pycolmap.Reconstruction]:
    """Run VGGT-Omega feed-forward reconstruction on a set of images.

    Args:
        image_paths: Paths to input images (already background-masked if desired).
        image_names: COLMAP image names — must match DataGraph.image_filename.
        weights_path: Local path to a vggt_omega checkpoint (gated on HF).
        device: Torch device.
        camera_K: Accepted for interface parity but NOT used — the reconstruction is
            self-consistent only with Omega's own predicted intrinsics (as with Pi3).
        conf_percentile: Drop the lowest X% of depth-confidence values, computed over
            in-mask pixels when segmentations are given (Omega's conf is unbounded and
            scene-relative, so a percentile is the released demo's convention).
        image_resolution: Omega preprocessing resolution ('balanced' mode; the released
            512 checkpoint was trained at 512).
        max_points: Maximum number of 3D points to include.
        segmentation_paths: Optional per-frame masks; points whose own source pixel is
            on background are dropped here. The multi-view carve runs at pipeline level.
        model: Pre-loaded model (from load_vggt_omega_model). If None, loads it.

    Returns:
        pycolmap.Reconstruction with poses and 3D points, or None on failure.
    """
    _ensure_omega_on_path()
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    if len(image_paths) < 2:
        print("VGGT-Omega requires at least 2 images")
        return None

    if model is None:
        model = load_vggt_omega_model(weights_path, device)

    orig_sizes = [Image.open(str(p)).size for p in image_paths]  # (W, H)

    images = load_and_preprocess_images(
        [str(p) for p in image_paths], image_resolution=image_resolution).to(device)

    with torch.no_grad():
        predictions = model(images)  # autocasts internally

    extrinsic, intrinsic = encoding_to_camera(
        predictions['pose_enc'], images.shape[-2:])
    extrinsic = extrinsic[0].float().cpu().numpy()          # (N, 3, 4) cam-from-world
    intrinsic = intrinsic[0].float().cpu().numpy()          # (N, 3, 3) working res
    depth = predictions['depth'][0].float().cpu().numpy()   # (N, H, W, 1)
    conf = predictions['depth_conf'][0].float().cpu().numpy()  # (N, H, W)

    points = _unproject_depth(depth, extrinsic, intrinsic)  # (N, H, W, 3)
    num_frames, height, width = conf.shape

    points_rgb = (images.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

    ys, xs = np.mgrid[0:height, 0:width]
    points_xyf = np.zeros((num_frames, height, width, 3), dtype=np.float32)
    for fidx in range(num_frames):
        points_xyf[fidx, :, :, 0] = xs
        points_xyf[fidx, :, :, 1] = ys
        points_xyf[fidx, :, :, 2] = fidx

    # --- Per-source-frame segmentation filter, then scene-relative confidence cut ---
    keep = np.isfinite(points).all(-1) & np.isfinite(conf)
    if segmentation_paths is not None:
        for fidx, seg_path in enumerate(segmentation_paths):
            seg_img = Image.open(seg_path).convert('L').resize((width, height), Image.NEAREST)
            keep[fidx] &= (np.array(seg_img) > 127)
    if conf_percentile > 0 and keep.any():
        thr = np.percentile(conf[keep], conf_percentile)
        keep &= conf >= thr

    n_true = int(keep.sum())
    if n_true > max_points:
        true_idx = np.flatnonzero(keep.reshape(-1))
        drop = np.random.choice(true_idx, n_true - max_points, replace=False)
        flat = keep.reshape(-1)
        flat[drop] = False
        keep = flat.reshape(keep.shape)

    filtered_pts3d = points[keep]
    filtered_xyf = points_xyf[keep]
    filtered_rgb = points_rgb[keep]

    if len(filtered_pts3d) == 0:
        print("VGGT-Omega produced no valid 3D points after filtering")
        return None

    # --- Build pycolmap.Reconstruction ---
    reconstruction = pycolmap.Reconstruction()

    for idx in range(len(filtered_pts3d)):
        reconstruction.add_point3D(
            filtered_pts3d[idx], pycolmap.Track(), filtered_rgb[idx])

    for fidx in range(num_frames):
        orig_w, orig_h = orig_sizes[fidx]
        sx = orig_w / width
        sy = orig_h / height

        # Predicted intrinsics scaled from working resolution to original image space,
        # so the camera pixel space matches the on-disk masks used by the carve.
        K = intrinsic[fidx]
        cam_params = np.array([K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy])

        camera = pycolmap.Camera(
            model='PINHOLE', width=int(orig_w), height=int(orig_h),
            params=cam_params, camera_id=fidx + 1)
        reconstruction.add_camera(camera)

        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsic[fidx, :3, :3]), extrinsic[fidx, :3, 3])

        points2D_list = []
        point2D_idx = 0
        points_in_frame = (filtered_xyf[:, 2].astype(np.int32) == fidx)
        for batch_idx in np.nonzero(points_in_frame)[0]:
            point3D_id = int(batch_idx) + 1
            gx, gy = filtered_xyf[batch_idx, :2]
            points2D_list.append(
                pycolmap.Point2D(np.array([gx * sx, gy * sy]), point3D_id))
            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        add_posed_image_to_reconstruction(
            reconstruction, fidx + 1, fidx + 1, image_names[fidx],
            cam_from_world, points2D=make_point2d_list(points2D_list))

    print(f"VGGT-Omega reconstruction: {num_frames} images, {len(filtered_pts3d)} 3D points")
    return reconstruction
