"""Adapter for the Pi3 (π³) external repository.

This is the SOLE location in GloPose that imports Pi3 internals.
All other modules import from here.

Pi3 (Permutation-Equivariant Visual Geometry Learning, arXiv:2507.13347) is a
feed-forward method: a single forward pass over all images jointly predicts, per
view, a camera-to-world pose and a point map. The global point cloud is the local
point maps transformed by the predicted poses, so the geometry is internally
consistent (points + poses) but lives in an arbitrary, scale-free world frame —
exactly like VGGT. Alignment to GT (Kabsch / depths) happens later in the pipeline.

Unlike VGGT, Pi3 does not output intrinsics. We recover self-consistent intrinsics
from the predicted ray directions via Pi3's own ``recover_intrinsic_from_rays_d``,
so that projecting the global points through the predicted cameras lands back on the
source pixel grid. Using GT intrinsics here would be inconsistent with Pi3's implied
focal and would break the multi-view mask carve; the reconstruction's intrinsics do
not enter pose / point-cloud evaluation anyway (those use camera centers and 3D
points), so the recovered intrinsics are the correct choice.
"""

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import pycolmap
from PIL import Image

from onboarding.colmap_utils import add_posed_image_to_reconstruction, make_point2d_list

PI3_REPO = Path(__file__).resolve().parent.parent / 'repositories' / 'Pi3'


def _ensure_pi3_on_path():
    pi3_str = str(PI3_REPO)
    if pi3_str not in sys.path:
        sys.path.insert(0, pi3_str)


def load_pi3_model(device: str = 'cuda'):
    """Load the Pi3 model from HuggingFace."""
    _ensure_pi3_on_path()
    from pi3.models.pi3 import Pi3

    model = Pi3.from_pretrained("yyfz233/Pi3")
    model.eval()
    model = model.to(device)
    return model


def _compute_pi3_target_size(w_orig: int, h_orig: int, pixel_limit: int = 255000) -> tuple[int, int]:
    """Replicate Pi3's load_images_as_tensor target-size logic.

    Pi3 resizes every image to a uniform size that is a multiple of 14 on each
    side and keeps total pixels under ``pixel_limit``, preserving aspect ratio.
    """
    scale = math.sqrt(pixel_limit / (w_orig * h_orig)) if w_orig * h_orig > 0 else 1.0
    w_target, h_target = w_orig * scale, h_orig * scale
    k, m = round(w_target / 14), round(h_target / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > w_target / h_target:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def reconstruct_with_pi3(
    image_paths: list[Path],
    image_names: list[str],
    device: str = 'cuda',
    camera_K: Optional[torch.Tensor] = None,
    conf_threshold: float = 0.1,
    max_points: int = 100_000,
    segmentation_paths: Optional[list[Path]] = None,
    model=None,
) -> Optional[pycolmap.Reconstruction]:
    """Run Pi3 feed-forward reconstruction on a set of images.

    Args:
        image_paths: Paths to input images (already background-masked if desired).
        image_names: COLMAP image names (e.g. '0.png', '5.png') — must match
            the names used in DataGraph.image_filename.
        device: Torch device.
        camera_K: Accepted for interface parity but NOT used — Pi3's geometry is
            self-consistent only with its own implied intrinsics (see module docstring).
        conf_threshold: Confidence threshold (post-sigmoid) for point filtering.
        max_points: Maximum number of 3D points to include.
        segmentation_paths: Optional per-frame masks; points whose own source pixel
            is on background are dropped here (cheap first pass). The multi-view
            visual-hull carve runs afterwards at the pipeline level.
        model: Pre-loaded Pi3 model. If None, loads from HuggingFace.

    Returns:
        pycolmap.Reconstruction with poses and 3D points, or None on failure.
    """
    _ensure_pi3_on_path()
    from pi3.utils.geometry import recover_intrinsic_from_rays_d

    if len(image_paths) < 2:
        print("Pi3 requires at least 2 images")
        return None

    if model is None:
        model = load_pi3_model(device)

    # --- Load + preprocess images, replicating Pi3's uniform sizing ---
    pil_imgs = [Image.open(str(p)).convert('RGB') for p in image_paths]
    orig_sizes = [im.size for im in pil_imgs]  # (W, H) per frame
    w0, h0 = pil_imgs[0].size
    target_w, target_h = _compute_pi3_target_size(w0, h0)

    tensors = []
    for im in pil_imgs:
        resized = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
        arr = np.asarray(resized).astype(np.float32) / 255.0  # (H, W, 3)
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    imgs = torch.stack(tensors, dim=0).to(device)  # (N, 3, H, W)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # --- Run Pi3 ---
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None])  # add batch dim -> (1, N, 3, H, W)

    points = res['points'][0].float().cpu().numpy()          # (N, H, W, 3) global
    conf = torch.sigmoid(res['conf'][0, ..., 0]).float().cpu().numpy()  # (N, H, W)
    camera_poses = res['camera_poses'][0].float().cpu().numpy()         # (N, 4, 4) cam-to-world

    # Recover self-consistent intrinsics (working resolution) from ray directions.
    rays_d = F.normalize(res['local_points'], dim=-1)        # (1, N, H, W, 3)
    K_work = recover_intrinsic_from_rays_d(
        rays_d, force_center_principal_point=True)[0].float().cpu().numpy()  # (N, 3, 3)

    num_frames, height, width = conf.shape

    # RGB colors at working resolution
    points_rgb = (imgs.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)  # (N, H, W, 3)

    # Pixel-coordinate + frame-index grid (working resolution)
    ys, xs = np.mgrid[0:height, 0:width]
    points_xyf = np.zeros((num_frames, height, width, 3), dtype=np.float32)
    for fidx in range(num_frames):
        points_xyf[fidx, :, :, 0] = xs
        points_xyf[fidx, :, :, 1] = ys
        points_xyf[fidx, :, :, 2] = fidx

    # --- Confidence + per-source-frame segmentation filter ---
    conf_mask = conf >= conf_threshold
    if segmentation_paths is not None:
        for fidx, seg_path in enumerate(segmentation_paths):
            seg_img = Image.open(seg_path).convert('L').resize((width, height), Image.NEAREST)
            conf_mask[fidx] &= (np.array(seg_img) > 127)

    # Randomly limit the number of kept points
    n_true = int(conf_mask.sum())
    if n_true > max_points:
        true_idx = np.flatnonzero(conf_mask.reshape(-1))
        drop = np.random.choice(true_idx, n_true - max_points, replace=False)
        flat = conf_mask.reshape(-1)
        flat[drop] = False
        conf_mask = flat.reshape(conf_mask.shape)

    filtered_pts3d = points[conf_mask]
    filtered_xyf = points_xyf[conf_mask]
    filtered_rgb = points_rgb[conf_mask]

    if len(filtered_pts3d) == 0:
        print("Pi3 produced no valid 3D points after filtering")
        return None

    # --- Build pycolmap.Reconstruction ---
    reconstruction = pycolmap.Reconstruction()

    # Add 3D points (global frame) with empty tracks first
    for idx in range(len(filtered_pts3d)):
        reconstruction.add_point3D(
            filtered_pts3d[idx], pycolmap.Track(), filtered_rgb[idx])

    for fidx in range(num_frames):
        orig_w, orig_h = orig_sizes[fidx]
        sx = orig_w / width
        sy = orig_h / height

        # Recovered intrinsics scaled from working resolution to original image space,
        # so the camera pixel space matches the on-disk masks used by the carve.
        K = K_work[fidx]
        fx = K[0, 0] * sx
        fy = K[1, 1] * sy
        cx = K[0, 2] * sx
        cy = K[1, 2] * sy
        cam_params = np.array([fx, fy, cx, cy])

        camera = pycolmap.Camera(
            model='PINHOLE', width=int(orig_w), height=int(orig_h),
            params=cam_params, camera_id=fidx + 1)
        reconstruction.add_camera(camera)

        # cam_from_world = inverse of predicted cam-to-world pose
        cam_from_world_mat = np.linalg.inv(camera_poses[fidx])
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(cam_from_world_mat[:3, :3]),
            cam_from_world_mat[:3, 3])

        # 2D observations for points belonging to this frame, scaled to original space
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

    print(f"Pi3 reconstruction: {num_frames} images, {len(filtered_pts3d)} 3D points")
    return reconstruction
