"""Adapter for the Mast3r external repository.

This is the SOLE location in GloPose that imports Mast3r/Dust3r internals.
All other modules import from here.
"""

import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pycolmap

from onboarding.colmap_utils import add_posed_image_to_reconstruction, make_point2d_list

MAST3R_REPO = Path(__file__).resolve().parent.parent / 'repositories' / 'mast3r'
DUST3R_REPO = MAST3R_REPO / 'dust3r'


def _ensure_mast3r_on_path():
    for p in [str(MAST3R_REPO), str(DUST3R_REPO)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def load_mast3r_model(device: str = 'cuda'):
    """Load the Mast3r ViT-Large model."""
    _ensure_mast3r_on_path()
    from mast3r.model import AsymmetricMASt3R

    model_name = 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    model.eval()
    return model


def reconstruct_with_mast3r(
    image_paths: list[Path],
    image_names: list[str],
    matching_pairs: list[tuple[int, int]],
    device: str = 'cuda',
    camera_K: Optional[torch.Tensor] = None,
    shared_intrinsics: bool = False,
    niter1: int = 300,
    niter2: int = 300,
    segmentation_paths: Optional[list[Path]] = None,
    model=None,
) -> Optional[pycolmap.Reconstruction]:
    """Run Mast3r sparse global alignment on a set of images.

    Args:
        image_paths: Paths to input images (already background-masked if desired).
        image_names: COLMAP image names (e.g. '0.png', '5.png') — must match
            the names used in DataGraph.image_filename.
        matching_pairs: List of (i, j) index pairs for matching. These come from
            our frame filter's keyframe graph edges.
        device: Torch device.
        camera_K: Optional known camera intrinsics (3x3 tensor). Not directly
            used by Mast3r optimization, but used for the output reconstruction.
        shared_intrinsics: Whether to optimize a single set of intrinsics.
        niter1: Coarse alignment iterations.
        niter2: Fine refinement iterations.
        model: Pre-loaded Mast3r model. If None, loads from HuggingFace.

    Returns:
        pycolmap.Reconstruction with poses and 3D points, or None on failure.
    """
    _ensure_mast3r_on_path()
    from dust3r.utils.image import load_images
    from mast3r.image_pairs import make_pairs
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

    if len(image_paths) < 2:
        print("Mast3r requires at least 2 images")
        return None

    if model is None:
        model = load_mast3r_model(device)

    # Load images via Mast3r's own preprocessing
    imgs = load_images([str(p) for p in image_paths], size=512, square_ok=False)

    # Build pairs from our matching_pairs indices
    # Mast3r's make_pairs creates (img_dict, img_dict) tuples
    # We create pairs directly from our index list, symmetrized
    pairs = []
    for i, j in matching_pairs:
        pairs.append((imgs[i], imgs[j]))
        pairs.append((imgs[j], imgs[i]))  # symmetrize

    if len(pairs) == 0:
        print("No matching pairs for Mast3r")
        return None

    # Run sparse global alignment
    img_identifiers = [img['instance'] for img in imgs]

    with tempfile.TemporaryDirectory(suffix='_mast3r_cache') as cache_dir:
        try:
            scene = sparse_global_alignment(
                img_identifiers,
                pairs,
                cache_dir,
                model,
                shared_intrinsics=shared_intrinsics,
                lr1=0.07, niter1=niter1,
                lr2=0.01, niter2=niter2,
                device=device,
            )
        except Exception as e:
            print(f"Mast3r sparse_global_alignment failed: {e}")
            return None

    # Extract results
    cam2w = scene.get_im_poses().detach().cpu().numpy()  # (N, 4, 4) cam-to-world
    focals = scene.get_focals().detach().cpu().numpy()  # (N,)
    pp = scene.get_principal_points().detach().cpu().numpy()  # (N, 2)

    # Get 3D points (sparse)
    sparse_pts3d = scene.get_sparse_pts3d()

    # Load segmentation masks if provided
    seg_masks = {}
    if segmentation_paths is not None:
        from PIL import Image
        for fidx, seg_path in enumerate(segmentation_paths):
            seg_masks[fidx] = np.array(Image.open(seg_path).convert('L')) > 127

    # Collect all 3D points, colors, source frame indices, and 2D pixel coordinates
    all_pts3d = []
    all_colors = []
    all_frame_indices = []
    all_pixel_coords = []  # (x, y) in Mast3r output resolution
    all_mast3r_shapes = []  # (H, W) per frame at Mast3r resolution
    pts3d_colors = scene.get_pts3d_colors()
    for fidx, (pts, colors) in enumerate(zip(sparse_pts3d, pts3d_colors)):
        pts_np = pts.detach().cpu().numpy() if isinstance(pts, torch.Tensor) else np.asarray(pts)
        colors_np = np.asarray(colors)
        if len(pts_np) == 0:
            continue
        # Filter out invalid points (NaN, Inf, very large)
        valid = np.all(np.isfinite(pts_np), axis=-1) & (np.abs(pts_np).max(axis=-1) < 1000)

        # Filter by segmentation mask
        if fidx in seg_masks and pts_np.ndim == 3:
            seg = seg_masks[fidx]
            h, w = pts_np.shape[:2]
            if seg.shape != (h, w):
                seg = np.array(Image.fromarray(seg).resize((w, h), Image.NEAREST))
            valid &= seg

        if pts_np.ndim == 3:
            # (H, W, 3) dense format — compute pixel coordinates before flattening
            h, w = pts_np.shape[:2]
            all_mast3r_shapes.append((h, w))
            ys, xs = np.mgrid[:h, :w]
            pixel_xy = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float64)

            valid_flat = valid.reshape(-1)
            pts_flat = pts_np.reshape(-1, 3)[valid_flat]
            colors_flat = colors_np.reshape(-1, 3)[valid_flat]
            pixel_coords = pixel_xy[valid_flat]
            frame_indices = np.full(pts_flat.shape[0], fidx)
        else:
            pts_flat = pts_np[valid]
            colors_flat = colors_np[valid] if len(colors_np) == len(pts_np) else np.zeros((len(pts_flat), 3))
            pixel_coords = np.zeros((pts_flat.shape[0], 2), dtype=np.float64)
            frame_indices = np.full(pts_flat.shape[0], fidx)

        all_pts3d.append(pts_flat)
        all_colors.append(colors_flat)
        all_frame_indices.append(frame_indices)
        all_pixel_coords.append(pixel_coords)

    if len(all_pts3d) == 0:
        print("Mast3r produced no valid 3D points")
        return None

    all_pts3d = np.concatenate(all_pts3d, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    all_colors = (np.clip(all_colors, 0, 1) * 255).astype(np.uint8)
    all_frame_indices = np.concatenate(all_frame_indices, axis=0)
    all_pixel_coords = np.concatenate(all_pixel_coords, axis=0)

    # Subsample if too many points
    max_points = 100_000
    if len(all_pts3d) > max_points:
        indices = np.random.choice(len(all_pts3d), max_points, replace=False)
        all_pts3d = all_pts3d[indices]
        all_colors = all_colors[indices]
        all_frame_indices = all_frame_indices[indices]
        all_pixel_coords = all_pixel_coords[indices]

    # Build pycolmap.Reconstruction
    reconstruction = pycolmap.Reconstruction()
    num_frames = len(image_paths)

    # Build the output reconstruction in the *original processed image* resolution
    # — i.e. the resolution of image_paths on disk, which is also the resolution of
    # the GT intrinsics (camera_K) and of the segmentation masks. Mast3r works
    # internally at a resized resolution (~512px, square_ok=False); using that for
    # the output cameras leaves the camera pixel space inconsistent with the masks,
    # which breaks reprojection and the multi-view mask carve.
    from PIL import Image
    orig_sizes = [Image.open(str(p)).size for p in image_paths]  # (W, H) per frame
    # Mast3r output resolution from the img tensor: (1, 3, H, W)
    mast3r_shapes = [img['img'].shape[2:] for img in imgs]  # (H, W)

    # Per-frame scale factors from Mast3r output resolution to original image resolution
    scale_factors = {}  # fidx -> (scale_x, scale_y)
    for fidx in range(num_frames):
        orig_w, orig_h = orig_sizes[fidx]
        mast3r_h, mast3r_w = mast3r_shapes[fidx]
        scale_factors[fidx] = (orig_w / mast3r_w, orig_h / mast3r_h)

    # Add cameras and images
    for fidx in range(num_frames):
        w2c = np.linalg.inv(cam2w[fidx])
        R = w2c[:3, :3]
        t = w2c[:3, 3]

        orig_w, orig_h = orig_sizes[fidx]
        sx, sy = scale_factors[fidx]

        if camera_K is not None:
            K_np = camera_K.cpu().numpy() if isinstance(camera_K, torch.Tensor) else camera_K
            fx, fy = K_np[0, 0], K_np[1, 1]
            cx, cy = K_np[0, 2], K_np[1, 2]
        else:
            # Mast3r's focals/pp are in its output resolution — scale to original image resolution
            fx = focals[fidx] * sx
            fy = focals[fidx] * sy
            cx = pp[fidx, 0] * sx
            cy = pp[fidx, 1] * sy

        cam_params = np.array([fx, fy, cx, cy])
        camera = pycolmap.Camera(
            model='PINHOLE', width=int(orig_w), height=int(orig_h),
            params=cam_params, camera_id=fidx + 1)
        reconstruction.add_camera(camera)

        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)

        add_posed_image_to_reconstruction(
            reconstruction, fidx + 1, fidx + 1, image_names[fidx], cam_from_world)

    # Group points by source frame and assign local indices
    frame_point_indices = {}  # fidx -> list of global point indices
    point_local_idx = {}  # global_pt_idx -> local index within its frame
    for pt_idx in range(len(all_pts3d)):
        fidx = int(all_frame_indices[pt_idx])
        local_idx = len(frame_point_indices.get(fidx, []))
        frame_point_indices.setdefault(fidx, []).append(pt_idx)
        point_local_idx[pt_idx] = local_idx

    # Step 1: Set points2D on each image BEFORE adding 3D points (pycolmap validates track indices)
    # Scale pixel coordinates from Mast3r resolution to true image resolution
    for fidx in range(num_frames):
        image_id = fidx + 1
        image = reconstruction.images[image_id]
        sx, sy = scale_factors[fidx]

        pt_indices = frame_point_indices.get(fidx, [])
        points2D = []
        for global_idx in pt_indices:
            px, py = all_pixel_coords[global_idx]
            p2d = pycolmap.Point2D(np.array([px * sx, py * sy]))
            points2D.append(p2d)

        image.points2D = make_point2d_list(points2D) if points2D else make_point2d_list()

    # Step 2: Add 3D points with tracks — pycolmap will set point3D_id on the referenced Point2D
    for pt_idx in range(len(all_pts3d)):
        fidx = int(all_frame_indices[pt_idx])
        image_id = fidx + 1
        pt2d_idx = point_local_idx[pt_idx]

        track = pycolmap.Track()
        track.add_element(image_id, pt2d_idx)
        reconstruction.add_point3D(all_pts3d[pt_idx], track, all_colors[pt_idx])

    print(f"Mast3r reconstruction: {num_frames} images, "
          f"{len(all_pts3d)} 3D points")
    return reconstruction
