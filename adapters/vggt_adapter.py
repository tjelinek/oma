"""Adapter for the VGGT external repository.

This is the SOLE location in GloPose that imports VGGT internals.
All other modules import from here.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import pycolmap

from onboarding.colmap_utils import add_posed_image_to_reconstruction, make_point2d_list

VGGT_REPO = Path(__file__).resolve().parent.parent / 'repositories' / 'vggt'


def _ensure_vggt_on_path():
    vggt_str = str(VGGT_REPO)
    if vggt_str not in sys.path:
        sys.path.insert(0, vggt_str)


def load_vggt_model(device: str = 'cuda', custom_weights_path: Optional[str] = None):
    """Load the VGGT-1B model.

    Args:
        device: Torch device.
        custom_weights_path: Optional path to a fine-tuned checkpoint. When given, the
            base weights are loaded first, then overlaid with the checkpoint's ``model``
            state dict (accepts either a bare state_dict or a ``{"model": ...}`` dict).
    """
    _ensure_vggt_on_path()
    from vggt.models.vggt import VGGT

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    if custom_weights_path is not None:
        sd = torch.load(custom_weights_path, map_location='cpu', weights_only=False)
        state = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
        model.load_state_dict(state)
        print(f"[VGGT] loaded custom fine-tuned weights from {custom_weights_path}")
    model.eval()
    model = model.to(device)
    return model


def _crop_array(arr: np.ndarray, box: tuple[int, int, int], fill=0) -> np.ndarray:
    """Crop (H, W[, C]) array to the square window ``box = (ox, oy, side)``, padding with
    ``fill`` where the window leaves the image (safe for masked/black-bg images)."""
    ox, oy, side = box
    if arr.ndim == 3:
        out = np.full((side, side, arr.shape[2]), fill, dtype=arr.dtype)
    else:
        out = np.full((side, side), fill, dtype=arr.dtype)
    H, W = arr.shape[:2]
    sx0, sy0 = max(ox, 0), max(oy, 0)
    sx1, sy1 = min(ox + side, W), min(oy + side, H)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox] = arr[sy0:sy1, sx0:sx1]
    return out


def _union_object_crop_box(segmentation_paths: list[Path], margin: float,
                           principal_point: Optional[tuple[float, float]] = None) \
        -> Optional[tuple[int, int, int]]:
    """Square crop window covering the union of the object masks across ALL frames
    (+``margin``). One FIXED window for the whole sequence = a single virtual pinhole
    camera (pure principal-point shift + scale), so relative camera geometry is exactly
    preserved — unlike per-frame crops, which would inject an unknown virtual rotation
    per frame.

    When ``principal_point`` (from GT K) is given, the window is CENTERED ON IT and grown
    to still cover the union bbox (+margin). VGGT's pose encoding hard-codes the principal
    point at the image center; a window centered elsewhere makes every recovered rotation
    uniformly off by ~atan(|pp - window_center| / f), which world-frame alignment cannot
    remove. Centering on pp eliminates that bias exactly (at the cost of a looser zoom for
    off-center objects). Returns (ox, oy, side) or None when no mask pixels exist."""
    from PIL import Image
    x0 = y0 = np.inf
    x1 = y1 = -np.inf
    for sp in segmentation_paths:
        seg = np.array(Image.open(sp).convert('L'))
        ys, xs = np.nonzero(seg > 127)
        if len(xs) == 0:
            continue
        x0, x1 = min(x0, xs.min()), max(x1, xs.max())
        y0, y1 = min(y0, ys.min()), max(y1, ys.max())
    if not np.isfinite(x0):
        return None
    if principal_point is not None:
        ppx, ppy = principal_point
        half = margin * max(ppx - x0, x1 - ppx, ppy - y0, y1 - ppy)
        side = int(np.ceil(2 * half))
        ox = int(round(ppx - side / 2))
        oy = int(round(ppy - side / 2))
    else:
        side = int(np.ceil(margin * max(x1 - x0 + 1, y1 - y0 + 1)))
        ox = int(round((x0 + x1) / 2 - side / 2))
        oy = int(round((y0 + y1) / 2 - side / 2))
    return ox, oy, side


def _load_images_cropped(image_paths: list[Path], box: tuple[int, int, int],
                         target_size: int) -> torch.Tensor:
    """Load images, crop each to the fixed square ``box`` (black padding), resize to
    ``target_size``. Returns (N, 3, target_size, target_size) floats in [0, 1] — the same
    convention as vggt's ``load_and_preprocess_images_square``."""
    from PIL import Image
    out = []
    for p in image_paths:
        img = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
        crop = _crop_array(img, box, 0.0)
        t = torch.from_numpy(crop).permute(2, 0, 1)[None]
        t = F.interpolate(t, size=(target_size, target_size),
                          mode='bilinear', align_corners=False)[0]
        out.append(t)
    return torch.stack(out)


def reconstruct_with_vggt(
    image_paths: list[Path],
    image_names: list[str],
    device: str = 'cuda',
    camera_K: Optional[torch.Tensor] = None,
    conf_threshold: float = 0.0,
    max_points: int = 100_000,
    segmentation_paths: Optional[list[Path]] = None,
    model=None,
    custom_weights_path: Optional[str] = None,
    crop_to_object: bool = False,
    crop_margin: float = 1.2,
    use_ba: bool = False,
) -> Optional[pycolmap.Reconstruction]:
    """Run VGGT feed-forward reconstruction on a set of images.

    Args:
        image_paths: Paths to input images (already background-masked if desired).
        image_names: COLMAP image names (e.g. '0.png', '5.png') — must match
            the names used in DataGraph.image_filename.
        device: Torch device.
        camera_K: Optional known camera intrinsics (3x3 tensor). If provided,
            VGGT-predicted intrinsics are replaced with these.
        conf_threshold: Depth confidence threshold for point filtering.
        max_points: Maximum number of 3D points to include.
        model: Pre-loaded VGGT model. If None, loads from HuggingFace.
        crop_to_object: Crop all frames to ONE fixed square window covering the union of
            the object masks (+crop_margin) before feeding VGGT. For masked (black-bg)
            inputs this replaces a mostly-black frame with an object-filling view. The
            fixed window is a pure principal-point shift + scale of the original camera,
            so the predicted extrinsics remain directly comparable to GT; all 2D points /
            intrinsics are mapped back to the ORIGINAL pixel space below.
        crop_margin: Side of the crop window = crop_margin * max union-bbox dimension.
        use_ba: The paper's "VGGT + BA" variant (demo_colmap.py --use_ba): predict
            tracks with the VGGSfM tracker (ALIKED+SuperPoint keypoints; the released
            VGGT TrackHead is too slow for this per upstream), build a track-based
            pycolmap reconstruction seeded from the feed-forward cameras/points, and
            refine with one global bundle adjustment. GT camera_K is then only used
            for the crop window; BA keeps its refined intrinsics.

    Returns:
        pycolmap.Reconstruction with poses and 3D points, or None on failure.
    """
    _ensure_vggt_on_path()
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues

    if len(image_paths) < 2:
        print("VGGT requires at least 2 images")
        return None

    model_loaded_locally = model is None
    if model is None:
        model = load_vggt_model(device, custom_weights_path=custom_weights_path)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    vggt_resolution = 518
    img_load_resolution = 1024

    # Load and preprocess images
    crop_box = None
    if crop_to_object and segmentation_paths is not None:
        pp = None
        if camera_K is not None:
            K_np = camera_K.cpu().numpy() if isinstance(camera_K, torch.Tensor) else np.asarray(camera_K)
            pp = (float(K_np[0, 2]), float(K_np[1, 2]))
        crop_box = _union_object_crop_box(segmentation_paths, crop_margin, principal_point=pp)
        if crop_box is None:
            print("[vggt-crop] no mask pixels in any frame — falling back to full frames")
    if crop_box is not None:
        ox_c, oy_c, side_c = crop_box
        centered = 'pp-centered' if camera_K is not None else 'union-centered (no GT K — rotation bias ~atan(pp_offset/f) possible)'
        print(f"[vggt-crop] fixed object crop ({centered}): origin=({ox_c},{oy_c}) side={side_c}")
        images = _load_images_cropped(image_paths, crop_box, img_load_resolution).to(device)
        original_coords = None
    else:
        images, original_coords = load_and_preprocess_images_square(
            [str(p) for p in image_paths], img_load_resolution)
        images = images.to(device)
        original_coords = original_coords.to(device)

    # Run VGGT
    images_resized = F.interpolate(
        images, size=(vggt_resolution, vggt_resolution),
        mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_batch = images_resized[None]  # add batch dim
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)

        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, images_resized.shape[-2:])
        depth_map, depth_conf = model.depth_head(
            aggregated_tokens_list, images_batch, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()

    # Unproject depth to 3D points
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    if use_ba:
        # Free the VGGT-1B weights + aggregator activations before tracking —
        # for 40+ keyframes they hold ~30GiB and the VGGSfM tracker OOMs even in
        # coarse mode with them resident. Everything the BA branch needs
        # (extrinsic/intrinsic/depth_conf/points_3d) is already numpy.
        del aggregated_tokens_list, ps_idx, pose_enc, images_batch, images_resized
        if model_loaded_locally:
            model.to('cpu')
            del model
        torch.cuda.empty_cache()
        return _reconstruct_vggt_ba(
            images=images, image_paths=image_paths, image_names=image_names,
            extrinsic=extrinsic, intrinsic=intrinsic, depth_conf=depth_conf,
            points_3d=points_3d, crop_box=crop_box, original_coords=original_coords,
            vggt_resolution=vggt_resolution, img_load_resolution=img_load_resolution,
            dtype=dtype)

    # Build reconstruction in feed-forward mode (no BA)
    num_frames = len(image_paths)
    image_size = np.array([vggt_resolution, vggt_resolution])
    height, width = vggt_resolution, vggt_resolution

    # Get RGB colors at VGGT resolution
    points_rgb = F.interpolate(
        images, size=(vggt_resolution, vggt_resolution),
        mode="bilinear", align_corners=False)
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)  # (N, H, W, 3)

    # Pixel coordinates + frame indices
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    # Filter by confidence
    conf_mask = depth_conf >= conf_threshold

    # Filter by segmentation masks (keep only object points).
    # The depth map / conf_mask live in VGGT's center-padded square space
    # (see load_and_preprocess_images_square: the original image is padded to a
    # square of max(w, h) with the content centered, then resized to the VGGT
    # resolution). The mask must undergo the *same* pad-to-square transform,
    # otherwise a naive resize stretches a non-square mask across the full
    # square and misaligns it with the depth grid.
    # Per-source-frame filter: keep only points whose own pixel is on the object.
    # The multi-view visual-hull carve (consistency across all views) is applied
    # afterwards at the pipeline level via carve_reconstruction_by_masks(), shared
    # with the other neural reconstruction methods.
    if segmentation_paths is not None:
        from PIL import Image
        for fidx, seg_path in enumerate(segmentation_paths):
            seg_img = Image.open(seg_path).convert('L')
            if crop_box is not None:
                # Same fixed crop window as the images, then resize to the VGGT grid.
                seg_crop = _crop_array(np.array(seg_img), crop_box, 0)
                seg = np.array(Image.fromarray(seg_crop).resize(
                    (vggt_resolution, vggt_resolution), Image.NEAREST))
            else:
                w, h = seg_img.size
                max_dim = max(w, h)
                left = (max_dim - w) // 2
                top = (max_dim - h) // 2
                square_seg = Image.new('L', (max_dim, max_dim), 0)
                square_seg.paste(seg_img, (left, top))
                seg = np.array(square_seg.resize(
                    (vggt_resolution, vggt_resolution), Image.NEAREST))
            conf_mask[fidx] &= (seg > 127)

    conf_mask = randomly_limit_trues(conf_mask, max_points)

    filtered_pts3d = points_3d[conf_mask]
    filtered_xyf = points_xyf[conf_mask]
    filtered_rgb = points_rgb[conf_mask]

    # Build pycolmap.Reconstruction
    reconstruction = pycolmap.Reconstruction()

    # Add 3D points
    for idx in range(len(filtered_pts3d)):
        reconstruction.add_point3D(
            filtered_pts3d[idx], pycolmap.Track(), filtered_rgb[idx])

    # Rescale intrinsics and 2D points from VGGT's 518px padded-square space
    # to the original image pixel space.
    #
    # original_coords: [x1, y1, x2, y2, width, height] per frame
    #   (x1,y1)-(x2,y2) = bounding box of original image content in the 1024px padded square
    #   width, height = original image dimensions
    #
    # Mapping: grid_518 → padded_1024 → original_image
    #   padded = grid * (1024 / 518)
    #   original_x = (padded_x - x1) * (width / (x2 - x1))
    #   original_y = (padded_y - y1) * (height / (y2 - y1))
    if crop_box is not None:
        # Crop mode: grid_518 -> crop pixels (side / 518) -> original pixels (+ crop origin).
        from PIL import Image
        with Image.open(image_paths[0]) as _im0:
            _orig_w, _orig_h = _im0.size
        ox_c, oy_c, side_c = crop_box
    else:
        original_coords_np = original_coords.cpu().numpy()
    grid_to_padded = img_load_resolution / vggt_resolution  # 1024 / 518

    for fidx in range(num_frames):
        if crop_box is not None:
            scale_x = scale_y = side_c / vggt_resolution
            # Exact inverse of crop->resize on pixel centers (align_corners=False):
            # px = (gx + 0.5) * scale - 0.5 + ox  — fold the half-pixel term into the offset.
            offset_x = -ox_c - 0.5 * (scale_x - 1.0)
            offset_y = -oy_c - 0.5 * (scale_y - 1.0)
            cam_w, cam_h = int(_orig_w), int(_orig_h)
        else:
            x1, y1, x2, y2, orig_w, orig_h = original_coords_np[fidx]
            content_w = x2 - x1  # width of original image in 1024px space
            content_h = y2 - y1  # height of original image in 1024px space

            # Scale and offset from 518px grid to original image
            scale_x = grid_to_padded * orig_w / content_w
            scale_y = grid_to_padded * orig_h / content_h
            offset_x = x1 * orig_w / content_w
            offset_y = y1 * orig_h / content_h

            cam_w, cam_h = int(orig_w), int(orig_h)

        # Override with known intrinsics if provided
        if camera_K is not None:
            K_np = camera_K.cpu().numpy() if isinstance(camera_K, torch.Tensor) else camera_K
            fx, fy = K_np[0, 0], K_np[1, 1]
            cx, cy = K_np[0, 2], K_np[1, 2]
            cam_params = np.array([fx, fy, cx, cy])
        else:
            # Transform VGGT intrinsics (518px padded-square) to original image space
            K = intrinsic[fidx].copy()
            fx = K[0, 0] * scale_x
            fy = K[1, 1] * scale_y
            cx = K[0, 2] * scale_x - offset_x
            cy = K[1, 2] * scale_y - offset_y
            cam_params = np.array([fx, fy, cx, cy])

        camera = pycolmap.Camera(
            model='PINHOLE', width=cam_w, height=cam_h,
            params=cam_params, camera_id=fidx + 1)
        reconstruction.add_camera(camera)

        # Extrinsics are already camera-from-world (OpenCV convention)
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsic[fidx][:3, :3]),
            extrinsic[fidx][:3, 3])

        # Add 2D point observations for points belonging to this frame
        # Transform grid coordinates to original image space
        points2D_list = []
        point2D_idx = 0
        points_in_frame = (filtered_xyf[:, 2].astype(np.int32) == fidx)
        for batch_idx in np.nonzero(points_in_frame)[0]:
            point3D_id = int(batch_idx) + 1
            gx, gy = filtered_xyf[batch_idx, :2]
            px = gx * scale_x - offset_x
            py = gy * scale_y - offset_y
            points2D_list.append(pycolmap.Point2D(np.array([px, py]), point3D_id))
            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        add_posed_image_to_reconstruction(
            reconstruction, fidx + 1, fidx + 1, image_names[fidx],
            cam_from_world, points2D=make_point2d_list(points2D_list))

    print(f"VGGT reconstruction: {num_frames} images, "
          f"{len(filtered_pts3d)} 3D points")
    return reconstruction


def _reconstruct_vggt_ba(images, image_paths, image_names, extrinsic, intrinsic,
                         depth_conf, points_3d, crop_box, original_coords,
                         vggt_resolution, img_load_resolution, dtype,
                         vis_thresh: float = 0.2, max_reproj_error: float = 8.0,
                         query_frame_num: int = 8, max_query_pts: int = 4096,
                         fine_tracking: bool = True) \
        -> Optional[pycolmap.Reconstruction]:
    """Port of demo_colmap.py's --use_ba branch (defaults match the demo args).

    VGGSfM-tracker tracks on the 1024px square images + feed-forward cameras and
    depth-unprojected 3D points seed a track-based pycolmap reconstruction, refined by
    one global pycolmap.bundle_adjustment. Cameras/points2D are then mapped back to
    the ORIGINAL pixel space and images renamed to our COLMAP names — exactly (crop
    mode) or with the demo's center-pp approximation (pad-to-square mode)."""
    from vggt.dependency.track_predict import predict_tracks
    from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap

    device = images.device
    image_size = np.array(images.shape[-2:])
    scale = img_load_resolution / vggt_resolution

    print(f"VGGT+BA: {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated "
          f"before tracking ({len(images)} frames)")

    def _run_tracking(track_images, fine):
        with torch.cuda.amp.autocast(dtype=dtype):
            return predict_tracks(track_images, conf=depth_conf, points_3d=points_3d,
                                  masks=None, max_query_pts=max_query_pts,
                                  query_frame_num=query_frame_num,
                                  keypoint_extractor="aliked+sp", fine_tracking=fine)

    # OOM fallback ladder for long chunks (other pipeline models stay resident,
    # unlike the standalone demo): fine@1024 -> coarse@1024 (the demo's own
    # suggestion) -> coarse@518 with tracks rescaled to the 1024 frame.
    # Flat loop, NOT nested try/except: a retry inside an except suite keeps the
    # failed attempt's whole GPU working set alive via the exception traceback.
    attempts = [
        ("fine@1024", fine_tracking, None, 1.0),
        ("coarse@1024", False, None, 1.0),
        (f"coarse@{vggt_resolution}", False, vggt_resolution,
         img_load_resolution / vggt_resolution),
    ]
    result, track_scale = None, 1.0
    with torch.no_grad():
        for label, fine, res, sc in attempts:
            track_images = images if res is None else F.interpolate(
                images, size=(res, res), mode="bilinear", align_corners=False)
            try:
                result = _run_tracking(track_images, fine)
                track_scale = sc
            except torch.cuda.OutOfMemoryError:
                print(f"VGGT+BA: {label} tracking OOM — falling back")
                torch.cuda.empty_cache()
            if result is not None:
                break
        torch.cuda.empty_cache()
    if result is None:
        print("VGGT+BA: tracking OOM at all fallback levels")
        return None
    pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = result
    if track_scale != 1.0:
        pred_tracks = pred_tracks * track_scale

    intrinsic = intrinsic.copy()
    intrinsic[:, :2, :] *= scale  # 518 grid -> 1024 track space
    track_mask = pred_vis_scores > vis_thresh

    reconstruction, _ = batch_np_matrix_to_pycolmap(
        points_3d, extrinsic, intrinsic, pred_tracks, image_size,
        masks=track_mask, max_reproj_error=max_reproj_error,
        shared_camera=False, camera_type="SIMPLE_PINHOLE", points_rgb=points_rgb)
    if reconstruction is None:
        print("VGGT+BA: no valid reconstruction could be built from tracks")
        return None

    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, ba_options)

    # Map cameras/points2D from the 1024px square space back to original pixels.
    if crop_box is not None:
        ox_c, oy_c, side_c = crop_box
        from PIL import Image
        with Image.open(image_paths[0]) as _im0:
            orig_w, orig_h = _im0.size
        s = side_c / img_load_resolution
        # Exact inverse of crop->resize on pixel centers: px = (gx + 0.5) * s - 0.5 + ox
        off = np.array([ox_c - 0.5 * (1.0 - s), oy_c - 0.5 * (1.0 - s)])
    else:
        original_coords_np = original_coords.cpu().numpy()

    for image_id in reconstruction.images:
        pyimage = reconstruction.images[image_id]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_names[image_id - 1]

        params = np.array(pycamera.params, dtype=float)  # SIMPLE_PINHOLE: [f, cx, cy]
        if crop_box is not None:
            params[0] *= s
            params[1:] = params[1:] * s + off
            pycamera.params = params
            pycamera.width, pycamera.height = int(orig_w), int(orig_h)
            for point2D in pyimage.points2D:
                point2D.xy = point2D.xy * s + off
        else:
            # Demo's approximation: uniform rescale, principal point forced to center.
            real_wh = original_coords_np[image_id - 1, -2:]
            resize_ratio = max(real_wh) / img_load_resolution
            top_left = original_coords_np[image_id - 1, :2]
            params = params * resize_ratio
            params[1:] = real_wh / 2
            pycamera.params = params
            pycamera.width, pycamera.height = int(real_wh[0]), int(real_wh[1])
            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

    print(f"VGGT+BA reconstruction: {reconstruction.num_images()} images, "
          f"{reconstruction.num_points3D()} 3D points")
    return reconstruction
