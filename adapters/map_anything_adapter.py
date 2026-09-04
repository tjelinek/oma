"""Adapter for the Map Anything external repository.

This is the SOLE location in GloPose that imports Map Anything internals.
All other modules import from here.

Install: clone https://github.com/facebookresearch/map-anything into repositories/
and run `pip install -e . --no-deps` from there. Then install the per-backend
extras you actually need (see SUPPORTED_BACKENDS below).

Supported backends (pass via `backend` parameter):
  mapanything            - Meta's MapAnything (default)
  modular_dust3r         - Modular DUSt3R
  vggt                   - VGGT (feed-forward 3D)
  mast3r                 - MASt3R sparse global alignment
  must3r                 - MUSt3R
  dust3r                 - DUSt3R bundle adjustment
  moge                   - MoGe monocular geometry
  pi3 / pi3x             - Pi3 / Pi3X
  pow3r / pow3r_ba       - Pow3R, with optional bundle adjustment
  anycalib               - AnyCalib
  da3                    - Depth Anything 3
"""

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pycolmap


# Mapping: backend name → image normalization type. Required by load_images()
# and export_predictions_to_colmap(). Pulled from configs/model/*.yaml.
BACKEND_NORM_TYPES: dict[str, str] = {
    "mapanything": "dinov2",
    "modular_dust3r": "dust3r",
    "vggt": "identity",
    "mast3r": "dust3r",
    "must3r": "dust3r",
    "dust3r": "dust3r",
    "moge": "identity",
    "pi3": "identity",
    "pi3x": "identity",
    "pow3r": "dust3r",
    "pow3r_ba": "dust3r",
    "anycalib": "identity",
    "da3": "dinov2",
}

SUPPORTED_BACKENDS: tuple[str, ...] = tuple(BACKEND_NORM_TYPES.keys())


_loaded_models: dict[str, object] = {}


def load_map_anything_model(backend: str = "mapanything", device: str = "cuda"):
    """Load a Map Anything model by backend name. Caches across calls.

    For 'mapanything', loads from HuggingFace via MapAnything.from_pretrained.
    For all other backends, uses init_model_from_config which composes the
    Hydra config and downloads weights through the wrapper itself.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported Map Anything backend '{backend}'. "
            f"Valid options: {', '.join(SUPPORTED_BACKENDS)}"
        )

    cache_key = f"{backend}_{device}"
    if cache_key in _loaded_models:
        return _loaded_models[cache_key]

    if backend == "mapanything":
        from mapanything.models.mapanything import MapAnything
        model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    else:
        from mapanything.models import init_model_from_config
        model = init_model_from_config(backend, device=device)

    model.eval()
    _loaded_models[cache_key] = model
    return model


def _build_reconstruction_pycolmap4(
    points_3d: np.ndarray,
    points_rgb: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    image_names: Optional[list[str]] = None,
    camera_type: str = "PINHOLE",
    skip_point2d: bool = False,
) -> pycolmap.Reconstruction:
    """pycolmap-4.x replacement for MapAnything's build_colmap_reconstruction.

    Builds a Reconstruction from world point cloud + per-frame world2cam extrinsics
    and intrinsics, using the project's pose-setting helper (Rig->Frame->Image chain).
    Point2D backprojection is intentionally omitted (not needed for pose/rate/time
    metrics). Mirrors the vggt/pi3 adapters.

    Args:
        points_3d: (P, 3) world-coordinate points.
        points_rgb: (P, 3) uint8 colours.
        extrinsics: (N, 3, 4) world2cam [R|t] per frame.
        intrinsics: (N, 3, 3) per-frame K (PINHOLE).
    """
    from onboarding.colmap_utils import add_posed_image_to_reconstruction

    rec = pycolmap.Reconstruction()
    for i in range(len(points_3d)):
        rec.add_point3D(points_3d[i], pycolmap.Track(), points_rgb[i])

    n_frames = extrinsics.shape[0]
    if image_names is None:
        image_names = [f"image_{i + 1}.jpg" for i in range(n_frames)]

    for fidx in range(n_frames):
        K = intrinsics[fidx]
        cam_params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64)
        camera = pycolmap.Camera(
            model=camera_type, width=int(image_width), height=int(image_height),
            params=cam_params, camera_id=fidx + 1)
        rec.add_camera(camera)

        ext = extrinsics[fidx]  # (3, 4) world2cam
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(np.asarray(ext[:3, :3])), np.asarray(ext[:3, 3]))
        add_posed_image_to_reconstruction(
            rec, fidx + 1, fidx + 1, image_names[fidx], cam_from_world)

    return rec


def reconstruct_with_map_anything(
    image_paths: list[Path],
    image_names: list[str],
    device: str = "cuda",
    camera_K: Optional[torch.Tensor] = None,
    segmentation_paths: Optional[list[Path]] = None,
    backend: str = "mapanything",
    voxel_fraction: float = 0.01,
    max_points: int = 100_000,
    model=None,
) -> Optional[pycolmap.Reconstruction]:
    """Run Map Anything reconstruction and export to pycolmap.

    Uses Map Anything's built-in COLMAP export. Segmentation masks are applied
    to filter the predicted 3D points to object-only regions.

    Args:
        image_paths: Paths to input images (already background-masked if desired).
        image_names: COLMAP image names (e.g. '0.png', '5.png').
        device: Torch device.
        camera_K: Optional known camera intrinsics (3x3 tensor). Overrides
            Map Anything's predicted intrinsics in the output reconstruction.
        segmentation_paths: Per-frame segmentation masks. Points outside the
            mask are filtered from the reconstruction.
        backend: Map Anything backend model name. See SUPPORTED_BACKENDS.
        voxel_fraction: Fraction of scene extent used as voxel size for
            downsampling (passed to export_predictions_to_colmap).
        max_points: Maximum number of 3D points after filtering.
        model: Pre-loaded model. If None, loads via load_map_anything_model.

    Returns:
        pycolmap.Reconstruction with poses and 3D points, or None on failure.
    """
    from mapanything.utils.image import load_images
    from mapanything.utils import colmap_export as _colmap_export
    from mapanything.utils.colmap_export import export_predictions_to_colmap

    # open3d has no wheels for Python 3.13, so it is not installed in the env and
    # MapAnything's voxel downsampling (called unconditionally inside
    # export_predictions_to_colmap) raises ImportError. Voxel downsampling only
    # thins the exported point cloud — it does not affect camera poses, and the
    # point count is already capped below via max_points — so when open3d is
    # absent we replace the downsampler with a passthrough rather than failing the
    # whole reconstruction. (No effect on any reported pose / rate / timing metric.)
    if not getattr(_colmap_export, 'OPEN3D_AVAILABLE', True):
        _colmap_export.voxel_downsample_point_cloud = (
            lambda points, colors, *a, **k: (points, colors))

    # MapAnything's vendored build_colmap_reconstruction targets the pycolmap 3.x
    # API (Image(id=..., cam_from_world=...), image.registered, no Rig/Frame), which
    # is incompatible with the pycolmap 4.x in this environment (poses need a
    # Rig->Frame->Image chain). Replace it with a 4.x-native builder that uses the
    # project's add_posed_image_to_reconstruction. We keep MapAnything's own
    # output->arrays parsing (extrinsics/intrinsics/points) and only swap the final
    # COLMAP assembly. Point2D backprojection is skipped: Table-5 metrics use camera
    # poses (extrinsics) and Kabsch aligns camera centres, neither of which needs
    # per-point 2D observations or tracks.
    _colmap_export.build_colmap_reconstruction = _build_reconstruction_pycolmap4

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported Map Anything backend '{backend}'. "
            f"Valid options: {', '.join(SUPPORTED_BACKENDS)}"
        )

    if len(image_paths) < 2 and backend != "moge":
        print(f"Map Anything backend '{backend}' requires at least 2 images")
        return None

    if model is None:
        model = load_map_anything_model(backend, device)

    norm_type = BACKEND_NORM_TYPES[backend]
    views = load_images([str(p) for p in image_paths], norm_type=norm_type)

    with torch.no_grad():
        outputs = model.infer(
            views,
            memory_efficient_inference=True,
            use_amp=True,
        )

    # model.infer() runs under torch.inference_mode(), so every tensor in `outputs`
    # is an inference tensor that cannot be updated in place. The mask filtering and
    # intrinsics override below (and the COLMAP export) do in-place updates, which
    # raise "Inplace update to inference tensor outside InferenceMode is not allowed".
    # Clone to normal tensors first (clone outside inference mode yields a normal tensor).
    outputs = [
        {k: (v.clone() if torch.is_tensor(v) else v) for k, v in pred.items()}
        for pred in outputs
    ]

    # outputs is a list[dict], one entry per frame.
    if segmentation_paths is not None:
        _apply_segmentation_masks(outputs, segmentation_paths)

    if camera_K is not None:
        _override_intrinsics(outputs, camera_K)

    with tempfile.TemporaryDirectory(suffix="_mapanything_colmap") as tmpdir:
        reconstruction = export_predictions_to_colmap(
            outputs=outputs,
            processed_views=views,
            image_names=image_names,
            output_dir=tmpdir,
            voxel_fraction=voxel_fraction,
            data_norm_type=norm_type,
            save_ply=False,
            save_images=False,
        )

    if reconstruction is None:
        print(f"Map Anything ({backend}) COLMAP export returned None")
        return None

    n_points = len(reconstruction.points3D)
    if n_points > max_points:
        point_ids = list(reconstruction.points3D.keys())
        np.random.shuffle(point_ids)
        for pid in point_ids[max_points:]:
            reconstruction.delete_point3D(pid)

    print(
        f"Map Anything ({backend}) reconstruction: {len(reconstruction.images)} images, "
        f"{len(reconstruction.points3D)} 3D points"
    )
    return reconstruction


def _apply_segmentation_masks(
    outputs: list[dict], segmentation_paths: list[Path]
) -> None:
    """Filter predicted 3D points by per-frame segmentation masks.

    Modifies outputs[i]['mask'] in-place so the COLMAP export skips
    points outside the object.
    """
    from PIL import Image

    for fidx, seg_path in enumerate(segmentation_paths):
        if fidx >= len(outputs):
            break
        pred = outputs[fidx]
        mask = pred.get("mask")
        if mask is None:
            continue

        # mask shape: (1, H, W, 1) as torch tensor
        _, h, w, _ = mask.shape
        seg = np.array(
            Image.open(seg_path).convert("L").resize((w, h), Image.NEAREST)
        )
        seg_bool = torch.from_numpy(seg > 127).to(mask.device)

        if mask.dtype == torch.bool:
            mask[0, :, :, 0] &= seg_bool
        else:
            mask[0, :, :, 0] = mask[0, :, :, 0] * seg_bool.to(mask.dtype)


def _override_intrinsics(outputs: list[dict], camera_K: torch.Tensor) -> None:
    """Replace predicted intrinsics with known camera K, scaled to model resolution.

    Map Anything wrappers predict intrinsics at the model's working resolution
    (after load_images preprocessing). Known K is at the original image
    resolution, so it must be scaled by (model_w / true_w).
    """
    K_np = (
        camera_K.cpu().numpy()
        if isinstance(camera_K, torch.Tensor)
        else np.asarray(camera_K)
    )

    for pred in outputs:
        intrinsics = pred.get("intrinsics")
        if intrinsics is None:
            continue

        pts3d = pred.get("pts3d")
        if pts3d is None:
            continue
        _, pred_h, pred_w, _ = pts3d.shape

        true_shape = pred.get("true_shape")
        if true_shape is not None:
            true_h, true_w = int(true_shape[0, 0]), int(true_shape[0, 1])
        else:
            img = pred.get("img") or pred.get("img_no_norm")
            if img is not None:
                # img shape varies: (1, 3, H, W) for normalized, (1, H, W, 3) for img_no_norm.
                if img.shape[1] == 3 and img.ndim == 4:
                    true_h, true_w = img.shape[-2], img.shape[-1]
                else:
                    true_h, true_w = img.shape[1], img.shape[2]
            else:
                true_h, true_w = pred_h, pred_w

        sx = pred_w / true_w
        sy = pred_h / true_h

        K_scaled = np.eye(3, dtype=np.float32)
        K_scaled[0, 0] = K_np[0, 0] * sx
        K_scaled[1, 1] = K_np[1, 1] * sy
        K_scaled[0, 2] = K_np[0, 2] * sx
        K_scaled[1, 2] = K_np[1, 2] * sy

        if isinstance(intrinsics, torch.Tensor):
            intrinsics[0] = torch.from_numpy(K_scaled).to(intrinsics.device)
        else:
            intrinsics[0] = K_scaled
