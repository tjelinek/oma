"""Adapter for the SAM-3D-Objects external repository.

This is the SOLE location in GloPose that imports SAM3D internals.
All other modules import from here.

SAM3D reconstructs a 3D mesh from a single image + binary segmentation mask.
It outputs the mesh in canonical object space [-0.5, 0.5]^3 (Y-up) along with
a local-to-camera pose (rotation, translation, scale) in PyTorch3D convention
(X-left, Y-up, Z-forward).
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pycolmap
import torch
import trimesh

from onboarding.colmap_utils import add_posed_image_to_reconstruction, make_point2d_list

logger = logging.getLogger(__name__)

SAM3D_REPO = Path(__file__).resolve().parent.parent / 'repositories' / 'sam3d'


def _ensure_sam3d_on_path():
    sam3d_str = str(SAM3D_REPO)
    if sam3d_str not in sys.path:
        sys.path.insert(0, sam3d_str)
    # The Inference class lives in notebook/inference.py (not in the sam3d_objects package)
    notebook_str = str(SAM3D_REPO / 'notebook')
    if notebook_str not in sys.path:
        sys.path.insert(0, notebook_str)


def load_sam3d_model(checkpoint_path: str, device: str = 'cuda', compile: bool = False):
    """Load the SAM3D Inference pipeline.

    Args:
        checkpoint_path: Path to SAM3D checkpoint directory (contains pipeline.yaml).
        device: Torch device.
        compile: Whether to use torch.compile for performance.

    Returns:
        SAM3D Inference pipeline object.
    """
    _ensure_sam3d_on_path()
    from inference import Inference

    # checkpoint_path should point to the directory containing pipeline.yaml
    # (e.g. weights/SAM3D/hf/ after the HuggingFace download)
    config_file = str(Path(checkpoint_path) / 'pipeline.yaml')
    inference = Inference(config_file=config_file, compile=compile)
    return inference


def _pytorch3d_to_opencv_pose(rotation_wxyz: np.ndarray, translation: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Convert a local-to-camera pose from PyTorch3D convention to OpenCV convention.

    PyTorch3D: X-left, Y-up, Z-forward (right-handed)
    OpenCV:    X-right, Y-down, Z-forward (right-handed)

    The conversion negates X and Y axes: diag(-1, -1, 1) applied to camera frame.

    Args:
        rotation_wxyz: Quaternion (w, x, y, z) in PyTorch3D convention.
        translation: Translation (3,) in PyTorch3D convention.

    Returns:
        (rotation_matrix_3x3, translation_3) in OpenCV convention.
    """
    from scipy.spatial.transform import Rotation as R

    # Convert quaternion wxyz → xyzw for scipy
    quat_xyzw = np.array([rotation_wxyz[1], rotation_wxyz[2], rotation_wxyz[3], rotation_wxyz[0]])
    rot_matrix_pt3d = R.from_quat(quat_xyzw).as_matrix()

    # PyTorch3D uses row-vector convention: p_cam = p_local @ R^T + t
    # Convert to column-vector convention: p_cam = R @ p_local + t
    rot_matrix_pt3d = rot_matrix_pt3d.T

    # Flip X and Y: R_opencv = diag(-1,-1,1) @ R_pt3d, t_opencv = diag(-1,-1,1) @ t_pt3d
    flip = np.diag([-1.0, -1.0, 1.0])
    rot_matrix_opencv = flip @ rot_matrix_pt3d
    translation_opencv = flip @ translation

    return rot_matrix_opencv, translation_opencv


def reconstruct_with_sam3d(
    image_path: Path,
    segmentation_path: Path,
    image_name: str,
    device: str = 'cuda',
    camera_K: Optional[torch.Tensor] = None,
    checkpoint_path: str = 'weights/SAM3D/',
    output_dir: Optional[Path] = None,
    seed: int = 42,
    model=None,
) -> tuple[Optional[trimesh.Trimesh], Optional[pycolmap.Reconstruction]]:
    """Run SAM3D single-image 3D reconstruction.

    Args:
        image_path: Path to the input RGB image.
        segmentation_path: Path to the binary segmentation mask (used as prompt).
        image_name: COLMAP image name (e.g. '5.png') — must match DataGraph.
        device: Torch device.
        camera_K: Optional known camera intrinsics (3x3 tensor).
        checkpoint_path: Path to SAM3D checkpoint directory.
        output_dir: Directory to save raw mesh files. If None, meshes are not saved.
        seed: Random seed for reproducibility.
        model: Pre-loaded SAM3D model. If None, loads from checkpoint_path.

    Returns:
        (mesh, reconstruction) tuple. mesh is a trimesh.Trimesh object with the
        reconstructed 3D model. reconstruction is a pycolmap.Reconstruction with
        all mesh vertices as 3D points and the SAM3D-estimated camera pose.
        Either can be None on failure.
    """
    from PIL import Image

    _ensure_sam3d_on_path()

    if model is None:
        model = load_sam3d_model(checkpoint_path, device)

    # Load image and mask
    image = np.array(Image.open(image_path).convert('RGB'))
    mask = np.array(Image.open(segmentation_path).convert('L'))
    mask = (mask > 127).astype(np.uint8) * 255

    img_h, img_w = image.shape[:2]

    try:
        output = model(image, mask, seed=seed)
    except Exception as e:
        logger.error(f"SAM3D inference failed: {e}")
        return None, None

    # Extract mesh (GLB post-processed trimesh)
    mesh = output.get('glb')
    if mesh is None:
        logger.error("SAM3D produced no mesh output")
        return None, None

    # Extract pose (local-to-camera in PyTorch3D convention)
    rotation_wxyz = output['rotation'].squeeze().cpu().numpy()   # (4,) wxyz
    translation = output['translation'].squeeze().cpu().numpy()  # (3,)
    scale = output['scale'].squeeze().cpu().numpy()              # (3,)

    logger.info(f"SAM3D pose — rotation(wxyz): {rotation_wxyz}, "
                f"translation: {translation}, scale: {scale}")

    # Save raw mesh
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        glb_path = output_dir / 'sam3d_mesh.glb'
        mesh.export(str(glb_path))
        logger.info(f"Saved GLB mesh to {glb_path}")

        ply_path = output_dir / 'sam3d_mesh.ply'
        mesh.export(str(ply_path))
        logger.info(f"Saved PLY mesh to {ply_path}")

        # Also save Gaussian splat if available
        gs = output.get('gs')
        if gs is not None:
            gs_path = output_dir / 'sam3d_gaussians.ply'
            gs.save_ply(str(gs_path))
            logger.info(f"Saved Gaussian splat to {gs_path}")

    # Get mesh vertices and colors
    vertices = np.array(mesh.vertices)  # (N, 3) in canonical object space
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        colors = np.array(mesh.visual.vertex_colors)[:, :3]  # (N, 3) RGB uint8
    else:
        colors = np.full((len(vertices), 3), 128, dtype=np.uint8)

    n_vertices = len(vertices)
    if n_vertices == 0:
        logger.error("SAM3D mesh has no vertices")
        return mesh, None

    logger.info(f"SAM3D mesh: {n_vertices} vertices, "
                f"bbox min={vertices.min(axis=0)}, max={vertices.max(axis=0)}")

    # Convert SAM3D l2c pose from PyTorch3D to OpenCV convention
    rot_opencv, trans_opencv = _pytorch3d_to_opencv_pose(rotation_wxyz, translation)

    # Build cam_from_world (object canonical frame = world frame)
    # SAM3D's l2c maps canonical vertices to camera frame, incorporating scale.
    # We need to account for scale: p_cam = scale * (R @ p_local) + t
    # So cam_from_world rotation = R, translation = t, and we scale the world points.
    # Alternatively, bake scale into the world→cam transform:
    # cam_from_world = [s*R | t] applied as p_cam = s*R @ p_world + t
    #
    # pycolmap Rigid3d is a pure rigid transform (no scale), so we scale the vertices instead.
    # The "world" frame uses scaled canonical coordinates.
    mean_scale = float(np.mean(scale))
    vertices_scaled = vertices * mean_scale

    cam_from_world = pycolmap.Rigid3d(
        pycolmap.Rotation3d(rot_opencv),
        trans_opencv)

    # Build pycolmap.Reconstruction
    reconstruction = pycolmap.Reconstruction()

    # Camera
    if camera_K is not None:
        K_np = camera_K.cpu().numpy() if isinstance(camera_K, torch.Tensor) else camera_K
        fx, fy = K_np[0, 0], K_np[1, 1]
        cx, cy = K_np[0, 2], K_np[1, 2]
    else:
        # Default intrinsics based on image size (approximate)
        fx = fy = max(img_w, img_h)
        cx, cy = img_w / 2.0, img_h / 2.0

    cam_params = np.array([fx, fy, cx, cy])
    camera = pycolmap.Camera(
        model='PINHOLE', width=img_w, height=img_h,
        params=cam_params, camera_id=1)
    reconstruction.add_camera(camera)

    # Add 3D points and compute 2D projections
    # Project: p_2d = K @ (R @ p_world + t)
    R_mat = rot_opencv
    t_vec = trans_opencv

    points2D_list = []
    point2D_idx = 0

    for idx in range(n_vertices):
        p_world = vertices_scaled[idx]
        p_cam = R_mat @ p_world + t_vec

        if p_cam[2] <= 0:
            # Behind camera — add point without observation
            reconstruction.add_point3D(p_world, pycolmap.Track(), colors[idx])
            continue

        # Project to image
        px = fx * p_cam[0] / p_cam[2] + cx
        py = fy * p_cam[1] / p_cam[2] + cy

        point3D_id = idx + 1
        reconstruction.add_point3D(p_world, pycolmap.Track(), colors[idx])

        # Only add observation if projection falls within image bounds
        if 0 <= px < img_w and 0 <= py < img_h:
            point2D = pycolmap.Point2D(np.array([px, py]), point3D_id)
            points2D_list.append(point2D)

            track = reconstruction.points3D[point3D_id].track
            track.add_element(1, point2D_idx)  # image_id=1, point2D_idx
            point2D_idx += 1

    # Add image with pose
    add_posed_image_to_reconstruction(
        reconstruction, image_id=1, camera_id=1, name=image_name,
        cam_from_world=cam_from_world,
        points2D=make_point2d_list(points2D_list))

    logger.info(f"SAM3D reconstruction: 1 image, {n_vertices} 3D points, "
                f"{len(points2D_list)} visible projections")

    return mesh, reconstruction
