import copy
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pycolmap
import torch
from kornia.geometry import Se3, Quaternion
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# pycolmap 4.0 introduces Rig→Frame→Image hierarchy; 3.x has Image.set_cam_from_world()
PYCOLMAP_MAJOR = int(pycolmap.__version__.split('.')[0])
PYCOLMAP4 = PYCOLMAP_MAJOR >= 4


def make_point2d_list(points: list | None = None):
    """Create a Point2D list container, compatible with both pycolmap 3.x and 4.x."""
    cls = pycolmap.Point2DList if hasattr(pycolmap, 'Point2DList') else pycolmap.ListPoint2D
    return cls(points) if points is not None else cls()


def carve_reconstruction_by_masks(reconstruction: pycolmap.Reconstruction,
                                  name_to_seg: Dict[str, Path],
                                  eps: float = 1e-6,
                                  max_remove_fraction: float = 0.95,
                                  min_bg_views: int = 1) -> Tuple[pycolmap.Reconstruction, int, int]:
    """Visual-hull carve: delete 3D points that project onto background in any view.

    A point is removed if, in any registered image where it projects in front of
    the camera and within image bounds, it falls on a background pixel of that
    image's segmentation mask. Points are projected through each image's own
    camera (so GT intrinsics if those were set on the reconstruction) and sampled
    against the on-disk masks — the same geometry as
    ``visualizations.rerun_utils.log_colmap_point_projections``, so this directly
    removes the points that show up red there.

    Works for any neural reconstruction pipeline that yields a
    ``pycolmap.Reconstruction`` (VGGT, Mast3r, MapAnything, …). The per-source-frame
    mask filtering done inside each adapter is the cheap first pass; this is the
    multi-view consistency pass on top.

    Args:
        reconstruction: reconstruction to carve (modified in place).
        name_to_seg: maps ``image.name`` → segmentation mask path on disk.
        eps: minimum positive depth for a point to count as in front of a camera.
        max_remove_fraction: safety valve — if carving would remove more than this
            fraction of points, the reconstruction's camera/mask pixel spaces are
            almost certainly inconsistent (e.g. cameras built at a different
            resolution than the masks), so skip carving and leave it untouched
            rather than destroying the reconstruction.

    Returns:
        (reconstruction, n_removed, n_kept)
    """
    import imageio.v3 as iio

    if len(reconstruction.points3D) == 0:
        return reconstruction, 0, 0

    pids = list(reconstruction.points3D.keys())
    pts = np.stack([reconstruction.points3D[p].xyz for p in pids], axis=0)  # (N, 3)
    # Count, per point, the views where it projects onto background; a point is
    # carved once that count reaches min_bg_views (1 = the strict visual-hull
    # rule; >1 tolerates isolated mask errors in single views).
    bg_views = np.zeros(len(pids), dtype=np.int32)

    for image in reconstruction.images.values():
        seg_path = name_to_seg.get(image.name)
        if seg_path is None or not Path(seg_path).exists():
            continue
        seg = iio.imread(seg_path)
        if seg.ndim == 3:
            seg = seg[..., 0]
        h, w = seg.shape[:2]

        camera = reconstruction.cameras[image.camera_id]
        cam_from_world = image.cam_from_world()
        R = cam_from_world.rotation.matrix()
        t = cam_from_world.translation

        p_cam = pts @ R.T + t                       # (N, 3) camera coords
        z = p_cam[:, 2]
        in_front = z > eps
        zc = np.where(in_front, z, 1.0)

        params = np.asarray(camera.params, dtype=np.float64)
        if params.shape[0] == 4:                    # PINHOLE: [fx, fy, cx, cy]
            fx, fy, cx, cy = params
            u = fx * p_cam[:, 0] / zc + cx
            v = fy * p_cam[:, 1] / zc + cy
        else:                                       # generic fallback (any camera model)
            uv = np.full((len(pids), 2), -1.0)
            idxf = np.flatnonzero(in_front)
            if len(idxf):
                uv[idxf] = np.stack([camera.img_from_cam(p) for p in p_cam[idxf]], axis=0)
            u, v = uv[:, 0], uv[:, 1]

        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        in_bounds = in_front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)

        inside = np.zeros(len(pids), dtype=bool)
        ib = np.flatnonzero(in_bounds)
        if len(ib):
            inside[ib] = seg[np.clip(vi[ib], 0, h - 1), np.clip(ui[ib], 0, w - 1)] > 127

        # Count points seen as background in this view.
        bg_views += (in_bounds & ~inside).astype(np.int32)

    survive = bg_views < min_bg_views
    remove_fraction = float((~survive).mean())
    if remove_fraction > max_remove_fraction:
        logger.warning(
            "Mask carve would remove %.1f%% of points (> %.0f%%); skipping — "
            "camera and mask pixel spaces are likely inconsistent.",
            100.0 * remove_fraction, 100.0 * max_remove_fraction)
        return reconstruction, 0, len(pids)

    n_removed = 0
    for keep, pid in zip(survive, pids):
        if not keep:
            reconstruction.delete_point3D(pid)
            n_removed += 1

    return reconstruction, n_removed, int(survive.sum())


def add_posed_image_to_reconstruction(
    rec: pycolmap.Reconstruction,
    image_id: int,
    camera_id: int,
    name: str,
    cam_from_world: pycolmap.Rigid3d,
    points2D=None,
) -> pycolmap.Image:
    """Add an image with its pose to a reconstruction.

    Handles both pycolmap 3.x (Image.set_cam_from_world) and
    4.x (Rig→Frame→Image chain) transparently.
    """
    if points2D is not None:
        img = pycolmap.Image(name=name, points2D=points2D, camera_id=camera_id, image_id=image_id)
    else:
        img = pycolmap.Image(image_id=image_id, camera_id=camera_id, name=name)

    if PYCOLMAP4:
        cam = rec.cameras[camera_id]

        # One rig per camera (reused across images sharing the same camera)
        rig_id = camera_id
        if rig_id not in rec.rigs:
            rig = pycolmap.Rig(rig_id=rig_id)
            rig.add_ref_sensor(cam.sensor_id)
            rec.add_rig(rig)

        # Create frame, link to image, add to reconstruction, then set pose
        frame_id = image_id  # 1:1 mapping between frames and images
        img.frame_id = frame_id
        frame = pycolmap.Frame(frame_id=frame_id, rig_id=rig_id)
        frame.add_data_id(img.data_id)
        rec.add_frame(frame)
        rec.frames[frame_id].set_cam_from_world(camera_id, cam_from_world)

        rec.add_image(img)
        rec.register_frame(frame_id)
    else:
        img.set_cam_from_world(camera_id, cam_from_world)
        rec.add_image(img)

    return img


def create_database_cache(database: pycolmap.Database):
    """Create a DatabaseCache, compatible with both pycolmap 3.x and 4.x."""
    if PYCOLMAP4:
        cache_opts = pycolmap.DatabaseCacheOptions()
        cache_opts.min_num_matches = 0
        return pycolmap.DatabaseCache.create(database, cache_opts)
    else:
        return pycolmap.DatabaseCache().create(database, 0, False, set())


def get_image_Se3_world2cam(image: pycolmap.Image, device: str) -> Se3:
    image_world2cam: pycolmap.Rigid3d = image.cam_from_world()
    image_t_cam = torch.tensor(image_world2cam.translation).to(device).to(torch.float)
    image_q_cam_xyzw = torch.tensor(image_world2cam.rotation.quat[[3, 0, 1, 2]]).to(device).to(torch.float)
    Se3_image_world2cam = Se3(Quaternion(image_q_cam_xyzw), image_t_cam)

    return Se3_image_world2cam


def world2cam_from_reconstruction(reconstruction: pycolmap.Reconstruction) -> Dict[int, Se3]:
    poses = {}
    for image_id, image in reconstruction.images.items():
        Se3_world2cam = get_image_Se3_world2cam(image, 'cpu')
        poses[image_id] = Se3_world2cam
    return poses


def merge_two_databases(colmap_db1_path: Path, colmap_db2_path: Path, merged_db_path: Path, db1_imgs_prefix="db1_",
                        db2_imgs_prefix="db2_") \
        -> Tuple[Dict[str, str], Dict[str, str]]:
    db1 = pycolmap.Database.open(str(colmap_db1_path))

    tmp_db1_path = merged_db_path.parent / 'tmp_db1.db'
    tmp_db2_path = merged_db_path.parent / 'tmp_db2.db'

    shutil.copy(colmap_db1_path, tmp_db1_path)
    shutil.copy(colmap_db2_path, tmp_db2_path)

    tmp_db1 = pycolmap.Database.open(str(tmp_db1_path))
    tmp_db2 = pycolmap.Database.open(str(tmp_db2_path))

    def rename_db_imgs(tmp_db: pycolmap.Database, db_imgs_prefix: str):
        db_rename_dict = {}
        for image in tmp_db.read_all_images():
            old_name = image.name
            new_name = db_imgs_prefix + image.name
            image.name = new_name
            tmp_db.update_image(image)

            db_rename_dict[old_name] = new_name

        return db_rename_dict

    db1_rename_dict = rename_db_imgs(tmp_db1, db1_imgs_prefix)
    db2_rename_dict = rename_db_imgs(tmp_db2, db2_imgs_prefix)

    merged_db = pycolmap.Database.open(str(merged_db_path))
    pycolmap.Database.merge(tmp_db1, tmp_db2, merged_db)

    tmp_db1_path.unlink()
    tmp_db2_path.unlink()

    db1.close()
    merged_db.close()

    return db1_rename_dict, db2_rename_dict


def merge_colmap_reconstructions(
        rec1: pycolmap.Reconstruction, rec2: pycolmap.Reconstruction,
        rec1_id_map: Dict[int, int], rec2_id_map: Dict[int, int],
        rec1_name_map: Dict[str, str], rec2_name_map: Dict[str, str],
) -> pycolmap.Reconstruction:
    """Merge two COLMAP reconstructions, remapping image IDs and names to match a merged DB.

    Args:
        rec1, rec2: Source reconstructions.
        rec1_id_map, rec2_id_map: old_colmap_image_id → new_merged_image_id for each reconstruction.
        rec1_name_map, rec2_name_map: old_image_name → new_prefixed_image_name for each reconstruction.
    """
    merged = pycolmap.Reconstruction()
    next_camera_id = 0

    def _add_reconstruction(rec, id_map, name_map):
        nonlocal next_camera_id
        camera_id_remap = {}

        for old_cam_id, camera in rec.cameras.items():
            new_cam_id = next_camera_id
            next_camera_id += 1
            camera.camera_id = new_cam_id
            merged.add_camera(camera)
            camera_id_remap[old_cam_id] = new_cam_id

        for old_image_id, image in rec.images.items():
            new_image_id = id_map[old_image_id]
            new_name = name_map[image.name]
            new_camera_id = camera_id_remap[image.camera_id]

            clean_points2D = make_point2d_list()
            for point2D in image.points2D:
                clean_points2D.append(pycolmap.Point2D(xy=point2D.xy))

            add_posed_image_to_reconstruction(
                merged, new_image_id, new_camera_id, new_name,
                image.cam_from_world(), points2D=clean_points2D)

        for point3D in rec.points3D.values():
            new_track = pycolmap.Track()
            for elem in point3D.track.elements:
                new_track.add_element(id_map[elem.image_id], elem.point2D_idx)
            merged.add_point3D(point3D.xyz, new_track, point3D.color)

    _add_reconstruction(rec1, rec1_id_map, rec1_name_map)
    _add_reconstruction(rec2, rec2_id_map, rec2_name_map)

    return merged


def _apply_rigid_to_reconstruction(
        rec: pycolmap.Reconstruction, rotation_matrix: np.ndarray, translation: np.ndarray,
) -> pycolmap.Reconstruction:
    """Apply a rigid transform (R, t) to a reconstruction via pycolmap Sim3d (scale=1)."""
    rec_out = copy.deepcopy(rec)
    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()  # scipy returns [x, y, z, w]
    sim3d = pycolmap.Sim3d(
        scale=1.0,
        rotation=pycolmap.Rotation3d(np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])),
        translation=translation,
    )
    rec_out.transform(sim3d)
    return rec_out


def _procrustes_svd(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve for rigid (R, t) that minimizes ||R @ source.T + t - target.T||.

    Args:
        source: (N, 3) matched points from source.
        target: (N, 3) matched points from target.

    Returns:
        R: (3, 3) rotation matrix.
        t: (3,) translation vector.
    """
    centroid_src = source.mean(axis=0)
    centroid_tgt = target.mean(axis=0)

    src_centered = source - centroid_src
    tgt_centered = target - centroid_tgt

    H = src_centered.T @ tgt_centered  # (3, 3)
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    t = centroid_tgt - R @ centroid_src

    return R, t


def align_reconstructions_icp(
        rec_target: pycolmap.Reconstruction,
        rec_source: pycolmap.Reconstruction,
        max_correspondence_distance: float | None = None,
        max_iterations: int = 10,
        skip_centroid_prewarp: bool = False,
) -> Tuple[pycolmap.Reconstruction, dict]:
    """Align rec_source to rec_target using centroid prewarp + gated Procrustes.

    Both reconstructions should already be roughly aligned (e.g. via GT Kabsch).
    This refines the residual misalignment:
    1. Translate rec_source so its centroid matches rec_target's centroid.
    2. Iteratively: find mutual nearest neighbors within a distance gate,
       solve rigid transform via SVD Procrustes on those pairs only.

    Args:
        rec_target: Reference reconstruction (kept fixed).
        rec_source: Source reconstruction (will be transformed).
        max_correspondence_distance: Max NN distance for a pair to count as inlier.
            If None, uses 2× median NN distance after centroid prewarp.
        max_iterations: Number of Procrustes refinement iterations.

    Returns:
        rec_source_aligned: Deep copy of rec_source with alignment applied.
        info: Dict with alignment metrics.
    """
    points_target = np.array([p.xyz for p in rec_target.points3D.values()])
    points_source = np.array([p.xyz for p in rec_source.points3D.values()])

    if len(points_target) < 10 or len(points_source) < 10:
        print(f"[align] Too few points (target={len(points_target)}, source={len(points_source)}), skipping")
        return copy.deepcopy(rec_source), {
            'num_points_target': len(points_target), 'num_points_source': len(points_source),
            'num_inliers': 0, 'converged': False,
        }

    # Step 1: Centroid prewarp (skip when input is already roughly aligned, e.g. after Procrustes)
    print(f"[align] points: target={len(points_target)}, source={len(points_source)}")
    if skip_centroid_prewarp:
        centroid_shift = np.zeros(3)
        print(f"[align] centroid prewarp: SKIPPED (input already aligned)")
    else:
        centroid_target = points_target.mean(axis=0)
        centroid_source = points_source.mean(axis=0)
        centroid_shift = centroid_target - centroid_source
        print(f"[align] centroid shift: {np.linalg.norm(centroid_shift):.4f}")

    current_source = points_source + centroid_shift

    # Step 2: Iterative gated Procrustes
    R_total = np.eye(3)
    t_total = centroid_shift.copy()

    for iteration in range(max_iterations):
        tree = KDTree(points_target)
        distances, indices = tree.query(current_source)

        # Distance gate
        if max_correspondence_distance is None:
            gate = 2.0 * np.median(distances)
        else:
            gate = max_correspondence_distance

        inlier_mask = distances < gate
        num_inliers = inlier_mask.sum()

        if num_inliers < 10:
            print(f"[align] iter {iteration}: only {num_inliers} inliers, stopping")
            break

        matched_source = current_source[inlier_mask]
        matched_target = points_target[indices[inlier_mask]]

        R_iter, t_iter = _procrustes_svd(matched_source, matched_target)

        # Apply incremental transform
        current_source = (R_iter @ current_source.T).T + t_iter

        # Accumulate: new_total = R_iter @ old_total, t = R_iter @ t_old + t_iter
        R_total = R_iter @ R_total
        t_total = R_iter @ t_total + t_iter

        mean_dist = np.mean(distances[inlier_mask])
        rot_angle = np.degrees(np.arccos(np.clip((np.trace(R_iter) - 1) / 2, -1, 1)))
        print(f"[align] iter {iteration}: inliers={num_inliers}, mean_dist={mean_dist:.4f}, "
              f"rot={rot_angle:.4f}°, trans={np.linalg.norm(t_iter):.4f}")

        if rot_angle < 0.01 and np.linalg.norm(t_iter) < 1e-5:
            print(f"[align] converged at iteration {iteration}")
            break

    # Apply total transform to reconstruction
    rec_source_aligned = _apply_rigid_to_reconstruction(rec_source, R_total, t_total)

    total_rot_deg = np.degrees(np.arccos(np.clip((np.trace(R_total) - 1) / 2, -1, 1)))
    total_trans = np.linalg.norm(t_total)
    print(f"[align] total: rotation={total_rot_deg:.4f}°, translation={total_trans:.4f}"
          f" (skip_centroid_prewarp={skip_centroid_prewarp})")

    info = {
        'num_points_target': len(points_target),
        'num_points_source': len(points_source),
        'num_inliers': int(num_inliers),
        'rotation_deg': total_rot_deg,
        'translation_norm': total_trans,
        'transform_R': R_total,
        'transform_t': t_total,
        'converged': True,
    }

    return rec_source_aligned, info


def _build_2d_to_3d_index(image: pycolmap.Image, reconstruction: pycolmap.Reconstruction
                           ) -> Tuple[np.ndarray, np.ndarray, KDTree | None]:
    """Build a KDTree of 2D projections for an image's 3D-linked points.

    Returns:
        pts_2d: (M, 2) array of 2D coordinates of points with 3D links.
        pts_3d: (M, 3) array of corresponding 3D world coordinates.
        tree: KDTree built on pts_2d, or None if no linked points.
    """
    pts_2d_list = []
    pts_3d_list = []
    for point2D in image.points2D:
        if point2D.has_point3D():
            pts_2d_list.append(point2D.xy)
            pts_3d_list.append(reconstruction.point3D(point2D.point3D_id).xyz)
    if not pts_2d_list:
        return np.empty((0, 2)), np.empty((0, 3)), None
    pts_2d = np.array(pts_2d_list)
    pts_3d = np.array(pts_3d_list)
    return pts_2d, pts_3d, KDTree(pts_2d)


def _select_diverse_images(images: List[pycolmap.Image], reconstruction: pycolmap.Reconstruction,
                           n: int) -> List[pycolmap.Image]:
    """Select n images: top half by 3D point count, then farthest point sampling for viewpoint diversity.

    1. Sort all images by number of 3D-linked points (descending).
    2. Take the top 2*n candidates (images with most structure).
    3. From those, greedily pick n images with maximally spread camera centers
       (farthest point sampling).
    """
    # Sort by number of 3D-linked points
    imgs_with_counts = []
    for img in images:
        n_3d = sum(1 for p in img.points2D if p.has_point3D())
        center = img.cam_from_world().inverse().translation
        imgs_with_counts.append((img, n_3d, np.array(center)))
    imgs_with_counts.sort(key=lambda x: x[1], reverse=True)

    # Take top 2*n candidates
    candidates = imgs_with_counts[:2 * n]
    print(f"[match-align] Top {len(candidates)} candidates by 3D points: "
          f"{[(c[0].name, c[1]) for c in candidates]}")

    if len(candidates) <= n:
        return [c[0] for c in candidates]

    # Farthest point sampling on camera centers
    centers = np.array([c[2] for c in candidates])
    selected_idx = [0]  # Start with the image with most 3D points
    for _ in range(n - 1):
        selected_centers = centers[selected_idx]
        # For each candidate, compute min distance to any selected center
        min_dists = np.full(len(centers), np.inf)
        for si in selected_idx:
            dists = np.linalg.norm(centers - centers[si], axis=1)
            min_dists = np.minimum(min_dists, dists)
        # Zero out already selected
        for si in selected_idx:
            min_dists[si] = -1.0
        # Pick the candidate farthest from all selected
        best = int(np.argmax(min_dists))
        selected_idx.append(best)

    selected = [candidates[i][0] for i in selected_idx]
    print(f"[match-align] After viewpoint diversity sampling: "
          f"{[(candidates[i][0].name, candidates[i][1]) for i in selected_idx]}")
    return selected


def find_3d_correspondences_via_matching(
        rec_target: pycolmap.Reconstruction,
        rec_source: pycolmap.Reconstruction,
        match_provider: 'MatchingProvider',
        target_images_dir: Path,
        source_images_dir: Path,
        target_segs_dir: Path,
        source_segs_dir: Path,
        sample_size: int,
        certainty_threshold: float,
        reliability_threshold: float,
        max_pairs: int = 5,
        max_2d_distance: float = 5.0,
        black_background: bool = False,
        device: str = 'cuda',
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Find 3D-3D correspondences between two reconstructions via 2D image matching.

    For each of up to `max_pairs` randomly selected source images, matches against
    all target images, picks the best match (by reliability), and extracts 3D-3D
    correspondences by looking up the nearest 3D-linked 2D point on each side.

    Args:
        rec_target: Reference (fixed) reconstruction.
        rec_source: Source reconstruction to be aligned.
        match_provider: Dense or sparse matcher (same type as onboarding filter_matcher).
        target_images_dir: Directory with target keyframe images ({colmap_image_name_stem}_image.png).
        source_images_dir: Directory with source keyframe images.
        target_segs_dir: Directory with target segmentations ({colmap_image_name_stem}_seg.png).
        source_segs_dir: Directory with source segmentations.
        sample_size: Number of match samples per image pair.
        certainty_threshold: Per-match certainty threshold for reliability computation.
        reliability_threshold: Minimum reliability to consider a pair acceptable.
        max_pairs: Maximum number of source images to try.
        max_2d_distance: Maximum pixel distance for 2D-to-3D lookup.
        device: PyTorch device string.

    Returns:
        source_pts_3d: (N, 3) matched 3D points in source frame.
        target_pts_3d: (N, 3) matched 3D points in target frame.
        match_pair_info: List of dicts with per-pair diagnostics.
    """
    from onboarding.frame_filter import compute_matching_reliability
    from onboarding.reconstruction import _load_image_and_segmentation

    # Select source images: top by 3D point count, then subsample for viewpoint diversity
    source_images = list(rec_source.images.values())
    print(f"[match-align] Source reconstruction: {len(source_images)} images, "
          f"{len(rec_source.points3D)} 3D points")
    print(f"[match-align] Target reconstruction: {len(rec_target.images)} images, "
          f"{len(rec_target.points3D)} 3D points")
    if len(source_images) > max_pairs:
        source_images = _select_diverse_images(source_images, rec_source, max_pairs)
    print(f"[match-align] Selected {len(source_images)} source images: "
          f"{[img.name for img in source_images]}")

    target_images = list(rec_target.images.values())

    # Pre-build 2D→3D indices for all target images
    target_indices = {}
    for tgt_img in target_images:
        pts_2d, pts_3d, tree = _build_2d_to_3d_index(tgt_img, rec_target)
        if tree is not None:
            target_indices[tgt_img.image_id] = (pts_2d, pts_3d, tree)
    print(f"[match-align] Target images with 3D points: {len(target_indices)}/{len(target_images)}")
    print(f"[match-align] Certainty threshold: {certainty_threshold}, sample size: {sample_size}")

    all_source_3d = []
    all_target_3d = []
    seen_pairs = set()
    match_pair_info = []  # Collected for rerun visualization

    for src_img in source_images:
        # Build 2D→3D index for this source image (with point3D_ids for dedup)
        src_pts_2d, src_pts_3d, src_tree = _build_2d_to_3d_index(src_img, rec_source)
        if src_tree is None:
            print(f"[match-align]   Source {src_img.name}: no 3D-linked points, skipping")
            continue
        src_point3d_ids = [p.point3D_id for p in src_img.points2D if p.has_point3D()]
        print(f"[match-align]   Source {src_img.name}: {len(src_pts_2d)} points with 3D links")

        # Load source image and segmentation
        src_stem = Path(src_img.name).stem
        src_img_path = source_images_dir / f'{src_stem}_image.png'
        src_seg_path = source_segs_dir / f'{src_stem}_seg.png'
        if not src_img_path.exists() or not src_seg_path.exists():
            print(f"[match-align]   Source {src_img.name}: MISSING files "
                  f"(img={src_img_path.exists()}, seg={src_seg_path.exists()})")
            continue
        src_image_tensor, src_seg_tensor, _, _ = _load_image_and_segmentation(
            src_img_path, src_seg_path, device)
        src_seg_tensor = src_seg_tensor.squeeze()
        if black_background:
            src_image_tensor = src_image_tensor * src_seg_tensor
        print(f"[match-align]   Source {src_img.name}: loaded, shape={src_image_tensor.shape}, "
              f"seg_shape={src_seg_tensor.shape}, seg_fg={float(src_seg_tensor.sum()):.0f}px")

        # Match against all target images, pick best by reliability
        best_reliability = -1.0
        best_target_img = None
        best_match_pts = None
        best_num_matches = 0
        best_median_cert = 0.0

        for tgt_img in target_images:
            if tgt_img.image_id not in target_indices:
                continue

            tgt_stem = Path(tgt_img.name).stem
            tgt_img_path = target_images_dir / f'{tgt_stem}_image.png'
            tgt_seg_path = target_segs_dir / f'{tgt_stem}_seg.png'
            if not tgt_img_path.exists() or not tgt_seg_path.exists():
                print(f"[match-align]     Target {tgt_img.name}: MISSING files")
                continue
            tgt_image_tensor, tgt_seg_tensor, _, _ = _load_image_and_segmentation(
                tgt_img_path, tgt_seg_path, device)
            tgt_seg_tensor = tgt_seg_tensor.squeeze()
            if black_background:
                tgt_image_tensor = tgt_image_tensor * tgt_seg_tensor

            # Get matches
            src_pts_xy, dst_pts_xy, certainty = match_provider.get_source_target_points(
                src_image_tensor, tgt_image_tensor, sample=sample_size,
                source_image_segmentation=src_seg_tensor, target_image_segmentation=tgt_seg_tensor,
                as_int=True, zero_certainty_outside_segmentation=True, only_foreground_matches=True)

            num_matches = len(src_pts_xy)
            if num_matches == 0:
                print(f"[match-align]     vs target {tgt_img.name}: 0 matches")
                continue

            reliability = compute_matching_reliability(
                src_pts_xy, certainty, src_seg_tensor,
                certainty_threshold, min_num_of_certain_matches=0)

            print(f"[match-align]     vs target {tgt_img.name}: {num_matches} matches, "
                  f"reliability={reliability:.3f} (cert: min={certainty.min():.3f}, "
                  f"med={certainty.median():.3f}, max={certainty.max():.3f})")

            if reliability > best_reliability:
                best_reliability = reliability
                best_target_img = tgt_img
                best_match_pts = (src_pts_xy.cpu().numpy(), dst_pts_xy.cpu().numpy())
                best_num_matches = num_matches
                best_median_cert = float(certainty.median())

        if best_target_img is None:
            print(f"[match-align]   Source {src_img.name}: no matches found")
            continue

        print(f"[match-align]   Source {src_img.name} -> Target {best_target_img.name} "
              f"(reliability={best_reliability:.3f}, {best_num_matches} matches)")

        # Extract 3D-3D correspondences from the best pair
        tgt_pts_2d, tgt_pts_3d, tgt_tree = target_indices[best_target_img.image_id]
        tgt_point3d_ids = [p.point3D_id for p in best_target_img.points2D if p.has_point3D()]
        match_src_2d, match_tgt_2d = best_match_pts

        # Query source 2D→3D
        src_dists, src_indices = src_tree.query(match_src_2d)
        # Query target 2D→3D
        tgt_dists, tgt_indices = tgt_tree.query(match_tgt_2d)

        # Accept only matches where both sides have a nearby 3D point
        valid = (src_dists < max_2d_distance) & (tgt_dists < max_2d_distance)
        n_valid = valid.sum()
        n_src_close = (src_dists < max_2d_distance).sum()
        n_tgt_close = (tgt_dists < max_2d_distance).sum()
        print(f"[match-align]   2D→3D lookup: src_close={n_src_close}, tgt_close={n_tgt_close}, "
              f"both_close={n_valid} (max_2d_dist={max_2d_distance}px)")
        if len(src_dists) > 0:
            print(f"[match-align]   src_dists: median={np.median(src_dists):.1f}, "
                  f"mean={np.mean(src_dists):.1f}, min={np.min(src_dists):.1f}, max={np.max(src_dists):.1f}")
            print(f"[match-align]   tgt_dists: median={np.median(tgt_dists):.1f}, "
                  f"mean={np.mean(tgt_dists):.1f}, min={np.min(tgt_dists):.1f}, max={np.max(tgt_dists):.1f}")

        pair_corr_count = 0
        for i in np.where(valid)[0]:
            si = src_indices[i]
            ti = tgt_indices[i]
            pair_key = (src_point3d_ids[si], tgt_point3d_ids[ti])
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                all_source_3d.append(src_pts_3d[si])
                all_target_3d.append(tgt_pts_3d[ti])
                pair_corr_count += 1

        print(f"[match-align]   -> {pair_corr_count} new 3D-3D correspondences from this pair")
        match_pair_info.append({
            'src_name': src_img.name, 'tgt_name': best_target_img.name,
            'reliability': best_reliability, 'reliability_threshold': reliability_threshold,
            'num_matches': best_num_matches,
            'median_certainty': best_median_cert, 'num_correspondences': pair_corr_count,
            'match_pts': best_match_pts,
        })

    print(f"[match-align] Total: {len(all_source_3d)} 3D-3D correspondences from "
          f"{len(match_pair_info)} accepted pairs")

    if all_source_3d:
        return np.array(all_source_3d), np.array(all_target_3d), match_pair_info
    return np.empty((0, 3)), np.empty((0, 3)), match_pair_info


def align_reconstructions_matching(
        rec_target: pycolmap.Reconstruction,
        rec_source: pycolmap.Reconstruction,
        match_provider: 'MatchingProvider',
        target_images_dir: Path,
        source_images_dir: Path,
        target_segs_dir: Path,
        source_segs_dir: Path,
        sample_size: int,
        certainty_threshold: float,
        reliability_threshold: float,
        max_pairs: int = 5,
        black_background: bool = False,
        use_procrustes: bool = True,
        refine_with_icp: bool = True,
        icp_centroid_prewarp: bool = False,
        device: str = 'cuda',
) -> Tuple[pycolmap.Reconstruction, dict]:
    """Align rec_source to rec_target using 2D-matching-derived 3D correspondences.

    1. Find 3D-3D correspondences by matching images across reconstructions.
    2. Optionally solve rigid alignment via Procrustes SVD.
    3. Optionally refine with ICP.

    Falls back to ICP-only if Procrustes is disabled or too few correspondences.

    Returns:
        rec_source_aligned: Transformed copy of rec_source.
        info: Dict with alignment metrics.
    """
    print(f"[match-align] black_background={black_background}, use_procrustes={use_procrustes}, "
          f"refine_with_icp={refine_with_icp}, icp_centroid_prewarp={icp_centroid_prewarp}")

    match_pair_info = []
    info = {
        'method': 'matching',
        'match_pairs': match_pair_info,
        'rec_after_procrustes': None,
        'rec_before_icp': None,
        'rec_after_icp': None,
    }

    # Step 1: Find 3D-3D correspondences via image matching
    source_pts, target_pts, match_pair_info = find_3d_correspondences_via_matching(
        rec_target, rec_source, match_provider,
        target_images_dir, source_images_dir,
        target_segs_dir, source_segs_dir,
        sample_size, certainty_threshold=certainty_threshold,
        reliability_threshold=reliability_threshold,
        max_pairs=max_pairs, black_background=black_background, device=device)
    info['match_pairs'] = match_pair_info

    num_correspondences = len(source_pts)
    print(f"[match-align] Found {num_correspondences} 3D-3D correspondences")
    info['num_correspondences'] = num_correspondences

    # Step 2: Procrustes alignment
    rec_current = rec_source
    if use_procrustes and num_correspondences >= 10:
        R, t = _procrustes_svd(source_pts, target_pts)
        rec_after_procrustes = _apply_rigid_to_reconstruction(rec_source, R, t)

        rot_deg = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        trans_norm = np.linalg.norm(t)
        print(f"[match-align] Procrustes: rotation={rot_deg:.4f} deg, translation={trans_norm:.4f}")

        aligned_source_pts = (R @ source_pts.T).T + t
        residuals = np.linalg.norm(aligned_source_pts - target_pts, axis=1)
        print(f"[match-align] Residual: mean={residuals.mean():.4f}, "
              f"median={np.median(residuals):.4f}, max={residuals.max():.4f}")

        info['rotation_deg'] = rot_deg
        info['translation_norm'] = trans_norm
        info['residual_mean'] = float(residuals.mean())
        info['residual_median'] = float(np.median(residuals))
        info['rec_after_procrustes'] = rec_after_procrustes
        rec_current = rec_after_procrustes
    elif use_procrustes:
        print(f"[match-align] Too few correspondences ({num_correspondences}) for Procrustes, skipping")

    # Step 3: ICP refinement
    info['rec_before_icp'] = rec_current
    if refine_with_icp:
        skip_prewarp = not icp_centroid_prewarp
        rec_after_icp, icp_info = align_reconstructions_icp(
            rec_target, rec_current, max_iterations=5, skip_centroid_prewarp=skip_prewarp)
        info['icp_refinement'] = icp_info
        info['rec_after_icp'] = rec_after_icp
        rec_current = rec_after_icp

    return rec_current, info


def colmap_K_params_vec(camera_K, camera_type=pycolmap.CameraModelId.PINHOLE):
    if camera_type == pycolmap.CameraModelId.PINHOLE:
        f_x = float(camera_K[0, 0])
        f_y = float(camera_K[1, 1])
        c_x = float(camera_K[0, 2])
        c_y = float(camera_K[1, 2])
        params_vec = [f_x, f_y, c_x, c_y]
    elif camera_type == pycolmap.CameraModelId.SIMPLE_PINHOLE:
        f_x = float(camera_K[0, 0])
        f_y = float(camera_K[1, 1])
        c_x = float(camera_K[0, 2])
        c_y = float(camera_K[1, 2])
        params_vec = [(f_x + f_y) / 2., c_x, c_y]
    else:
        raise ValueError(f'Unknown camera model {camera_type}')

    return params_vec
