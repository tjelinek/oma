"""Decoupled densification: fixed-pose dense triangulation + outlier filtering.

Given a finished (aligned) reconstruction, re-match its keyframes densely,
triangulate new points with the camera poses held fixed (no BA on cameras),
and filter the resulting cloud:
  1. reprojection-error / track-length gates (kills 2-view chance
     triangulations and high-residual points),
  2. multi-view visual-hull carve (a point dies if it projects onto background
     in any view that sees it) via carve_reconstruction_by_masks.

Used by the pipeline stage (OnboardingConfig.densify_reconstruction) and by
scripts/densify_from_poses.py.
"""

import copy
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import pycolmap

from onboarding.colmap_utils import add_posed_image_to_reconstruction, carve_reconstruction_by_masks
from onboarding.reconstruction import (_match_image_pairs, _merge_tracks, _write_colmap_database,
                                       two_view_geometry)


def filter_densified_points(reconstruction: pycolmap.Reconstruction,
                            min_track_len: int = 3,
                            max_reproj_error: float = 2.0) -> Tuple[int, int]:
    """Delete 3D points with short tracks or high reprojection error (in place).

    Returns (n_removed, n_kept)."""
    remove = [pid for pid, p in reconstruction.points3D.items()
              if p.track.length() < min_track_len
              or (p.has_error() and p.error > max_reproj_error)]
    for pid in remove:
        reconstruction.delete_point3D(pid)
    return len(remove), reconstruction.num_points3D()


def make_windowed_pairs(n_img: int, max_pairs: int = 800) -> List[Tuple[int, int]]:
    """All pairs; falls back to a temporal window for large keyframe sets (on an
    orbit the covisible pairs are the near-in-time ones and poses are fixed, so
    loop closures are not needed)."""
    pairs = [(i, j) for i in range(n_img) for j in range(i + 1, n_img)]
    if len(pairs) > max_pairs:
        window = max(3, max_pairs // n_img)
        pairs = [(i, j) for i in range(n_img) for j in range(i + 1, min(i + 1 + window, n_img))]
    return pairs


def contiguous_pose_model(reconstruction: pycolmap.Reconstruction) \
        -> Tuple[pycolmap.Reconstruction, List[Tuple[int, str]]]:
    """Model with contiguous 1..N image ids (the triangulator matches DB ids).

    Returns (pose_model, [(new_id, image_name)] in id order). The input model is
    returned unchanged when its ids are already contiguous."""
    ids_names = sorted((img_id, img.name) for img_id, img in reconstruction.images.items())
    if [i for i, _ in ids_names] == list(range(1, len(ids_names) + 1)):
        return reconstruction, ids_names
    renum = pycolmap.Reconstruction()
    cam = next(iter(reconstruction.cameras.values()))
    renum.add_camera(pycolmap.Camera(camera_id=1, model=cam.model, width=cam.width,
                                     height=cam.height, params=cam.params))
    for new_id, (old_id, name) in enumerate(ids_names, start=1):
        add_posed_image_to_reconstruction(renum, new_id, 1, name,
                                          reconstruction.images[old_id].cam_from_world())
    return renum, [(i + 1, name) for i, (_, name) in enumerate(ids_names)]


def densify_reconstruction(reconstruction: pycolmap.Reconstruction,
                           images: List[Path], segmentations: List[Path],
                           match_provider, sample_size: int, workdir: Path,
                           device: str = 'cuda',
                           add_track_merging: bool = True,
                           max_pairs: int = 800,
                           min_track_len: int = 2,
                           max_reproj_error: float = 4.0,
                           carve: bool = True,
                           carve_min_bg_views: int = 2) -> Optional[pycolmap.Reconstruction]:
    """Fixed-pose dense triangulation of `reconstruction`'s registered views.

    Args:
        reconstruction: aligned model providing the fixed camera poses. `images`
            and `segmentations` must be ordered by its (contiguous-renumbered)
            image ids — use contiguous_pose_model() to derive the order.
        match_provider: dense MatchingProvider (UFM) for pair matching.
        workdir: scratch dir for the COLMAP database + triangulated model.

    Returns the densified reconstruction (same poses; new points), or None.
    """
    pose_model, _ = contiguous_pose_model(reconstruction)
    n_img = pose_model.num_images()
    if n_img < 3 or len(images) != n_img:
        print(f'[densify] skipped: {n_img} registered views, {len(images)} images supplied')
        return None

    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    matching_pairs = make_windowed_pairs(n_img, max_pairs)
    print(f'[densify] {n_img} keyframes -> {len(matching_pairs)} pairs')
    matching_edges, certainties, view_graph, single_camera = _match_image_pairs(
        images, segmentations, matching_pairs, match_provider, sample_size, device,
        use_background_points=False)
    if add_track_merging:
        matching_edges, certainties = _merge_tracks(
            images, segmentations, matching_pairs, matching_edges, certainties,
            view_graph, match_provider, device, use_background_points=False)

    import torch
    cam = next(iter(pose_model.cameras.values()))
    camera_K = torch.tensor([[cam.focal_length_x, 0.0, cam.principal_point_x],
                             [0.0, cam.focal_length_y, cam.principal_point_y],
                             [0.0, 0.0, 1.0]], dtype=torch.float64)
    db_path = workdir / 'database.db'
    _write_colmap_database(images, matching_edges, certainties, single_camera, camera_K,
                           db_path, device, camera_K_is_gt=True)
    two_view_geometry(db_path)

    opts = pycolmap.IncrementalPipelineOptions()
    opts.triangulation.ignore_two_view_tracks = not add_track_merging
    opts.triangulation.max_transitivity = 2
    out_dir = workdir / 'model'
    out_dir.mkdir()
    try:
        densified = pycolmap.triangulate_points(copy.deepcopy(pose_model), str(db_path),
                                                str(images[0].parent), str(out_dir),
                                                clear_points=True, options=opts,
                                                refine_intrinsics=False)
    except Exception as e:
        print(f'[densify] triangulation failed: {e}')
        return None
    n_raw = densified.num_points3D()

    n_gated, n_kept = filter_densified_points(densified, min_track_len, max_reproj_error)
    if carve:
        name_to_seg = {img.name: seg for img, seg in zip(images, segmentations)}
        densified, n_carved, n_kept = carve_reconstruction_by_masks(
            densified, name_to_seg, min_bg_views=carve_min_bg_views)
    else:
        n_carved = 0
    print(f'[densify] {n_raw} triangulated -> {n_kept} after filters '
          f'(track/reproj gate removed {n_gated}, carve removed {n_carved})')
    densified.write(str(out_dir))
    return densified
