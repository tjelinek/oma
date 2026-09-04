import copy
import logging
import os
import select
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import networkx as nx
import numpy as np
import pycolmap
import torch
from kornia.geometry import Se3
from kornia.image import ImageSize
from pycolmap import TwoViewGeometryOptions

logger = logging.getLogger(__name__)
from tqdm import tqdm

from data_providers.flow_provider import MatchingProvider
from data_providers.frame_provider import PrecomputedSegmentationProvider, PrecomputedFrameProvider
from onboarding.colmap_utils import colmap_K_params_vec
from utils.conversions import Se3_to_Rigid3d
from utils.image_utils import get_intrinsics_from_exif


def _load_image_and_segmentation(img_path: Path, seg_path: Path, device: str) \
        -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    image = PrecomputedFrameProvider.load_and_downsample_image(img_path, 1., device).squeeze()
    h, w = image.shape[-2:]
    segmentation = PrecomputedSegmentationProvider.load_and_downsample_segmentation(
        seg_path, ImageSize(h, w), device=device)
    return image, segmentation, h, w


def _match_image_pairs(images: List[Path], segmentations: List[Path], matching_pairs: List[Tuple[int, int]],
                       match_provider: MatchingProvider, match_sample_size: int, device: str,
                       progress=None, use_background_points: bool = False) \
        -> Tuple[Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
                 Dict[Tuple[int, int], torch.Tensor],
                 nx.DiGraph, bool]:
    # When use_background_points is True the matcher keeps correspondences outside the
    # segmentation mask so background points enter the COLMAP reconstruction (ablation).
    keep_only_fg = not use_background_points
    matching_edges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    matching_edges_certainties: Dict[Tuple[int, int], torch.Tensor] = {}
    view_graph = nx.DiGraph()
    single_camera = True

    for pair_idx, (i1, i2) in tqdm(enumerate(matching_pairs)):
        img1_pth, img2_pth = images[i1], images[i2]
        img1_id = i1 + 1
        img2_id = i2 + 1

        if img1_pth.parent != img2_pth.parent:
            raise ValueError(f"Image pair must be in the same directory: {img1_pth} vs {img2_pth}")

        if progress is not None:
            progress(0.5 * pair_idx / float(len(matching_pairs)), desc="Matching image pairs for reconstruction")

        img1, img1_seg, h1, w1 = _load_image_and_segmentation(img1_pth, segmentations[i1], device)
        img2, img2_seg, h2, w2 = _load_image_and_segmentation(img2_pth, segmentations[i2], device)

        if h1 != h2 or w1 != w2:
            single_camera = False

        src_pts_xy_int, dst_pts_xy_int, certainty = \
            match_provider.get_source_target_points(img1, img2, match_sample_size, img1_seg.squeeze(),
                                                    img2_seg.squeeze(), Path(img1_pth.name),
                                                    Path(img2_pth.name), as_int=True,
                                                    zero_certainty_outside_segmentation=keep_only_fg,
                                                    only_foreground_matches=keep_only_fg)
        view_graph.add_edge(img1_id, img2_id)
        edge = (img1_id, img2_id)
        # Store matches on CPU: for a dense Complete view graph the accumulated per-edge
        # matches (especially after track merging) reach tens of GB and otherwise compete
        # with each pair's flow computation on the GPU, OOMing the 40 GB A100. They are
        # only needed later (COLMAP DB writing), where they are moved back to `device`
        # per node in unique_keypoints_from_matches().
        matching_edges[edge] = (src_pts_xy_int.cpu(), dst_pts_xy_int.cpu())
        matching_edges_certainties[edge] = certainty.cpu()

    return matching_edges, matching_edges_certainties, view_graph, single_camera


def _merge_tracks(images: List[Path], segmentations: List[Path], matching_pairs: List[Tuple[int, int]],
                  matching_edges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
                  matching_edges_certainties: Dict[Tuple[int, int], torch.Tensor],
                  view_graph: nx.DiGraph, match_provider: MatchingProvider, device: str,
                  progress=None, use_background_points: bool = False) \
        -> Tuple[Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
                 Dict[Tuple[int, int], torch.Tensor]]:
    keep_only_fg = not use_background_points
    for pair_idx, (i1, i2) in tqdm(enumerate(matching_pairs)):
        img1_pth, img2_pth = images[i1], images[i2]
        img1_id = i1 + 1
        img2_id = i2 + 1

        if progress is not None:
            progress(0.5 * pair_idx / float(len(matching_pairs)), desc="Densifying matching...")

        img1, img1_seg, h1, w1 = _load_image_and_segmentation(img1_pth, segmentations[i1], device)
        img2, img2_seg, h2, w2 = _load_image_and_segmentation(img2_pth, segmentations[i2], device)

        previous_matching_pairs = view_graph.in_edges(img1_id)
        src_pts_xy_roma_int_can_be_added = []
        dst_pts_xy_roma_int_can_be_added = []
        certainties_can_be_added = []
        for (edge_u, edge_v) in previous_matching_pairs:
            src_pts_xy_int_nonsampled, dst_pts_xy_int_nonsampled, certainty_nonsampled = \
                match_provider.get_source_target_points(img1, img2, None, img1_seg.squeeze(),
                                                        img2_seg.squeeze(), Path(img1_pth.name),
                                                        Path(img2_pth.name), as_int=True,
                                                        zero_certainty_outside_segmentation=keep_only_fg,
                                                        only_foreground_matches=keep_only_fg)

            # Move the dense full-resolution matches to CPU before merging so the growing
            # track sets never accumulate on the GPU (see _match_image_pairs note). The
            # set operations below then run consistently on CPU against the CPU-stored edges.
            src_pts_xy_int_nonsampled = src_pts_xy_int_nonsampled.cpu()
            dst_pts_xy_int_nonsampled = dst_pts_xy_int_nonsampled.cpu()
            certainty_nonsampled = certainty_nonsampled.cpu()

            prev_match_certain_dst_pts = matching_edges[edge_u, edge_v][1]

            # Degenerate edges (e.g. a keyframe whose mask collapsed to empty yields
            # zero matches) have nothing to merge through — .max() on empty crashes.
            if prev_match_certain_dst_pts.numel() == 0 or src_pts_xy_int_nonsampled.numel() == 0:
                continue

            max_coord = max(prev_match_certain_dst_pts.max().item(), src_pts_xy_int_nonsampled.max().item()) + 1
            A_hash = prev_match_certain_dst_pts[:, 0] * max_coord + prev_match_certain_dst_pts[:, 1]
            B_hash = src_pts_xy_int_nonsampled[:, 0] * max_coord + src_pts_xy_int_nonsampled[:, 1]
            mask = torch.isin(B_hash, A_hash)

            src_pts_xy_roma_int_can_be_added.append(src_pts_xy_int_nonsampled[mask])
            dst_pts_xy_roma_int_can_be_added.append(dst_pts_xy_int_nonsampled[mask])
            certainties_can_be_added.append(certainty_nonsampled[mask])

        edge = (img1_id, img2_id)

        src_pts_xy_int = torch.cat([matching_edges[edge][0]] + src_pts_xy_roma_int_can_be_added)
        dst_pts_xy_int = torch.cat([matching_edges[edge][1]] + dst_pts_xy_roma_int_can_be_added)
        certainty = torch.cat([matching_edges_certainties[edge]] + certainties_can_be_added)

        matching_edges[edge] = (src_pts_xy_int, dst_pts_xy_int)
        matching_edges_certainties[edge] = certainty

    return matching_edges, matching_edges_certainties


def _apply_seq_consistency_gate(matching_edges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
                                matching_edges_certainties: Dict[Tuple[int, int], torch.Tensor],
                                images: List[Path], segmentations: List[Path],
                                matching_pairs: List[Tuple[int, int]],
                                seq_gate_provider: MatchingProvider, sample_size: int, device: str,
                                tau_px: float, assoc_px: float, min_edge_matches: int,
                                use_background_points: bool = False) \
        -> Tuple[Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
                 Dict[Tuple[int, int], torch.Tensor]]:
    """Sequential-consistency gate (VidMap-inspired, docs/archive/report_seqconsistency_gate.md).

    For each view-graph edge, obtain point-track correspondences chained through the
    intermediate captured frames (temporal provenance) and keep a dense match only if its
    displacement agrees with the displacement of its nearest tracked query:

        accept (x_a, x_b)  <=>  exists track (t_a -> t_b) with ||x_a - t_a|| <= assoc_px
                                and ||(x_b - x_a) - (t_b - t_a)|| <= tau_px

    Dense matches with no chain support within assoc_px are rejected (temporal trust is
    the point — 2-view self-consistency cannot break coherent aliasing). Edges left with
    fewer than min_edge_matches matches are dropped entirely: an edge whose matches the
    chain cannot support is an unsupported bridge (the obj_000010 duplicate mechanism).
    """
    keep_only_fg = not use_background_points
    kept_edges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    kept_certainties: Dict[Tuple[int, int], torch.Tensor] = {}
    n_dropped_edges = 0
    for (i1, i2) in matching_pairs:
        edge = (i1 + 1, i2 + 1)
        if edge not in matching_edges:
            continue
        src, dst = matching_edges[edge]
        cert = matching_edges_certainties[edge]
        if src.numel() == 0:
            kept_edges[edge] = (src, dst)
            kept_certainties[edge] = cert
            continue

        img1, img1_seg, _, _ = _load_image_and_segmentation(images[i1], segmentations[i1], device)
        img2, img2_seg, _, _ = _load_image_and_segmentation(images[i2], segmentations[i2], device)
        try:
            t_src, t_dst, _t_cert = seq_gate_provider.get_source_target_points(
                img1, img2, sample_size, img1_seg.squeeze(), img2_seg.squeeze(),
                Path(images[i1].name), Path(images[i2].name), as_int=True,
                zero_certainty_outside_segmentation=keep_only_fg,
                only_foreground_matches=keep_only_fg)
        except Exception as e:
            # Fail-open per edge: a tracker crash must not silently delete an edge the
            # baseline pipeline would have used — keep the edge ungated and log.
            print(f"seq_gate: tracking failed on edge {edge} ({e}) — keeping edge ungated")
            kept_edges[edge] = (src, dst)
            kept_certainties[edge] = cert
            continue

        if t_src is None or t_src.numel() == 0:
            # Zero surviving tracks across this edge: no chain support at all.
            n_dropped_edges += 1
            print(f"seq_gate: edge {edge} has NO chain support (0 surviving tracks) — dropped")
            continue

        src_f = src.float().to(device)
        dst_f = dst.float().to(device)
        ts = t_src.float().to(device)
        td = t_dst.float().to(device)
        # nearest tracked query per dense match, chunked over dense matches
        keep_mask = torch.zeros(len(src_f), dtype=torch.bool, device=device)
        disp_track = td - ts
        for s in range(0, len(src_f), 4096):
            chunk = src_f[s:s + 4096]
            d = torch.cdist(chunk, ts)
            nn_d, nn_i = d.min(dim=1)
            disp_dense = dst_f[s:s + 4096] - chunk
            ok = (nn_d <= assoc_px) & \
                 ((disp_dense - disp_track[nn_i]).norm(dim=1) <= tau_px)
            keep_mask[s:s + 4096] = ok
        n_in, n_keep = len(src_f), int(keep_mask.sum())
        if n_keep < min_edge_matches:
            n_dropped_edges += 1
            print(f"seq_gate: edge {edge} kept {n_keep}/{n_in} matches (< {min_edge_matches}) — dropped")
            continue
        keep_cpu = keep_mask.cpu()
        kept_edges[edge] = (src[keep_cpu], dst[keep_cpu])
        kept_certainties[edge] = cert[keep_cpu]
        print(f"seq_gate: edge {edge} kept {n_keep}/{n_in} matches "
              f"({len(ts)} tracks, reject {100 * (1 - n_keep / max(n_in, 1)):.1f}%)")

    print(f"seq_gate: {len(kept_edges)}/{len(matching_edges)} edges survive "
          f"({n_dropped_edges} dropped)")
    return kept_edges, kept_certainties


def _write_colmap_database(images: List[Path],
                           matching_edges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
                           matching_edges_certainties: Dict[Tuple[int, int], torch.Tensor],
                           single_camera: bool, camera_K: Optional[torch.Tensor],
                           database_path: Path, device: str,
                           camera_K_is_gt: bool = False) -> Path:
    print(f"[pycolmap4-debug] _write_colmap_database: opening {database_path}")
    database = pycolmap.Database.open(str(database_path))

    keypoints, edge_match_indices = unique_keypoints_from_matches(matching_edges, None, matching_edges_certainties,
                                                                  eliminate_one_to_many_matches=True, device=device)

    new_cam_id = 1
    if single_camera:
        if camera_K is None:
            camera_K = get_intrinsics_from_exif(images[0])

        if camera_K_is_gt:
            # Caspar GPU BA (colmap 4.1.x) supports PINHOLE but not SIMPLE_PINHOLE —
            # unsupported models are silently skipped, turning BA into a no-op. With
            # exact GT intrinsics the model choice does not affect calibration
            # (refinement is disabled in run_mapper), so use PINHOLE.
            camera_model = pycolmap.CameraModelId.PINHOLE
        else:
            # Self-calibrating from an EXIF guess: keep the tighter single-focal model.
            camera_model = pycolmap.CameraModelId.SIMPLE_PINHOLE
        params_vec = colmap_K_params_vec(camera_K, camera_model)

        h, w = PrecomputedFrameProvider.load_and_downsample_image(images[0], 1.).shape[-2:]

        new_camera = pycolmap.Camera(camera_id=new_cam_id, model=camera_model, width=w, height=h, params=params_vec)
        if camera_K_is_gt:
            # Exact (GT) intrinsics: mark them as a trusted prior so the mapper's initial
            # two-view geometry uses them as-is. Focal refinement is additionally disabled
            # in run_mapper (masked object-only points cannot self-calibrate focal, and a
            # drifted focal warps the whole orbit into a uniform per-camera rotation bias).
            new_camera.has_prior_focal_length = True
        database.write_camera(new_camera, use_camera_id=True)

    for i, img in enumerate(images):
        if not single_camera:
            raise NotImplementedError("To be added")

        img_id = i + 1
        image = pycolmap.Image(image_id=img_id, camera_id=new_cam_id, name=str(img.name))
        database.write_image(image, use_image_id=True)

    for colmap_image_id in sorted(keypoints.keys()):
        keypoints_np = keypoints[colmap_image_id].numpy(force=True).astype(np.float32)
        database.write_keypoints(colmap_image_id, keypoints_np)

    for colmap_image_u, colmap_image_v in edge_match_indices.keys():
        match_indices_np = edge_match_indices[colmap_image_u, colmap_image_v].numpy(force=True)
        if match_indices_np.ndim != 2 or match_indices_np.shape[1] != 2:
            continue
        match_indices_np = match_indices_np.astype(np.uint32)
        database.write_matches(colmap_image_u, colmap_image_v, match_indices_np)

    database.close()
    print(f"[pycolmap4-debug] _write_colmap_database: done, wrote {len(keypoints)} image keypoints, "
          f"{len(edge_match_indices)} match pairs")
    return database_path


def _forensic_db_copy(database_path: Path, name: str) -> None:
    """Snapshot the COLMAP database for post-mortem inspection, opt-in.

    These two snapshots (pre-RANSAC and post-verification) are written on every run
    but are read by nothing in the codebase, so they tripled the database footprint
    for no benefit: they accounted for most of the 465 GB of .db under the results
    tree, and filling the quota is what stopped a 480-cell sweep mid-flight. Keep
    them available for debugging a specific failure, just not by default.

    Set GLOPOSE_KEEP_FORENSIC_DB=1 to restore the old behaviour.
    """
    if os.environ.get('GLOPOSE_KEEP_FORENSIC_DB', '') not in ('', '0'):
        shutil.copy(database_path, database_path.parent / name)


def reconstruct_images_using_sfm(images: List[Path], segmentations: List[Path], matching_pairs: List[Tuple[int, int]],
                                 init_with_first_two_images: bool, mapper: str, match_provider: MatchingProvider,
                                 match_sample_size: int, colmap_working_dir: Path, add_track_merging_matches: bool,
                                 camera_K: Optional[torch.Tensor] = None, device: str = 'cpu',
                                 progress=None, filter_points_by_seg: bool = False,
                                 use_background_points: bool = False,
                                 filter_degenerate_edges: bool = False,
                                 filter_degenerate_edges_mode: str = 'all',
                                 ba_backend: str = 'caspar',
                                 seq_gate_provider: Optional[MatchingProvider] = None,
                                 seq_gate_tau_px: float = 5.0,
                                 seq_gate_assoc_px: float = 16.0,
                                 seq_gate_min_edge_matches: int = 15,
                                 min_track_length: int = 0) \
        -> Tuple[Optional[pycolmap.Reconstruction], int]:
    if len(matching_pairs) == 0:
        raise ValueError("Needed at least 1 match.")
    if len(images) == 0:
        raise ValueError("No images provided for SfM reconstruction")

    database_path = colmap_working_dir / 'database.db'
    colmap_output_path = colmap_working_dir / 'output'
    colmap_image_path = colmap_working_dir / 'images'

    if database_path.exists():
        raise FileExistsError(f"COLMAP database already exists: {database_path}")

    matching_pairs = sorted(matching_pairs)

    # --- Matching phase (GPU): dense flow over the view-graph pairs + track merging. ---
    matching_start = time.time()

    matching_edges, matching_edges_certainties, view_graph, single_camera = _match_image_pairs(
        images, segmentations, matching_pairs, match_provider, match_sample_size, device, progress,
        use_background_points=use_background_points)

    if seq_gate_provider is not None:
        # Sequential-consistency gate (docs/archive/report_seqconsistency_gate.md): dense matches
        # must agree with a point track chained through the intermediate frames. Applied
        # BEFORE track merging so merged tracks build on gated matches only.
        matching_edges, matching_edges_certainties = _apply_seq_consistency_gate(
            matching_edges, matching_edges_certainties, images, segmentations, matching_pairs,
            seq_gate_provider, match_sample_size, device,
            tau_px=seq_gate_tau_px, assoc_px=seq_gate_assoc_px,
            min_edge_matches=seq_gate_min_edge_matches,
            use_background_points=use_background_points)
        # Dropped edges must disappear everywhere downstream: _merge_tracks indexes
        # matching_edges for every pair in matching_pairs, and the view graph drives
        # its transitive-merge lookups.
        matching_pairs = [(i1, i2) for (i1, i2) in matching_pairs if (i1 + 1, i2 + 1) in matching_edges]
        view_graph = nx.DiGraph()
        for (i1, i2) in matching_edges:
            view_graph.add_edge(i1, i2)

    if add_track_merging_matches:
        matching_edges, matching_edges_certainties = _merge_tracks(
            images, segmentations, matching_pairs, matching_edges, matching_edges_certainties,
            view_graph, match_provider, device, progress, use_background_points=use_background_points)

    matching_time = time.time() - matching_start

    # --- Reconstruction phase (CPU): COLMAP database, geometric verification, mapper. ---
    reconstruction_start = time.time()

    _write_colmap_database(images, matching_edges, matching_edges_certainties, single_camera, camera_K,
                           database_path, device, camera_K_is_gt=camera_K is not None)
    _forensic_db_copy(database_path, 'database_before_ransac.db')

    two_view_geometry(database_path)

    _forensic_db_copy(database_path, 'database_after_ransac_before_rec.db')

    if filter_degenerate_edges:
        # The copy above keeps the unpruned verified DB for forensics; the mapper
        # sees only the pruned edge set (policy: filter_degenerate_edges_mode).
        prune_degenerate_two_view_edges(database_path, mode=filter_degenerate_edges_mode)

    first_image_id = None
    second_image_id = None
    if init_with_first_two_images:
        first_image_id = 1
        second_image_id = 2

    if progress is not None:
        progress(0.5, desc="Running reconstruction...")

    ignore_two_view_tracks = not add_track_merging_matches
    num_reconstructions = run_mapper(colmap_output_path, database_path, colmap_image_path, mapper, first_image_id,
                                     second_image_id, ignore_two_view_tracks,
                                     fix_camera_K=camera_K is not None, ba_backend=ba_backend)

    if progress is not None:
        progress(1.0, desc="Reconstruction finished.")

    path_to_rec = colmap_output_path / '0'
    print(f"[pycolmap4-debug] reconstruct_images_using_sfm: loading reconstruction from {path_to_rec}")
    try:
        reconstruction = pycolmap.Reconstruction(path_to_rec)
        print(f"[pycolmap4-debug] reconstruct_images_using_sfm: loaded OK")
        print(reconstruction.summary())
    except Exception as e:
        print(f"[pycolmap4-debug] reconstruct_images_using_sfm: load FAILED: {e}")
        timings = {'matching_time': matching_time, 'reconstruction_time': time.time() - reconstruction_start}
        return None, num_reconstructions, timings

    if filter_points_by_seg:
        reconstruction = filter_points_by_segmentation(reconstruction, segmentations, images)

    if min_track_length > 0:
        reconstruction = filter_points_by_track_length(reconstruction, min_track_length)

    timings = {'matching_time': matching_time, 'reconstruction_time': time.time() - reconstruction_start}
    return reconstruction, num_reconstructions, timings


def filter_points_by_track_length(reconstruction: pycolmap.Reconstruction,
                                  min_track_length: int) -> pycolmap.Reconstruction:
    """Delete 3D points seen in fewer than min_track_length distinct images."""
    to_delete = [pid for pid, p in reconstruction.points3D.items()
                 if len({e.image_id for e in p.track.elements}) < min_track_length]
    num_before = len(reconstruction.points3D)
    for pid in to_delete:
        reconstruction.delete_point3D(pid)
    print(f"Track-length filtering (>= {min_track_length} images): removed {len(to_delete)}/{num_before} points, "
          f"{len(reconstruction.points3D)} remaining")
    return reconstruction


def filter_points_by_segmentation(reconstruction: pycolmap.Reconstruction,
                                   segmentations: List[Path],
                                   images: List[Path], run_ba: bool = True) -> pycolmap.Reconstruction:
    """Remove 3D points whose 2D observation falls outside the segmentation mask in any image.
    Then run bundle adjustment on the cleaned reconstruction."""

    # Build mapping: image name -> segmentation path
    image_name_to_seg: Dict[str, Path] = {}
    for img_path, seg_path in zip(images, segmentations):
        image_name_to_seg[img_path.name] = seg_path

    # Load and cache segmentation masks as numpy arrays (H, W), values 0-255
    seg_cache: Dict[str, np.ndarray] = {}

    def get_seg_mask(image_name: str) -> np.ndarray | None:
        if image_name in seg_cache:
            return seg_cache[image_name]
        seg_path = image_name_to_seg.get(image_name)
        if seg_path is None or not seg_path.exists():
            return None
        import imageio
        mask = imageio.v3.imread(seg_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        seg_cache[image_name] = mask
        return mask

    # Collect point3D IDs to delete
    point3d_ids_to_delete = set()
    for point3d_id, point3d in reconstruction.points3D.items():
        for track_element in point3d.track.elements:
            image_id = track_element.image_id
            point2d_idx = track_element.point2D_idx

            image = reconstruction.images[image_id]
            point2d = image.points2D[point2d_idx]
            x, y = point2d.xy.astype(int)

            mask = get_seg_mask(image.name)
            if mask is None:
                continue

            h, w = mask.shape
            if y < 0 or y >= h or x < 0 or x >= w or mask[y, x] < 128:
                point3d_ids_to_delete.add(point3d_id)
                break

    num_before = len(reconstruction.points3D)
    for point3d_id in point3d_ids_to_delete:
        reconstruction.delete_point3D(point3d_id)
    num_after = len(reconstruction.points3D)
    print(f"Segmentation filtering: removed {num_before - num_after}/{num_before} points, "
          f"{num_after} remaining")

    if not run_ba:
        return reconstruction
    # Run bundle adjustment on the cleaned reconstruction
    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, ba_options)

    return reconstruction


def align_with_kabsch(reconstruction: pycolmap.Reconstruction, gt_Se3_world2cam_poses: Dict[str, Se3]) \
        -> Tuple[pycolmap.Reconstruction, bool]:
    """Align a reconstruction to per-frame GT camera poses with a full-pose similarity.

    The global rotation is taken from the camera ORIENTATIONS (Procrustes over the
    world2cam rotations), not from the camera centres. Centres-only Sim3 estimation
    is ill-conditioned when the camera-centre cloud is near-degenerate, which is
    exactly the moving-object / static-camera geometry (the centres orbit an object
    that is largely turning in place): the fit matches the centres but picks a wrong
    global rotation, inflating every per-frame rotation error to near-chance while
    the reconstruction is internally correct. The orientations are well conditioned,
    so we solve the rotation from them and recover only scale and translation from
    the centres. On well-distributed static captures the two rotations coincide, so
    static results are unchanged; on dynamic captures this removes the spurious
    ~constant global-rotation offset.
    """
    print(f"[pycolmap4-debug] align_with_kabsch: starting (deepcopy)")
    reconstruction = copy.deepcopy(reconstruction)

    gt_camera_centers = []
    pred_camera_centers = []
    gt_rotations = []
    pred_rotations = []

    for image_name, gt_Se3_world2cam in gt_Se3_world2cam_poses.items():

        pred_image = reconstruction.find_image_with_name(image_name)
        if pred_image is None:
            continue

        gt_cam_center = gt_Se3_world2cam.inverse().translation.numpy(force=True)
        pred_cam_from_world = pred_image.cam_from_world()
        pred_cam_center = pred_cam_from_world.inverse().translation

        gt_camera_centers.append(np.asarray(gt_cam_center).reshape(3))
        pred_camera_centers.append(np.asarray(pred_cam_center).reshape(3))
        gt_rotations.append(np.asarray(gt_Se3_world2cam.rotation.matrix().squeeze()
                                       .numpy(force=True)).reshape(3, 3))
        pred_rotations.append(np.asarray(pred_cam_from_world.rotation.matrix()).reshape(3, 3))

    if len(gt_camera_centers) < 3:
        print(f"[pycolmap4-debug] align_with_kabsch: too few matched cameras ({len(gt_camera_centers)}), skipping")
        return reconstruction, False

    gt_camera_centers = np.stack(gt_camera_centers)
    pred_camera_centers = np.stack(pred_camera_centers)
    gt_rotations = np.stack(gt_rotations)      # (N, 3, 3), world2cam
    pred_rotations = np.stack(pred_rotations)  # (N, 3, 3), world2cam

    # Global rotation from orientations. reconstruction.transform applies a world
    # similarity x -> s R x + t, under which a camera's world2cam rotation becomes
    # pred_R @ R^T; setting that equal to gt_R gives the per-frame estimate
    # R = gt_R^T @ pred_R, averaged over frames by projecting the accumulator onto
    # SO(3): B = sum_i gt_R_i^T @ pred_R_i = U S V^T -> R = U diag(1,1,det(UV^T)) V^T.
    B = np.einsum('nba,nbc->ac', gt_rotations, pred_rotations)
    U, _, Vt = np.linalg.svd(B)
    d = np.sign(np.linalg.det(U @ Vt))
    R_align = U @ np.diag([1.0, 1.0, d]) @ Vt

    # Scale and translation from the centres with the rotation held fixed:
    # gt_center_i = scale * (R_align @ pred_center_i) + t (least squares).
    rotated_pred = (R_align @ pred_camera_centers.T).T
    rp_mean = rotated_pred.mean(axis=0)
    gt_mean = gt_camera_centers.mean(axis=0)
    rp_c = rotated_pred - rp_mean
    gt_c = gt_camera_centers - gt_mean
    denom = float((rp_c ** 2).sum())
    if denom <= 0.0:
        print(f"[pycolmap4-debug] align_with_kabsch: degenerate centres, cannot recover scale")
        return reconstruction, False
    scale = float((rp_c * gt_c).sum() / denom)
    translation = gt_mean - scale * rp_mean

    sim3d = pycolmap.Sim3d(scale=scale, rotation=pycolmap.Rotation3d(R_align),
                           translation=translation)

    print(f"[pycolmap4-debug] align_with_kabsch: applying transform (scale={scale:.4f})")
    reconstruction.transform(sim3d)
    print(f"[pycolmap4-debug] align_with_kabsch: done")

    return reconstruction, True


def two_view_geometry(colmap_db_path: Path):
    print(f"[pycolmap4-debug] two_view_geometry: starting match_exhaustive")
    opts = TwoViewGeometryOptions()
    opts.detect_watermark = False
    pycolmap.match_exhaustive(str(colmap_db_path), verification_options=opts)
    print(f"[pycolmap4-debug] two_view_geometry: done")


def prune_degenerate_two_view_edges(colmap_db_path: Path, mode: str = 'all') -> int:
    """Delete two_view_geometries edges that COLMAP's RANSAC verification classified as
    degenerate (anything but CALIBRATED=2). The mapper builds its correspondence graph
    exclusively from two_view_geometries, so this removes the pruned edges with exactly
    the semantics of the verification that already ran (same options).

    Rationale (docs/archive/report_dyn_failure_analysis.md): PLANAR_OR_PANORAMIC edges are
    homography-degenerate match fields (texture aliasing / near-pure translation) that
    RANSAC cannot reject; a single such edge can glue two disconnected temporal blocks
    into an offset duplicate of the object. With GT intrinsics provided (prior focal),
    a healthy edge should verify as CALIBRATED; an UNCALIBRATED-F fallback signals
    inconsistency with the known K.

    Modes:
      'all'          — delete every non-CALIBRATED edge. Cures duplicates but amputates
                       sparse/planar graphs (HOPE) whose H-classified chain edges are
                       legitimate (val-sweep evidence in the report).
      'connectivity' — degenerate evidence may confirm, never merge: group keyframes by
                       CALIBRATED edges only, then (a) delete degenerate edges inside a
                       group (redundant, can only distort), (b) keep a degenerate edge
                       that is the sole link between groups iff it connects TEMPORAL
                       NEIGHBOURS (consecutive keyframes — a planar chain doing its job),
                       (c) refuse degenerate sole bridges across a temporal gap (the
                       duplicate-maker). A refused bridge may split the graph; the mapper
                       then reconstructs the component of the init pair — an honest
                       partial model.

    Returns the number of deleted edges.
    """
    import sqlite3
    MAXIM = 2147483647
    con = sqlite3.connect(str(colmap_db_path))
    try:
        hist = dict(con.execute(
            'SELECT config, COUNT(*) FROM two_view_geometries WHERE rows > 0 GROUP BY config'))
        if mode == 'all':
            cur = con.execute('DELETE FROM two_view_geometries WHERE config != 2')
            deleted = cur.rowcount
            con.commit()
            print(f"prune_degenerate_two_view_edges[all]: config histogram {hist}, "
                  f"deleted {deleted} non-CALIBRATED edges")
            return deleted

        if mode != 'connectivity':
            raise ValueError(f"Unknown prune mode: {mode}")

        # Temporal order of keyframes from the decimated frame index in the image name.
        frames = {}
        for iid, name in con.execute('SELECT image_id, name FROM images'):
            f = Path(name).name.split('_')[0].split('.')[0]
            frames[iid] = int(f) if f.isdigit() else -1
        order = sorted(frames, key=lambda i: frames[i])
        pos = {iid: k for k, iid in enumerate(order)}

        parent = {iid: iid for iid in frames}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        clean, suspicious = [], []
        for pid, cfg, r in con.execute('SELECT pair_id, config, rows FROM two_view_geometries'):
            if (r or 0) == 0:
                continue
            a, b = pid // MAXIM, pid % MAXIM
            (clean if cfg == 2 else suspicious).append((pid, a, b, cfg))

        for _, a, b, _ in clean:
            union(a, b)

        # Ascending temporal gap: chain-repairing neighbour edges merge groups first, so
        # farther bridges are judged against the fullest clean+chain connectivity.
        kept_bridges, deleted_pids = [], []
        n_redundant = n_refused = 0
        for pid, a, b, cfg in sorted(suspicious, key=lambda e: abs(pos[e[1]] - pos[e[2]])):
            if find(a) == find(b):
                deleted_pids.append(pid)
                n_redundant += 1
            elif abs(pos[a] - pos[b]) == 1:
                kept_bridges.append((frames[a], frames[b], cfg))
                union(a, b)
            else:
                deleted_pids.append(pid)
                n_refused += 1
                print(f"prune[connectivity]: REFUSED degenerate bridge frames "
                      f"{frames[a]}<->{frames[b]} (config {cfg}, temporal gap "
                      f"{abs(pos[a] - pos[b])})")

        for pid in deleted_pids:
            con.execute('DELETE FROM two_view_geometries WHERE pair_id=?', (pid,))
        con.commit()

        n_comp = len({find(i) for i in frames})
        print(f"prune_degenerate_two_view_edges[connectivity]: config histogram {hist}; "
              f"deleted {n_redundant} redundant + {n_refused} refused-bridge edges, kept "
              f"{len(kept_bridges)} neighbour bridges {kept_bridges[:6]}; graph has "
              f"{n_comp} component(s) after pruning")
        return len(deleted_pids)
    finally:
        con.close()


def _caspar_usable() -> bool:
    """True iff the installed pycolmap exposes the Caspar GPU BA backend and a GPU is present.

    `pycolmap.has_cuda` is a compile-time flag, so it alone does not guarantee a usable
    GPU at runtime (e.g. CUDA-built wheel running on a CPU-only node) — hence the torch check.
    """
    return (hasattr(pycolmap, 'BundleAdjustmentBackend')
            and hasattr(pycolmap.BundleAdjustmentBackend, 'CASPAR')
            and getattr(pycolmap, 'has_cuda', False)
            and torch.cuda.is_available())


def _caspar_supports_db_cameras(colmap_db_path: Path) -> bool:
    """Caspar (colmap 4.1.x) implements adapters only for PINHOLE and SIMPLE_RADIAL;
    observations of any other camera model are silently skipped, which turns bundle
    adjustment into a no-op ('No residuals to optimize') — not faster, just wrong.
    Refuse Caspar unless every camera in the database uses a supported model."""
    import sqlite3
    with sqlite3.connect(f"file:{colmap_db_path}?mode=ro", uri=True) as con:
        models = {row[0] for row in con.execute("SELECT model FROM cameras")}
    return len(models) > 0 and models <= {int(pycolmap.CameraModelId.PINHOLE),
                                          int(pycolmap.CameraModelId.SIMPLE_RADIAL)}


def run_mapper(colmap_output_path: Path, colmap_db_path: Path, colmap_image_path: Path, mapper: str = 'pycolmap',
               first_image_id: Optional[int] = None, second_image_id: Optional[int] = None,
               ignore_two_view_tracks: bool = True, fix_camera_K: bool = False,
               ba_backend: str = 'caspar') -> int:
    """Run COLMAP/glomap mapper. Returns the number of reconstructions produced."""
    colmap_output_path.mkdir(exist_ok=True, parents=True)

    initial_pair_provided = first_image_id is not None and second_image_id is not None
    if mapper == 'glomap':
        # GLOMAP is deprecated as a standalone project and lives on inside COLMAP;
        # pycolmap >= 4.0 exposes it as the global mapping pipeline. Calling it
        # in-process guarantees the reader matches the schema pycolmap wrote
        # (the old ~/bin/glomap binaries predate the 3.12+ database schema and
        # abort with 'SQLite error: SQL logic error').
        opts = pycolmap.GlobalPipelineOptions()
        opts.mapper.track_min_num_views_per_track = 3 if ignore_two_view_tracks else 2
        maps = pycolmap.global_mapping(str(colmap_db_path), str(colmap_image_path), str(colmap_output_path),
                                       options=opts)
        if len(maps) > 0:
            best = max(maps.values(), key=lambda r: r.num_reg_images())
            if len(maps) > 1:
                sizes = [r.num_reg_images() for r in maps.values()]
                logger.warning("GLOMAP produced %d reconstructions (sizes: %s), using largest (%d images)",
                               len(maps), sizes, best.num_reg_images())
            best_output_path = colmap_output_path / '0'
            best_output_path.mkdir(exist_ok=True, parents=True)
            best.write(str(best_output_path))
        return len(maps)

    elif mapper == 'colmap':
        command = [
            "colmap",
            "mapper",
            "--database_path", str(colmap_db_path),
            "--output_path", str(colmap_output_path),
            "--image_path", str(colmap_image_path),
            "--Mapper.tri_ignore_two_view_tracks", str(int(ignore_two_view_tracks)),
            *("--Mapper.init_image_id1", str(first_image_id) if initial_pair_provided else ""),
            *("--Mapper.init_image_id2", str(second_image_id) if initial_pair_provided else ""),
            "--log_to_stderr", str(1),
        ]

        with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
        ) as process:
            fds = [process.stdout.fileno(), process.stderr.fileno()]
            while True:
                ready_fds, _, _ = select.select(fds, [], [])
                for fd in ready_fds:
                    if fd == process.stdout.fileno():
                        line = process.stdout.readline()
                        if line:
                            print(f"STDOUT: {line.strip()}")
                    elif fd == process.stderr.fileno():
                        line = process.stderr.readline()
                        if line:
                            print(f"STDERR: {line.strip()}")
                if process.poll() is not None:
                    break

            process.wait()
            if process.returncode != 0:
                error_message = process.stderr.read()
                print(f"Error: {error_message}")
                raise subprocess.CalledProcessError(process.returncode, command, output=None, stderr=error_message)

        # Count output reconstruction directories
        num_recs = sum(1 for d in colmap_output_path.iterdir() if d.is_dir() and d.name.isdigit())
        return num_recs

    elif mapper == 'pycolmap':

        opts = pycolmap.IncrementalPipelineOptions()
        opts.triangulation.ignore_two_view_tracks = ignore_two_view_tracks
        opts.triangulation.max_transitivity = 2
        if fix_camera_K:
            # Exact (GT) intrinsics provided — do not let bundle adjustment drift them.
            # Masked object-only observations cannot self-calibrate focal; a drifted focal
            # warps the orbit into a uniform per-camera rotation bias.
            opts.ba_refine_focal_length = False
            opts.ba_refine_principal_point = False
            opts.ba_refine_extra_params = False
        # opts.mapper.ba_local_num_images = 3
        # opts.ba_global_frames_freq = 3
        if ba_backend == 'caspar':
            if not _caspar_usable():
                logger.info("Caspar BA requested but unavailable (pycolmap %s, cuda build: %s, gpu: %s) "
                            "— falling back to Ceres",
                            getattr(pycolmap, '__version__', '?'), getattr(pycolmap, 'has_cuda', False),
                            torch.cuda.is_available())
            elif not _caspar_supports_db_cameras(colmap_db_path):
                logger.warning("Caspar BA requested but the database contains camera models Caspar does not "
                               "support (only PINHOLE and SIMPLE_RADIAL are) — falling back to Ceres to avoid "
                               "a silent BA no-op")
            else:
                opts.ba_local_backend = pycolmap.BundleAdjustmentBackend.CASPAR
                opts.ba_global_backend = pycolmap.BundleAdjustmentBackend.CASPAR
                opts.ba_gpu_index = '0'  # index within CUDA_VISIBLE_DEVICES; must be a string
                logger.info("Using Caspar GPU bundle-adjustment backend")
        if initial_pair_provided:
            opts.init_image_id1 = first_image_id
            opts.init_image_id2 = second_image_id

        print(f"[pycolmap4-debug] run_mapper: starting incremental_mapping")
        maps = pycolmap.incremental_mapping(str(colmap_db_path), str(colmap_image_path), str(colmap_output_path),
                                            options=opts)
        print(f"[pycolmap4-debug] run_mapper: incremental_mapping returned {len(maps)} maps")
        if len(maps) > 0:
            # Pick the largest reconstruction (most registered images)
            best = max(maps.values(), key=lambda r: r.num_reg_images())
            if len(maps) > 1:
                sizes = [r.num_reg_images() for r in maps.values()]
                logger.warning("COLMAP produced %d reconstructions (sizes: %s), using largest (%d images)",
                               len(maps), sizes, best.num_reg_images())
            best_output_path = colmap_output_path / '0'
            print(f"[pycolmap4-debug] run_mapper: writing best map ({best.num_reg_images()} images) to {best_output_path}")
            best.write(str(best_output_path))
            print(f"[pycolmap4-debug] run_mapper: write done")
        return len(maps)
    else:
        raise ValueError(f"Need to run either glomap or colmap, got mapper={mapper}")


def get_match_points_indices(keypoints, match_pts):
    N = keypoints.shape[0]
    keypoints_and_match_pts = torch.cat([keypoints, match_pts], dim=0)
    _, kpts_and_match_pts_indices = torch.unique(keypoints_and_match_pts, return_inverse=True, dim=0)

    if kpts_and_match_pts_indices.max() >= N:
        raise ValueError("Not all src_pts included in keypoints")
    assert torch.equal(keypoints[kpts_and_match_pts_indices[:N]], keypoints)
    match_pts_indices = kpts_and_match_pts_indices[N:]
    return match_pts_indices


def keypoints_unique_preserve_order(keypoints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    first_kpt_idx_occurrence, inverse_indices = get_first_occurrence_indices(keypoints, dim=0)
    first_kpt_idx_occurrence_sorted, index_sort_permutation = torch.sort(first_kpt_idx_occurrence)
    unique_order_preserving = keypoints[first_kpt_idx_occurrence_sorted]

    idx_mapping = torch.zeros(len(index_sort_permutation), dtype=torch.long).to(keypoints.device)

    # Use scatter_ to place indices at the positions specified by index_sort_permutation
    # This creates our mapping from original positions to new positions after reordering
    idx_mapping.scatter_(0, index_sort_permutation, torch.arange(len(index_sort_permutation), device=keypoints.device))

    # Apply the mapping to get correct inverse indices
    inverse_indices_order_preserving = idx_mapping[inverse_indices]

    assert torch.all(torch.eq(keypoints, unique_order_preserving[inverse_indices_order_preserving]).view(-1))

    return unique_order_preserving, inverse_indices_order_preserving


def unique_keypoints_from_matches(matching_edges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
                                  existing_database: pycolmap.Database = None,
                                  matching_edges_certainties: Dict[Tuple[int, int], torch.Tensor] = None,
                                  eliminate_one_to_many_matches: bool = True, device: str = 'cpu') -> (
        Tuple)[Dict[int, torch.Tensor], Dict[Tuple[int, int], torch.Tensor]]:
    # Run keypoint uniquification entirely on CPU. For a dense Complete view graph one
    # node's merged matches (after track densification) total tens of GB; moving them to
    # the GPU here (per-node cat) OOMs the 40 GB A100 inside _write_colmap_database — this
    # is where the every-4 Complete HOPE OOM actually lands once the matching-phase dicts
    # are already CPU-resident. CPU RAM (256 GB) holds one node's matches comfortably, and
    # every output of this function is consumed via numpy(force=True), so device is
    # irrelevant to correctness. The COLMAP mapper, not this step, dominates reconstruction
    # time, so the runtime impact is small (and t_match is untouched).
    device = 'cpu'
    G = nx.DiGraph()
    G.add_edges_from(matching_edges.keys())

    existing_database_image_ids = []
    if existing_database is not None:
        existing_database_image_ids = [img.image_id for img in existing_database.read_all_images()]

    keypoints_for_node: Dict[int, torch.Tensor] = {}
    edge_match_indices: Dict[Tuple[int, int, int], torch.Tensor] = defaultdict(lambda: torch.zeros(0, ).to(device))

    for u in G.nodes():

        incoming_edges = list(G.in_edges(u))
        outgoing_edges = list(G.out_edges(u))

        if u in existing_database_image_ids and existing_database.read_keypoints(u).shape[0] > 0:
            existing_keypoints_u = [torch.from_numpy(existing_database.read_keypoints(u)).to(device)]
        else:
            existing_keypoints_u = [torch.zeros((0, 2)).to(torch.int).to(device)]
        existing_keypoints_lengths = [existing_keypoints_u[0].shape[0]]

        # matching_edges are stored on CPU (see _match_image_pairs); move only this node's
        # keypoints to `device` for the cat/unique below (bounded by one node's matches).
        keypoints_u_incoming_list = [matching_edges[v, u][1].to(device) for v, _ in incoming_edges]
        keypoints_u_incoming_list_lengths = [matching_edges[v, u][1].shape[0] for v, _ in incoming_edges]
        keypoints_u_outgoing_list = [matching_edges[u, v][0].to(device) for _, v in outgoing_edges]
        keypoints_u_outgoing_list_lengths = [matching_edges[u, v][0].shape[0] for _, v in outgoing_edges]

        keypoints_u_all_lists = existing_keypoints_u + keypoints_u_incoming_list + keypoints_u_outgoing_list
        keypoints_u_all = torch.cat(keypoints_u_all_lists)

        keypoints_u_unique, match_indices_order_preserving = keypoints_unique_preserve_order(keypoints_u_all)

        num_existing = existing_keypoints_lengths[0]
        num_incoming = int(np.sum(keypoints_u_incoming_list_lengths))
        num_outgoing = int(np.sum(keypoints_u_outgoing_list_lengths))
        match_indices_sizes = [num_existing, num_incoming, num_outgoing]
        match_indices_delimiters = np.cumsum(match_indices_sizes)

        match_indices_existing, match_indices_incoming, match_indices_outgoing = (
            torch.split(match_indices_order_preserving, match_indices_sizes))

        if match_indices_incoming.shape[0] > 0:
            keypoints_matches_incoming_indices = match_indices_order_preserving[match_indices_delimiters[0]:
                                                                                match_indices_delimiters[1]]

            keypoints_matches_incoming_indices_split = torch.split(keypoints_matches_incoming_indices,
                                                                   keypoints_u_incoming_list_lengths)

            for i, (v, _) in enumerate(incoming_edges):
                # Handle if both (u, v), and (v, u) exists
                edge_match_indices[v, u, 1] = torch.cat([edge_match_indices[v, u, 1],
                                                         keypoints_matches_incoming_indices_split[i]], dim=0)

        if match_indices_outgoing.shape[0] > 0:
            keypoints_matches_outgoing_indices = match_indices_order_preserving[match_indices_delimiters[1]:]
            keypoints_matches_outgoing_indices_split = torch.split(keypoints_matches_outgoing_indices,
                                                                   keypoints_u_outgoing_list_lengths)

            for i, (_, v) in enumerate(outgoing_edges):
                # Handle if both (u, v), and (v, u) exists
                edge_match_indices[u, v, 0] = torch.cat([edge_match_indices[u, v, 0],
                                                         keypoints_matches_outgoing_indices_split[i]], dim=0)

        keypoints_for_node[u] = keypoints_u_unique

    edge_match_indices_concatenated = {}
    for u, v in G.to_undirected().edges():
        # `to_undirected()` does not preserve the orientation the edge was stored under: a
        # pair matched only as (v, u) can be reported here as (u, v). That orientation is
        # absent from matching_edges_certainties (KeyError, which killed whole sequences)
        # and from edge_match_indices, whose defaultdict would otherwise hand back empty
        # tensors and silently drop the edge. Resolve to the direction actually present;
        # when both directions exist, keep (u, v) as before.
        if not G.has_edge(u, v) and G.has_edge(v, u):
            u, v = v, u

        keypoints_indices_u = edge_match_indices[u, v, 0]
        keypoints_indices_v = edge_match_indices[u, v, 1]

        if eliminate_one_to_many_matches:
            if matching_edges_certainties is not None:
                certainty = matching_edges_certainties[u, v].to(device)
            else:
                certainty = torch.zeros(keypoints_indices_v.shape[0], device=device)

            certainty_sort_idx = torch.argsort(certainty, descending=True)
            keypoints_indices_u_sorted = keypoints_indices_u[certainty_sort_idx]
            keypoints_indices_v_sorted = keypoints_indices_v[certainty_sort_idx]

            unique_keypoints_indices_u, _ = get_first_occurrence_indices(keypoints_indices_u_sorted)
            unique_keypoints_indices_v, _ = get_first_occurrence_indices(keypoints_indices_v_sorted)

            unique_keypoints_mask_u = torch.zeros_like(keypoints_indices_u, device=device, dtype=torch.bool)
            unique_keypoints_mask_v = torch.zeros_like(keypoints_indices_v, device=device, dtype=torch.bool)

            unique_keypoints_mask_u[unique_keypoints_indices_u] = True
            unique_keypoints_mask_v[unique_keypoints_indices_v] = True

            ono_to_one_mask = unique_keypoints_mask_u & unique_keypoints_mask_v

            keypoints_indices_u = keypoints_indices_u_sorted[ono_to_one_mask]
            keypoints_indices_v = keypoints_indices_v_sorted[ono_to_one_mask]

        stacked_indices = torch.stack([keypoints_indices_u, keypoints_indices_v], dim=1)

        edge_match_indices_concatenated[(u, v)] = stacked_indices

    return keypoints_for_node, edge_match_indices_concatenated


def get_first_occurrence_indices(elements: torch.Tensor, dim: Optional[int] = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the indices where each unique element first appears in the input tensor. Unlike torch.unique, this function
    guarantees ordering.

    This function identifies all unique elements in the input tensor and returns:
    1. The indices where each unique element first appears in the original tensor
    2. A mapping from each position in the original tensor to its corresponding unique element index

    Args:
        elements: Input tensor containing potentially duplicate elements
        dim: The dimension along which to find unique elements, if elements is multi-dimensional

    Returns:
        Tuple containing:
        - first_occurrence_indices: Indices where each unique element first appears
        - element_to_unique_mapping: Mapping from original positions to unique element indices
    """
    # Find unique elements and get mapping information
    unique_elements, element_to_unique_mapping, occurrence_counts = torch.unique(
        elements,
        sorted=True,
        dim=dim,
        return_inverse=True,
        return_counts=True
    )

    # Get indices that would sort the mapping array while preserving order of equal elements
    _, sorted_positions = torch.sort(element_to_unique_mapping, stable=True)

    # Calculate cumulative occurrence counts
    cumulative_counts = occurrence_counts.cumsum(0)

    # Shift cumulative counts to get starting positions of each unique value
    N = occurrence_counts.size(0)
    zero = torch.tensor([0], device=cumulative_counts.device, dtype=cumulative_counts.dtype)
    starting_positions = torch.cat((zero, cumulative_counts[:-1]))[:N]  # [:N] when called the function on empty Tensor

    # first-occurrence indices
    first_occurrence_indices = sorted_positions[starting_positions]

    return first_occurrence_indices, element_to_unique_mapping


def _first_frame_depth_pairs(reconstruction: pycolmap.Reconstruction, first_image: pycolmap.Image,
                             depth_map: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Sample (pred_depth, gt_depth) correspondences at the first frame for scale recovery.

    Primary path: read the first image's own 2D-3D observations (COLMAP, and the
    feed-forward methods that emit per-image tracks — VGGT, Pi3).

    Fallback path: when the first image carries no 3D tracks — feed-forward methods
    that emit a *global* point cloud with no per-image observations, e.g. Mast3r /
    MapAnything — project the entire reconstructed cloud into the first camera and
    sample the depth there instead. Without this, alignment fails outright ("empty
    tracks") and those methods get no dynamic numbers.

    In both paths the projected pixel coordinates (in the reconstruction camera's
    resolution) are rescaled to the GT depth-map grid, so a resolution mismatch
    between the camera and the depth map does not corrupt the lookup; GT pixels with
    non-positive depth (holes / background) are dropped so they cannot bias the
    median-ratio scale.
    """
    cam_from_world = first_image.cam_from_world()
    camera = reconstruction.cameras[first_image.camera_id]
    depth_h, depth_w = depth_map.shape[:2]
    sx = depth_w / camera.width
    sy = depth_h / camera.height

    pred, gt = [], []

    def _gt_at(u_img: float, v_img: float):
        u = int(round(u_img * sx))
        v = int(round(v_img * sy))
        if 0 <= u < depth_w and 0 <= v < depth_h:
            d = depth_map[v, u].item()
            return d if d > 0 else None
        return None

    # Primary: the first image's stored 2D-3D observations.
    for point2D in first_image.points2D:
        if not point2D.has_point3D():
            continue
        point3D_cam = cam_from_world * reconstruction.point3D(point2D.point3D_id).xyz
        if point3D_cam[2] <= 0:
            continue
        depth_gt = _gt_at(point2D.xy[0], point2D.xy[1])
        if depth_gt is not None:
            pred.append(float(point3D_cam[2]))
            gt.append(depth_gt)

    if pred:
        return np.asarray(pred), np.asarray(gt)

    # Fallback: project the whole cloud into the first camera (global-cloud methods).
    pids = list(reconstruction.points3D.keys())
    if not pids:
        return np.empty(0), np.empty(0)
    xyz_world = np.stack([reconstruction.point3D(pid).xyz for pid in pids])  # (P, 3)
    R = cam_from_world.rotation.matrix()
    t = cam_from_world.translation
    p_cam = xyz_world @ R.T + t                                             # (P, 3) camera coords
    z = p_cam[:, 2]
    in_front = z > 1e-6
    zc = np.where(in_front, z, 1.0)
    params = np.asarray(camera.params, dtype=np.float64)
    if params.shape[0] == 4:                                               # PINHOLE: [fx, fy, cx, cy]
        fx, fy, cx, cy = params
        u = fx * p_cam[:, 0] / zc + cx
        v = fy * p_cam[:, 1] / zc + cy
    else:                                                                  # generic fallback (any model)
        u = np.full(len(z), -1.0)
        v = np.full(len(z), -1.0)
        idxf = np.flatnonzero(in_front)
        if len(idxf):
            uv = np.stack([camera.img_from_cam(p_cam[i]) for i in idxf], axis=0)
            u[idxf], v[idxf] = uv[:, 0], uv[:, 1]
    ui = np.round(u * sx).astype(np.int64)
    vi = np.round(v * sy).astype(np.int64)
    ok = in_front & (ui >= 0) & (ui < depth_w) & (vi >= 0) & (vi < depth_h)
    for i in np.flatnonzero(ok):
        d_gt = depth_map[vi[i], ui[i]].item()
        if d_gt > 0:
            pred.append(float(z[i]))
            gt.append(d_gt)
    return np.asarray(pred), np.asarray(gt)


def align_reconstruction_with_pose(reconstruction: pycolmap.Reconstruction, first_image_gt_Se3_world2cam: Se3,
                                   image_depths: Dict[str, torch.Tensor], first_image_name: str) \
        -> Tuple[pycolmap.Reconstruction, bool]:
    reconstruction = copy.deepcopy(reconstruction)

    if not (first_image_colmap := reconstruction.find_image_with_name(first_image_name)):
        print("Alignment error. The 1st image was not registered.")
        return reconstruction, False

    first_image = reconstruction.find_image_with_name(first_image_name)
    first_image_name = first_image.name

    depth_map = image_depths[first_image_name]

    pred_first_image_point_depths, gt_first_image_point_depths = _first_frame_depth_pairs(
        reconstruction, first_image, depth_map)

    if len(pred_first_image_point_depths) == 0:
        print("Alignment error: no usable depth correspondences in the first image "
              "(empty tracks and the projected cloud falls outside the GT depth map).")
        return reconstruction, False

    median_gt = np.median(gt_first_image_point_depths)
    median_pred = np.median(pred_first_image_point_depths)
    if median_pred == 0 or median_gt == 0:
        print(f"Alignment error: degenerate depths (median_gt={median_gt:.4f}, median_pred={median_pred:.4f}).")
        return reconstruction, False

    # scale = metric depth (GT, mm) / colmap depth at the same first-frame points
    #       = millimetres per COLMAP unit. The aligning similarity must MULTIPLY the
    # colmap coordinates by this scale (mapping colmap units -> mm), so the composed
    # Sim3d below carries `scale`, NOT `1.0/scale`. Using 1/scale inverts the mapping:
    # it collapses the cloud (extent -> ~0) onto the GT camera-centre translation,
    # so every point lands far past the eval's 20 mm clamp (fscore -> 0). Requires the
    # GT depth in mm (the caller converts) and a metrically-consistent reconstruction
    # (true intrinsics, use_default_colmap_K=False), else the size/distance ratio is off.
    scale = median_gt / median_pred

    colmap_cam_from_world = first_image_colmap.cam_from_world()
    gt_cam_from_world = Se3_to_Rigid3d(first_image_gt_Se3_world2cam)
    gt_world_from_cam = gt_cam_from_world.inverse()

    colmap_world_from_cam = colmap_cam_from_world.inverse()  # world_from_cam

    gt_world_from_cam_scaled = pycolmap.Sim3d(
        scale=scale,
        rotation=gt_world_from_cam.rotation,
        translation=gt_world_from_cam.translation
    )

    colmap_world_from_cam_sim3d = pycolmap.Sim3d(
        scale=1.0,
        rotation=colmap_world_from_cam.rotation,
        translation=colmap_world_from_cam.translation
    )

    Sim3d_first_image_colmap2gt = gt_world_from_cam_scaled * colmap_world_from_cam_sim3d.inverse()

    reconstruction.transform(Sim3d_first_image_colmap2gt)

    return reconstruction, True

