from pathlib import Path
from typing import Dict

import numpy as np
from kornia.geometry import Se3

from data_structures.view_graph import ViewGraph
from eval.eval_point_cloud import (
    sample_points_from_mesh,
    extract_reconstruction_points,
    compute_reconstruction_metrics,
)
from eval.eval_reconstruction import (
    evaluate_reconstruction,
    update_sequence_reconstructions_stats,
)
from eval.aggregate_stats import (
    SEQUENCE_STATS, KEYFRAME_STATS, experiment_root_of, rebuild_experiment_aggregates,
)
from configs.glopose_config import RunConfig, PathsConfig
from configs.components.bop_config import BaseBOPConfig


def gt_mesh_unit_for_dataset(dataset: str) -> str:
    """Return the coordinate unit of the GT mesh for a given dataset.

    BOP-format datasets store models in millimetres; HO3D and Google Scanned
    Objects use metres. NAVI's 3d_scan meshes AND its annotation camera world
    are both in scanner units ~= millimetres (e.g. 3d_dollhouse_sink extent
    ~82x126x77, camera |t| ~604 — an ~10 cm toy filmed from ~60 cm), NOT
    metres: labelling them 'm' scales both clouds x1000 and clamps every
    mesh metric at max_dist.
    """
    dataset_lower = dataset.lower()
    # 'googlescanned' with no separator: run.dataset is the string
    # 'GoogleScannedObjects', which contains neither 'gso' nor 'google_scanned',
    # so both of those spellings silently fell through to the mm default. That
    # would have clamped every GSO mesh metric at max_dist and reported F@5 = 0,
    # i.e. a fabricated total failure, the moment the meshes were wired up.
    metre_datasets = {'ho3d', 'gso', 'google_scanned', 'googlescanned'}
    for name in metre_datasets:
        if name in dataset_lower:
            return 'm'
    # BOP datasets (handal, hope, tless, lmo, ycbv, …) and NAVI use mm
    return 'mm'


def renderer_mesh_normalization(mesh_path: Path) -> tuple:
    """The (centre, scale) that the synthetic renderer applies before rendering.

    Synthetic sequences are rendered from a mesh that models/encoder.py first passes
    through utils.general.normalize_vertices: centre on the vertex mean, then divide
    by the largest absolute centred coordinate so the object fits max|v| = 1. The
    reconstruction therefore lives in NORMALISED object units, while the GT mesh on
    disk is in metres. Comparing them directly leaves the clouds a factor 1/scale
    apart (measured ~12.8x on GSO, isotropic), which pins accuracy and completeness
    at the max_dist ceiling and reports F@5 = 0 for every configuration including
    ones whose poses are accurate to 2 degrees.

    Returns (centre, scale) so predicted points can be mapped back to mesh units as
    pred * scale + centre.
    """
    import trimesh
    mesh = trimesh.load(str(mesh_path), force='mesh')
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    centre = verts.mean(axis=0)
    scale = float(np.abs(verts - centre).max())
    return centre, (scale if scale != 0 else 1.0)


def uses_renderer_normalization(dataset: str) -> bool:
    """Datasets whose frames are rendered from a normalised mesh at eval time."""
    d = dataset.lower()
    return 'googlescanned' in d or 'google_scanned' in d or 'syntheticobjects' in d


def evaluate_onboarding(
    view_graph: ViewGraph,
    gt_Se3_world2cam: Dict[int, Se3] | None,
    run: RunConfig,
    bop: BaseBOPConfig,
    write_folder: Path,
) -> None:
    """Evaluate an onboarding result and write CSV statistics.

    Reads metadata (timing, success flags, image_name_to_frame_id) from the
    ViewGraph. Writes per-keyframe, per-sequence, and per-dataset CSVs to the
    experiment results folder (two levels above write_folder).

    Args:
        view_graph: The ViewGraph returned by run_pipeline(), carrying metadata.
        gt_Se3_world2cam: Ground-truth world-to-camera poses keyed by frame index,
            or None if GT is unavailable.
        run: The RunConfig used for this run.
        bop: The BaseBOPConfig used for this run.
        write_folder: The per-sequence output folder (e.g. results/exp/dataset/seq/).
    """
    keyframe_nodes = sorted(view_graph.view_graph.nodes)
    num_keyframes = len(keyframe_nodes)

    # Determine whether we know GT poses for any keyframe. For dynamic onboarding
    # only a subset of frames have GT in scene_gt.json (sometimes only the first
    # frame), so we evaluate against whatever subset is available rather than
    # skipping the sequence entirely. evaluate_reconstruction() and the
    # per-sequence aggregation both tolerate keyframes without a GT pose — those
    # rows are simply skipped when computing pose errors.
    if gt_Se3_world2cam is not None and len(gt_Se3_world2cam) > 0:
        have_gt_poses = any(idx in gt_Se3_world2cam for idx in keyframe_nodes)
    else:
        have_gt_poses = False

    # Dynamic onboarding sequences (HANDAL/HOPE/HOT3D _dynamic) only have GT for the
    # first frame, so per-keyframe pose error/accuracy/AUC are meaningless. Suppress
    # those columns (left blank) while still recording reconstruction stats. Pose
    # columns are populated solely from the per-keyframe CSV, so skipping the
    # per-keyframe evaluation below naturally leaves them None.
    is_dynamic = bop.onboarding_type == 'dynamic'

    # Without GT poses there is nothing to evaluate — except for dynamic sequences,
    # where we still want the reconstruction stats (keyframes, registration, success).
    if not have_gt_poses and not is_dynamic:
        return

    # Build dataset/sequence names for CSV columns
    dataset_name_for_eval = run.dataset
    if bop.onboarding_type is not None:
        dataset_name_for_eval = f'{dataset_name_for_eval}_{bop.onboarding_type}_onboarding'

    sequence_name = run.sequence
    if run.special_hash is not None and len(run.special_hash) > 0:
        sequence_name = f'{sequence_name}_{run.special_hash}'

    # Per-sequence CSVs live IN the sequence folder (<experiment>/<dataset>/<sequence>/):
    # each run only ever writes its own files, so concurrent jobs of one
    # experiment cannot race on a shared CSV. The experiment-level aggregates two levels
    # up are rebuilt atomically from these at the end (see eval/aggregate_stats.py).
    rec_csv_detailed_stats = write_folder / KEYFRAME_STATS
    rec_csv_per_sequence_stats = write_folder / SEQUENCE_STATS

    # Load COLMAP reconstruction from disk if it succeeded. The directory may exist but
    # lack the COLMAP binary files (e.g. external methods whose reconstruction path was not
    # written, or a failed/empty mapper run) — in that case pycolmap raises. Treat any load
    # failure as "no reconstruction" so the per-sequence CSV row is still written (as a
    # failure) instead of crashing the whole eval and leaving the sequence with no result.
    reconstruction = None
    if view_graph.reconstruction_success:
        import pycolmap
        rec_path = view_graph.colmap_reconstruction_path
        if rec_path is not None and rec_path.exists():
            try:
                reconstruction = pycolmap.Reconstruction(str(rec_path))
            except Exception as e:
                print(f"Warning: failed to load reconstruction from {rec_path}: {e}")
                reconstruction = None

    # Per-keyframe evaluation (rotation/translation errors).
    # Skipped for dynamic sequences: with only first-frame GT, pose errors are
    # degenerate. Skipping leaves all pose columns blank in the sequence CSV.
    if reconstruction is not None and have_gt_poses and not is_dynamic:
        evaluate_reconstruction(
            reconstruction, gt_Se3_world2cam, view_graph.image_name_to_frame_id,
            rec_csv_detailed_stats, dataset_name_for_eval, sequence_name,
        )

    # 3D reconstruction quality (point cloud vs GT mesh)
    reconstruction_quality = None
    if reconstruction is not None and view_graph.gt_model_path is not None and view_graph.gt_model_path.exists():
        try:
            gt_pts = sample_points_from_mesh(view_graph.gt_model_path)
            pred_pts = extract_reconstruction_points(reconstruction)
            unit = gt_mesh_unit_for_dataset(run.dataset)
            if uses_renderer_normalization(run.dataset):
                # Undo the renderer's normalisation so the prediction is back in the
                # GT mesh's own units before any millimetre threshold is applied.
                centre, scale = renderer_mesh_normalization(view_graph.gt_model_path)
                pred_pts = pred_pts * scale + centre
            # The reconstruction is aligned into the dataset's GT-pose world units
            # (Kabsch to GT poses; depth alignment converts to mm explicitly for the
            # BOP dynamic path), which coincide with the GT mesh units — metres for
            # NAVI/HO3D, millimetres for BOP. Passing pred_points_unit='mm' for a
            # metre-unit dataset clamps every metric at max_dist (the NAVI/HO3D bug).
            reconstruction_quality = compute_reconstruction_metrics(
                pred_pts, gt_pts, gt_mesh_unit=unit, pred_points_unit=unit,
            )
        except Exception as e:
            print(f"Warning: point cloud evaluation failed: {e}")

    # Per-sequence summary statistics
    reconstruction_success = view_graph.reconstruction_success
    alignment_success = view_graph.alignment_success

    update_sequence_reconstructions_stats(
        rec_csv_detailed_stats, rec_csv_per_sequence_stats, num_keyframes,
        view_graph.num_input_frames, reconstruction, dataset_name_for_eval,
        sequence_name, reconstruction_success, alignment_success,
        view_graph.frame_filtering_time, view_graph.reconstruction_time,
        reconstruction_quality=reconstruction_quality,
        colmap_num_reconstructions=view_graph.colmap_num_reconstructions,
        matching_time=view_graph.matching_time,
    )

    # Experiment-level aggregates (<experiment>/reconstruction_{sequence,keyframe,dataset}_stats.csv),
    # rebuilt from all per-sequence files and replaced atomically.
    rebuild_experiment_aggregates(experiment_root_of(write_folder))


def resolve_gt_model_path(run: RunConfig, paths: PathsConfig) -> Path | None:
    """Resolve the path to the GT 3D model for the current dataset/object.

    Returns None if no GT model is available for the dataset.
    """
    dataset = run.dataset
    object_id = run.object_id

    # BOP datasets (handal, hope, tless, lmo, icbin, etc.)
    bop_datasets = {'handal', 'handal_native', 'hope', 'tless', 'lmo', 'icbin', 'itodd', 'tudl', 'ycbv', 'hb'}
    dataset_lower = dataset.lower()
    for bop_name in bop_datasets:
        if bop_name in dataset_lower:
            try:
                obj_int = int(object_id)
            except (ValueError, TypeError):
                return None
            bop_dataset_name = bop_name
            if bop_dataset_name == 'handal_native':
                bop_dataset_name = 'handal'
            dataset_dir = paths.bop_data_folder / bop_dataset_name
            # Prefer the plain `models/` mesh (keeps every existing experiment's numbers
            # identical). Some BOP datasets (e.g. tless) ship no `models/`, only
            # `models_eval/` (canonical BOP eval mesh, mm units) — fall back to it so
            # F-score isn't silently blank. Additive: never overrides an existing models/.
            model_path = dataset_dir / 'models' / f'obj_{obj_int:06d}.ply'
            if model_path.exists():
                return model_path
            eval_model_path = dataset_dir / 'models_eval' / f'obj_{obj_int:06d}.ply'
            return eval_model_path if eval_model_path.exists() else None

    # NAVI
    if 'navi' in dataset_lower:
        if object_id is not None:
            obj_name = str(object_id)
            model_path = paths.navi_data_folder / obj_name / '3d_scan' / f'{obj_name}.obj'
            return model_path if model_path.exists() else None

    # HO3D
    if 'ho3d' in dataset_lower:
        if object_id is not None:
            model_path = paths.ho3d_data_folder / 'models' / str(object_id) / 'textured_simple.obj'
            return model_path if model_path.exists() else None

    # Google Scanned Objects. Keyed on run.sequence, not run.object_id: the synthetic
    # runner sets only the sequence (the object folder name) and leaves object_id
    # unset, so an object_id lookup returns None and the shape metrics stay blank.
    # That is why every GSO run so far reports pose but an empty F@5 column.
    if 'googlescanned' in dataset_lower or 'google_scanned' in dataset_lower:
        if run.sequence:
            model_path = (paths.google_scanned_objects_data_folder / 'models'
                          / str(run.sequence) / 'meshes' / 'model.obj')
            return model_path if model_path.exists() else None
        return None

    # YCBInEOAT — objects are YCB, so use the canonical BOP ycbv meshes (mm units).
    # run_YCBInEOAT sets run.object_id to the ycbv object integer.
    if 'ycbineoat' in dataset_lower:
        try:
            obj_int = int(object_id)
        except (ValueError, TypeError):
            return None
        model_path = paths.bop_data_folder / 'ycbv' / 'models' / f'obj_{obj_int:06d}.ply'
        return model_path if model_path.exists() else None

    return None
