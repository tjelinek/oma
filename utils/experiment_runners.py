import copy
import hashlib
import inspect
import logging
import os
from pathlib import Path
from typing import Dict, List, Any

import pycolmap
import torch
from kornia.geometry import Se3, Quaternion

from data_providers.frame_provider import PrecomputedSegmentationProvider
from dataset_generators import scenarios
from eval.eval_onboarding import evaluate_onboarding
from configs.glopose_config import GloPoseConfig
from onboarding.pipeline import OnboardingPipeline
from utils.bop_challenge import get_bop_images_and_segmentations, read_gt_Se3_cam2obj_transformations, \
    read_pinhole_params, read_static_onboarding_world2cam, add_extrinsics_to_pinhole_params, read_object_id, \
    get_sequence_folder, _scene_camera_filename, _hot3d_camera_suffix
from utils.data_utils import load_texture, load_mesh
from utils.math_utils import Se3_obj_relative_to_Se3_cam2obj

logger = logging.getLogger(__name__)


def resolve_write_folder(paths, experiment_name: str, dataset: str, sequence_dirname: str,
                         output_folder=None) -> Path:
    """Canonical per-run results folder: ``<root>/<experiment>/<dataset>/<sequence_dirname>/``.

    ``output_folder`` (the runners' ``--output_folder``) IS the ``<root>/<experiment>`` level;
    the ``<dataset>/<sequence>`` levels are always appended here. Submission scripts must
    therefore pass ``RESULTS_ROOT/<experiment>`` and nothing deeper. ``<experiment>`` itself
    may be nested (``final_ablation/ablation1``, ``table5/<stem>``); the aggregates are then
    rebuilt at that level. Only a folder whose LAST component is the dataset or sequence name
    is refused, because that is exactly what produced the ``.../<ds>/<seq>/<ds>/<seq>/`` trees
    (see CLAUDE.md, "Results layout"). ``GLOPOSE_ALLOW_NESTED_OUTPUT=1`` bypasses the check.
    """
    if output_folder is not None:
        root = Path(output_folder)
        nested = root.name in (sequence_dirname, dataset)
        if nested and not os.environ.get('GLOPOSE_ALLOW_NESTED_OUTPUT'):
            raise ValueError(
                f"--output_folder '{root}' already ends in the dataset/sequence name. It must be the "
                f"'<results_root>/<experiment>' level; '{dataset}/{sequence_dirname}' is appended by the "
                f"runner (would have produced {root / dataset / sequence_dirname}).")
    else:
        root = Path(paths.results_folder) / experiment_name
    return root / dataset / sequence_dirname


def run_on_synthetic_data(config: GloPoseConfig, dataset: str, sequence: str, experiment=None, output_folder=None,
                          gt_mesh_path: Path = None, gt_texture_path: Path = None, rotation_generator=None):
    """
    Common function to run 6D tracking across different datasets

    Parameters:
    -----------
    config : GloPoseConfig
        Configuration object that has already been loaded and customized
    dataset : str
        Name of the dataset (e.g., 'SyntheticObjects', 'GoogleScannedObjects')
    sequence : str
        Specific sequence to run tracking on
    experiment : str, optional
        Experiment name
    output_folder : str, optional
        Output folder to save results
    gt_mesh_path : Path, optional
        Path to the ground truth mesh
    gt_texture_path : Path, optional
        Path to the ground truth texture
    rotation_generator : function, optional
        Function to generate rotations (defaults to y-axis rotations)
    skip_frames : int, optional
        Number of frames to skip in the sequence (defaults to 1 - no skipping)
    """
    from models.rendering import get_Se3_obj2cam_from_kaolin_params

    # Use provided experiment name or get from config if available
    if experiment is None and hasattr(config.run, 'experiment_name'):
        experiment = config.run.experiment_name

    # Update configuration
    config.run.experiment_name = experiment
    config.run.sequence = sequence
    config.run.dataset = dataset
    config.paths.purge_cache = False
    config.visualization.large_images_write_frequency = 1

    # Set mesh and texture paths
    config.renderer.gt_texture_path = gt_texture_path
    config.renderer.gt_mesh_path = gt_mesh_path

    # Load mesh and texture
    gt_texture = load_texture(Path(config.renderer.gt_texture_path), config.renderer.texture_size)
    gt_mesh = load_mesh(Path(config.renderer.gt_mesh_path))

    # Generate rotations
    if rotation_generator is None:
        rotation_generator = scenarios.generate_rotations_y

    # Fixed-step generators (generate_rotations_*) take `step`; the random-walk
    # scenarios take `n_steps`/`seed` and raise TypeError on `step`, which used to
    # kill every GoogleScannedObjects run before the first frame.
    if 'step' in inspect.signature(rotation_generator).parameters:
        scenario = rotation_generator(step=5)
    else:
        scenario = rotation_generator()
    gt_rotations = torch.deg2rad(scenario.rotations).to(config.run.device)
    gt_rotations = torch.cat([gt_rotations, gt_rotations], dim=0)

    # Create image paths
    images_paths = [Path(f'{i}.png') for i in range(gt_rotations.shape[0])]

    # Generate translations (zero by default)
    gt_translations = scenarios.generate_sinusoidal_translations(steps=gt_rotations.shape[0]).translations * 0
    gt_translations = gt_translations.to(config.run.device)

    trace_hash = hashlib.md5(gt_rotations.numpy(force=True).tobytes() + gt_translations.numpy(force=True).tobytes())
    config.run.special_hash = trace_hash.hexdigest()

    # Set number of input frames
    config.input.input_frames = gt_rotations.shape[0]

    # Create Se3 transformations
    gt_obj_1_to_obj_i_Se3 = Se3(Quaternion.from_axis_angle(gt_rotations), gt_translations)

    # Set up camera parameters
    camera_trans = torch.FloatTensor(config.renderer.camera_position)[None].to(config.run.device)
    up = torch.FloatTensor(config.renderer.camera_up)[None].to(config.run.device)
    obj_center = torch.FloatTensor(config.renderer.obj_center)[None].to(config.run.device)

    gt_Se3_obj2cam = get_Se3_obj2cam_from_kaolin_params(camera_trans, up, obj_center)
    gt_Se3_cam2obj = Se3_obj_relative_to_Se3_cam2obj(gt_obj_1_to_obj_i_Se3, gt_Se3_obj2cam)
    gt_Se3_cam2obj_dict = {i: gt_Se3_cam2obj[i] for i in range(config.input.input_frames)}

    # Object-relative camera poses. The camera is physically static and the object
    # rotates, so in the object frame the cameras trace a sphere of radius
    # |camera_position|; passing identity here (the previous behaviour) made both
    # Kabsch alignment (N identical poses, degenerate) and the per-keyframe pose
    # error (every camera compared against identity) meaningless.
    #
    # Despite its name, `Se3_obj_relative_to_Se3_cam2obj` already returns the
    # world-to-camera (cam_from_world) transform for the object-centric world, so
    # it is used DIRECTLY here and not inverted (the BOP path inverts because its
    # `cam2obj` comes from a different reader with the opposite direction).
    # Verified against the rendered frames: with this convention the reconstructed
    # keyframe rotations sit 2.1 deg from GT after removing the unobservable global
    # rotation, versus 109 deg (chance) when inverted. Note that neither the
    # camera-centre radius nor the inter-keyframe rotation magnitudes can tell the
    # two apart, since the inverted convention differs by a conjugation.
    gt_Se3_world2cam_dict = dict(gt_Se3_cam2obj_dict)

    # Set up output folder (canonical <root>/<experiment>/<dataset>/<sequence>/)
    write_folder = resolve_write_folder(config.paths, experiment, dataset, sequence, output_folder)

    # Create and run tracker
    tracker = OnboardingPipeline(config, write_folder, input_images=images_paths, gt_texture=gt_texture,
                                 gt_mesh=gt_mesh,
                                 gt_Se3_cam2obj=gt_Se3_cam2obj_dict, gt_Se3_world2cam=gt_Se3_world2cam_dict)

    view_graph = tracker.run_pipeline()
    # Pipeline GT copy, not the local dict: skip_indices decimation remaps the GT
    # keys inside OnboardingPipeline.__init__, and the full-rate dict then
    # misindexes every keyframe except frame 0 (see commit 015595a0).
    evaluate_onboarding(view_graph, tracker.gt_Se3_world2cam, config.run, config.bop, write_folder)

    return tracker


def reindex_frame_dict(frame_dict: Dict[int, Any], valid_frames: List[int]):
    frame_dict = {
        i: frame_dict[frame]
        for i, frame in enumerate(valid_frames) if frame in frame_dict
    }
    return frame_dict


def run_on_bop_sequences(dataset: str, experiment_name: str, sequence_type: str, config: GloPoseConfig, gt_cam_scale,
                         output_folder: Path = None, scene_obj_id: int = None, merge_only: bool = False):
    onboarding_type = config.bop.onboarding_type
    sequence = config.run.sequence

    # Path to BOP dataset
    bop_folder = config.paths.bop_data_folder

    # HOT3D: prefix special_hash with device to avoid aria/quest3 cache collisions
    hot3d_prefix = f'{config.input.hot3d_device}_' if dataset == 'hot3d' else ''

    if onboarding_type == 'static':
        static_onboarding_sequence = config.bop.static_onboarding_sequence
        config.run.special_hash = f'{hot3d_prefix}{static_onboarding_sequence or "static"}'
    elif onboarding_type == 'dynamic':
        config.run.special_hash = f'{hot3d_prefix}dynamic'
        static_onboarding_sequence = None
    elif sequence_type in ['test', 'train', 'val']:
        config.run.special_hash = f'{scene_obj_id:06d}'
        static_onboarding_sequence = None
    else:
        raise ValueError("This should not happen")

    # --- Strategy: 'separate' for _both sequences — run up and down independently, then merge ---
    if (static_onboarding_sequence == 'both'
            and config.onboarding.both_merge_strategy == 'separate'):
        _run_separate_merge(dataset, experiment_name, sequence_type, config, gt_cam_scale,
                            output_folder, scene_obj_id, merge_only=merge_only)
        return

    # Determine output folder (canonical <root>/<experiment>/<dataset>/<sequence>_<hash>/)
    write_folder = resolve_write_folder(config.paths, experiment_name, dataset,
                                        f'{sequence}_{config.run.special_hash}', output_folder)

    # Load images and segmentations
    hot3d_dev = config.input.hot3d_device
    gt_images, gt_segs, gt_depths, sequence_starts = \
        get_bop_images_and_segmentations(bop_folder, dataset, sequence, sequence_type,
                                         onboarding_type, static_onboarding_sequence, scene_obj_id=scene_obj_id,
                                         hot3d_device=hot3d_dev)

    # HOT3D fisheye undistortion: replace distorted images/masks with pinhole-undistorted versions
    hot3d_pinhole_params = None
    if dataset == 'hot3d':
        from adapters.hot3d_adapter import undistort_hot3d_sequence
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type,
                                               direction=static_onboarding_sequence, hot3d_device=hot3d_dev)
        scene_camera_path = sequence_folder / _scene_camera_filename(dataset, hot3d_dev)
        undistort_cache = config.paths.cache_folder / 'hot3d_undistorted' / f'{hot3d_prefix}{sequence}_{onboarding_type}'
        gt_images, gt_segs, hot3d_pinhole_params = undistort_hot3d_sequence(
            scene_camera_path, gt_images, gt_segs, undistort_cache,
            scale=config.input.image_downsample, device=config.run.device)

    # Get camera-to-object transformations
    dict_gt_Se3_cam2obj = read_gt_Se3_cam2obj_transformations(bop_folder, dataset, sequence, sequence_type,
                                                              gt_cam_scale, onboarding_type,
                                                              sequence_starts, static_onboarding_sequence, scene_obj_id,
                                                              device=config.run.device, hot3d_device=hot3d_dev)

    object_id = read_object_id(bop_folder, dataset, sequence, sequence_type, onboarding_type,
                               static_onboarding_sequence, scene_obj_id, sequence_starts,
                               hot3d_device=hot3d_dev)
    config.run.object_id = object_id

    # Apply frame skipping
    if config.input.run_only_on_frames_with_known_pose:
        valid_frames = sorted(dict_gt_Se3_cam2obj.keys())
    else:
        valid_frames = list(range(min(gt_images.keys()), max(gt_images.keys()) + 1))

    gt_images = [gt_images[i] for i in valid_frames]
    gt_segs = [gt_segs.get(i) for i in valid_frames]
    if gt_depths is not None:
        gt_depths = [gt_depths[i] for i in valid_frames]
    dict_gt_Se3_cam2obj = reindex_frame_dict(dict_gt_Se3_cam2obj, valid_frames)

    # Compute sequence boundaries for the frame filter (after reindexing to 0-based)
    sequence_boundaries = None
    if static_onboarding_sequence == 'both' and sequence_starts:
        # sequence_starts contains the original boundary index; map it through valid_frames reindexing
        boundary_orig = sequence_starts[0] if len(sequence_starts) == 1 else sequence_starts[1]
        # Find the reindexed position of the first frame >= boundary_orig
        boundary_reindexed = None
        for new_idx, orig_idx in enumerate(valid_frames):
            if orig_idx >= boundary_orig:
                boundary_reindexed = new_idx
                break
        if boundary_reindexed is not None and boundary_reindexed > 0:
            sequence_boundaries = [boundary_reindexed]

    # Get initial image and segmentation
    initial_bbox = None
    if gt_segs[0] is not None:
        first_segmentation = PrecomputedSegmentationProvider.get_initial_segmentation(gt_images, gt_segs,
                                                                                      segmentation_channel=0)
    else:
        # No mask available (e.g. HOT3D dynamic) — read bbox from scene_gt_info for SAM2 box prompt
        first_segmentation = None
        import json
        gt_info_filename = f'scene_gt_info_{_hot3d_camera_suffix(hot3d_dev)}.json' if dataset == 'hot3d' else 'scene_gt_info.json'
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type,
                                               hot3d_device=hot3d_dev)
        gt_info_path = sequence_folder / gt_info_filename
        if gt_info_path.exists():
            with open(gt_info_path, 'r') as f:
                gt_info = json.load(f)
            frame_0_info = gt_info.get('0', [])
            if frame_0_info:
                bbox_xywh = frame_0_info[0].get('bbox_obj', [-1, -1, -1, -1])
                if bbox_xywh[0] >= 0:
                    x, y, w, h = bbox_xywh
                    s = config.input.image_downsample
                    initial_bbox = [x * s, y * s, (x + w) * s, (y + h) * s]  # xyxy, scaled to output resolution

    # Get camera parameters (HOT3D: use undistorted pinhole params, others: read from JSON)
    if hot3d_pinhole_params is not None:
        pinhole_params = hot3d_pinhole_params
    else:
        pinhole_params = read_pinhole_params(bop_folder, dataset, sequence, sequence_type, config.input.image_downsample,
                                             onboarding_type, static_onboarding_sequence, sequence_starts,
                                             config.run.device, hot3d_device=hot3d_dev)

    gt_Se3_world2cam = None
    if onboarding_type == 'static' or sequence_type in ['val', 'train']:
        gt_Se3_world2cam = read_static_onboarding_world2cam(bop_folder, dataset, sequence, sequence_type,
                                                            onboarding_type, static_onboarding_sequence,
                                                            sequence_starts, config.run.device,
                                                            hot3d_device=hot3d_dev)
    if gt_Se3_world2cam is not None:
        pinhole_params = add_extrinsics_to_pinhole_params(pinhole_params, gt_Se3_world2cam)

    pinhole_params = reindex_frame_dict(pinhole_params, valid_frames)
    if gt_Se3_world2cam is not None:
        gt_Se3_world2cam = reindex_frame_dict(gt_Se3_world2cam, valid_frames)

    if dict_gt_Se3_cam2obj is not None:
        gt_Se3_world2cam = {i: cam2obj.inverse() for i, cam2obj in dict_gt_Se3_cam2obj.items()}
        dict_gt_Se3_cam2obj = None

    # Update config with frame information
    config.input.input_frames = len(gt_images)
    config.input.frame_provider = 'precomputed'
    config.input.segmentation_provider = 'SAM2'

    # Initialize and run the tracker
    tracker = OnboardingPipeline(config, write_folder, input_images=gt_images, gt_Se3_world2cam=gt_Se3_world2cam,
                                 gt_pinhole_params=pinhole_params, input_segmentations=gt_segs, depth_paths=gt_depths,
                                 initial_segmentation=first_segmentation, initial_bbox=initial_bbox,
                                 sequence_boundaries=sequence_boundaries)

    view_graph = tracker.run_pipeline()
    # Pipeline GT copy, not the local dict: when skip_indices != 1 (e.g. HOT3D's 4x)
    # OnboardingPipeline.__init__ remaps GT to the decimated frame ids; the local
    # full-rate dict misindexes every keyframe except frame 0 during eval. Identical
    # to the local dict when skip_indices == 1 (HANDAL/HOPE).
    evaluate_onboarding(view_graph, tracker.gt_Se3_world2cam, config.run, config.bop, write_folder)


def _run_separate_merge(dataset: str, experiment_name: str, sequence_type: str, config: GloPoseConfig, gt_cam_scale,
                        output_folder: Path = None, scene_obj_id: int = None, merge_only: bool = False):
    """Run up and down onboarding separately, then merge the two ViewGraphs.

    Called from run_on_bop_sequences() when both_merge_strategy='separate' and
    static_onboarding_sequence='both'.

    Args:
        merge_only: If True, skip the up/down onboarding runs and go straight to
            merging. Requires cached ViewGraphs from a previous run.
    """
    import shutil
    from data_structures.view_graph import merge_two_view_graphs, load_view_graph
    from utils.results_logging import build_onboarding_blueprint, log_merge_to_rerun, log_reconstruction_to_rerun
    from visualizations.rerun_utils import init_rerun_recording

    sequence = config.run.sequence
    hot3d_prefix = f'{config.input.hot3d_device}_' if dataset == 'hot3d' else ''

    if not merge_only:
        # Run 'down' pass
        config_down = copy.deepcopy(config)
        config_down.bop.static_onboarding_sequence = 'down'
        config_down.run.special_hash = f'{hot3d_prefix}down'
        logger.info("Separate merge: running 'down' pass for %s", sequence)
        run_on_bop_sequences(dataset, experiment_name, sequence_type, config_down, gt_cam_scale,
                             output_folder, scene_obj_id)

        # Run 'up' pass
        config_up = copy.deepcopy(config)
        config_up.bop.static_onboarding_sequence = 'up'
        config_up.run.special_hash = f'{hot3d_prefix}up'
        logger.info("Separate merge: running 'up' pass for %s", sequence)
        run_on_bop_sequences(dataset, experiment_name, sequence_type, config_up, gt_cam_scale,
                             output_folder, scene_obj_id)
    else:
        logger.info("Separate merge: --merge-only, skipping up/down onboarding for %s", sequence)

    # Check cached ViewGraphs exist
    cache_root = config.paths.cache_folder / 'view_graph_cache' / experiment_name / dataset
    down_cache = cache_root / f'{sequence}_{hot3d_prefix}down'
    up_cache = cache_root / f'{sequence}_{hot3d_prefix}up'
    merged_cache = cache_root / f'{sequence}_{hot3d_prefix}both'

    if not down_cache.exists() or not up_cache.exists():
        logger.warning("Separate merge: missing cached ViewGraph(s) for %s (down=%s, up=%s)",
                       sequence, down_cache.exists(), up_cache.exists())
        return

    # Load sub-ViewGraphs to read their metadata before merge overwrites them
    vg_down = load_view_graph(down_cache, device='cpu')
    vg_up = load_view_graph(up_cache, device='cpu')

    # Load unaligned reconstructions for visualization (before merge overwrites them)
    rec_down = pycolmap.Reconstruction(str(vg_down.colmap_reconstruction_path))
    rec_up = pycolmap.Reconstruction(str(vg_up.colmap_reconstruction_path))

    # Merge
    logger.info("Separate merge: merging down + up ViewGraphs for %s", sequence)
    merged_vg, merged_rec, db1_rename, db2_rename, align_info = merge_two_view_graphs(
        down_cache, up_cache, merged_cache,
        onboarding_config=config.onboarding, device=config.run.device)

    # --- Set up write folder ---
    config.run.special_hash = f'{hot3d_prefix}both'
    write_folder = resolve_write_folder(config.paths, experiment_name, dataset,
                                        f'{sequence}_{config.run.special_hash}', output_folder)
    write_folder.mkdir(parents=True, exist_ok=True)

    # --- (a) Copy reconstruction to results folder ---
    glomap_folder = write_folder / f'glomap_{sequence}'
    glomap_output = glomap_folder / 'output' / '0'
    glomap_output.mkdir(parents=True, exist_ok=True)
    merged_rec.write(str(glomap_output))
    shutil.copy(merged_cache / 'database.db', glomap_folder / 'database.db')
    # Copy keyframe images/segmentations
    for subdir in ('images', 'segmentations'):
        src = merged_cache / subdir
        dst = glomap_folder / subdir
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # --- (c) Populate ViewGraph metadata ---
    merged_vg.reconstruction_success = vg_down.reconstruction_success and vg_up.reconstruction_success
    merged_vg.alignment_success = vg_down.alignment_success and vg_up.alignment_success
    merged_vg.num_input_frames = vg_down.num_input_frames + vg_up.num_input_frames
    merged_vg.colmap_num_reconstructions = vg_down.colmap_num_reconstructions + vg_up.colmap_num_reconstructions
    merged_vg.frame_filtering_time = vg_down.frame_filtering_time + vg_up.frame_filtering_time
    merged_vg.reconstruction_time = vg_down.reconstruction_time + vg_up.reconstruction_time
    merged_vg.gt_model_path = vg_down.gt_model_path

    # Build combined GT poses for evaluation (union of down and up GT)
    bop_folder = config.paths.bop_data_folder
    hot3d_dev = config.input.hot3d_device
    _, _, _, sequence_starts = get_bop_images_and_segmentations(
        bop_folder, dataset, sequence, sequence_type, 'static', 'both',
        scene_obj_id=scene_obj_id, hot3d_device=hot3d_dev)

    # Build image_name_to_frame_id: map prefixed reconstruction image names to combined GT frame indices
    # db1_rename: {"0.png" -> "db1_0.png"} for down sequence (frame indices stay as-is)
    # db2_rename: {"0.png" -> "db2_0.png"} for up sequence (frame indices offset by num_down_frames)
    up_offset = sequence_starts[0] if len(sequence_starts) == 1 else sequence_starts[1]
    image_name_to_frame_id = {}
    for orig_name, prefixed_name in db1_rename.items():
        frame_id = int(Path(orig_name).stem)
        image_name_to_frame_id[prefixed_name] = frame_id
    for orig_name, prefixed_name in db2_rename.items():
        frame_id = int(Path(orig_name).stem) + up_offset
        image_name_to_frame_id[prefixed_name] = frame_id
    merged_vg.image_name_to_frame_id = image_name_to_frame_id

    dict_gt_Se3_cam2obj = read_gt_Se3_cam2obj_transformations(
        bop_folder, dataset, sequence, sequence_type, gt_cam_scale, 'static',
        sequence_starts, 'both', scene_obj_id, device='cpu', hot3d_device=hot3d_dev)

    if dict_gt_Se3_cam2obj is not None:
        gt_Se3_world2cam = {i: cam2obj.inverse() for i, cam2obj in dict_gt_Se3_cam2obj.items()}
    else:
        gt_Se3_world2cam = None

    # --- (b) Create rerun file with merge-specific blueprint ---
    import rerun as rr
    rerun_file = write_folder / f'rerun_{experiment_name}_{sequence}_{config.run.special_hash}.rrd'
    blueprint = build_onboarding_blueprint(config)
    rerun_name = f'{sequence}-{experiment_name}-merged'
    init_rerun_recording(rerun_name, rerun_file, blueprint)
    rr.set_time('frame', sequence=0)
    log_merge_to_rerun(rec_down, rec_up, merged_rec, align_info,
                       down_cache / 'images', up_cache / 'images',
                       down_cache / 'segmentations', up_cache / 'segmentations',
                       gt_model_path=merged_vg.gt_model_path,
                       gt_Se3_world2cam=gt_Se3_world2cam)

    # Evaluate
    evaluate_onboarding(merged_vg, gt_Se3_world2cam, config.run, config.bop, write_folder)
