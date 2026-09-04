import copy
import logging
import os
import random
import shutil
import threading
import time
import zlib
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import imageio
import networkx as nx
import numpy as np
import torch
from PIL import Image
from kornia.geometry import Se3, PinholeCamera
from pycolmap import Reconstruction

from data_providers.flow_provider import FlowCache, create_matching_provider
from data_providers.frame_provider import FrameProviderAll
from data_structures.data_graph import DataGraph
from data_structures.view_graph import ViewGraph, view_graph_from_datagraph
from eval.eval_onboarding import resolve_gt_model_path
from configs.glopose_config import GloPoseConfig
from onboarding.colmap_utils import carve_reconstruction_by_masks
from onboarding.frame_filter import create_frame_filter
from onboarding.reconstruction import align_reconstruction_with_pose, align_with_kabsch, reconstruct_images_using_sfm
from utils.math_utils import Se3_cam_to_obj_to_Se3_obj_1_to_obj_i
from utils.results_logging import WriteResults


logger = logging.getLogger(__name__)


def seed_run(seed: int, sequence: Optional[str] = None) -> int:
    """Make one onboarding run reproducible.

    Correspondence sampling uses torch.multinomial (inherited from RoMa), so without
    this two runs of the same config draw different match subsets, and the keyframe
    filter, which thresholds matchability computed from those samples, then selects
    different keyframes. Measured on the validation set: two runs differing only in
    view-graph topology, which is applied after selection finishes and cannot affect
    it, disagreed on the keyframe count in 25 of 39 cells.

    The sequence name is mixed in so that different sequences in a sweep do not all
    replay the identical draw order, while any single (config, sequence) pair stays
    reproducible across runs.
    """
    mixed = seed if not sequence else (seed * 1_000_003 + zlib.crc32(sequence.encode())) % (2 ** 31)
    random.seed(mixed)
    np.random.seed(mixed)
    torch.manual_seed(mixed)
    torch.cuda.manual_seed_all(mixed)
    return mixed


class OnboardingPipeline:

    def __init__(self, config: GloPoseConfig, write_folder: Path, input_images: Union[List[Path], Path],
                 gt_texture=None, gt_mesh=None, gt_Se3_cam2obj: Optional[Dict[int, Se3]] = None,
                 gt_Se3_world2cam: Optional[Dict[int, Se3]] = None,
                 gt_pinhole_params: Optional[Dict[int, PinholeCamera]] = None,
                 input_segmentations: Union[List[Path], Path] = None, depth_paths: Optional[List[Path]] = None,
                 initial_segmentation: Union[torch.Tensor, List[torch.Tensor]] = None,
                 initial_bbox=None,
                 progress=None,
                 sequence_boundaries: list[int] | None = None):

        self.write_folder: Path = write_folder
        self.config: GloPoseConfig = config

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'Processing sequence written into {write_folder}')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        self.progress = progress

        cache_root = config.paths.cache_folder
        self.matching_cache_folder: Path = \
            (cache_root / f'{self.config.onboarding.filter_matcher}_cache' /
             config.onboarding.roma.config_name / config.run.dataset / f'{config.run.sequence}_{config.run.special_hash}')
        self.cache_folder_SIFT: Path = (cache_root / 'SIFT_cache' /
                                        config.onboarding.sift.config_name / config.run.dataset /
                                        f'{config.run.sequence}_{config.run.special_hash}')
        self.cache_folder_SAM2: Path = ((cache_root / 'SAM_cache' / self.config.run.dataset /
                                         f'{self.config.run.sequence}_{self.config.run.special_hash}') /
                                        str(self.config.input.image_downsample))

        self.cache_folder_view_graph: Path = (cache_root / 'view_graph_cache' /
                                              config.run.experiment_name / config.run.dataset /
                                              f'{config.run.sequence}_{config.run.special_hash}')

        self.colmap_base_path: Path = self.write_folder / f'glomap_{self.config.run.sequence}'
        self.colmap_image_path = self.colmap_base_path / 'images'
        self.colmap_seg_path = self.colmap_base_path / 'segmentations'

        self.prepare_output_folder()

        seed_run(config.run.seed, config.run.sequence)

        skip = config.input.skip_indices
        if skip != 1:
            used_indices = range(0, config.input.input_frames, skip)
            config.input.input_frames = config.input.input_frames // skip

            if gt_Se3_cam2obj is not None:
                gt_Se3_cam2obj = {i // skip: gt_Se3_cam2obj[i] for i in used_indices if i in gt_Se3_cam2obj}
            if gt_pinhole_params is not None:
                gt_pinhole_params = {i // skip: gt_pinhole_params[i] for i in used_indices if i in gt_pinhole_params}
            if gt_Se3_world2cam is not None:
                gt_Se3_world2cam = {i // skip: gt_Se3_world2cam[i] for i in used_indices if i in gt_Se3_world2cam}

        # Paths
        self.input_images: Union[List[Path], Path] = input_images
        self.input_segmentations: Optional[Union[List[Path], Path]] = input_segmentations
        self.depth_paths: Optional[List[Path]] = depth_paths

        # Ground truth related
        self.gt_Se3_cam2obj: Optional[Dict[int, Se3]] = gt_Se3_cam2obj
        self.gt_Se3_world2cam: Optional[Dict[int, Se3]] = gt_Se3_world2cam
        self.gt_pinhole_params: Optional[Dict[int, PinholeCamera]] = gt_pinhole_params

        # Initialization stuff
        self.initial_segmentation: torch.Tensor = initial_segmentation

        # Cameras
        self.pinhole_params: Optional[PinholeCamera] = None

        # Frame provider
        self.tracker: Optional[FrameProviderAll] = None
        self.densified_reconstruction: Optional[Reconstruction] = None

        # Other utilities and flags
        self.results_writer = None
        self._colmap_num_reconstructions: int = 1

        self.data_graph: DataGraph = DataGraph(out_device=self.config.run.device)

        self.initialize_frame_provider(gt_mesh, gt_texture, input_images, initial_segmentation,
                                       input_segmentations, depth_paths, initial_bbox=initial_bbox)

        self.results_writer = WriteResults(write_folder=self.write_folder, tracking_config=self.config,
                                           data_graph=self.data_graph)

        if self.config.onboarding.frame_filter in ('vggt_covis', 'vggt_trackvis'):
            # VGGT pair-score filters need no matching provider — don't load the filter matcher.
            self.match_provider_filtering = None
        else:
            filtering_cache = FlowCache(self.config.run.device, self.matching_cache_folder, self.data_graph,
                                        allow_disk_cache=self.config.onboarding.allow_disk_cache,
                                        purge_cache=self.config.paths.purge_cache)
            self.match_provider_filtering = create_matching_provider(
                self.config.onboarding.filter_matcher, self.config.onboarding, self.config.run.device,
                cache=filtering_cache)
            if self.config.onboarding.crop_matching_reconstruction_only:
                # Keyframe selection stays on full frames; only reconstruction matching zooms.
                self.match_provider_filtering.crop_matching = False
        self.match_provider_reconstruction = create_matching_provider(
            self.config.onboarding.reconstruction_matcher, self.config.onboarding, self.config.run.device,
            data_graph=self.data_graph, frame_provider=self.tracker.frame_provider)
        # Sequential-consistency gate: secondary point-tracking provider supplying the
        # chained (through-frames) predictions that gate the dense reconstruction
        # matches (docs/archive/report_seqconsistency_gate.md). Only meaningful when the
        # reconstruction matcher is dense — the tracking matcher IS the chain already.
        self.seq_gate_provider = None
        if self.config.onboarding.seq_consistency_gate:
            if self.config.onboarding.reconstruction_matcher == 'CoTracker':
                raise ValueError("seq_consistency_gate requires a dense reconstruction_matcher; "
                                 "the point-tracking matcher is already the sequential chain.")
            self.seq_gate_provider = create_matching_provider(
                'CoTracker', self.config.onboarding, self.config.run.device,
                data_graph=self.data_graph, frame_provider=self.tracker.frame_provider)
        self.frame_filter = create_frame_filter(
            self.config.onboarding, self.config.run.device, self.config.input.input_frames,
            self.data_graph, self.match_provider_filtering, sequence_boundaries=sequence_boundaries,
            frame_provider=self.tracker.frame_provider)

    def initialize_frame_provider(self, gt_mesh: torch.Tensor, gt_texture: torch.Tensor,
                                  images_paths: List[Path] | Path, initial_segmentation: torch.Tensor,
                                  input_segmentations: List[Path] | Path, depth_paths: List[Path],
                                  initial_bbox=None):

        if self.gt_Se3_cam2obj is not None:

            if self.config.input.segmentation_provider == 'synthetic' or self.config.input.frame_provider == 'synthetic':
                assert set(range(self.config.input.input_frames)).issubset(self.gt_Se3_cam2obj.keys()), \
                    f"Missing keys: {set(range(self.config.input.input_frames)) - self.gt_Se3_cam2obj.keys()}"

            all_gt_T_cam2obj = [self.gt_Se3_cam2obj[i].matrix() for i in sorted(self.gt_Se3_cam2obj.keys())]
            gt_T_cam2obj = torch.stack(all_gt_T_cam2obj)
            gt_Se3_cam2obj = Se3.from_matrix(gt_T_cam2obj)
            Se3_obj_1_to_obj_i = Se3_cam_to_obj_to_Se3_obj_1_to_obj_i(gt_Se3_cam2obj)
        else:
            Se3_obj_1_to_obj_i = None

        self.tracker = FrameProviderAll(self.config, gt_mesh=gt_mesh, gt_texture=gt_texture,
                                        gt_Se3_obj1_to_obj_i=Se3_obj_1_to_obj_i,
                                        initial_segmentation=initial_segmentation, input_images=images_paths,
                                        input_segmentations=input_segmentations, depth_paths=depth_paths,
                                        initial_bbox=initial_bbox,
                                        sam2_cache_folder=self.cache_folder_SAM2, write_folder=self.write_folder,
                                        progress=self.progress)

    def dump_frame_node_for_glomap(self, frame_idx: int):

        device = self.config.run.device

        frame_data = self.data_graph.get_frame_data(frame_idx)

        img = frame_data.frame_observation.observed_image.squeeze().permute(1, 2, 0).to(device)
        img_seg = frame_data.frame_observation.observed_segmentation.squeeze(0).permute(1, 2, 0).to(device)

        image_filename = f'{frame_idx}.png'
        seg_filename = f'{frame_idx}.png.png'

        node_save_path = self.colmap_image_path / image_filename
        img_np = (img * 255).to(torch.uint8).numpy(force=True)
        img_pil = Image.fromarray(img_np, mode='RGB')
        img_pil.save(node_save_path)

        segmentation_save_path = self.colmap_seg_path / seg_filename
        img_seg_np = (img_seg * 255).squeeze().to(torch.uint8).numpy(force=True)
        img_seg_pil = Image.fromarray(img_seg_np, mode='L')
        img_seg_pil.save(segmentation_save_path)

        frame_data.image_save_path = copy.deepcopy(node_save_path)
        frame_data.segmentation_save_path = copy.deepcopy(segmentation_save_path)

        occluded = getattr(self.tracker.frame_provider, 'expost_occluded_segmentations', {}).get(frame_idx)
        if occluded is not None:
            occ_dir = self.colmap_base_path / 'segmentations_occluded'
            occ_dir.mkdir(exist_ok=True, parents=True)
            occ_np = (occluded.squeeze() * 255).to(torch.uint8).numpy(force=True)
            Image.fromarray(occ_np, mode='L').save(occ_dir / seg_filename)

    def run_pipeline(self) -> ViewGraph:

        start_time = time.time()
        keyframe_graph = self.filter_frames()

        keyframe_nodes_idxs = list(sorted(keyframe_graph.nodes()))

        end_time = time.time()
        frame_filtering_time = end_time - start_time

        # Defaults; _reconstruct_colmap overwrites these with the instrumented split.
        # For external methods (vggt/mast3r) matching is not separable, so matching_time
        # stays 0 and reconstruction_time falls back to the wall-clock below.
        self._matching_time = 0.0
        self._colmap_reconstruction_time = None

        if len(keyframe_nodes_idxs) <= 2:
            logger.warning("Too few keyframes (%d) for reconstruction in %s/%s — skipping COLMAP",
                           len(keyframe_nodes_idxs), self.config.run.dataset, self.config.run.sequence)
            reconstruction = None
            alignment_success = False
            matching_time = 0.0
            reconstruction_time = 0.0
        else:
            images_paths, segmentation_paths, matching_pairs = self.prepare_input_for_colmap(keyframe_graph)

            start_time = time.time()
            reconstruction, alignment_success = self.run_reconstruction(images_paths, segmentation_paths, matching_pairs)
            end_time = time.time()
            reconstruction_wall = end_time - start_time
            # Prefer the instrumented split (COLMAP path); fall back to wall-clock otherwise.
            matching_time = self._matching_time
            reconstruction_time = (self._colmap_reconstruction_time
                                   if self._colmap_reconstruction_time is not None else reconstruction_wall)

        # Always create a ViewGraph (even if reconstruction failed)
        colmap_db_path = self.colmap_base_path / 'database.db'
        colmap_output_path = self.colmap_base_path / 'output'
        print(f"[pycolmap4-debug] run_pipeline: creating ViewGraph from datagraph")
        view_graph = view_graph_from_datagraph(keyframe_graph, self.data_graph, reconstruction, colmap_db_path,
                                               colmap_output_path, self.config.run.object_id)
        print(f"[pycolmap4-debug] run_pipeline: ViewGraph created")

        # Populate metadata on the ViewGraph
        view_graph.alignment_success = alignment_success and reconstruction is not None
        view_graph.frame_filtering_time = frame_filtering_time
        view_graph.matching_time = matching_time
        view_graph.reconstruction_time = reconstruction_time
        view_graph.num_input_frames = self.config.input.input_frames
        view_graph.colmap_num_reconstructions = self._colmap_num_reconstructions
        view_graph.gt_model_path = resolve_gt_model_path(self.config.run, self.config.paths)

        # Build image_name_to_frame_id mapping
        image_name_to_frame_id = {}
        for i in range(self.config.input.input_frames):
            frame_data = self.data_graph.get_frame_data(i)
            image_name_to_frame_id[str(frame_data.image_filename.name)] = i
        view_graph.image_name_to_frame_id = image_name_to_frame_id

        # Determine GT pose availability for visualization
        if self.gt_Se3_world2cam is not None and len(self.gt_Se3_world2cam.keys()) > 0:
            known_gt_poses = all(frm_idx in self.gt_Se3_world2cam.keys() for frm_idx in keyframe_nodes_idxs)
        else:
            known_gt_poses = None

        # Save ViewGraph and visualize
        if reconstruction is not None and alignment_success:
            view_graph.save_viewgraph(self.cache_folder_view_graph, reconstruction, save_images=True,
                                      overwrite=True, to_cpu=True)
            self.results_writer.visualize_colmap_track(self.config.input.input_frames - 1, reconstruction,
                                                       known_gt_poses,
                                                       self.colmap_image_path, self.colmap_seg_path,
                                                       gt_model_path=view_graph.gt_model_path)
        elif reconstruction is not None:
            self.results_writer.visualize_colmap_track(self.config.input.input_frames - 1, reconstruction, False,
                                                       self.colmap_image_path, self.colmap_seg_path,
                                                       gt_model_path=view_graph.gt_model_path)
            logger.warning("Reconstruction succeeded but alignment failed for %s/%s",
                           self.config.run.dataset, self.config.run.sequence)
        else:
            logger.warning("Reconstruction failed for %s/%s (%d keyframes from %d input frames)",
                           self.config.run.dataset, self.config.run.sequence,
                           len(keyframe_nodes_idxs), self.config.input.input_frames)

        return view_graph

    def prepare_input_for_colmap(self, keyframe_graph: nx.DiGraph) -> \
            Tuple[List[Path], List[Path], List[Tuple[int, int]]]:
        keyframe_nodes_idxs = list(sorted(keyframe_graph.nodes()))
        if self.config.onboarding.shuffle_keyframes:
            # Permute the input image order (seeded). images_paths and matching_pairs are
            # both built from this list below, so they stay consistent; for a complete graph
            # the pair set is unchanged, only the ordering (COLMAP init pair / neural input
            # sequence) differs.
            import random
            random.Random(self.config.onboarding.shuffle_seed).shuffle(keyframe_nodes_idxs)

        images_paths = []
        segmentation_paths = []
        matching_pairs = []
        for node_idx in keyframe_nodes_idxs:
            self.dump_frame_node_for_glomap(node_idx)
            frame_data = self.data_graph.get_frame_data(node_idx)

            images_paths.append(frame_data.image_save_path)
            segmentation_paths.append(frame_data.segmentation_save_path)
        for frame1_idx, frame2_idx in keyframe_graph.edges:
            u_index = keyframe_nodes_idxs.index(frame1_idx)
            v_index = keyframe_nodes_idxs.index(frame2_idx)
            matching_pairs.append((u_index, v_index))
        print(sorted(keyframe_graph.edges))

        return images_paths, segmentation_paths, matching_pairs

    def filter_frames(self, progress=None, stop_event: threading.Event = None) -> nx.DiGraph:

        for frame_i in range(0, self.tracker.frame_provider.get_input_length()):

            if progress is not None:
                progress(frame_i / float(self.tracker.frame_provider.get_input_length()), desc="Filtering frames...")

            if stop_event is not None and stop_event.is_set():
                print('Computation stopped by the user.')
                return self.frame_filter.get_keyframe_graph()

            self.init_datagraph_frame(frame_i)

            new_frame_observation = self.tracker.next(frame_i)

            new_frame_node = self.data_graph.get_frame_data(frame_i)
            new_frame_node.frame_observation = new_frame_observation.send_to_device('cpu')
            new_frame_node.image_shape = self.tracker.get_image_size()

            start = time.time()

            self.frame_filter.filter_frames(frame_i)

            self.results_writer.write_results(frame_i=frame_i, keyframe_graph=self.frame_filter.keyframe_graph)

            print(f'Elapsed time in seconds: {time.time() - start:.3f}s, frame {frame_i} out of '
                  f'{self.config.input.input_frames - 1}')

        keyframe_graph = self.frame_filter.get_keyframe_graph()

        return keyframe_graph

    def run_reconstruction(self, images_paths, segmentation_paths, matching_pairs) -> \
            Tuple[Optional[Reconstruction], bool]:

        first_frame_data = self.data_graph.get_frame_data(0)
        camera_K = first_frame_data.gt_pinhole_K if not self.config.onboarding.use_default_colmap_K else None

        method = self.config.onboarding.reconstruction_method

        if method == 'colmap':
            reconstruction = self._reconstruct_colmap(images_paths, segmentation_paths, matching_pairs, camera_K)
        elif method in ('vggt', 'vggt_omega', 'mast3r', 'pi3', 'vggsfm'):
            reconstruction = self._reconstruct_external(method, images_paths, segmentation_paths,
                                                        matching_pairs, camera_K)
        elif method == 'map_anything':
            reconstruction = self._reconstruct_map_anything(images_paths, segmentation_paths, camera_K)
        elif method == 'sam3d':
            reconstruction = self._reconstruct_sam3d(images_paths, segmentation_paths, camera_K)
        else:
            raise ValueError(f'Unknown reconstruction method: {method}')

        # Multi-view visual-hull carve: drop 3D points that project onto background
        # in any view. Neural reconstruction methods (VGGT/Mast3r/MapAnything) build
        # per-frame point clouds that are not multi-view consistent, so their output
        # contains points that fall outside the segmentation masks of other views.
        # COLMAP points are triangulated from multi-view matches and don't need it.
        if (reconstruction is not None and method != 'colmap'
                and self.config.onboarding.multiview_mask_filter
                and segmentation_paths is not None):
            name_to_seg = {p.name: s for p, s in zip(images_paths, segmentation_paths)}
            reconstruction, n_removed, n_kept = carve_reconstruction_by_masks(reconstruction, name_to_seg)
            print(f"[mask-carve] {method}: removed {n_removed} points, kept {n_kept}")

        if reconstruction is not None:
            # Save reconstruction to disk so evaluation can load it
            rec_output_path = self.colmap_base_path / 'output' / '0'
            rec_output_path.mkdir(exist_ok=True, parents=True)
            print(f"[pycolmap4-debug] run_reconstruction: writing reconstruction to {rec_output_path}")
            reconstruction.write(str(rec_output_path))
            print(f"[pycolmap4-debug] run_reconstruction: write done")

        # Alignment
        if reconstruction is None or self.gt_Se3_world2cam is None:
            return reconstruction, False
        if method == 'sam3d':
            # SAM3D produces a single-camera reconstruction with its own pose estimate.
            # Kabsch/depths alignment requires multiple cameras — skip for now.
            # TODO: implement SAM3D-specific alignment (see CLAUDE.md Phase 3.1)
            logger.warning("Skipping alignment for SAM3D reconstruction (single camera)")
            return reconstruction, False
        if self.config.onboarding.similarity_transformation == 'depths':

            first_image_filename = str(first_frame_data.image_filename)

            # Depth-based alignment anchors on the first frame's GT pose. For
            # dynamic onboarding only a subset of frames carry GT; if frame 0 has
            # no GT pose we cannot anchor the alignment, so skip it gracefully
            # rather than raising a KeyError.
            gt_Se3_obj2cam = self.gt_Se3_world2cam.get(0)
            if gt_Se3_obj2cam is None:
                logger.warning("Skipping depth alignment: no GT pose for first frame")
                return reconstruction, False

            # The depth provider returns depth in METRES (raw_mm * depth_scale_to_meter,
            # with depth_scale_to_meter=0.001 for BOP). The first-frame GT pose translation
            # and the GT mesh are in BOP MILLIMETRES, so align_reconstruction_with_pose must
            # see depth in mm for its (mm-depth / colmap-depth) scale ratio to be metric.
            depth_to_mm = (1.0 / self.config.input.depth_scale_to_meter
                           if self.config.input.depth_scale_to_meter else 1.0)
            image_depths = {}
            for i in self.data_graph.G.nodes:
                frame_data = self.data_graph.get_frame_data(i)
                if frame_data.frame_observation.depth is not None:
                    image_depths[str(frame_data.image_filename)] = (
                        frame_data.frame_observation.depth.squeeze() * depth_to_mm)

            reconstruction, align_success = align_reconstruction_with_pose(reconstruction, gt_Se3_obj2cam, image_depths,
                                                                           first_image_filename)
        elif self.config.onboarding.similarity_transformation == 'kabsch':
            gt_Se3_world2cam_poses = {
                str(self.data_graph.get_frame_data(n).image_filename):
                    self.data_graph.get_frame_data(n).gt_Se3_world2cam
                for n in self.data_graph.G.nodes
            }
            reconstruction, align_success = align_with_kabsch(reconstruction, gt_Se3_world2cam_poses)
        else:
            raise ValueError(f'Unknown similarity transform method {self.config.onboarding.similarity_transformation}')

        if self.config.onboarding.densify_reconstruction and reconstruction is not None and align_success:
            self._run_densification(reconstruction, images_paths, segmentation_paths)

        return reconstruction, align_success

    def _run_densification(self, reconstruction, images_paths, segmentation_paths):
        """Fixed-pose dense triangulation of the aligned model (see onboarding/densify.py).
        Stores the filtered densified model on self, saves it under the colmap dir and
        logs it to the 'Densified' rerun view. Failures never break the run."""
        try:
            from data_providers.flow_provider import create_matching_provider
            from data_structures.rerun_annotations import RerunAnnotations
            from onboarding.densify import contiguous_pose_model, densify_reconstruction

            _, ids_names = contiguous_pose_model(reconstruction)
            by_name = {p.name: (p, s) for p, s in zip(images_paths, segmentation_paths)}
            ordered = [by_name[name] for _, name in ids_names if name in by_name]
            if len(ordered) != len(ids_names):
                logger.warning('Densification skipped: %d/%d registered views have images on disk',
                               len(ordered), len(ids_names))
                return
            images = [p for p, _ in ordered]
            segmentations = [s for _, s in ordered]

            provider = create_matching_provider('UFM', self.config.onboarding, self.config.run.device)
            densified = densify_reconstruction(
                reconstruction, images, segmentations, provider,
                self.config.onboarding.densify_sample_size or self.config.onboarding.sample_size,
                self.colmap_base_path / 'densified', device=self.config.run.device,
                add_track_merging=self.config.onboarding.add_track_merging_matches,
                min_track_len=self.config.onboarding.densify_min_track_len,
                max_reproj_error=self.config.onboarding.densify_max_reproj_error,
                carve=self.config.onboarding.densify_carve,
                carve_min_bg_views=self.config.onboarding.densify_carve_min_bg_views)
            self.densified_reconstruction = densified
            if densified is not None and densified.num_points3D() > 0:
                import numpy as np
                import rerun as rr
                pts = np.stack([p.xyz for p in densified.points3D.values()], axis=0)
                cols = np.stack([p.color for p in densified.points3D.values()], axis=0)
                rr.log(RerunAnnotations.colmap_pointcloud_densified,
                       rr.Points3D(pts, colors=cols), static=True)
        except Exception as e:
            import traceback
            logger.warning('Densification failed: %s', e)
            traceback.print_exc()

    def _reconstruct_colmap(self, images_paths, segmentation_paths, matching_pairs, camera_K) \
            -> Optional[Reconstruction]:
        occ_dir = self.colmap_base_path / 'segmentations_occluded'
        if occ_dir.is_dir():
            # Ex-post synthetic occlusion: the matcher sees the UN-occluded images, but
            # the occluded masks gate the matches, so every correspondence with an
            # endpoint inside the occluder (in either image of every pair, incl. track
            # merging) is dropped before reconstruction. Per image, hence per all its
            # pairs. Nothing is deleted after reconstruction.
            segmentation_paths = [occ_dir / p.name for p in segmentation_paths]
        try:
            reconstruction, self._colmap_num_reconstructions, timings = reconstruct_images_using_sfm(
                images_paths, segmentation_paths, matching_pairs,
                self.config.onboarding.init_with_first_two_images,
                self.config.onboarding.mapper,
                self.match_provider_reconstruction,
                self.config.onboarding.sample_size,
                self.colmap_base_path,
                self.config.onboarding.add_track_merging_matches,
                camera_K, self.config.run.device,
                filter_points_by_seg=self.config.onboarding.filter_points_by_segmentation,
                use_background_points=self.config.onboarding.reconstruction_use_background_points,
                filter_degenerate_edges=self.config.onboarding.filter_degenerate_two_view_edges,
                filter_degenerate_edges_mode=self.config.onboarding.filter_degenerate_edges_mode,
                ba_backend=self.config.onboarding.ba_backend,
                seq_gate_provider=self.seq_gate_provider,
                seq_gate_tau_px=self.config.onboarding.seq_gate_tau_px,
                seq_gate_assoc_px=self.config.onboarding.seq_gate_assoc_px,
                seq_gate_min_edge_matches=self.config.onboarding.seq_gate_min_edge_matches,
                min_track_length=self.config.onboarding.min_track_length)
            # Split timings: matching (GPU dense flow) vs reconstruction (CPU COLMAP).
            self._matching_time = timings.get('matching_time', 0.0)
            self._colmap_reconstruction_time = timings.get('reconstruction_time', 0.0)
        except Exception as e:
            import traceback
            print(f"Reconstruction failed: {e}")
            traceback.print_exc()
            reconstruction = None
        return reconstruction

    def _reconstruct_external(self, method, images_paths, segmentation_paths, matching_pairs, camera_K) \
            -> Optional[Reconstruction]:
        """Run reconstruction with an external method (VGGT or Mast3r)."""
        image_names = [p.name for p in images_paths]

        try:
            if method == 'vggt':
                from adapters.vggt_adapter import reconstruct_with_vggt
                conf_threshold = self.config.onboarding.vggt_depth_conf_threshold
                reconstruction = reconstruct_with_vggt(
                    image_paths=images_paths,
                    image_names=image_names,
                    device=self.config.run.device,
                    camera_K=camera_K,
                    conf_threshold=conf_threshold,
                    segmentation_paths=segmentation_paths,
                    custom_weights_path=self.config.onboarding.vggt_custom_weights_path,
                    crop_to_object=self.config.onboarding.vggt_crop_to_object,
                    crop_margin=self.config.onboarding.vggt_crop_margin,
                    use_ba=self.config.onboarding.vggt_use_ba,
                )
            elif method == 'mast3r':
                from adapters.mast3r_adapter import reconstruct_with_mast3r
                reconstruction = reconstruct_with_mast3r(
                    image_paths=images_paths,
                    image_names=image_names,
                    matching_pairs=matching_pairs,
                    device=self.config.run.device,
                    camera_K=camera_K,
                    segmentation_paths=segmentation_paths,
                )
            elif method == 'vggt_omega':
                from adapters.vggt_omega_adapter import reconstruct_with_vggt_omega
                reconstruction = reconstruct_with_vggt_omega(
                    image_paths=images_paths,
                    image_names=image_names,
                    weights_path=self.config.onboarding.vggt_omega_weights_path,
                    device=self.config.run.device,
                    camera_K=camera_K,
                    conf_percentile=self.config.onboarding.vggt_omega_conf_percentile,
                    segmentation_paths=segmentation_paths,
                )
            elif method == 'pi3':
                from adapters.pi3_adapter import reconstruct_with_pi3
                reconstruction = reconstruct_with_pi3(
                    image_paths=images_paths,
                    image_names=image_names,
                    device=self.config.run.device,
                    camera_K=camera_K,
                    conf_threshold=self.config.onboarding.pi3_conf_threshold,
                    segmentation_paths=segmentation_paths,
                )
            elif method == 'vggsfm':
                from adapters.vggsfm_adapter import reconstruct_with_vggsfm
                reconstruction = reconstruct_with_vggsfm(
                    image_paths=images_paths,
                    image_names=image_names,
                    device=self.config.run.device,
                    camera_K=camera_K,
                    segmentation_paths=segmentation_paths,
                    python_bin=self.config.onboarding.vggsfm_python_bin,
                    fine_tracking=self.config.onboarding.vggsfm_fine_tracking,
                )
            else:
                raise ValueError(f'Unknown external method: {method}')
        except Exception as e:
            print(f"External reconstruction ({method}) failed: {e}")
            reconstruction = None

        return reconstruction

    def _reconstruct_map_anything(self, images_paths, segmentation_paths, camera_K) \
            -> Optional[Reconstruction]:
        """Run Map Anything reconstruction."""
        image_names = [p.name for p in images_paths]
        try:
            from adapters.map_anything_adapter import reconstruct_with_map_anything
            reconstruction = reconstruct_with_map_anything(
                image_paths=images_paths,
                image_names=image_names,
                device=self.config.run.device,
                camera_K=camera_K,
                segmentation_paths=segmentation_paths,
                backend=self.config.onboarding.map_anything_backend,
                voxel_fraction=self.config.onboarding.map_anything_voxel_fraction,
            )
        except Exception as e:
            print(f"Map Anything reconstruction failed: {e}")
            reconstruction = None
        return reconstruction

    def _reconstruct_sam3d(self, images_paths, segmentation_paths, camera_K) \
            -> Optional[Reconstruction]:
        """Run SAM3D single-image 3D reconstruction."""
        if len(images_paths) == 0:
            print("SAM3D: no images provided")
            return None

        # SAM3D uses a single image (the best frame selected by FrameFilterMaxVisible)
        image_path = images_paths[0]
        segmentation_path = segmentation_paths[0]
        image_name = image_path.name

        try:
            from adapters.sam3d_adapter import reconstruct_with_sam3d
            mesh, reconstruction = reconstruct_with_sam3d(
                image_path=image_path,
                segmentation_path=segmentation_path,
                image_name=image_name,
                device=self.config.run.device,
                camera_K=camera_K,
                checkpoint_path=self.config.onboarding.sam3d_checkpoint_path,
                output_dir=self.colmap_base_path,
                seed=self.config.onboarding.sam3d_seed,
            )
        except Exception as e:
            print(f"SAM3D reconstruction failed: {e}")
            reconstruction = None

        return reconstruction

    def init_datagraph_frame(self, frame_i):
        self.data_graph.add_new_frame(frame_i)

        frame_node = self.data_graph.get_frame_data(frame_i)

        if self.gt_Se3_cam2obj is not None and (gt_Se3_cam2obj := self.gt_Se3_cam2obj.get(frame_i)):
            frame_node.gt_Se3_cam2obj = gt_Se3_cam2obj

        if self.gt_pinhole_params is not None and (gt_pinhole_params := self.gt_pinhole_params.get(frame_i)):
            frame_node.gt_pinhole_K = gt_pinhole_params.intrinsics.squeeze()

        if self.gt_Se3_world2cam is not None and (gt_Se3_world2cam := self.gt_Se3_world2cam.get(frame_i)):
            frame_node.gt_Se3_world2cam = gt_Se3_world2cam

        frame_node.image_filename = Path(f'{frame_i}.png')

        if type(self.input_segmentations) is list:
            frame_node.segmentation_filename = Path(f'{frame_i}.png')

    def prepare_output_folder(self):
        """Wipe and recreate the output folder. Called early in __init__."""
        if os.path.exists(self.write_folder):
            shutil.rmtree(self.write_folder)
        self.write_folder.mkdir(exist_ok=True, parents=True)
        self.colmap_image_path.mkdir(exist_ok=True, parents=True)
        self.colmap_seg_path.mkdir(exist_ok=True, parents=True)
