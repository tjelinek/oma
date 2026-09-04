from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from time import time
from typing import Tuple

import networkx as nx
import torch

from data_providers.flow_provider import MatchingProvider
from data_structures.data_graph import DataGraph
from configs.glopose_config import OnboardingConfig
from onboarding.ransac import estimate_inlier_mask
from utils.image_utils import otsu_threshold


class BaseFrameFilter:
    def __init__(self, onboarding: OnboardingConfig, n_frames: int, data_graph: DataGraph, device: str = 'cuda',
                 sequence_boundaries: list[int] | None = None):
        self.onboarding: OnboardingConfig = onboarding
        self.n_frames: int = n_frames
        self.device: str = device
        self.data_graph: DataGraph = data_graph
        self.keyframe_graph: nx.DiGraph = nx.DiGraph()
        self.sequence_boundaries: set[int] = set(sequence_boundaries) if sequence_boundaries else set()

    def get_keyframe_graph(self) -> nx.DiGraph:
        # if len(self.keyframe_graph.nodes) <= 2:
        #     nodes_list = sorted(list(self.keyframe_graph.nodes))
        #     middle_node = (nodes_list[-1] - nodes_list[0]) // 2
        #     if nodes_list[0] < middle_node < nodes_list[-1]:
        #         self.keyframe_graph.add_edge(nodes_list[0], middle_node)
        #         self.keyframe_graph.add_edge(middle_node, nodes_list[-1])

        if self.onboarding.view_graph_strategy == 'dense':
            nodes_list = list(self.keyframe_graph.nodes)
            for i in range(len(nodes_list)):
                for j in range(len(nodes_list)):
                    if i != j:  # Don't add self-loops
                        self.keyframe_graph.add_edge(nodes_list[i], nodes_list[j])
        elif self.onboarding.view_graph_strategy == 'linear':
            # Sequential chain over the sorted keyframes: edges (k0,k1),(k1,k2),(k2,k3),...
            # Any pre-existing edges (e.g. the all-to-all set built by FrameFilterPassThrough)
            # are discarded first so the graph fed to SfM is purely linear, not dense.
            nodes_list = sorted(self.keyframe_graph.nodes)
            self.keyframe_graph.remove_edges_from(list(self.keyframe_graph.edges))
            for a, b in zip(nodes_list, nodes_list[1:]):
                self.keyframe_graph.add_edge(a, b)
        else:
            assert self.onboarding.view_graph_strategy == 'from_matching'

        return self.keyframe_graph

    @abstractmethod
    def add_keyframe(self, frame_i: int):
        pass

    @abstractmethod
    def filter_frames(self, frame_i: int):
        pass


class RoMaFrameFilter(BaseFrameFilter):

    def __init__(self, onboarding: OnboardingConfig, n_frames: int, data_graph: DataGraph, flow_provider: MatchingProvider,
                 device: str = 'cuda', sequence_boundaries: list[int] | None = None):

        super().__init__(onboarding, n_frames, data_graph, device, sequence_boundaries)

        self.flow_provider: MatchingProvider = flow_provider

        self.matching_reliability_threshold = self.onboarding.flow_reliability_threshold

    def add_keyframe(self, frame_i: int):
        self.keyframe_graph.add_node(frame_i)

        src_pts_xy_int, dst_pts_xy_int, certainty = (
            self.flow_provider.get_source_target_points_datagraph(frame_i, frame_i,
                                                                  self.onboarding.sample_size, as_int=True,
                                                                  zero_certainty_outside_segmentation=True,
                                                                  only_foreground_matches=True))

        kf_data = self.data_graph.get_frame_data(frame_i)
        if self.onboarding.certainty_threshold_strategy == 'otsu':
            certainty_threshold = otsu_threshold(certainty)
            if certainty_threshold is None and frame_i > 0:
                prev_kf = kf_data.matching_source_keyframe
                if prev_kf is not None:
                    certainty_threshold = self.data_graph.get_frame_data(prev_kf).roma_certainty_threshold
                else:
                    # Degenerate sequence: this frame was never reliably matched to any
                    # keyframe (e.g. mask collapse) and Otsu failed on the last-frame
                    # fallback keyframe — use the configured minimum instead of crashing.
                    certainty_threshold = self.onboarding.min_certainty_threshold
            else:
                certainty_threshold = self.onboarding.min_certainty_threshold
        else:
            certainty_threshold = self.onboarding.min_certainty_threshold
        kf_data.roma_certainty_threshold = certainty_threshold

        if self.onboarding.matchability_based_reliability:
            image_shape = self.data_graph.get_frame_data(frame_i).image_shape
            img_h, img_w = image_shape.height, image_shape.width
            arc_data = self.data_graph.get_edge_observations(frame_i, frame_i)
            if arc_data.roma_flow_warp_certainty is None:
                # crop_matching bypasses the flow-warp datagraph cache (only shifted points
                # are stored), so the certainty map is unavailable for matchability.
                raise RuntimeError("matchability_based_reliability requires the flow-warp "
                                   "certainty map, which crop_matching does not populate — "
                                   "disable one of the two.")
            roma_shape = arc_data.roma_flow_warp_certainty.shape
            certainty_map = arc_data.roma_flow_warp_certainty[:, :roma_shape[1] // 2]
            certainty_map_img_size = torch.nn.functional.interpolate(certainty_map[None, None], (img_h, img_w),
                                                                     mode='bilinear').squeeze()
            matchability_map = certainty_map_img_size > certainty_threshold
            kf_data.matchability_mask = matchability_map
        kf_data.is_keyframe = True
        print(frame_i)

    def _init_first_frame(self):
        self.add_keyframe(0)
        first_frame_node = self.data_graph.get_frame_data(0)
        first_frame_node.reliable_sources = {0}
        first_frame_node.matching_source_keyframe = 0
        first_frame_node.current_flow_reliability_threshold = self.matching_reliability_threshold

    def _find_source(self, frame_i: int, preceding_source: int, preceding_reliable: bool
                     ) -> Tuple[int, set]:
        is_boundary = frame_i in self.sequence_boundaries

        if is_boundary:
            # Sequence boundary (e.g. transition from down to up sub-sequence).
            # Force last frame of previous sub-sequence as keyframe, then match
            # against all existing keyframes — pick best even if below threshold.
            prev = frame_i - 1
            if prev >= 0:
                self._ensure_last_frame_keyframe(prev, preceding_source, set())
            reliable_kfs, best_source = self._match_to_all_keyframes(frame_i)
            if best_source is not None:
                return best_source, reliable_kfs
            # No reliable match — still use the best-matching keyframe (don't
            # fall back to adding frame_i-1 which belongs to a different sub-seq)
            best_kf = self._best_matching_keyframe(frame_i)
            return best_kf, set()

        need_all_keyframes = (
            self.onboarding.edge_strategy == 'always' or not preceding_reliable
        )

        if need_all_keyframes:
            reliable_kfs, best_source = self._match_to_all_keyframes(frame_i)
            if best_source is not None:
                return best_source, reliable_kfs
            # No reliable match anywhere — add frame_i-1 as new keyframe
            new_source = frame_i - 1
            self.add_keyframe(new_source)
            self.keyframe_graph.add_edge(preceding_source, new_source)
            return new_source, {new_source}

        # Preceding match is reliable and strategy is 'on_unreliable'
        return preceding_source, {preceding_source}

    def _ensure_last_frame_keyframe(self, frame_i: int, source: int, reliable_kfs: set
                                    ) -> Tuple[int, set]:
        if frame_i in self.keyframe_graph.nodes:
            return source, reliable_kfs

        self.add_keyframe(frame_i)

        if reliable_kfs:
            for kf in reliable_kfs:
                self.keyframe_graph.add_edge(kf, frame_i)
            return source, reliable_kfs

        # No reliable match — try matching against preceding frame
        prev = frame_i - 1
        if prev not in self.keyframe_graph.nodes:
            self.add_keyframe(prev)
            # Connect prev to its own source
            prev_source = self.data_graph.get_frame_data(prev).matching_source_keyframe
            if prev_source is not None and prev_source in self.keyframe_graph.nodes:
                self.keyframe_graph.add_edge(prev_source, prev)

        self.flow_reliability(prev, frame_i)
        edge = self.data_graph.get_edge_observations(prev, frame_i)
        if edge.is_match_reliable:
            self.keyframe_graph.add_edge(prev, frame_i)
            return prev, {prev}

        return source, reliable_kfs

    @torch.no_grad()
    def filter_frames(self, frame_i: int):
        start_time = time()

        if frame_i == 0:
            self._init_first_frame()
            return

        # Step 1: Check preceding frame's match reliability
        preceding_source = self.data_graph.get_frame_data(frame_i - 1).matching_source_keyframe
        self.flow_reliability(preceding_source, frame_i - 1)
        preceding_reliable = self.data_graph.get_edge_observations(
            preceding_source, frame_i - 1).is_match_reliable

        # Step 2: Determine source and collect reliable keyframe matches
        if frame_i == 1:
            source, reliable_kfs = 0, {0}
        else:
            source, reliable_kfs = self._find_source(frame_i, preceding_source, preceding_reliable)

        # Step 3: Handle last frame
        is_last = (frame_i == self.n_frames - 1)
        if is_last and self.onboarding.always_add_last_frame:
            source, reliable_kfs = self._ensure_last_frame_keyframe(
                frame_i, source, reliable_kfs)

        # Step 4: Final reliability + metadata
        flow_reliability = self.flow_reliability(source, frame_i)
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{flow_reliability}')
        node = self.data_graph.get_frame_data(frame_i)
        node.pose_estimation_time = time() - start_time
        node.current_flow_reliability_threshold = self.matching_reliability_threshold
        node.reliable_sources = {source} | reliable_kfs
        node.matching_source_keyframe = source

    def _best_matching_keyframe(self, frame_i: int) -> int:
        """Return the keyframe with highest reliability to frame_i, regardless of threshold."""
        best_kf = 0
        best_rel = -1.
        for kf in list(self.keyframe_graph.nodes):
            rel = self.flow_reliability(kf, frame_i)
            if rel > best_rel:
                best_kf = kf
                best_rel = rel
        return best_kf

    def _match_to_all_keyframes(self, frame_i: int) -> Tuple[set, int | None]:
        best_source: int = 0
        best_source_reliability: float = 0.
        reliable_kfs = set()

        for kf in list(self.keyframe_graph.nodes):
            reliability = self.flow_reliability(kf, frame_i)
            if reliability >= self.matching_reliability_threshold:
                reliable_kfs.add(kf)
            if reliability > best_source_reliability:
                best_source = kf
                best_source_reliability = reliability

        if best_source_reliability < self.matching_reliability_threshold:
            return set(), None
        return reliable_kfs, best_source

    def flow_reliability(self, source_frame: int, target_frame: int) -> float:
        dev = self.device
        source_datagraph_node = self.data_graph.get_frame_data(source_frame)
        source_segmentation_mask = source_datagraph_node.frame_observation.observed_segmentation.squeeze().to(dev)

        H_A, W_A = source_datagraph_node.image_shape.height, source_datagraph_node.image_shape.width
        assert source_segmentation_mask.shape[-2:] == (H_A, W_A)

        src_pts_xy_int, dst_pts_xy_int, certainty = (
            self.flow_provider.get_source_target_points_datagraph(source_frame, target_frame,
                                                                  self.onboarding.sample_size, as_int=True,
                                                                  zero_certainty_outside_segmentation=True,
                                                                  only_foreground_matches=True))

        edge_data = self.data_graph.get_edge_observations(source_frame, target_frame)

        assert ((src_pts_xy_int[:, 0] >= 0) & (src_pts_xy_int[:, 0] < W_A)).all()
        assert ((src_pts_xy_int[:, 1] >= 0) & (src_pts_xy_int[:, 1] < H_A)).all()
        assert certainty.shape[0] == src_pts_xy_int.shape[0] and certainty.shape[0] == dst_pts_xy_int.shape[0]

        if self.onboarding.matchability_based_reliability:
            matchability_mask = source_datagraph_node.matchability_mask
            source_segmentation_mask = source_segmentation_mask * matchability_mask
            matchable_fg_matches_mask = source_segmentation_mask[src_pts_xy_int[:, 1], src_pts_xy_int[:, 0]].bool()

            fg_matches_mask = source_segmentation_mask[src_pts_xy_int[:, 1], src_pts_xy_int[:, 0]].bool()
            in_segmentation_items = float(fg_matches_mask.sum())
            relative_area_matchable = float(fg_matches_mask.sum()) / (in_segmentation_items + 1e-5)

            edge_data.src_pts_xy_roma_matchable = src_pts_xy_int[matchable_fg_matches_mask]
            edge_data.dst_pts_xy_roma_matchable = dst_pts_xy_int[matchable_fg_matches_mask]
            edge_data.src_dst_certainty_roma_matchable = certainty[matchable_fg_matches_mask]
            source_datagraph_node.relative_area_matchable = relative_area_matchable

        min_num_of_certain_matches = self.onboarding.min_number_of_reliable_matches
        certain_matches_share_threshold = self.matching_reliability_threshold
        match_certainty_threshold = source_datagraph_node.roma_certainty_threshold

        reliability = compute_matching_reliability(src_pts_xy_int, certainty, source_segmentation_mask,
                                                   match_certainty_threshold, min_num_of_certain_matches)

        edge_data.reliability_score = reliability
        edge_data.is_match_reliable = reliability >= certain_matches_share_threshold

        return reliability


class FrameFilterRANSAC(RoMaFrameFilter):

    def flow_reliability(self, source_frame: int, target_frame: int) -> float:
        src_pts_xy, dst_pts_xy, certainty = (
            self.flow_provider.get_source_target_points_datagraph(source_frame, target_frame,
                                                                  self.onboarding.sample_size, as_int=False,
                                                                  zero_certainty_outside_segmentation=True,
                                                                  only_foreground_matches=True))
        src_pts_xy_np = src_pts_xy.numpy(force=True)
        dst_pts_xy_np = dst_pts_xy.numpy(force=True)
        certainty_np = certainty.numpy(force=True)

        frame_data_source = self.data_graph.get_frame_data(source_frame)
        frame_data_target = self.data_graph.get_frame_data(target_frame)
        camera_K1 = frame_data_source.gt_pinhole_K
        camera_K2 = frame_data_target.gt_pinhole_K

        K1_np = camera_K1.numpy(force=True) if camera_K1 is not None else None
        K2_np = camera_K2.numpy(force=True) if camera_K2 is not None else None

        ransac_config = self.onboarding.ransac
        inlier_mask = estimate_inlier_mask(
            src_pts_xy_np, dst_pts_xy_np, ransac_config,
            K1=K1_np, K2=K2_np,
            source_shape=frame_data_source.image_shape,
            target_shape=frame_data_target.image_shape,
            confidences=certainty_np)

        edge_data = self.data_graph.get_edge_observations(source_frame, target_frame)

        if inlier_mask is not None:
            reliability = float(inlier_mask.sum() / len(inlier_mask))
            edge_data.ransac_inliers = torch.from_numpy(src_pts_xy_np[inlier_mask])
            edge_data.ransac_outliers = torch.from_numpy(src_pts_xy_np[~inlier_mask])
        else:
            reliability = 0.

        edge_data.reliability_score = reliability
        edge_data.is_match_reliable = reliability >= self.matching_reliability_threshold

        return reliability


class FrameFilterDepth(RoMaFrameFilter):
    """Trust-but-verify reliability: instead of trusting how many matches the matcher
    *reports*, measure what fraction are *geometrically consistent* with a single rigid
    object motion.

    Reuses everything RoMaFrameFilter does (match gathering, foreground masking, keyframe
    graph, source selection) and overrides only ``flow_reliability``. The relative motion
    between the two frames is recovered from the matches themselves via a robust 3D-3D
    similarity fit (depth-lifted points + RANSAC) — so no GT poses are needed. Each match
    is then scored by its pixel reprojection error: lift the source point to 3D with the
    source depth, apply the recovered transform, project into the target image, and compare
    to where the matcher said it lands. The reliability is the fraction of foreground
    matches whose pixel error is below ``depth_reprojection_threshold_px``.

    Depth comes from the existing per-frame depth provider (``frame_observation.depth``);
    intrinsics from ``gt_pinhole_K`` (same source as FrameFilterRANSAC).
    """

    @torch.no_grad()
    def flow_reliability(self, source_frame: int, target_frame: int) -> float:
        source_node = self.data_graph.get_frame_data(source_frame)
        target_node = self.data_graph.get_frame_data(target_frame)

        src_pts_xy_int, dst_pts_xy_int, _ = (
            self.flow_provider.get_source_target_points_datagraph(
                source_frame, target_frame, self.onboarding.sample_size, as_int=True,
                zero_certainty_outside_segmentation=True, only_foreground_matches=True))

        edge_data = self.data_graph.get_edge_observations(source_frame, target_frame)

        num_fg = src_pts_xy_int.shape[0]
        inlier_mask, sim3d = self._geometric_inlier_mask(src_pts_xy_int, dst_pts_xy_int,
                                                         source_node, target_node)

        if inlier_mask is None or num_fg == 0:
            reliability = 0.0
        else:
            # Denominator is ALL sampled foreground matches; matches we could not verify
            # (missing/invalid depth, or that disagree with the recovered motion) do not
            # count as confirmed visible surface.
            reliability = float(inlier_mask.sum()) / (num_fg + 1e-5)
            edge_data.ransac_inliers = src_pts_xy_int[inlier_mask]
            edge_data.ransac_outliers = src_pts_xy_int[~inlier_mask]
            # Stash the recovered relative pose and (when GT is available) its error.
            # Skip degenerate self-pairs (source == target → identity, zero translation).
            if source_frame != target_frame:
                self._store_estimated_pose(edge_data, source_node, target_node, sim3d)

        enough_matches = num_fg > self.onboarding.min_number_of_reliable_matches
        reliability *= float(enough_matches)

        edge_data.reliability_score = reliability
        edge_data.is_match_reliable = reliability >= self.matching_reliability_threshold

        if edge_data.depth_rotation_error_deg is not None:
            t_err = edge_data.depth_translation_error_deg
            print(f"[FrameFilterDepth] pair ({source_frame},{target_frame}) reliability={reliability:.3f} "
                  f"rot_err={edge_data.depth_rotation_error_deg:.2f} deg "
                  f"trans_dir_err={t_err if t_err is not None else float('nan'):.2f} deg")
        return reliability

    def _store_estimated_pose(self, edge_data, source_node, target_node, sim3d) -> None:
        """Record the sim3d-recovered relative pose (target_from_source, camera frame) on the
        edge, and compare it against the GT relative camera pose when both frames have GT."""
        import numpy as np

        M = np.asarray(sim3d.matrix(), dtype=np.float64)   # 3x4: [scale*R | t]
        scale = float(np.cbrt(max(np.linalg.det(M[:, :3]), 1e-12)))
        R_est = M[:, :3] / scale
        t_est = M[:, 3]
        edge_data.depth_estimated_R = torch.from_numpy(R_est)
        edge_data.depth_estimated_t = torch.from_numpy(t_est)
        edge_data.depth_estimated_scale = scale

        gt_src = source_node.gt_Se3_world2cam
        gt_tgt = target_node.gt_Se3_world2cam
        if gt_src is None or gt_tgt is None:
            return
        # GT relative camera pose: target_from_source = world2cam_tgt @ (world2cam_src)^-1
        M_src = gt_src.matrix().squeeze().numpy(force=True).astype(np.float64)
        M_tgt = gt_tgt.matrix().squeeze().numpy(force=True).astype(np.float64)
        M_rel = M_tgt @ np.linalg.inv(M_src)
        R_gt, t_gt = M_rel[:3, :3], M_rel[:3, 3]

        # Rotation error: geodesic angle (deg) — unit/scale independent.
        cos_rot = (np.trace(R_est.T @ R_gt) - 1.0) / 2.0
        edge_data.depth_rotation_error_deg = float(np.degrees(np.arccos(np.clip(cos_rot, -1.0, 1.0))))

        # Translation error: angle between directions (deg) — robust to depth's scale/units.
        n_est, n_gt = np.linalg.norm(t_est), np.linalg.norm(t_gt)
        if n_est > 1e-9 and n_gt > 1e-9:
            cos_t = float(np.dot(t_est, t_gt) / (n_est * n_gt))
            edge_data.depth_translation_error_deg = float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))

    def _geometric_inlier_mask(self, src_pts_xy_int: torch.Tensor, dst_pts_xy_int: torch.Tensor,
                               source_node, target_node) -> tuple[torch.Tensor | None, object]:
        """Return (mask, sim3d): a boolean mask (over the foreground matches) of matches whose
        pixel reprojection error under the robustly-recovered similarity transform is below
        threshold, plus the recovered ``pycolmap.Sim3d`` (target_from_source).

        Returns (None, None) when depth/intrinsics are unavailable or too few matches have
        valid depth on both sides for a robust fit.
        """
        import numpy as np
        import pycolmap

        src_depth = source_node.frame_observation.depth
        tgt_depth = target_node.frame_observation.depth
        K_src = source_node.gt_pinhole_K
        K_tgt = target_node.gt_pinhole_K
        if src_depth is None or tgt_depth is None or K_src is None or K_tgt is None:
            if not getattr(self, '_warned_missing_depth', False):
                missing = [n for n, v in (('src_depth', src_depth), ('tgt_depth', tgt_depth),
                                          ('K_src', K_src), ('K_tgt', K_tgt)) if v is None]
                print(f"[FrameFilterDepth] WARNING: missing {missing} during filtering — depth verification "
                      f"cannot run, reliability will be 0 for all pairs (every frame becomes a keyframe). "
                      f"The 'depth' frame filter requires per-frame depth maps (e.g. dynamic onboarding "
                      f"sequences); static onboarding sequences ship no depth.")
                self._warned_missing_depth = True
            return None, None

        src_depth = src_depth.squeeze()
        tgt_depth = tgt_depth.squeeze()
        K_src = K_src.numpy(force=True).astype(np.float64)
        K_tgt = K_tgt.numpy(force=True).astype(np.float64)

        src_xy = src_pts_xy_int.numpy(force=True).astype(np.int64)
        dst_xy = dst_pts_xy_int.numpy(force=True).astype(np.int64)

        # Depth is indexed [y, x]; matches are already clamped to image bounds by the matcher.
        z_src = src_depth[src_xy[:, 1], src_xy[:, 0]].numpy(force=True).astype(np.float64)
        z_tgt = tgt_depth[dst_xy[:, 1], dst_xy[:, 0]].numpy(force=True).astype(np.float64)

        valid = (z_src > 0) & np.isfinite(z_src) & (z_tgt > 0) & np.isfinite(z_tgt)
        if int(valid.sum()) < self.onboarding.ransac.min_num_matches:
            return None, None

        # Back-project both sides to 3D in their own camera frames.
        p_src = _backproject_to_cam(src_xy[valid], z_src[valid], K_src)
        p_tgt = _backproject_to_cam(dst_xy[valid], z_tgt[valid], K_tgt)

        # Robust 3D-3D similarity (absorbs the global up-to-scale ambiguity of predicted depth).
        report = pycolmap.estimate_sim3d_robust(p_src, p_tgt)
        if report is None:
            return None, None
        sim3d = report['tgt_from_src']

        # Reproject source 3D into the target image and compare to the matched pixel.
        p_src_in_tgt = sim3d * p_src                       # (N, 3) in target camera frame
        proj = p_src_in_tgt @ K_tgt.T                       # (N, 3)
        depth_ok = proj[:, 2] > 1e-6
        uv = proj[:, :2] / np.where(depth_ok[:, None], proj[:, 2:3], 1.0)

        pixel_err = np.linalg.norm(uv - dst_xy[valid].astype(np.float64), axis=1)
        valid_inliers = depth_ok & (pixel_err < self.onboarding.depth_reprojection_threshold_px)

        # Scatter back to full foreground length: unverifiable matches stay False (outliers).
        full_mask = torch.zeros(src_pts_xy_int.shape[0], dtype=torch.bool)
        full_mask[torch.from_numpy(valid)] = torch.from_numpy(valid_inliers)
        return full_mask, sim3d


class FrameFilterMaxVisible(BaseFrameFilter):
    """Selects the single frame with the maximum number of visible object pixels.

    Iterates through all frames, tracking the one with the highest foreground
    segmentation pixel count. After all frames are processed, get_keyframe_graph()
    returns a graph with only that single best frame as a node.
    """

    def __init__(self, onboarding: OnboardingConfig, n_frames: int, data_graph: DataGraph, **kwargs):
        super().__init__(onboarding, n_frames, data_graph, **kwargs)
        self.best_frame: int = 0
        self.best_pixel_count: int = 0

    def filter_frames(self, frame_i: int):
        seg = self.data_graph.get_frame_data(frame_i).frame_observation.observed_segmentation
        pixel_count = int((seg > 0.5).sum())
        if pixel_count > self.best_pixel_count:
            self.best_pixel_count = pixel_count
            self.best_frame = frame_i
        self.data_graph.get_frame_data(frame_i).matching_source_keyframe = self.best_frame

    def add_keyframe(self, frame_i: int):
        self.keyframe_graph.add_node(frame_i)

    def get_keyframe_graph(self) -> nx.DiGraph:
        self.keyframe_graph = nx.DiGraph()
        self.keyframe_graph.add_node(self.best_frame)
        print(f"FrameFilterMaxVisible: selected frame {self.best_frame} "
              f"with {self.best_pixel_count} visible pixels")
        return self.keyframe_graph


class FrameFilterPassThrough(BaseFrameFilter):

    def filter_frames(self, frame_i: int):
        if frame_i % self.onboarding.passthrough_skip == 0:
            self.add_keyframe(frame_i)

        # Set matching_source_keyframe for every frame (nearest preceding keyframe)
        frame_data = self.data_graph.get_frame_data(frame_i)
        nearest_keyframe = frame_i - (frame_i % self.onboarding.passthrough_skip)
        frame_data.matching_source_keyframe = nearest_keyframe if nearest_keyframe >= 0 else 0

    def add_keyframe(self, frame_i: int):
        # Connect the new keyframe to the previously-selected keyframes only — not to
        # every preceding frame index. Using range(frame_i) here would implicitly create
        # a node per skipped frame (add_edges_from auto-creates nodes), inflating the
        # graph to the full frame set and matching O(N^2) pairs regardless of
        # passthrough_skip. Wiring to existing keyframe nodes yields a complete graph
        # over exactly the every-k keyframes (the 'linear' strategy later reduces it to a
        # chain, 'dense' keeps it complete).
        existing_keyframes = list(self.keyframe_graph.nodes)
        self.keyframe_graph.add_node(frame_i)
        for kf in existing_keyframes:
            self.keyframe_graph.add_edge(kf, frame_i)
            self.keyframe_graph.add_edge(frame_i, kf)


class FrameFilterLinearTransition(RoMaFrameFilter):
    """Linear (sequential) keyframe pose graph with a matchability-bridged transition.

    Keyframe SELECTION is every-kth (``passthrough_skip``), like FrameFilterPassThrough.
    Keyframe GRAPH is a linear chain (k0,k1),(k1,k2),... WITHIN each sub-sequence.

    At a sub-sequence boundary (e.g. the up/down split of a ``_both`` onboarding run) a
    naive linear chain would bridge the two halves with the arbitrary consecutive pair
    (last-frame-of-prev, first-frame-of-next), which usually shows opposite sides of the
    object and matches poorly. Instead, for the first keyframe of each new sub-sequence
    (the "transition"), we score it against all already-selected keyframes ("the current
    templates") with the dense matcher and add a single bridge edge at the best-matching
    template. Inherits the matcher + ``flow_reliability`` machinery from RoMaFrameFilter.

    Does not consult ``view_graph_strategy``: it builds its own topology in
    ``get_keyframe_graph``. With no sequence boundaries it degenerates to a plain linear
    chain over all keyframes.
    """

    def filter_frames(self, frame_i: int):
        # Passthrough selection: keep every k-th frame as a keyframe (no adaptive matching).
        frame_data = self.data_graph.get_frame_data(frame_i)
        nearest_keyframe = frame_i - (frame_i % self.onboarding.passthrough_skip)
        frame_data.matching_source_keyframe = nearest_keyframe if nearest_keyframe >= 0 else 0
        if frame_i % self.onboarding.passthrough_skip == 0:
            # RoMaFrameFilter.add_keyframe adds the node AND sets roma_certainty_threshold
            # (needed later by flow_reliability for the transition matchability scoring).
            self.add_keyframe(frame_i)

    def _segments(self, nodes: list[int]) -> list[list[int]]:
        """Split sorted keyframe nodes into sub-sequences at sequence_boundaries."""
        bounds = sorted(self.sequence_boundaries)
        segments: list[list[int]] = []
        seg: list[int] = []
        bi = 0
        for n in nodes:
            while bi < len(bounds) and n >= bounds[bi]:
                if seg:
                    segments.append(seg)
                seg = []
                bi += 1
            seg.append(n)
        if seg:
            segments.append(seg)
        return segments

    def get_keyframe_graph(self) -> nx.DiGraph:
        nodes = sorted(self.keyframe_graph.nodes)
        self.keyframe_graph.remove_edges_from(list(self.keyframe_graph.edges))
        segments = self._segments(nodes)

        templates: list[int] = []  # keyframes accumulated from all prior sub-sequences
        for seg in segments:
            # Linear chain within the sub-sequence.
            for a, b in zip(seg, seg[1:]):
                self.keyframe_graph.add_edge(a, b)
            # Bridge this sub-sequence to the prior templates at the best-matching pair.
            if templates:
                transition = seg[0]
                best_kf, best_rel = None, -1.0
                for kf in templates:
                    rel = self.flow_reliability(kf, transition)
                    if rel > best_rel:
                        best_kf, best_rel = kf, rel
                if best_kf is not None:
                    self.keyframe_graph.add_edge(best_kf, transition)
                    print(f"LinearTransition: bridge {best_kf} -> {transition} "
                          f"(matchability {best_rel:.3f})")
            templates.extend(seg)

        return self.keyframe_graph


class FrameFilterSift(BaseFrameFilter):

    def __init__(self, onboarding: OnboardingConfig, n_frames: int, data_graph: DataGraph, sift_matcher: MatchingProvider):

        super().__init__(onboarding, n_frames, data_graph)

        self.sift_matcher: MatchingProvider = sift_matcher

    @torch.no_grad()
    def filter_frames(self, current_frame_idx: int):

        start_time = time()

        if current_frame_idx == 0:
            self.keyframe_graph.add_node(current_frame_idx)
            self.data_graph.get_frame_data(current_frame_idx).matching_source_keyframe = current_frame_idx
            # Create self-edge so visualization code can access it
            if not self.data_graph.G.has_edge(current_frame_idx, current_frame_idx):
                self.data_graph.add_new_arc(current_frame_idx, current_frame_idx)
            edge_data = self.data_graph.get_edge_observations(current_frame_idx, current_frame_idx)
            edge_data.num_matches = 0
            edge_data.is_match_reliable = True
            return

        preceding_frame_idx = current_frame_idx - 1
        preceding_frame_node = self.data_graph.get_frame_data(preceding_frame_idx)

        if current_frame_idx == 1:
            keyframe_idx = 0
        else:
            keyframe_idx = preceding_frame_node.matching_source_keyframe

        print("Detection features")

        reliable_sources = set()

        selected_keyframe_idxs = list(self.keyframe_graph.nodes())

        more_than_enough_matches = self.onboarding.sift_filter_good_to_add_matches
        min_matches = self.onboarding.sift_filter_min_matches

        reliable_keyframe_found = False
        we_stepped_back = False

        while not reliable_keyframe_found:

            num_matches = self.compute_sift_reliability(keyframe_idx, current_frame_idx)
            print(f'{num_matches}, {min_matches}, {more_than_enough_matches}')

            if num_matches >= self.onboarding.sift_filter_min_matches:
                self.keyframe_graph.add_edge(keyframe_idx, current_frame_idx)

            if num_matches >= more_than_enough_matches:
                print(f'{keyframe_idx} has more than enough matches')
                if we_stepped_back:
                    print(f"Step back was good, adding keyframe_idx={keyframe_idx}")
                    selected_keyframe_idxs.append(keyframe_idx)

                    if not self.keyframe_graph.has_node(keyframe_idx):
                        self.keyframe_graph.add_node(keyframe_idx)
                    we_stepped_back = False

                reliable_keyframe_found = True

            if (num_matches <= more_than_enough_matches) and (num_matches >= min_matches):
                print("Adding keyframe")

                if not self.keyframe_graph.has_node(keyframe_idx):
                    self.keyframe_graph.add_node(keyframe_idx)
                if not self.keyframe_graph.has_node(current_frame_idx):
                    self.keyframe_graph.add_node(current_frame_idx)

                reliable_keyframe_found = True

            if num_matches < min_matches:  # try going back
                print("Too few matches, going back")
                keyframe_idx = max(0, keyframe_idx - 1)
                we_stepped_back = True
                if keyframe_idx <= 0:
                    reliable_keyframe_found = True
                elif keyframe_idx in selected_keyframe_idxs:
                    print(f"We cannot match {current_frame_idx}, skipping it")
                    return

        flow_frames_idxs = (keyframe_idx, current_frame_idx)

        long_jump_source, long_jump_target = flow_frames_idxs

        duration = time() - start_time
        datagraph_node = self.data_graph.get_frame_data(current_frame_idx)
        datagraph_node.pose_estimation_time = duration

        datagraph_node.reliable_sources |= ({long_jump_source} | reliable_sources)
        datagraph_node.matching_source_keyframe = keyframe_idx

    def compute_sift_reliability(self, frame_idx1: int, frame_idx2: int):

        source_frame_observation = self.data_graph.get_frame_data(frame_idx1)
        target_frame_observation = self.data_graph.get_frame_data(frame_idx2)

        source_img = source_frame_observation.frame_observation.observed_image.squeeze()
        target_img = target_frame_observation.frame_observation.observed_image.squeeze()

        source_seg = source_frame_observation.frame_observation.observed_segmentation.squeeze()
        target_seg = target_frame_observation.frame_observation.observed_segmentation.squeeze()

        src_pts, dst_pts, certainty = self.sift_matcher.get_source_target_points(
            source_img, target_img, source_image_segmentation=source_seg,
            target_image_segmentation=target_seg)

        num_matches = len(src_pts)

        if not self.data_graph.G.has_edge(frame_idx1, frame_idx2):
            self.data_graph.add_new_arc(frame_idx1, frame_idx2)
        edge_data = self.data_graph.get_edge_observations(frame_idx1, frame_idx2)

        edge_data.num_matches = num_matches
        edge_data.is_match_reliable = num_matches >= self.onboarding.sift_filter_min_matches
        edge_data.src_pts_xy_roma = src_pts
        edge_data.dst_pts_xy_roma = dst_pts
        edge_data.src_dst_certainty_roma = certainty

        return num_matches


class FrameFilterVGGTPairScore(RoMaFrameFilter):
    """Base for matcher-free keyframe filters that score (keyframe, candidate) pairs
    with a VGGT-based model on ORIGINAL (unmasked) frames — the frozen VGGT trunk is
    in-domain there; the segmentation mask enters only through score aggregation.

    Inherits all source-selection / keyframe-graph logic from RoMaFrameFilter and
    replaces the two matcher-touching methods (add_keyframe, flow_reliability).
    Subclasses implement _pair_score(source, target) -> float in [0, 1].
    """

    RESOLUTION = 518          # VGGT native inference resolution (37 * 14)
    IMAGE_CACHE_SIZE = 256    # LRU over resized original frames (CPU, ~3 MB each)

    def __init__(self, onboarding: OnboardingConfig, n_frames: int, data_graph: DataGraph,
                 device: str = 'cuda', sequence_boundaries: list[int] | None = None,
                 frame_provider=None):
        super().__init__(onboarding, n_frames, data_graph, flow_provider=None,
                         device=device, sequence_boundaries=sequence_boundaries)
        if frame_provider is None:
            raise ValueError(f"frame_filter='{onboarding.frame_filter}' requires a "
                             "FrameProvider — the DataGraph only holds background-masked "
                             "observations and the pair scorer runs on original frames")
        if onboarding.matchability_based_reliability:
            raise ValueError("matchability_based_reliability is not supported by "
                             "VGGT pair-score filters (there is no certainty map)")
        self.frame_provider = frame_provider
        self._image_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        # (source, target) -> reliability. DataGraph edge observations default
        # reliability_score to 0.0 (not None), so they cannot serve as the cache.
        self._reliability_cache: dict[Tuple[int, int], float] = {}

    def _pair_score(self, source_frame: int, target_frame: int) -> float:
        raise NotImplementedError

    def add_keyframe(self, frame_i: int):
        self.keyframe_graph.add_node(frame_i)
        kf_data = self.data_graph.get_frame_data(frame_i)
        # Unused by this filter, but downstream logging expects the field.
        kf_data.roma_certainty_threshold = self.onboarding.min_certainty_threshold
        kf_data.is_keyframe = True
        # The matcher-based filters leave a (i, i) self-edge in the DataGraph as a side
        # effect of their self-matching; some consumers expect it to exist.
        self.flow_reliability(frame_i, frame_i)
        print(frame_i)

    def _frame_518(self, frame_i: int) -> torch.Tensor:
        """Original (unmasked) frame resized to 518x518, cached on CPU."""
        cached = self._image_cache.get(frame_i)
        if cached is not None:
            self._image_cache.move_to_end(frame_i)
            return cached
        img = self.frame_provider.next_image(frame_i).to(self.device)   # (1, 3, H, W)
        r = self.RESOLUTION
        img = torch.nn.functional.interpolate(img, size=(r, r), mode='bilinear',
                                              align_corners=False).squeeze(0)
        self._image_cache[frame_i] = img.cpu()
        if len(self._image_cache) > self.IMAGE_CACHE_SIZE:
            self._image_cache.popitem(last=False)
        return self._image_cache[frame_i]

    @torch.no_grad()
    def flow_reliability(self, source_frame: int, target_frame: int) -> float:
        cached = self._reliability_cache.get((source_frame, target_frame))
        if cached is not None:                           # deterministic — reuse
            return cached
        # Unlike the matcher-based filters, nothing has created this DataGraph edge
        # yet (the flow provider does it as a side effect of matching there).
        if not self.data_graph.G.has_edge(source_frame, target_frame):
            self.data_graph.add_new_arc(source_frame, target_frame)
        edge_data = self.data_graph.get_edge_observations(source_frame, target_frame)

        if source_frame == target_frame:                 # self-pair: trivially covisible
            self._reliability_cache[(source_frame, target_frame)] = 1.0
            edge_data.reliability_score = 1.0
            edge_data.is_match_reliable = True
            return 1.0

        reliability = self._pair_score(source_frame, target_frame)

        self._reliability_cache[(source_frame, target_frame)] = reliability
        edge_data.reliability_score = reliability
        edge_data.is_match_reliable = reliability >= self.matching_reliability_threshold
        return reliability

    def _seg_518(self, frame_i: int) -> torch.Tensor:
        """Frame's object mask at the VGGT resolution, boolean, on device."""
        seg = self.data_graph.get_frame_data(frame_i) \
            .frame_observation.observed_segmentation.squeeze()[None, None].float()
        r = self.RESOLUTION
        return torch.nn.functional.interpolate(
            seg.to(self.device), size=(r, r), mode='nearest').squeeze().bool()


class FrameFilterVGGTCovis(FrameFilterVGGTPairScore):
    """Learned covisibility head (training/vggt_covis/) as the pair score:
    fraction of the source keyframe's object pixels whose predicted covisibility
    probability in the target frame exceeds covis_prob_threshold."""

    def __init__(self, onboarding: OnboardingConfig, n_frames: int, data_graph: DataGraph,
                 device: str = 'cuda', sequence_boundaries: list[int] | None = None,
                 frame_provider=None):
        super().__init__(onboarding, n_frames, data_graph, device, sequence_boundaries,
                         frame_provider)
        # Import the covis model by FILE PATH, not as `training.vggt_covis.model`:
        # adapters/vggt_adapter.py sys.path-inserts repositories/vggt, whose own
        # top-level `training` package shadows ours once the tracking provider has
        # been constructed (which the pipeline does before the frame filter).
        import importlib.util
        model_py = Path(__file__).resolve().parent.parent / 'training' / 'vggt_covis' / 'model.py'
        spec = importlib.util.spec_from_file_location('glopose_vggt_covis_model', model_py)
        covis_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(covis_module)
        self.covis_model = covis_module.load_covis_model(
            onboarding.covis_head_weights_path, device)
        self.covis_model.covis_head.eval()
        self.covis_prob_threshold = onboarding.covis_prob_threshold

    @torch.no_grad()
    def _pair_score(self, source_frame: int, target_frame: int) -> float:
        pair = torch.stack([self._frame_518(source_frame),
                            self._frame_518(target_frame)]).to(self.device)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            logits = self.covis_model(pair[None])        # (1, 2, 518, 518)
        prob = torch.sigmoid(logits[0, 0].float())       # source-frame covis probability

        seg = self._seg_518(source_frame)
        n_fg = seg.sum().clamp_min(1)
        return float(((prob > self.covis_prob_threshold) & seg).sum() / n_fg)


class FrameFilterVGGTTrackVis(FrameFilterVGGTPairScore):
    """Training-free tracker-native matchability: track a grid of the source
    keyframe's object pixels THROUGH the intervening frames to the candidate with
    the VGGT TrackHead and score the fraction still visible at the candidate.

    The chunk matters: on a bare 2-frame video the vis head saturates (~0.95+ even
    for opposite orbit sides — appearance matching finds "something" everywhere),
    while tracking through a real temporal chunk shows honest attrition. Scoring
    the chunk measures exactly what the downstream chunked reconstruction tracking
    (PointTrackingMatchingProvider, same TrackHead, same vis gate) will be able to
    do with this keyframe pair."""

    MAX_QUERIES = 512
    CHUNK_LEN = 8             # frames per scoring chunk (both endpoints included)

    def __init__(self, onboarding: OnboardingConfig, n_frames: int, data_graph: DataGraph,
                 device: str = 'cuda', sequence_boundaries: list[int] | None = None,
                 frame_provider=None):
        super().__init__(onboarding, n_frames, data_graph, device, sequence_boundaries,
                         frame_provider)
        from adapters.vggt_track_adapter import VGGTTrackAdapter
        self.tracker = VGGTTrackAdapter(device, onboarding.tracking.custom_weights_path)
        # Pixel-level vis gate for the SCORE (covis_prob_threshold, calibrated 0.7 on
        # HANDAL val chunks) — deliberately not tracking.visibility_threshold, which
        # gates the reconstruction correspondences and stays at its own value.
        self.vis_threshold = onboarding.covis_prob_threshold

    @torch.no_grad()
    def _pair_score(self, source_frame: int, target_frame: int) -> float:
        seg = self._seg_518(source_frame)
        pts_yx = seg.nonzero()                            # (N, 2) y, x
        if len(pts_yx) == 0:
            return 0.0
        sel = torch.linspace(0, len(pts_yx) - 1, min(self.MAX_QUERIES, len(pts_yx)),
                             device=pts_yx.device).long().unique()
        queries_xy = pts_yx[sel][:, [1, 0]].float()
        lo, hi = sorted((source_frame, target_frame))
        chunk = torch.linspace(lo, hi, min(self.CHUNK_LEN, hi - lo + 1)).round().long()
        chunk = chunk.unique().tolist()
        if source_frame > target_frame:                   # query frame must be first
            chunk = chunk[::-1]
        video = torch.stack([self._frame_518(i) for i in chunk]).to(self.device)
        _, certainty = self.tracker.track(video, queries_xy)   # (T, N)
        return float((certainty[-1] > self.vis_threshold).float().mean())


def _backproject_to_cam(pts_xy: 'np.ndarray', depths: 'np.ndarray', K: 'np.ndarray') -> 'np.ndarray':
    """Back-project integer pixel coords (x, y) at given depths into 3D camera coords.

    X = (x - cx) / fx * z, Y = (y - cy) / fy * z, Z = z. Returns (N, 3).
    """
    import numpy as np
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (pts_xy[:, 0].astype(np.float64) - cx) / fx * depths
    y = (pts_xy[:, 1].astype(np.float64) - cy) / fy * depths
    return np.stack([x, y, depths], axis=1)


def compute_matching_reliability(src_pts_xy_int: torch.Tensor, certainty: torch.Tensor,
                                 source_segmentation_mask: torch.Tensor, match_certainty_threshold: float,
                                 min_num_of_certain_matches: int = 0) -> float:
    fg_matches_mask = source_segmentation_mask[src_pts_xy_int[:, 1], src_pts_xy_int[:, 0]].bool()
    fg_certainties = certainty[fg_matches_mask]
    fg_certainties_above_threshold = fg_certainties > match_certainty_threshold
    reliability = fg_certainties_above_threshold.sum() / (fg_certainties.numel() + 1e-5)
    enough_certain_matches = (fg_certainties_above_threshold.numel() > min_num_of_certain_matches)
    reliability *= float(enough_certain_matches)
    reliability = reliability.item()
    return reliability


def create_frame_filter(onboarding: OnboardingConfig, device: str, n_frames: int,
                        data_graph: DataGraph,
                        flow_provider: MatchingProvider = None,
                        sequence_boundaries: list[int] | None = None,
                        frame_provider=None) -> BaseFrameFilter:
    """Factory that maps a config string to a BaseFrameFilter instance.

    Args:
        onboarding: OnboardingConfig with frame_filter, sift sub-configs.
        device: PyTorch device string (e.g. 'cuda').
        n_frames: Total number of input frames.
        data_graph: The shared DataGraph.
        flow_provider: Flow provider for dense-matching-based filters.
        sequence_boundaries: Frame indices where a new sub-sequence starts (e.g. [N] for
            concatenated up+down with N down frames). Used for boundary-aware filtering.
        frame_provider: FrameProvider serving ORIGINAL (unmasked) frames; required by
            the 'vggt_covis' filter.
    """

    def _dense_matching():
        return RoMaFrameFilter(onboarding, n_frames, data_graph, flow_provider, device, sequence_boundaries)

    def _ransac():
        return FrameFilterRANSAC(onboarding, n_frames, data_graph, flow_provider, device, sequence_boundaries)

    def _passthrough():
        return FrameFilterPassThrough(onboarding, n_frames, data_graph, sequence_boundaries=sequence_boundaries)

    def _sift():
        from data_providers.matching_provider_sift import (
            SparseMatchingProvider, SIFTKeypointDetector, LightGlueKeypointMatcher)
        detector = SIFTKeypointDetector(device)
        matcher = LightGlueKeypointMatcher(device)
        sift_provider = SparseMatchingProvider(detector, matcher,
                                               num_features=onboarding.sift.sift_filter_num_feats,
                                               device=device)
        return FrameFilterSift(onboarding, n_frames, data_graph, sift_provider)

    def _max_visible():
        return FrameFilterMaxVisible(onboarding, n_frames, data_graph, sequence_boundaries=sequence_boundaries)

    def _depth():
        return FrameFilterDepth(onboarding, n_frames, data_graph, flow_provider, device, sequence_boundaries)

    def _linear_transition():
        return FrameFilterLinearTransition(onboarding, n_frames, data_graph, flow_provider, device,
                                           sequence_boundaries)

    def _vggt_covis():
        return FrameFilterVGGTCovis(onboarding, n_frames, data_graph, device,
                                    sequence_boundaries, frame_provider=frame_provider)

    def _vggt_trackvis():
        return FrameFilterVGGTTrackVis(onboarding, n_frames, data_graph, device,
                                       sequence_boundaries, frame_provider=frame_provider)

    filters = {
        'dense_matching': _dense_matching,
        'RANSAC': _ransac,
        'passthrough': _passthrough,
        'SIFT': _sift,
        'max_visible': _max_visible,
        'depth': _depth,
        'linear_transition': _linear_transition,
        'vggt_covis': _vggt_covis,
        'vggt_trackvis': _vggt_trackvis,
    }
    if onboarding.frame_filter not in filters:
        raise ValueError(f"Unknown frame filter '{onboarding.frame_filter}'. Options: {list(filters.keys())}")
    return filters[onboarding.frame_filter]()
