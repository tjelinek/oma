"""Point-tracking matching provider: reconstruction correspondences from point tracks.

Instead of matching a keyframe pair directly, query points are tracked from the source
keyframe THROUGH all intermediate pipeline frames (read from the DataGraph) to the
target keyframe (CoTracker3, see adapters/cotracker_adapter.py). The track endpoints
become the (src, dst) correspondences handed to reconstruction. Temporal continuity
makes tracks robust to the coherent wrong-warp failure mode of pairwise dense matching
on symmetric / low-texture masked objects: a tracker cannot "teleport" onto the
symmetric flip because it follows the point frame to frame.

Track re-seeding: the surviving track endpoints at keyframe b are reused (together
with fresh queries topping the budget up) as the query set for every chunk that starts
at b. With add_track_merging_matches, COLMAP then links the per-edge correspondences
that share these integer keypoints into multi-keyframe feature tracks.

The provider implements the MatchingProvider interface, so it composes with the
existing reconstruction code path unchanged; select it with
OnboardingConfig.reconstruction_matcher = 'CoTracker'.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import binary_erosion

from configs.matching.tracking_configs.base_tracking_config import BaseTrackingConfig
from data_providers.flow_provider import MatchingProvider
from data_structures.data_graph import DataGraph


class PointTrackingMatchingProvider(MatchingProvider):

    def __init__(self, device: str, tracking_config: BaseTrackingConfig, data_graph: DataGraph,
                 frame_provider=None):
        super().__init__(device, data_graph)
        self.config = tracking_config
        if tracking_config.tracker == 'vggt':
            from adapters.vggt_track_adapter import VGGTTrackAdapter
            self.tracker = VGGTTrackAdapter(device, tracking_config.custom_weights_path)
        elif tracking_config.tracker == 'spatrack2':
            from adapters.spatrack2_adapter import SpaTrack2Adapter
            self.tracker = SpaTrack2Adapter(device, tracking_config.custom_weights_path)
        elif tracking_config.tracker == 'alltracker':
            from adapters.alltracker_adapter import AllTrackerAdapter
            self.tracker = AllTrackerAdapter(device, tracking_config.custom_weights_path)
        elif tracking_config.tracker == 'cowtracker':
            from adapters.cowtracker_adapter import CoWTrackerAdapter
            self.tracker = CoWTrackerAdapter(device, tracking_config.custom_weights_path)
        else:
            from adapters.cotracker_adapter import CoTrackerAdapter
            self.tracker = CoTrackerAdapter(device, tracking_config.tracker,
                                            tracking_config.custom_weights_path)
        # FrameProvider serving the original (unmasked) frames at pipeline resolution;
        # required when track_on_original_frames is set.
        self.frame_provider = frame_provider
        if tracking_config.track_on_original_frames and frame_provider is None:
            raise ValueError("track_on_original_frames=True requires a FrameProvider "
                             "(the DataGraph only holds background-masked observations)")
        # keyframe idx -> (N, 2) int xy endpoints of tracks that survived into it
        self._seeds: dict[int, torch.Tensor] = {}
        # keyframe idx -> the query set actually used for chunks starting there (so all
        # out-edges of a keyframe share identical integer keypoints)
        self._queries: dict[int, torch.Tensor] = {}
        # (a, b) -> (src_int, dst_int, certainty); repeat calls (track merging phase)
        # must return identical correspondences
        self._pair_cache: dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def get_source_target_points(self, source_image: torch.Tensor, target_image: torch.Tensor,
                                 sample=None, source_image_segmentation: torch.Tensor = None,
                                 target_image_segmentation: torch.Tensor = None,
                                 source_image_name: Path = None, target_image_name: Path = None,
                                 source_image_index: int = None, target_image_index: int = None,
                                 as_int: bool = False, zero_certainty_outside_segmentation: bool = False,
                                 only_foreground_matches=False) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if source_image_name is None or target_image_name is None:
            raise ValueError("PointTrackingMatchingProvider needs image names "
                             "('{frame_idx}.png') to locate the video chunk in the DataGraph")
        frame_a = int(Path(source_image_name).stem)
        frame_b = int(Path(target_image_name).stem)
        if frame_a == frame_b:
            raise ValueError(f"Expected two distinct keyframes, got {frame_a} twice")
        # A dense (all-directed-pairs) keyframe graph requests both (a,b) and (b,a).
        # Chunks only run forward in time, so serve the backward edge from the
        # forward-tracked pair with source/target swapped (shares its cache entry).
        reverse = frame_a > frame_b
        if reverse:
            frame_a, frame_b = frame_b, frame_a
            source_image, target_image = target_image, source_image
            source_image_segmentation, target_image_segmentation = \
                target_image_segmentation, source_image_segmentation

        cached = self._pair_cache.get((frame_a, frame_b))
        if cached is None:
            cached = self._track_pair(frame_a, frame_b, source_image, target_image,
                                      source_image_segmentation, target_image_segmentation,
                                      only_foreground_matches)
            self._pair_cache[(frame_a, frame_b)] = cached
        src_pts_xy, dst_pts_xy, certainty = cached
        if reverse:
            src_pts_xy, dst_pts_xy = dst_pts_xy, src_pts_xy

        if sample is not None and len(certainty) > sample:
            top = torch.topk(certainty, sample).indices
            src_pts_xy, dst_pts_xy, certainty = src_pts_xy[top], dst_pts_xy[top], certainty[top]
        if not as_int:
            src_pts_xy, dst_pts_xy = src_pts_xy.float(), dst_pts_xy.float()
        return src_pts_xy, dst_pts_xy, certainty

    def _track_pair(self, frame_a: int, frame_b: int, source_image: torch.Tensor,
                    target_image: torch.Tensor, source_seg: torch.Tensor,
                    target_seg: torch.Tensor, only_foreground_matches: bool) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_idxs = self._chunk_indices(frame_a, frame_b)
        video, chunk_segs = self._load_chunk(frame_idxs)

        x0, y0, x1, y1 = self._chunk_crop_window(chunk_segs, video.shape[-2:])
        queries_xy = self._queries_for_keyframe(frame_a, source_seg)
        if len(queries_xy) == 0:
            empty = torch.zeros((0, 2), dtype=torch.int64, device=self.device)
            return empty, empty.clone(), torch.zeros(0, device=self.device)

        offset = torch.tensor([x0, y0], device=self.device, dtype=torch.float32)
        tracks, visibility = self.tracker.track(video[:, :, y0:y1, x0:x1],
                                                queries_xy.float().to(self.device) - offset)
        dst_xy = tracks[-1] + offset
        certainty = visibility[-1]

        h, w = target_image.shape[-2], target_image.shape[-1]
        keep = ((certainty >= self.config.visibility_threshold)
                & (dst_xy[:, 0] >= 0) & (dst_xy[:, 0] <= w - 1)
                & (dst_xy[:, 1] >= 0) & (dst_xy[:, 1] <= h - 1))
        src_int, dst_int = self.keypoints_to_int(queries_xy.float().to(self.device).clone(),
                                                 dst_xy.clone(), source_image, target_image)
        if only_foreground_matches and target_seg is not None:
            keep &= target_seg.squeeze()[dst_int[:, 1], dst_int[:, 0]].bool()
        if keep.sum() == 0:
            # Never return an empty match set: downstream track merging assumes at
            # least one correspondence per edge. One low-confidence match cannot
            # anchor a wrong pose, COLMAP's verification discards it.
            keep[int(certainty.argmax())] = True

        src_int, dst_int, certainty = src_int[keep], dst_int[keep], certainty[keep]
        if self.config.reseed_from_tracks and len(dst_int) > 0:
            prior = self._seeds.get(frame_b)
            merged = dst_int if prior is None else torch.cat([prior, dst_int])
            self._seeds[frame_b] = torch.unique(merged, dim=0)
        print(f"[Tracking] {frame_a}->{frame_b}: {len(frame_idxs)} frames, "
              f"{len(queries_xy)} queries -> {len(src_int)} tracks kept", flush=True)
        return src_int, dst_int, certainty

    def _chunk_indices(self, frame_a: int, frame_b: int) -> List[int]:
        idxs = [i for i in range(frame_a, frame_b + 1) if self.data_graph.G.has_node(i)]
        if len(idxs) > self.config.max_video_len:
            sel = np.unique(np.linspace(0, len(idxs) - 1, self.config.max_video_len)
                            .round().astype(int))
            idxs = [idxs[i] for i in sel]
        return idxs

    def _load_chunk(self, frame_idxs: List[int]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        images, segs = [], []
        for i in frame_idxs:
            obs = self.data_graph.get_frame_data(i).frame_observation
            if self.config.track_on_original_frames:
                images.append(self.frame_provider.next_image(i).squeeze().to(self.device))
            else:
                images.append(obs.observed_image.squeeze().to(self.device))
            segs.append(obs.observed_segmentation.squeeze().to(self.device))
        return torch.stack(images), segs

    def _chunk_crop_window(self, chunk_segs: List[torch.Tensor],
                           image_hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Fixed square window covering all chunk segmentations (+margin); identity
        window when cropping is disabled or no foreground exists."""
        h, w = image_hw
        if not self.config.crop_to_object:
            return 0, 0, w, h
        union = torch.stack([s.bool() for s in chunk_segs]).any(dim=0)
        ys, xs = torch.nonzero(union, as_tuple=True)
        if len(xs) == 0:
            return 0, 0, w, h
        cx = (xs.min() + xs.max()).item() / 2
        cy = (ys.min() + ys.max()).item() / 2
        side = max((xs.max() - xs.min()).item(), (ys.max() - ys.min()).item())
        side = min(int(side * self.config.crop_margin) + 1, min(h, w))
        x0 = int(np.clip(cx - side / 2, 0, w - side))
        y0 = int(np.clip(cy - side / 2, 0, h - side))
        return x0, y0, x0 + side, y0 + side

    def _queries_for_keyframe(self, frame_idx: int, seg: torch.Tensor) -> torch.Tensor:
        """Query set for chunks starting at frame_idx: surviving track endpoints from
        incoming chunks, topped up with fresh points sampled in the (eroded) mask.
        Cached so every out-edge of the keyframe uses identical keypoints."""
        if frame_idx in self._queries:
            return self._queries[frame_idx]
        budget = self.config.max_queries
        seeds = self._seeds.get(frame_idx)
        # Seeds are stored via torch.unique(dim=0), i.e. sorted by x. Taking the first
        # `budget` rows selected the leftmost band of the object and, with a complete
        # graph, the seeds always exceeded the budget, so no fresh queries were ever
        # sampled after the first keyframe: surfaces that appear later in the orbit were
        # never queried and never reconstructed. Subsample the seeds at random and cap
        # them so a fixed share of the budget is always fresh mask samples.
        parts = []
        n_seeds = 0
        if seeds is not None and len(seeds) > 0:
            cap = int(budget * self.config.max_seed_fraction)
            if len(seeds) > cap:
                rng = np.random.RandomState(frame_idx)
                seeds = seeds[torch.from_numpy(rng.permutation(len(seeds))[:cap])]
            parts.append(seeds.to(self.device))
            n_seeds = len(seeds)
        if n_seeds < budget and seg is not None:
            if self.config.query_full_frame:
                # "w/o mask" control: background features participate in tracking.
                mask = np.ones(tuple(seg.squeeze().shape[-2:]), dtype=bool)
            else:
                mask = seg.squeeze().bool().numpy(force=True)
                if self.config.erode_query_mask_iters > 0:
                    mask = binary_erosion(mask, iterations=self.config.erode_query_mask_iters)
            ys, xs = np.nonzero(mask)
            if len(xs) > 0:
                rng = np.random.RandomState(frame_idx)
                order = rng.permutation(len(xs))[:budget - n_seeds]
                fresh = torch.from_numpy(np.stack([xs[order], ys[order]], axis=1)) \
                    .to(dtype=torch.int64, device=self.device)
                parts.append(fresh)
        queries = torch.unique(torch.cat(parts), dim=0) if parts \
            else torch.zeros((0, 2), dtype=torch.int64, device=self.device)
        self._queries[frame_idx] = queries
        return queries
