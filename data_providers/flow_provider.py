import os
import shutil
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Tuple, Optional

import torch
import torchvision
from einops import rearrange
from romatch import roma_outdoor
from romatch.models.model_zoo import roma_model
from romatch.utils.kde import kde

from configs.glopose_config import OnboardingConfig
from configs.matching.roma_configs.base_roma_config import BaseRomaConfig
from configs.matching.ufm_configs.base_ufm_config import BaseUFMConfig
from data_structures.data_graph import DataGraph
from utils.flow import roma_warp_to_pixel_coordinates, convert_to_roma_warp, convert_certainty_to_roma_format


class FlowCache:
    """Standalone cache for flow computation results.

    Handles both disk-based and DataGraph-based caching of flow warps,
    certainties, and sampled source/target points.
    """

    def __init__(self, device: str, cache_dir: Path, data_graph: DataGraph = None,
                 allow_missing: bool = True, allow_disk_cache: bool = True, purge_cache: bool = False):
        self.device = device
        self.data_graph: Optional[DataGraph] = data_graph

        self.warps_path = cache_dir / 'warps'
        self.certainties_path = cache_dir / 'certainties'

        if purge_cache and self.warps_path.exists():
            shutil.rmtree(self.warps_path)
        if purge_cache and self.certainties_path.exists():
            shutil.rmtree(self.certainties_path)

        self.warps_path.mkdir(exist_ok=True, parents=True)
        self.certainties_path.mkdir(exist_ok=True, parents=True)

        self.allow_missing: bool = allow_missing
        self.allow_disk_cache: bool = False

    def datagraph_edge_exists(self, source_image_index, target_image_index) -> bool:
        return (source_image_index is not None and target_image_index is not None and
                self.data_graph is not None and
                self.data_graph.G.has_edge(source_image_index, target_image_index))

    def get_cache_filenames(self, source_image_index, source_image_name,
                            target_image_index, target_image_name) -> Tuple[Optional[Path], Optional[Path]]:
        if source_image_name is not None and target_image_name is not None:
            saved_filename = f'{source_image_name.stem}___{target_image_name.stem}.pt'
            warp_filename = self.warps_path / saved_filename
            certainty_filename = self.certainties_path / saved_filename
        elif (source_image_index is not None and target_image_index is not None and
              self.data_graph is not None and
              self.data_graph.G.has_node(source_image_index) and
              self.data_graph.G.has_node(target_image_index)):
            source_data = self.data_graph.get_frame_data(source_image_index)
            target_data = self.data_graph.get_frame_data(target_image_index)
            saved_filename = f'{source_data.image_filename.stem}___{target_data.image_filename.stem}.pt'
            warp_filename = self.warps_path / saved_filename
            certainty_filename = self.certainties_path / saved_filename
        else:
            warp_filename = None
            certainty_filename = None
        return warp_filename, certainty_filename

    def try_load_flow(self, source_image_index, target_image_index,
                      warp_filename, certainty_filename) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        warp, certainty = None, None
        if self.datagraph_edge_exists(source_image_index, target_image_index):
            edge_data = self.data_graph.get_edge_observations(source_image_index, target_image_index)
            if edge_data.roma_flow_warp is not None and edge_data.roma_flow_warp_certainty is not None:
                warp, certainty = edge_data.roma_flow_warp, edge_data.roma_flow_warp_certainty
        if (warp is None or certainty is None) and warp_filename is not None and certainty_filename is not None:
            if warp_filename.exists() and certainty_filename.exists() and self.allow_disk_cache:
                warp = torch.load(warp_filename, weights_only=True, map_location=self.device)
                certainty = torch.load(certainty_filename, weights_only=True, map_location=self.device)
        return warp, certainty

    def save_flow_to_disk(self, warp, certainty, warp_filename, certainty_filename,
                          source_image_name, target_image_name):
        if source_image_name and target_image_name and self.allow_missing and self.allow_disk_cache:
            torch.save(warp, warp_filename)
            torch.save(certainty, certainty_filename)

    def save_flow_to_datagraph(self, source_image_index, target_image_index, warp, certainty):
        if self.data_graph is not None:
            if source_image_index is not None and target_image_index is not None:
                if not self.data_graph.G.has_edge(source_image_index, target_image_index):
                    self.data_graph.add_new_arc(source_image_index, target_image_index)

                edge_data = self.data_graph.get_edge_observations(source_image_index, target_image_index)
                if edge_data.roma_flow_warp is None:
                    edge_data.roma_flow_warp = warp
                if edge_data.roma_flow_warp_certainty is None:
                    edge_data.roma_flow_warp_certainty = certainty

    def try_load_points(self, source_image_index, target_image_index) \
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.datagraph_edge_exists(source_image_index, target_image_index):
            edge_data = self.data_graph.get_edge_observations(source_image_index, target_image_index)
            if (edge_data.src_pts_xy_roma is not None and
                    edge_data.dst_pts_xy_roma is not None and
                    edge_data.src_dst_certainty_roma is not None):
                return edge_data.src_pts_xy_roma, edge_data.dst_pts_xy_roma, edge_data.src_dst_certainty_roma
        return None, None, None

    def save_points_to_datagraph(self, source_image_index, target_image_index,
                                 src_pts_xy, dst_pts_xy, certainty):
        if self.data_graph is not None and source_image_index is not None and target_image_index is not None:
            if not self.data_graph.G.has_edge(source_image_index, target_image_index):
                self.data_graph.add_new_arc(source_image_index, target_image_index)

            if self.datagraph_edge_exists(source_image_index, target_image_index):
                edge_data = self.data_graph.get_edge_observations(source_image_index, target_image_index)
                if edge_data.src_pts_xy_roma is None:
                    edge_data.src_pts_xy_roma = src_pts_xy
                if edge_data.dst_pts_xy_roma is None:
                    edge_data.dst_pts_xy_roma = dst_pts_xy
                if edge_data.src_dst_certainty_roma is None:
                    edge_data.src_dst_certainty_roma = certainty


class MatchingProvider(ABC):
    """Base class for all matching providers (dense flow-based and sparse keypoint-based).

    Defines the common interface: get_source_target_points returns matched
    (src_pts_xy, dst_pts_xy, certainty) tensors for a pair of images.
    """

    def __init__(self, device: str, data_graph: Optional[DataGraph] = None):
        self.device = device
        self.data_graph: Optional[DataGraph] = data_graph

    @abstractmethod
    def get_source_target_points(self, source_image: torch.Tensor, target_image: torch.Tensor,
                                 sample=None, source_image_segmentation: torch.Tensor = None,
                                 target_image_segmentation: torch.Tensor = None, source_image_name: Path = None,
                                 target_image_name: Path = None, source_image_index: int = None,
                                 target_image_index: int = None, as_int: bool = False,
                                 zero_certainty_outside_segmentation: bool = False, only_foreground_matches=False) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def get_source_target_points_datagraph(self, source_image_index: int, target_image_index: int,
                                           sample: int = None, as_int: bool = False,
                                           zero_certainty_outside_segmentation: bool = False,
                                           only_foreground_matches: bool = False) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.data_graph is not None
        source_data = self.data_graph.get_frame_data(source_image_index)
        target_data = self.data_graph.get_frame_data(target_image_index)

        return self.get_source_target_points(source_data.frame_observation.observed_image.squeeze(),
                                             target_data.frame_observation.observed_image.squeeze(), sample,
                                             source_data.frame_observation.observed_segmentation.squeeze(),
                                             target_data.frame_observation.observed_segmentation.squeeze(),
                                             source_data.image_filename, target_data.image_filename,
                                             source_image_index, target_image_index, as_int,
                                             zero_certainty_outside_segmentation, only_foreground_matches)

    @staticmethod
    def keypoints_to_int(src_pts_xy_roma, dst_pts_xy_roma, source_image_tensor, target_image_tensor):
        h1 = source_image_tensor.shape[-2]
        w1 = source_image_tensor.shape[-1]
        h2 = target_image_tensor.shape[-2]
        w2 = target_image_tensor.shape[-1]
        src_pts_xy_roma = src_pts_xy_roma.to(torch.int)
        dst_pts_xy_roma = dst_pts_xy_roma.to(torch.int)
        src_pts_xy_roma[:, 0] = torch.clamp(src_pts_xy_roma[:, 0], 0, w1 - 1)
        src_pts_xy_roma[:, 1] = torch.clamp(src_pts_xy_roma[:, 1], 0, h1 - 1)
        dst_pts_xy_roma[:, 0] = torch.clamp(dst_pts_xy_roma[:, 0], 0, w2 - 1)
        dst_pts_xy_roma[:, 1] = torch.clamp(dst_pts_xy_roma[:, 1], 0, h2 - 1)
        return src_pts_xy_roma, dst_pts_xy_roma


class FlowMatchingProvider(MatchingProvider):
    """Matching provider based on dense optical flow (RoMa, UFM).

    Computes dense flow warps, then samples source/target point correspondences.
    """

    def __init__(self, device: str, cache: FlowCache = None):
        super().__init__(device, data_graph=cache.data_graph if cache else None)
        self.cache = cache
        # Zoomed matching: crop source/target to their segmentation bboxes (+margin) before
        # matching and map the points back to full-frame coordinates. For masked (black-bg)
        # inputs this multiplies the effective object resolution seen by the matcher, whose
        # internal working resolution is fixed. Set from the matcher config by subclasses.
        self.crop_matching = False
        self.crop_matching_margin = 1.15

    def _compute_raw(self, source_image_tensor: torch.Tensor,
                     target_image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def _unify_window_sizes(sbox, tbox, s_shape, t_shape):
        """Expand both rect windows to a common (w, h) — matchers that batch source and
        target together (UFM backward=True) require equal tensor sizes. Windows are
        re-centered and shifted to stay inside their image bounds; returns (sbox, tbox)
        or None when a common size does not fit both images."""
        def dims(b):
            return b[2] - b[0], b[3] - b[1]

        sw, sh = dims(sbox)
        tw, th = dims(tbox)
        w, h = max(sw, tw), max(sh, th)

        def refit(box, shape):
            H, W = shape[-2], shape[-1]
            if w > W or h > H:
                return None
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            x0 = int(round(cx - w / 2))
            y0 = int(round(cy - h / 2))
            x0 = min(max(x0, 0), W - w)
            y0 = min(max(y0, 0), H - h)
            return x0, y0, x0 + w, y0 + h

        sbox2 = refit(sbox, s_shape)
        tbox2 = refit(tbox, t_shape)
        if sbox2 is None or tbox2 is None:
            return None
        return sbox2, tbox2

    @staticmethod
    def _seg_bbox_window(seg: torch.Tensor, margin: float, min_side: int = 64):
        """Rect window around the segmentation bbox (+margin), clamped to image bounds.
        Returns (x0, y0, x1, y1) exclusive-end, or None for an (almost) empty mask."""
        ys, xs = torch.nonzero(seg > 0, as_tuple=True)
        if len(xs) < 5:
            return None
        H, W = seg.shape[-2], seg.shape[-1]
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w = max((x1 - x0 + 1) * margin, min_side)
        h = max((y1 - y0 + 1) * margin, min_side)
        nx0 = max(0, int(round(cx - w / 2)))
        ny0 = max(0, int(round(cy - h / 2)))
        nx1 = min(W, int(round(cx + w / 2)))
        ny1 = min(H, int(round(cy + h / 2)))
        if nx1 - nx0 < 8 or ny1 - ny0 < 8:
            return None
        return nx0, ny0, nx1, ny1

    def compute_flow(self, source_image_tensor: torch.Tensor, target_image_tensor: torch.Tensor, sample=None,
                     source_image_segmentation: torch.Tensor = None, target_image_segmentation: torch.Tensor = None,
                     source_image_name: Path = None, target_image_name: Path = None, source_image_index: int = None,
                     target_image_index: int = None, zero_certainty_outside_segmentation: bool = False):

        warp, certainty = None, None
        warp_filename, certainty_filename = None, None

        if self.cache is not None:
            warp_filename, certainty_filename = self.cache.get_cache_filenames(
                source_image_index, source_image_name, target_image_index, target_image_name)
            warp, certainty = self.cache.try_load_flow(
                source_image_index, target_image_index, warp_filename, certainty_filename)

        if warp is None or certainty is None:
            warp, certainty = self._compute_raw(source_image_tensor, target_image_tensor)

            if self.cache is not None:
                self.cache.save_flow_to_disk(warp, certainty, warp_filename, certainty_filename,
                                             source_image_name, target_image_name)

        if zero_certainty_outside_segmentation:
            certainty = self.zero_certainty_outside_segmentation(certainty, source_image_segmentation,
                                                                 target_image_segmentation)

        if self.cache is not None:
            self.cache.save_flow_to_datagraph(source_image_index, target_image_index, warp, certainty)

        if sample:
            if (((source_image_segmentation is not None and source_image_segmentation.sum() <= 5) or
                 (target_image_segmentation is not None and target_image_segmentation.sum() <= 5)) and
                    zero_certainty_outside_segmentation):
                warp = torch.zeros(0, 4).to(warp.device).to(warp.dtype)
                certainty = torch.zeros(0, ).to(certainty.device).to(certainty.dtype)
            else:
                warp, certainty = self.sample(warp, certainty, sample)

        return warp, certainty

    def get_source_target_points(self, source_image: torch.Tensor, target_image: torch.Tensor,
                                 sample=None, source_image_segmentation: torch.Tensor = None,
                                 target_image_segmentation: torch.Tensor = None, source_image_name: Path = None,
                                 target_image_name: Path = None, source_image_index: int = None,
                                 target_image_index: int = None, as_int: bool = False,
                                 zero_certainty_outside_segmentation: bool = False, only_foreground_matches=False) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (self.crop_matching and source_image_segmentation is not None
                and target_image_segmentation is not None):
            sbox = self._seg_bbox_window(source_image_segmentation, self.crop_matching_margin)
            tbox = self._seg_bbox_window(target_image_segmentation, self.crop_matching_margin)
            if sbox is not None and tbox is not None:
                unified = self._unify_window_sizes(sbox, tbox, source_image_segmentation.shape,
                                                   target_image_segmentation.shape)
                sbox, tbox = unified if unified is not None else (None, None)
            if sbox is not None and tbox is not None:
                # Previously stored (already full-frame, shifted) points for this pair.
                if self.cache is not None:
                    c_src, c_dst, c_cert = self.cache.try_load_points(
                        source_image_index, target_image_index)
                    if c_src is not None and c_dst is not None and c_cert is not None:
                        if as_int:
                            c_src, c_dst = self.keypoints_to_int(
                                c_src, c_dst, source_image, target_image)
                        return c_src, c_dst, c_cert
                sx0, sy0, sx1, sy1 = sbox
                tx0, ty0, tx1, ty1 = tbox
                # Match on the crops (bypass flow/point caches — they are keyed on the
                # full-frame pair; the shifted full-frame points are cached below instead).
                cache_saved, self.cache = self.cache, None
                try:
                    src_pts_xy, dst_pts_xy, certainty = self._get_source_target_points_impl(
                        source_image[..., sy0:sy1, sx0:sx1], target_image[..., ty0:ty1, tx0:tx1],
                        sample,
                        source_image_segmentation[..., sy0:sy1, sx0:sx1],
                        target_image_segmentation[..., ty0:ty1, tx0:tx1],
                        None, None, None, None, False,
                        zero_certainty_outside_segmentation, only_foreground_matches)
                finally:
                    self.cache = cache_saved
                src_pts_xy = src_pts_xy + torch.tensor(
                    [sx0, sy0], device=src_pts_xy.device, dtype=src_pts_xy.dtype)
                dst_pts_xy = dst_pts_xy + torch.tensor(
                    [tx0, ty0], device=dst_pts_xy.device, dtype=dst_pts_xy.dtype)
                if getattr(self, 'crop_matching_add_fullframe', False):
                    # Multi-scale union: full-frame matches anchor the globally-consistent
                    # mode when the zoomed view amplifies symmetric-confusion matches.
                    cache_saved, self.cache = self.cache, None
                    try:
                        ff_src, ff_dst, ff_cert = self._get_source_target_points_impl(
                            source_image, target_image, sample,
                            source_image_segmentation, target_image_segmentation,
                            None, None, None, None, False,
                            zero_certainty_outside_segmentation, only_foreground_matches)
                    finally:
                        self.cache = cache_saved
                    src_pts_xy = torch.cat([src_pts_xy, ff_src.to(src_pts_xy.dtype)], dim=0)
                    dst_pts_xy = torch.cat([dst_pts_xy, ff_dst.to(dst_pts_xy.dtype)], dim=0)
                    certainty = torch.cat([certainty, ff_cert.to(certainty.dtype)], dim=0)
                if as_int:
                    src_pts_xy, dst_pts_xy = self.keypoints_to_int(
                        src_pts_xy, dst_pts_xy, source_image, target_image)
                if self.cache is not None:
                    self.cache.save_points_to_datagraph(source_image_index, target_image_index,
                                                        src_pts_xy, dst_pts_xy, certainty)
                return src_pts_xy, dst_pts_xy, certainty
        return self._get_source_target_points_impl(
            source_image, target_image, sample, source_image_segmentation,
            target_image_segmentation, source_image_name, target_image_name,
            source_image_index, target_image_index, as_int,
            zero_certainty_outside_segmentation, only_foreground_matches)

    def _get_source_target_points_impl(self, source_image: torch.Tensor, target_image: torch.Tensor,
                                       sample=None, source_image_segmentation: torch.Tensor = None,
                                       target_image_segmentation: torch.Tensor = None, source_image_name: Path = None,
                                       target_image_name: Path = None, source_image_index: int = None,
                                       target_image_index: int = None, as_int: bool = False,
                                       zero_certainty_outside_segmentation: bool = False,
                                       only_foreground_matches=False) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        src_pts_xy, dst_pts_xy, certainty = None, None, None
        if self.cache is not None:
            src_pts_xy, dst_pts_xy, certainty = self.cache.try_load_points(
                source_image_index, target_image_index)

        if src_pts_xy is None or dst_pts_xy is None or certainty is None:
            warp, certainty = self.compute_flow(source_image, target_image, sample,
                                                source_image_segmentation, target_image_segmentation,
                                                source_image_name, target_image_name, source_image_index,
                                                target_image_index, zero_certainty_outside_segmentation)

            h1 = source_image.shape[-2]
            w1 = source_image.shape[-1]
            h2 = target_image.shape[-2]
            w2 = target_image.shape[-1]
            src_pts_xy, dst_pts_xy = roma_warp_to_pixel_coordinates(warp, h1, w1, h2, w2)

            if len(src_pts_xy.shape) == 3 or len(dst_pts_xy.shape) == 3 or len(certainty.shape) == 2:
                assert len(src_pts_xy.shape) == 3 and len(dst_pts_xy.shape) == 3 and len(certainty.shape) == 2

                src_pts_xy = src_pts_xy.flatten(0, 1)
                dst_pts_xy = dst_pts_xy.flatten(0, 1)
                certainty = certainty.flatten(0, 1)

            src_pts_xy_int, dst_pts_xy_int = self.keypoints_to_int(src_pts_xy, dst_pts_xy, source_image, target_image)
            if as_int:
                src_pts_xy, dst_pts_xy = src_pts_xy_int, dst_pts_xy_int

            if only_foreground_matches:
                assert source_image_segmentation is not None or target_image_segmentation is not None

                if source_image_segmentation is not None:
                    assert len(source_image_segmentation.shape) == 2
                    in_segment_mask_src = source_image_segmentation[src_pts_xy_int[:, 1], src_pts_xy_int[:, 0]].bool()
                else:
                    in_segment_mask_src = torch.ones_like(src_pts_xy_int[:, 0], dtype=torch.bool)

                if target_image_segmentation is not None:
                    assert len(target_image_segmentation.shape) == 2
                    in_segment_mask_tgt = target_image_segmentation[dst_pts_xy_int[:, 1], dst_pts_xy_int[:, 0]].bool()
                else:
                    in_segment_mask_tgt = torch.ones_like(dst_pts_xy_int[:, 0], dtype=torch.bool)

                fg_matches = in_segment_mask_src * in_segment_mask_tgt

                src_pts_xy = src_pts_xy[fg_matches]
                dst_pts_xy = dst_pts_xy[fg_matches]
                certainty = certainty[fg_matches]

            if self.cache is not None:
                self.cache.save_points_to_datagraph(source_image_index, target_image_index,
                                                    src_pts_xy, dst_pts_xy, certainty)
        else:
            if as_int:
                src_pts_xy, dst_pts_xy = self.keypoints_to_int(src_pts_xy, dst_pts_xy, source_image, target_image)

        return src_pts_xy, dst_pts_xy, certainty

    @abstractmethod
    def sample(self, warp: torch.Tensor, certainty: torch.Tensor, sample: int) -> Tuple[torch.Tensor, torch.Tensor]:

        pass

    def zero_certainty_outside_segmentation(self, certainty: torch.Tensor,
                                            source_image_segmentation: torch.Tensor = None,
                                            target_image_segmentation: torch.Tensor = None) -> torch.Tensor:

        assert source_image_segmentation is not None or target_image_segmentation is not None

        certainty = certainty.clone()

        h, w = certainty.shape
        w //= 2
        if source_image_segmentation is not None:
            certainty[:, :w] *= source_image_segmentation.squeeze().bool().float()
        if target_image_segmentation is not None:
            certainty[:, w:2 * w] *= target_image_segmentation.squeeze().bool().float()

        return certainty


class RoMaMatchingProvider(FlowMatchingProvider):

    def __init__(self, device, roma_config: BaseRomaConfig, cache: FlowCache = None):
        FlowMatchingProvider.__init__(self, device, cache)

        if roma_config.use_custom_weights:
            custom_path = getattr(roma_config, 'custom_weights_path', None) \
                or os.environ.get('ROMA_WEIGHTS', 'weights/roma_outdoor_latest.pth')
            weights = torch.load(custom_path, map_location=device, weights_only=True)
            if isinstance(weights, dict) and "model" in weights:
                weights = weights["model"]
            print(f"[RoMa] loaded custom fine-tuned weights from {custom_path}")
        else:
            weights = torch.load(os.environ.get('ROMA_WEIGHTS', 'weights/roma_outdoor.pth'),
                                 map_location=device, weights_only=True)

        self.flow_model: roma_model = roma_outdoor(device=self.device, weights=weights)
        # roma_outdoor builds ConvRefiners with use_custom_corr=True, which imports an
        # uncompiled CUDA extension ('local_corr') and crashes match() at runtime. Disable it
        # so the native torch fallback is used (numerically correct; inference-only path).
        for m in self.flow_model.modules():
            if hasattr(m, "use_custom_corr"):
                m.use_custom_corr = False
        self.flow_model.sample_mode = 'balanced'  # This ensures that the matches are sampled ~ certainties
        self.roma_size_hw = (864, 864)

    def _compute_raw(self, source_image_tensor: torch.Tensor,
                     target_image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        source_image_roma = torchvision.transforms.functional.to_pil_image(source_image_tensor.squeeze())
        target_image_roma = torchvision.transforms.functional.to_pil_image(target_image_tensor.squeeze())
        warp, certainty = self.flow_model.match(source_image_roma, target_image_roma, device=self.device)
        # RoMa's match() returns a leading batch dim -> warp (1, H, 2W, 4), certainty (1, H, 2W).
        # The rest of the RoMa consumption path (warp->pixel coords, certainty masking, sampling,
        # COLMAP point extraction) is written for the un-batched (H, 2W, 4)/(H, 2W) form. Drop the
        # size-1 batch dim here so every consumer sees the expected shape (no-op if already un-batched).
        return warp.squeeze(0), certainty.squeeze(0)

    def sample(self, warp, certainty, sample):
        return self.flow_model.sample(warp, certainty, sample)

    def zero_certainty_outside_segmentation(self, certainty: torch.Tensor,
                                            source_image_segmentation: torch.Tensor = None,
                                            target_image_segmentation: torch.Tensor = None) -> torch.Tensor:
        roma_h, roma_w = self.roma_size_hw
        assert source_image_segmentation is not None or target_image_segmentation is not None

        certainty = certainty.clone()
        # RoMa's match() returns certainty as (H, 2W) with a leading batch dim -> (1, H, 2W):
        # forward half is the first W width-columns, backward half the next W. Slice the LAST
        # (width) axis via ellipsis so this is correct whether or not the batch dim is present
        # (indexing dim 1 masked the height when a leading singleton dim was present).
        if source_image_segmentation is not None:
            source_image_segment_roma_size = torchvision.transforms.functional.resize(source_image_segmentation[None],
                                                                                      size=self.roma_size_hw)
            certainty[..., :roma_w] *= source_image_segment_roma_size.squeeze().bool().float()
        if target_image_segmentation is not None:
            target_image_segment_roma_size = torchvision.transforms.functional.resize(target_image_segmentation[None],
                                                                                      size=self.roma_size_hw)
            certainty[..., roma_w:2 * roma_w] *= target_image_segment_roma_size.squeeze().bool().float()
        return certainty


class UFMMatchingProvider(FlowMatchingProvider):

    def __init__(self, device, ufm_config: BaseUFMConfig, cache: FlowCache = None):
        FlowMatchingProvider.__init__(self, device, cache)
        self.ufm_config = ufm_config

        from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement
        self.model = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine").to(self.device)

        if ufm_config.use_custom_weights:
            ckpt_path = getattr(ufm_config, 'custom_weights_path', None)
            assert ckpt_path is not None, "use_custom_weights=True requires ufm_config.custom_weights_path"
            sd = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(sd['model'] if isinstance(sd, dict) and 'model' in sd else sd)
            print(f"[UFM] loaded custom fine-tuned weights from {ckpt_path}")

        self.crop_matching = getattr(ufm_config, 'crop_matching', False)
        self.crop_matching_margin = getattr(ufm_config, 'crop_matching_margin', 1.15)
        self.crop_matching_add_fullframe = getattr(ufm_config, 'crop_matching_add_fullframe', False)
        if self.crop_matching:
            print(f"[UFM] zoomed matching enabled (seg-bbox crops, margin={self.crop_matching_margin}, "
                  f"union_fullframe={self.crop_matching_add_fullframe})")

        self.model.eval()

        self.sample_mode = 'balanced'
        self.sample_thresh = 0.5

    @torch.no_grad()
    def _compute_raw(self, source_image_tensor: torch.Tensor,
                     target_image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        h, w = source_image_tensor.shape[-2:]
        assert len(source_image_tensor.shape) == 3
        assert len(target_image_tensor.shape) == 3

        source_image_tensor_bhwc = rearrange(source_image_tensor[None], 'b c h w -> b h w c').to(torch.float)
        target_image_tensor_bhwc = rearrange(target_image_tensor[None], 'b c h w -> b h w c').to(torch.float)

        if self.ufm_config.backward:
            source_tensor_bhwc = torch.cat([source_image_tensor_bhwc, target_image_tensor_bhwc], dim=0)
            target_tensor_bhwc = torch.cat([target_image_tensor_bhwc, source_image_tensor_bhwc], dim=0)
        else:
            source_tensor_bhwc = source_image_tensor_bhwc
            target_tensor_bhwc = target_image_tensor_bhwc

        result = self.model.predict_correspondences_batched(source_image=source_tensor_bhwc,
                                                            target_image=target_tensor_bhwc,
                                                            data_norm_type='identity', )

        flow_forward = result.flow.flow_output[0]
        covisibility_forward = result.covisibility.mask[0]
        flow_backward = None
        covisibility_backward = None
        if self.ufm_config.backward:
            flow_backward = result.flow.flow_output[1]
            covisibility_backward = result.covisibility.mask[1]

        dst_pts_xy_forward = self.get_dst_pts(flow_forward, h, w)

        dst_pts_xy_roma = convert_to_roma_warp(dst_pts_xy_forward, flow_backward)
        covisibility = convert_certainty_to_roma_format(covisibility_forward, covisibility_backward)

        return dst_pts_xy_roma, covisibility

    def get_dst_pts(self, flow_forward, h, w):
        y, x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=self.device),
            torch.arange(w, dtype=torch.float32, device=self.device),
            indexing='ij'
        )
        coords = torch.stack([x, y], dim=0)
        dst_pts_xy = coords + flow_forward
        return dst_pts_xy

    def sample(self, warp: torch.Tensor, certainty: torch.Tensor, sample: int) -> Tuple[torch.Tensor, torch.Tensor]:

        # Taken from RoMa implementation

        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            warp.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty,
                                         num_samples=min(expansion_factor * sample, len(certainty)),
                                         replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1)
        p = 1 / (density + 1)
        p[density < 10] = 1e-7  # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p,
                                             num_samples=min(sample, len(good_certainty)),
                                             replacement=False)

        match_samples = good_matches[balanced_samples]
        certainty_samples = good_certainty[balanced_samples]

        return match_samples, certainty_samples


def create_matching_provider(name: str, onboarding: OnboardingConfig, device: str,
                             cache: FlowCache = None, data_graph=None,
                             frame_provider=None) -> MatchingProvider:
    """Factory that maps a config string to a MatchingProvider instance.

    Args:
        name: One of 'RoMa', 'UFM', 'SIFT', 'CoTracker'.
        onboarding: OnboardingConfig with roma, ufm, sift, tracking sub-configs.
        device: PyTorch device string (e.g. 'cuda').
        cache: Optional FlowCache for caching flow results.
        data_graph: DataGraph with per-frame observations; required by 'CoTracker'
            (point tracks run through the intermediate frames between keyframes).
        frame_provider: FrameProvider serving original (unmasked) frames; used by
            'CoTracker' when tracking.track_on_original_frames is set.
    """

    def _roma():
        return RoMaMatchingProvider(device, onboarding.roma, cache=cache)

    def _ufm():
        return UFMMatchingProvider(device, onboarding.ufm, cache=cache)

    def _sift():
        from data_providers.matching_provider_sift import (
            SparseMatchingProvider, SIFTKeypointDetector, LightGlueKeypointMatcher)
        detector = SIFTKeypointDetector(device)
        matcher = LightGlueKeypointMatcher(device)
        return SparseMatchingProvider(detector, matcher,
                                      num_features=onboarding.sift.sift_filter_num_feats,
                                      device=device,
                                      data_graph=cache.data_graph if cache else None)

    def _cotracker():
        from data_providers.matching_provider_tracking import PointTrackingMatchingProvider
        graph = data_graph if data_graph is not None else (cache.data_graph if cache else None)
        if graph is None:
            raise ValueError("'CoTracker' matching provider requires a DataGraph "
                             "(it tracks through the intermediate frames)")
        return PointTrackingMatchingProvider(device, onboarding.tracking, graph,
                                             frame_provider=frame_provider)

    providers = {'RoMa': _roma, 'UFM': _ufm, 'SIFT': _sift, 'CoTracker': _cotracker}
    if name not in providers:
        raise ValueError(f"Unknown matching provider '{name}'. Options: {list(providers.keys())}")
    return providers[name]()
