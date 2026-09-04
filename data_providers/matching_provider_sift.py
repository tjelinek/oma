from abc import abstractmethod, ABC
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from kornia.feature import get_laf_center, match_adalam, LightGlueMatcher
from kornia_moons.feature import laf_from_opencv_SIFT_kpts
from torchvision.transforms.functional import to_pil_image

from data_providers.flow_provider import MatchingProvider

# ---------------------------------------------------------------------------
# Keypoint detector / matcher ABCs
# ---------------------------------------------------------------------------

class KeypointDetector(ABC):
    @abstractmethod
    def detect(self, image: torch.Tensor, num_features: int,
               segmentation: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (lafs, keypoints_xy, descriptors)."""
        pass


class KeypointMatcher(ABC):
    @abstractmethod
    def match(self, descriptors1: torch.Tensor, descriptors2: torch.Tensor,
              lafs1: torch.Tensor, lafs2: torch.Tensor,
              hw1: Tuple[int, int], hw2: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (scores, match_indices).

        scores: per-match confidence in [0, 1].
        match_indices: (N, 2) tensor of matched keypoint index pairs.
        """
        pass


# ---------------------------------------------------------------------------
# Concrete detectors
# ---------------------------------------------------------------------------

class SIFTKeypointDetector(KeypointDetector):

    def __init__(self, device: str = 'cpu'):
        self.device = device

    def detect(self, image: torch.Tensor, num_features: int,
               segmentation: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return detect_sift_features(image, num_features, segmentation, self.device)


# ---------------------------------------------------------------------------
# Concrete matchers
# ---------------------------------------------------------------------------

class LightGlueKeypointMatcher(KeypointMatcher):

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self._matcher = LightGlueMatcher('sift').eval().to(self.device)

    def match(self, descriptors1: torch.Tensor, descriptors2: torch.Tensor,
              lafs1: torch.Tensor, lafs2: torch.Tensor,
              hw1: Tuple[int, int], hw2: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            scores, idxs = self._matcher(descriptors1, descriptors2, lafs1, lafs2, hw1=hw1, hw2=hw2)
        return scores, idxs


class AdaLAMKeypointMatcher(KeypointMatcher):

    def __init__(self, device: str = 'cpu'):
        self.device = device

    def match(self, descriptors1: torch.Tensor, descriptors2: torch.Tensor,
              lafs1: torch.Tensor, lafs2: torch.Tensor,
              hw1: Tuple[int, int], hw2: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            scores, idxs = match_adalam(descriptors1, descriptors2, lafs1, lafs2, hw1=hw1, hw2=hw2)
        return scores, idxs


# ---------------------------------------------------------------------------
# SparseMatchingProvider — composes a KeypointDetector + KeypointMatcher
# ---------------------------------------------------------------------------

class SparseMatchingProvider(MatchingProvider):
    """Sparse keypoint matching provider (e.g. SIFT + LightGlue).

    Composes a KeypointDetector and a KeypointMatcher to produce matched
    source/target points with real matching confidence scores.
    """

    def __init__(self, detector: KeypointDetector, matcher: KeypointMatcher,
                 num_features: int, device: str = 'cpu', data_graph=None):
        super().__init__(device, data_graph)
        self.detector = detector
        self.matcher = matcher
        self.num_features = num_features

    def get_source_target_points(self, source_image: torch.Tensor, target_image: torch.Tensor,
                                 sample=None, source_image_segmentation: torch.Tensor = None,
                                 target_image_segmentation: torch.Tensor = None, source_image_name=None,
                                 target_image_name=None, source_image_index: int = None,
                                 target_image_index: int = None, as_int: bool = False,
                                 zero_certainty_outside_segmentation: bool = False, only_foreground_matches=False) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        lafs1, kpts1_xy, descriptors1 = self.detector.detect(
            source_image, self.num_features, source_image_segmentation)
        lafs2, kpts2_xy, descriptors2 = self.detector.detect(
            target_image, self.num_features, target_image_segmentation)

        # Filter keypoints to foreground using segmentation masks
        if source_image_segmentation is not None:
            kpts1_xy_int = kpts1_xy.to(torch.long)
            in_seg1 = source_image_segmentation[kpts1_xy_int[:, 1], kpts1_xy_int[:, 0]].to(torch.bool)
            kpts1_xy = kpts1_xy[in_seg1]
            lafs1 = lafs1[:, in_seg1]
            descriptors1 = descriptors1[in_seg1]

        if target_image_segmentation is not None:
            kpts2_xy_int = kpts2_xy.to(torch.long)
            in_seg2 = target_image_segmentation[kpts2_xy_int[:, 1], kpts2_xy_int[:, 0]].to(torch.bool)
            kpts2_xy = kpts2_xy[in_seg2]
            lafs2 = lafs2[:, in_seg2]
            descriptors2 = descriptors2[in_seg2]

        if descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
            empty = torch.zeros(0, 2, device=self.device, dtype=torch.float)
            return empty, empty.clone(), torch.zeros(0, device=self.device, dtype=torch.float)

        hw1 = tuple(source_image.shape[-2:])
        hw2 = tuple(target_image.shape[-2:])

        scores, idxs = self.matcher.match(descriptors1, descriptors2, lafs1, lafs2, hw1, hw2)

        src_pts_xy = kpts1_xy[idxs[:, 0]]
        dst_pts_xy = kpts2_xy[idxs[:, 1]]

        # Use real matching scores as certainty (not torch.ones). kornia's LightGlueMatcher
        # returns them as an (N, 1) column; downstream (unique_keypoints_from_matches)
        # argsorts the certainty along its last dim, which for (N, 1) yields (N, 1, ...)
        # index tensors that _write_colmap_database rejects (ndim != 2), silently
        # writing zero matches for every pair. Flatten to (N,) like the dense providers.
        certainty = scores.to(torch.float).to(self.device).reshape(-1)

        if as_int:
            src_pts_xy, dst_pts_xy = MatchingProvider.keypoints_to_int(
                src_pts_xy, dst_pts_xy, source_image, target_image)

        if only_foreground_matches:
            src_pts_int = src_pts_xy.to(torch.long) if not as_int else src_pts_xy
            dst_pts_int = dst_pts_xy.to(torch.long) if not as_int else dst_pts_xy

            if source_image_segmentation is not None:
                fg_src = source_image_segmentation[src_pts_int[:, 1], src_pts_int[:, 0]].bool()
            else:
                fg_src = torch.ones(src_pts_xy.shape[0], dtype=torch.bool, device=self.device)

            if target_image_segmentation is not None:
                fg_tgt = target_image_segmentation[dst_pts_int[:, 1], dst_pts_int[:, 0]].bool()
            else:
                fg_tgt = torch.ones(dst_pts_xy.shape[0], dtype=torch.bool, device=self.device)

            fg_mask = fg_src & fg_tgt
            src_pts_xy = src_pts_xy[fg_mask]
            dst_pts_xy = dst_pts_xy[fg_mask]
            certainty = certainty[fg_mask]

        return src_pts_xy, dst_pts_xy, certainty


# ---------------------------------------------------------------------------
# Helper functions (kept as-is for reuse)
# ---------------------------------------------------------------------------

def detect_sift_features(image: torch.Tensor, num_features: int, segmentation: Optional[torch.Tensor] = None,
                         device: str = 'cpu'):
    sift = cv2.SIFT_create(num_features, edgeThreshold=-1000, contrastThreshold=-1000)

    if segmentation is not None:
        segmentation_np = np.array(to_pil_image(segmentation))
    else:
        segmentation_np = None

    image_rgb = np.array(to_pil_image(image))
    image_cv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    keypoints, descriptors = sift.detectAndCompute(image_cv, segmentation_np)
    lafs = laf_from_opencv_SIFT_kpts(keypoints).to(device)
    descriptors = sift_to_rootsift(torch.from_numpy(descriptors)).to(device)
    desc_dim = descriptors.shape[-1]
    keypoints = get_laf_center(lafs).reshape(-1, 2).to(device)
    descriptors = descriptors.reshape(-1, desc_dim).to(device)

    return lafs, keypoints, descriptors


def match_features_sift(descriptors1, descriptors2, lafs1, lafs2, hw1, hw2, alg='lightglue'):
    if alg == 'lightglue':
        matcher = LightGlueMatcher('sift').eval().to(descriptors1.device)
    elif alg == 'adalam':
        matcher = match_adalam
    else:
        raise ValueError('unknown algorithm')

    with torch.inference_mode():
        dists, idxs = matcher(descriptors1, descriptors2, lafs1, lafs2, hw1=hw1, hw2=hw2)
        # Adalam takes into account also geometric information and image sizes

    return dists, idxs


def sift_to_rootsift(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, p=1, dim=-1, eps=eps)
    x.clip_(min=eps).sqrt_()
    return torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)
