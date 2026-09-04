"""Adapter for descriptor extraction and detection containers.

This module provides the DescriptorExtractor protocol and concrete
implementation using vendored DINOv2/v3 code (no Hydra dependency).

The sys.path manipulation for cnos is kept solely for pickle compatibility:
old ViewGraph pickles may reference cnos types (e.g. src.model.utils.Detections)
and need cnos on the path at deserialization time.
"""

import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

# TODO: Remove once all ViewGraph caches are regenerated without cnos types
_cnos_path = './repositories/cnos'
if _cnos_path not in sys.path:
    sys.path.append(_cnos_path)


# ---------------------------------------------------------------------------
# DescriptorExtractor protocol — consumers program against this
# ---------------------------------------------------------------------------

@runtime_checkable
class DescriptorExtractor(Protocol):
    """Protocol for DINOv2/v3 descriptor extraction.

    Callers pass raw numpy images, mask tensors, and box tensors.
    """

    def extract_descriptors(self, image_np, masks: Tensor, boxes: Tensor) -> tuple[Tensor, Tensor]:
        """Extract (cls_descriptors, patch_descriptors) for the given proposals.

        Args:
            image_np: HxWx3 uint8 numpy array.
            masks: [N, H, W] boolean/float mask tensor.
            boxes: [N, 4] long tensor in xyxy format.

        Returns:
            (cls_descriptors, patch_descriptors) tensors.
        """
        ...

    def get_detections_from_files(self, image_path: Path, segmentation_path: Path) -> tuple[Tensor, Tensor]:
        """Extract descriptors from image and segmentation file paths.

        Returns:
            (cls_descriptors, patch_descriptors) tensors.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete implementation using vendored DINOv2/v3
# ---------------------------------------------------------------------------

class GloPoseDescriptorExtractor:
    """Wraps vendored CustomDINOv2 behind the DescriptorExtractor interface.

    Uses descriptor_from_config() instead of Hydra-based descriptor_from_hydra().
    """

    def __init__(self, model: str = 'dinov3', mask_detections: bool = True, device: str = 'cuda'):
        from adapters.dino_descriptor import descriptor_from_config
        self._dino_model = descriptor_from_config(model=model, mask_detections=mask_detections, device=device)

    def extract_descriptors(self, image_np, masks: Tensor, boxes: Tensor) -> tuple[Tensor, Tensor]:
        class _Proposals:
            def __init__(self, masks, boxes):
                self.masks = masks
                self.boxes = boxes

        proposals = _Proposals(masks, boxes)
        return self._dino_model(image_np, proposals)

    def get_detections_from_files(self, image_path: Path, segmentation_path: Path) -> tuple[Tensor, Tensor]:
        return self._dino_model.get_detections_from_files(image_path, segmentation_path)


# Backward-compat alias
CnosDescriptorExtractor = GloPoseDescriptorExtractor


def create_descriptor_extractor(model: str = 'dinov3', mask_detections: bool = True,
                                device: str = 'cuda') -> GloPoseDescriptorExtractor:
    """Factory: primary entry point for creating a descriptor extractor."""
    return GloPoseDescriptorExtractor(model=model, mask_detections=mask_detections, device=device)


