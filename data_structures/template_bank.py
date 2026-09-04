from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from data_structures.types import ObjectId


@dataclass
class TemplateBank:
    """Detection-ready template representations for one or more objects.

    Created by the condensation module (B1) from onboarding results.
    Used by the detection module (B2) for descriptor matching and scoring.
    """
    images: Dict[ObjectId, List[torch.Tensor]] = None
    masks: Dict[ObjectId, List[torch.Tensor]] = None
    cls_desc: Dict[ObjectId, torch.Tensor] = None
    patch_desc: Dict[ObjectId, torch.Tensor] = None
    template_thresholds: Dict[ObjectId, torch.Tensor] = None
    whitening_mean: Optional[torch.Tensor] = None
    whitening_W: Optional[torch.Tensor] = None
    sigma_inv: Optional[torch.Tensor] = None
    class_means: Optional[Dict[ObjectId, torch.Tensor]] = None
    maha_thresh_per_class: Optional[torch.Tensor] = None
    maha_thresh_global: Optional[torch.Tensor] = None
    template_csls_avg: Optional[Dict[ObjectId, torch.Tensor]] = None
    orig_onboarding_images: int = None
    orig_pbr_images: int = None
    orig_onboarding_sam_detections: int = None