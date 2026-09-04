from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class BaseUFMConfig:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.config_name: str = self.__class__.__name__

        self.use_custom_weights: bool = False
        self.custom_weights_path: str | None = None
        self.backward: bool = True
        # Zoomed matching: crop both images to their segmentation bboxes (+margin) before
        # matching and map points back to full-frame coordinates (see FlowMatchingProvider).
        self.crop_matching: bool = False
        self.crop_matching_margin: float = 1.15
        # Multi-scale: ALSO match the full frames and return the union of both point sets.
        # The full-frame matches anchor the globally-consistent mode when the zoomed view
        # amplifies confidently-wrong symmetric matches (2x matcher cost).
        self.crop_matching_add_fullframe: bool = False
