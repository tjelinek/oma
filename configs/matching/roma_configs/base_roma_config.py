from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class BaseRomaConfig:

    def __init__(self, **kwargs):
        # Defaults first, so an explicit kwarg (e.g. BaseRomaConfig(custom_weights_path=...))
        # is not clobbered by the default assignments below.
        self.config_name: str = self.__class__.__name__
        self.use_custom_weights: bool = False
        # Optional explicit path to a fine-tuned RoMa checkpoint ({"model": ...} or bare
        # state_dict). When use_custom_weights=True and this is None, falls back to the
        # ROMA_WEIGHTS environment variable (see data_providers/flow_provider.py).
        self.custom_weights_path: str | None = None

        for key, value in kwargs.items():
            setattr(self, key, value)
