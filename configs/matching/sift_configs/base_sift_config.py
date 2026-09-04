from dataclasses import dataclass
from typing import Tuple



@dataclass
class BaseSiftConfig:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.config_name: str = self.__class__.__name__
        self.resize_to: Tuple[int, int] = (800, 600)
        self.sift_filter_num_feats: int = 8192
        self.sift_filter_sift_matcher: str = 'adalam'
