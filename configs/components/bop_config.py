from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class BaseBOPConfig:
    scene_id: int = None
    tracked_obj_ids: List = field(default_factory=list)
    frame_id_to_im_id: Dict = field(default_factory=dict)
    onboarding_type: str = None  # Either 'static' or 'dynamic'
    static_onboarding_sequence = None  # Either 'up', 'down', or 'both'

    def __post_init__(self):
        self.config_name = self.__class__.__name__
