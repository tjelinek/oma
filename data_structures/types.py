from dataclasses import dataclass
from typing import TypeAlias

import torch
from kornia.geometry import Se3

ObjectId: TypeAlias = int | str


@dataclass
class Detection:
    """A single detected object instance in an image.

    Output of the detection module (B2), input to pose estimation (C).
    """
    object_id: ObjectId
    score: float
    bbox_xywh: list[float]                    # COCO format [x, y, w, h] in pixels
    mask: torch.Tensor | None = None          # (H, W) binary mask
    matched_template_idx: int | None = None   # index into TemplateBank for the matched object


@dataclass
class PoseEstimate:
    """A 6DoF pose estimate for a detected object.

    Output of the pose estimation module (C).
    Se3_obj2cam is the object/model-to-camera transform, equivalent to
    BOP's cam_R_m2c / cam_t_m2c convention.
    """
    object_id: ObjectId
    score: float
    Se3_obj2cam: Se3

    @property
    def R(self) -> torch.Tensor:
        """(3, 3) rotation matrix, object/model-to-camera."""
        return self.Se3_obj2cam.matrix().squeeze()[:3, :3]

    @property
    def t_meters(self) -> torch.Tensor:
        """(3,) translation vector in meters."""
        return self.Se3_obj2cam.translation.squeeze()


@dataclass
class DetectionSet:
    """Batched detection results for an image."""
    masks: torch.Tensor                             # (N, H, W)
    scores: torch.Tensor                            # (N,)
    object_ids: torch.Tensor                        # (N,)
    boxes: torch.Tensor                             # (N, 4) xyxy, long
    score_distribution: torch.Tensor | None = None  # (N, num_objects)

    def __post_init__(self):
        self.boxes = self.boxes.long()

    def __len__(self) -> int:
        return self.masks.shape[0]

    def filter(self, indices: torch.Tensor) -> None:
        self.masks = self.masks[indices]
        self.scores = self.scores[indices]
        self.object_ids = self.object_ids[indices]
        self.boxes = self.boxes[indices]
        if self.score_distribution is not None:
            self.score_distribution = self.score_distribution[indices]
