"""Vendored utility classes from cnos for DINOv2/v3 descriptor extraction.

CropResizePad: from repositories/cnos/src/utils/bbox_utils.py
BatchedData: from repositories/cnos/src/model/utils.py
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F


class CropResizePad:
    """Crops image proposals to bbox, scales to target size, pads to square."""

    def __init__(self, target_size, pad_value=0.0):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.target_ratio = self.target_size[1] / self.target_size[0]
        self.target_h, self.target_w = target_size
        self.target_max = max(self.target_h, self.target_w)
        self.pad_value = pad_value

    def __call__(self, images, boxes):
        box_sizes = boxes[:, 2:] - boxes[:, :2]
        scale_factor = self.target_max / torch.max(box_sizes, dim=-1)[0]
        processed_images = []
        for image, box, scale in zip(images, boxes, scale_factor):
            # crop and scale
            box = box.int()
            image = image[:, box[1]: box[3], box[0]: box[2]]
            image = F.interpolate(image.unsqueeze(0), scale_factor=scale.item())[0]
            # pad and resize
            original_h, original_w = image.shape[1:]
            original_ratio = original_w / original_h

            # check if the original and final aspect ratios are the same within a margin
            if self.target_ratio != original_ratio:
                padding_top = max((self.target_h - original_h) // 2, 0)
                padding_bottom = self.target_h - original_h - padding_top
                padding_left = max((self.target_w - original_w) // 2, 0)
                padding_right = self.target_w - original_w - padding_left
                image = F.pad(
                    image, (padding_left, padding_right, padding_top, padding_bottom), value=self.pad_value
                )
            assert image.shape[1] == image.shape[2], logging.info(
                f"image {image.shape} is not square after padding"
            )
            image = F.interpolate(
                image.unsqueeze(0), scale_factor=self.target_h / image.shape[1]
            )[0]
            processed_images.append(image)
        return torch.stack(processed_images)


class BatchedData:
    """Simple batch chunker for processing large batches."""

    def __init__(self, batch_size, data=None, **kwargs) -> None:
        self.batch_size = batch_size
        if data is not None:
            self.data = data
        else:
            self.data = []

    def __len__(self):
        assert self.batch_size is not None, "batch_size is not defined"
        return np.ceil(len(self.data) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        assert self.batch_size is not None, "batch_size is not defined"
        return self.data[idx * self.batch_size: (idx + 1) * self.batch_size]

    def cat(self, data, dim=0):
        if len(self.data) == 0:
            self.data = data
        else:
            self.data = torch.cat([self.data, data], dim=dim)

    def append(self, data):
        self.data.append(data)

    def stack(self, dim=0):
        self.data = torch.stack(self.data, dim=dim)
