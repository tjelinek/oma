"""Vendored DINOv2/v3 descriptor model from cnos.

CustomDINOv2: from repositories/cnos/src/model/dinov2.py
Replaces pl.LightningModule with torch.nn.Module.
Replaces Hydra-based loading with descriptor_from_config().

descriptor_size dict: from repositories/cnos/src/model/dinov2.py
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.ops import masks_to_boxes

from adapters.dino_utils import CropResizePad, BatchedData

descriptor_size = {
    # DINOv2 models
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
    # DINOv3 ViT models (web images - LVD-1689M)
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
    "dinov3_vith16plus": 1280,
    "dinov3_vit7b16": 1536,
    # DINOv3 ConvNeXt models
    "dinov3_convnext_tiny": 768,
    "dinov3_convnext_small": 768,
    "dinov3_convnext_base": 1024,
    "dinov3_convnext_large": 1536,
}


@dataclass
class DescriptorModelConfig:
    """Configuration for DINOv2/v3 descriptor model, replacing Hydra YAML."""
    model_name: str = 'dinov2_vitl14'
    model_type: str = 'dinov2'
    token_name: str = 'x_norm_clstoken'
    image_size: int = 224
    chunk_size: int = 16
    descriptor_width_size: int = 640
    apply_image_mask: bool = True
    normalization: dict | None = None  # None -> ImageNet defaults
    weights: str | None = None  # local .pth path for DINOv3
    repo_or_dir: str = 'facebookresearch/dinov2'
    source: str = 'github'  # 'github' or 'local'


# Preset configs matching the cnos YAML files
DINOV2_CONFIG = DescriptorModelConfig(
    model_name='dinov2_vitl14',
    model_type='dinov2',
    repo_or_dir='facebookresearch/dinov2',
    source='github',
)

DINOV3_CONFIG = DescriptorModelConfig(
    model_name='dinov3_vitl16',
    model_type='dinov3',
    # Local clone of https://github.com/facebookresearch/dinov3 (torch.hub source='local').
    repo_or_dir=os.environ.get('DINOV3_REPO', 'repositories/dinov3'),
    source='local',
    weights=os.environ.get('DINOV3_WEIGHTS', 'weights/dinov3_vitl16.pth'),
    normalization={
        'mean': [0.430, 0.411, 0.296],  # SAT-493M (satellite images)
        'std': [0.213, 0.156, 0.143],
    },
)


class CustomDINOv2(nn.Module):
    """DINOv2/v3 descriptor extraction model.

    Vendored from cnos CustomDINOv2 (pl.LightningModule -> nn.Module).
    Accepts any object with .masks and .boxes attributes for proposals (duck typing).
    """

    def __init__(
            self,
            model_name,
            model,
            token_name,
            image_size,
            chunk_size,
            descriptor_width_size,
            patch_size=14,
            model_type="dinov2",
            normalization=None,
            apply_image_mask=True,
    ):
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.token_name = token_name
        self.chunk_size = chunk_size
        self.model_type = model_type
        self.apply_image_mask: bool = apply_image_mask

        # Determine patch size based on model type
        if model_type == "dinov3":
            self.patch_size = 16 if "16" in model_name else 14
        else:
            self.patch_size = patch_size

        self.proposal_size = image_size
        self.descriptor_width_size = descriptor_width_size

        logging.info(f"Init CustomDINOv2 wrapper for {model_type} model: {model_name}")

        # Setup normalization parameters
        if normalization is None:
            norm_mean = (0.485, 0.456, 0.406)
            norm_std = (0.229, 0.224, 0.225)
        else:
            norm_mean = tuple(normalization.get("mean", [0.485, 0.456, 0.406]))
            norm_std = tuple(normalization.get("std", [0.229, 0.224, 0.225]))

        self.rgb_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=norm_mean, std=norm_std),
            ]
        )

        # use for global feature
        self.rgb_proposal_processor = CropResizePad(self.proposal_size)

        logging.info(
            f"Init {model_type} wrapper with full size={descriptor_width_size}, "
            f"proposal size={self.proposal_size}, patch size={self.patch_size} done!"
        )

    def process_rgb_proposals(self, image_np, masks, boxes):
        """Normalize image, mask and crop each proposal, resize to target size."""
        num_proposals = len(masks)
        rgb = self.rgb_normalize(image_np).to(masks.device).float()
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1)
        if self.apply_image_mask:
            masked_rgbs = rgbs * masks.unsqueeze(1)
        else:
            masked_rgbs = rgbs
        processed_masked_rgbs = self.rgb_proposal_processor(
            masked_rgbs, boxes
        )  # [N, 3, target_size, target_size]
        return processed_masked_rgbs

    @torch.no_grad()
    def compute_features(self, images, token_name):
        if token_name == "x_norm_clstoken":
            if images.shape[0] > self.chunk_size:
                cls_features, spatial_patch_features = self.forward_by_chunk(images)
            else:
                model_output = self.model.forward_features(images)
                cls_features = model_output['x_norm_clstoken']
                patch_features = model_output['x_norm_patchtokens']

                batch_size, num_patches, hidden_dim = patch_features.shape
                grid_size = int(num_patches ** 0.5)
                spatial_patch_features = patch_features.reshape(batch_size, grid_size, grid_size, hidden_dim)
        else:
            raise NotImplementedError
        return cls_features, spatial_patch_features

    @torch.no_grad()
    def forward_by_chunk(self, processed_rgbs):
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        del processed_rgbs  # free memory
        cls_features = BatchedData(batch_size=self.chunk_size)
        spatial_patch_features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            cls_feats, spatial_patch_feats = self.compute_features(batch_rgbs[idx_batch], token_name="x_norm_clstoken")
            cls_features.cat(cls_feats)
            spatial_patch_features.cat(spatial_patch_feats)

        return cls_features.data, spatial_patch_features.data

    @torch.no_grad()
    def forward_cls_token(self, image_np, proposals):
        processed_rgbs = self.process_rgb_proposals(
            image_np, proposals.masks, proposals.boxes
        )
        return self.forward_by_chunk(processed_rgbs)

    @torch.no_grad()
    def forward(self, image_np, proposals):
        return self.forward_cls_token(image_np, proposals)

    @property
    def device(self):
        """Return device of the underlying model parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            # Fallback: check the wrapped model
            return next(self.model.parameters()).device

    def get_detections_from_files(self, image_path: Path, segmentation_path: Path):
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).to(self.device)
        segmentation = Image.open(segmentation_path).convert('L')
        segmentation_np = np.array(segmentation)
        segmentation_mask = torch.from_numpy(segmentation_np).unsqueeze(0).to(self.device)
        segmentation_mask = segmentation_mask.to(torch.float32).clamp(0, 1)  # From 0-255 to binary
        segmentation_bbox = masks_to_boxes(segmentation_mask)
        image_np = image_tensor.to(torch.uint8).numpy(force=True)

        # Duck-typed proposals object with .masks and .boxes
        class _Proposals:
            def __init__(self, masks, boxes):
                self.masks = masks
                self.boxes = boxes

        proposals = _Proposals(segmentation_mask, segmentation_bbox)
        dino_cls_descriptor, dino_dense_descriptor = self.forward(image_np, proposals)

        return dino_cls_descriptor, dino_dense_descriptor


def descriptor_from_config(config: DescriptorModelConfig | None = None,
                           model: str = 'dinov2',
                           mask_detections: bool = True,
                           device: str = 'cuda') -> CustomDINOv2:
    """Load a DINOv2/v3 descriptor model from config, replacing Hydra-based loading.

    Args:
        config: Explicit config. If None, selects preset based on `model` arg.
        model: 'dinov2' or 'dinov3' — used to select preset when config is None.
        mask_detections: Whether to apply image mask to proposals.
        device: Target device.

    Returns:
        CustomDINOv2 model ready for inference.
    """
    if config is None:
        if model == 'dinov3':
            config = DINOV3_CONFIG
        else:
            config = DINOV2_CONFIG

    # Load the backbone via torch.hub
    hub_kwargs = {
        'repo_or_dir': config.repo_or_dir,
        'model': config.model_name,
    }
    if config.source == 'local':
        hub_kwargs['source'] = 'local'
        hub_kwargs['pretrained'] = False
    backbone = torch.hub.load(**hub_kwargs)

    # Load local weights if specified (DINOv3)
    if config.weights:
        logging.info(f"Loading weights from {config.weights}")
        state_dict = torch.load(config.weights, map_location=device)
        backbone.load_state_dict(state_dict, strict=True)

    dino_model = CustomDINOv2(
        model_name=config.model_name,
        model=backbone,
        token_name=config.token_name,
        image_size=config.image_size,
        chunk_size=config.chunk_size,
        descriptor_width_size=config.descriptor_width_size,
        model_type=config.model_type,
        normalization=config.normalization,
        apply_image_mask=mask_detections,
    )
    dino_model = dino_model.to(device)
    dino_model.model.device = device

    return dino_model
