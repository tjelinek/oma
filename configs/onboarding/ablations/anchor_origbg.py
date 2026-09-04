"""Ablation arm 'anchor_origbg': the anchor with the matcher shown the original
background (the 'w/' rows of the comparison tables). Background points are still
filtered before reconstruction, so the output is object-only either way.
"""
from configs.glopose_config import GloPoseConfig
from configs.onboarding.ours import get_config as anchor


def get_config() -> GloPoseConfig:
    cfg = anchor()
    cfg.input.frame_provider_config.background_color = 'original'
    return cfg
