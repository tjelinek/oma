"""Ablation arm 'sift': SIFT + LightGlue sparse correspondences (kornia
LightGlueMatcher('sift')) instead of fine-tuned UFM dense matching, everything else as
the anchor (adaptive matchability keyframes, linear graph, masked points, COLMAP).
"""
from configs.glopose_config import GloPoseConfig
from configs.onboarding.ours import get_config as anchor


def get_config() -> GloPoseConfig:
    cfg = anchor()
    cfg.onboarding.reconstruction_matcher = 'SIFT'
    return cfg
