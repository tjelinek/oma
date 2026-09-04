"""Ablation arm 'every8': fixed every-8th keyframes instead of adaptive matchability selection (GSO stride doubled by the runner).

Anchor = OMa (configs/onboarding/ours.py: fine-tuned UFM dense matching, adaptive
matchability keyframes, linear view graph). Every arm changes exactly one field from
the anchor. The paper reports them over the static validation split defined in
configs/splits.py (HANDAL 8, HOPE 8, NAVI 12, GSO 8 sequences).
"""
from configs.glopose_config import GloPoseConfig
from configs.onboarding.ours import get_config as anchor


def get_config() -> GloPoseConfig:
    cfg = anchor()
    cfg.onboarding.frame_filter = 'passthrough'
    cfg.onboarding.passthrough_skip = 8
    return cfg
