"""OMa, the full pipeline reported as **Ours** in the paper.

Recipe:
  keyframes       adaptive matchability selection (``dense_matching`` frame filter,
                  fine-tuned UFM, min_certainty 0.975, flow_reliability 0.5)
  correspondences fine-tuned UFM dense matching on zoomed segmentation-bbox crops
  view graph      linear chain over the selected keyframes (k-1 matcher calls)
  SfM             COLMAP incremental, ground-truth intrinsics locked in BA,
                  free initial pair, track merging on
  masking         the matcher sees a black background; only in-mask points are
                  reconstructed
  densification   off
  point filters   none (``min_track_length`` 0)

Run it with any of the dataset entry points, for example::

    python run_HOPE.py --config configs/onboarding/ours.py \\
        --experiment my_first_run --sequences obj_000001_up

The single-factor knockouts of this configuration used for the ablation table live
in ``configs/onboarding/ablations/``.
"""
from configs.glopose_config import GloPoseConfig
from configs.onboarding._anchor import anchor_config


def get_config() -> GloPoseConfig:
    cfg = anchor_config()
    cfg.onboarding.reconstruction_matcher = 'UFM'
    cfg.onboarding.view_graph_strategy = 'linear'
    return cfg
