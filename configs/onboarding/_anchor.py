"""Shared anchor configuration for OMa and its ablation arms.

``configs/onboarding/ours.py`` and every arm in ``configs/onboarding/ablations/``
call ``anchor_config()`` and then change EXACTLY ONE field. Writing the arms as
full standalone configs was the alternative, but with ten arms that silently
invites a second unintended difference (a stale checkpoint path, a forgotten
``crop_matching``) and then the ablation table no longer isolates the factor it
claims to. Keeping the delta to one assignment makes the claim checkable by
reading the arm file alone.

Anchor:
  keyframes      adaptive, masked fine-tuned-UFM matchability filter
  matcher        UFM fine-tuned on background-zeroed synthetic render pairs
  reconstruction point tracking (``ours.py`` overrides this to UFM dense matching)
  init           free COLMAP initial-pair selection
  view graph     dense, all-to-all over keyframes (``ours.py`` overrides to linear)
  background     black, the matcher sees no background
  densification  off

Background note: ``reconstruction_use_background_points`` stays False in EVERY arm,
including the with-background arm. The background axis is only about what the
matcher is shown; background matches are never reconstructed either way.

The fine-tuned UFM checkpoint is not bundled with the code. Point ``OMA_UFM_WEIGHTS``
at your downloaded copy, or edit ``FT_UFM_WEIGHTS`` below.
"""
import os

from configs.glopose_config import GloPoseConfig

# Mask-robust UFM fine-tune (see training/ufm/ to reproduce it, and the README for
# the download link). Trained on an object-disjoint pool of BOP-PBR scenes so that
# the evaluation objects stay unseen.
FT_UFM_WEIGHTS = os.environ.get('OMA_UFM_WEIGHTS', 'weights/ufm_ft_last.pth')


def anchor_config() -> GloPoseConfig:
    cfg = GloPoseConfig()

    cfg.onboarding.allow_disk_cache = False
    cfg.onboarding.use_default_colmap_K = False  # GT intrinsics, locked in BA

    # --- keyframe selection ---
    cfg.onboarding.frame_filter = 'dense_matching'
    cfg.onboarding.filter_matcher = 'UFM'
    cfg.onboarding.min_certainty_threshold = 0.975
    cfg.onboarding.flow_reliability_threshold = 0.5

    # --- background (matcher visibility only) ---
    cfg.input.frame_provider_config.background_color = 'black'
    cfg.onboarding.reconstruction_use_background_points = False

    # --- matcher weights ---
    cfg.onboarding.ufm.use_custom_weights = True
    cfg.onboarding.ufm.custom_weights_path = FT_UFM_WEIGHTS
    cfg.onboarding.ufm.crop_matching = True
    cfg.onboarding.crop_matching_reconstruction_only = True
    cfg.onboarding.ufm.crop_matching_margin = 1.15

    # --- reconstruction backend ---
    cfg.onboarding.reconstruction_matcher = 'CoTracker'
    cfg.onboarding.tracking.tracker = 'vggt'
    cfg.onboarding.tracking.max_queries = 2048
    # The 1B aggregator attends globally over T x 1369 patch tokens; cap chunks well
    # below the CoTracker setting so it fits beside the resident UFM filter model.
    cfg.onboarding.tracking.max_video_len = 24
    cfg.onboarding.tracking.crop_to_object = True
    cfg.onboarding.tracking.track_on_original_frames = True

    # --- COLMAP initial pair ---
    cfg.onboarding.init_with_first_two_images = False

    # --- view graph ---
    cfg.onboarding.view_graph_strategy = 'dense'

    # --- densification ---
    cfg.onboarding.densify_reconstruction = False

    return cfg
