"""Adapter for the SAM2 external repository (video predictor).

This is the SOLE location in GloPose that imports sam2 internals and handles
the Hydra config initialization required by SAM2. All other modules import
from here.

Only wraps the video predictor (for mask propagation / tracking). The CNOS
automatic mask generator (repositories/cnos/src/model/sam2.py) is a separate
concern behind the cnos adapter boundary.
"""

import os
from pathlib import Path

import numpy as np
import torch

# Download sam2.1_hiera_large.pt from https://github.com/facebookresearch/sam2 and
# either drop it at weights/sam2.1_hiera_large.pt or set SAM2_CHECKPOINT.
_DEFAULT_CHECKPOINT = Path(os.environ.get("SAM2_CHECKPOINT", "weights/sam2.1_hiera_large.pt"))
_DEFAULT_MODEL_CFG = "sam2.1/sam2.1_hiera_l.yaml"


def build_video_predictor(
    checkpoint: Path = _DEFAULT_CHECKPOINT,
    model_cfg: str = _DEFAULT_MODEL_CFG,
    device: str = "cuda",
):
    """Build a SAM2 video predictor with Hydra config initialization.

    Handles clearing any existing Hydra GlobalHydra state and initializing
    the SAM2 config directory, which SAM2's build function requires.

    Returns:
        A SAM2VideoPredictor instance.
    """
    from sam2.build_sam import build_sam2_video_predictor
    import sam2
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

    print(f"[SAM2 adapter] Building video predictor from {checkpoint.name} on {device}")

    cfg_dir = Path(sam2.__file__).parent / "configs"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(cfg_dir.resolve()), version_base=None, job_name="sam2"):
        predictor = build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)

    print(f"[SAM2 adapter] Video predictor built successfully")
    return predictor


def image_to_sam(image: torch.Tensor) -> np.ndarray:
    """Convert a [1, 3, H, W] or [3, H, W] image tensor to SAM2's HxWx3 numpy array."""
    return image.squeeze().permute(1, 2, 0).numpy(force=True)


def mask_to_sam_prompt(mask: torch.Tensor) -> np.ndarray:
    """Convert a segmentation mask tensor to SAM2's 2D boolean numpy array."""
    return mask.squeeze().to(torch.bool).numpy(force=True)
