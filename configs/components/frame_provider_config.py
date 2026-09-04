from dataclasses import dataclass


@dataclass
class BaseFrameProviderConfig:

    erode_segmentation: bool = False
    erode_segmentation_iters: int = 2
    background_color: str = 'original'  # 'original' | 'black' | 'white' | 'gray' (neutral 0.5)

    # Synthetic occlusion (HO3D-failure diagnostic): cut a circular patch out of the object
    # segmentation each frame, sized to remove ~this fraction of the mask area. Mimics a hand
    # occluding the object: with background_color='black' the patch turns black exactly like a
    # masked-out hand. 0.0 disables.
    synthetic_occlusion_fraction: float = 0.0
    # Moving occluder (default): patch center drifts smoothly across the object bbox with
    # ``synthetic_occlusion_period`` (frames per sweep) — the occluded surface CHANGES between
    # keyframes, as with a manipulating hand. Static: patch stays at one seeded bbox-relative
    # position — a (near-)consistent missing region, the control condition.
    synthetic_occlusion_static: bool = False
    synthetic_occlusion_period: float = 50.0
    synthetic_occlusion_seed: int = 0
    # Ex-post variant: the matcher sees the UN-occluded image and mask; the occluded
    # mask is written alongside and used to discard matches whose endpoint falls in
    # the occluder (per image, so for every pair that image is in) before
    # reconstruction. Separates "the occluder removed observations" from "the black
    # patch confused the matcher".
    synthetic_occlusion_expost: bool = False

    def __post_init__(self):
        self.config_name = self.__class__.__name__
