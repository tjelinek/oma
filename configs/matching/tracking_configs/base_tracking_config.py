from dataclasses import dataclass


@dataclass
class BaseTrackingConfig:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.config_name: str = self.__class__.__name__

        # 'cotracker3_offline' / 'cotracker3_online' (torch.hub, facebookresearch/
        # co-tracker) or 'vggt' (VGGT-1B TrackHead, adapters/vggt_track_adapter.py).
        self.tracker: str = 'cotracker3_offline'
        # Optional fine-tuned checkpoint overlaid on the hub weights.
        self.custom_weights_path: str | None = None
        # Maximum number of simultaneously tracked query points per keyframe.
        self.max_queries: int = 2048
        # Temporal cap on a tracked chunk; longer keyframe gaps are subsampled evenly
        # (both keyframe endpoints are always kept).
        self.max_video_len: int = 64
        # A track becomes a correspondence iff the tracker marks it visible at the
        # target keyframe with at least this confidence.
        self.visibility_threshold: float = 0.5
        # Erosion (iterations) of the source segmentation before query sampling, so
        # queries avoid mask-boundary pixels (mirrors the matcher pipeline's erosion).
        self.erode_query_mask_iters: int = 2
        # Re-seed the next chunk's queries from surviving track endpoints so COLMAP
        # track merging chains per-edge correspondences into multi-keyframe tracks.
        self.reseed_from_tracks: bool = True
        # Share of the query budget that reseeded endpoints may occupy; the rest is
        # always filled with fresh samples from the (eroded) mask, so surfaces that
        # become visible only later in the sequence still receive queries.
        self.max_seed_fraction: float = 0.5
        # Crop each chunk to the union of its segmentation bboxes (+margin) before
        # tracking (pure pixel shift, mapped back afterwards). CoTracker works at
        # ~512x384 internally, so cropping recovers effective resolution for masked
        # objects that are small in the frame — the tracking analogue of crop_matching.
        self.crop_to_object: bool = True
        self.crop_margin: float = 1.15
        # Track on the ORIGINAL (unmasked) frames instead of the pipeline's
        # background-masked observations; queries stay seeded on the object mask.
        # Point trackers are trained on natural video and handle dynamic scenes, so
        # they need neither the black background nor its artificial mask boundary,
        # and occlusion re-detection can work through the real occluder (hand).
        self.track_on_original_frames: bool = False
        # Seed queries uniformly over the FULL frame instead of inside the object
        # mask — the "w/o mask" control arm, where background features participate
        # in tracking and reconstruction. Combine with crop_to_object=False and
        # reconstruction_use_background_points=True.
        self.query_full_frame: bool = False
