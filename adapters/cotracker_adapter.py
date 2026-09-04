"""CoTracker point-tracking adapter — the sole location for CoTracker imports/loading.

Wraps the torch.hub CoTracker3 predictor behind a minimal track() interface used by
PointTrackingMatchingProvider. The predictor resizes the video to its internal
resolution and returns tracks in the input pixel coordinate frame, so callers never
deal with the model resolution.
"""
import torch


class CoTrackerAdapter:

    def __init__(self, device: str, variant: str = 'cotracker3_offline',
                 custom_weights_path: str | None = None):
        self.device = device
        self.model = torch.hub.load('facebookresearch/co-tracker', variant)
        if custom_weights_path is not None:
            state_dict = torch.load(custom_weights_path, map_location=device, weights_only=False)
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            self.model.load_state_dict(state_dict)
            print(f"[CoTracker] loaded custom weights from {custom_weights_path}")
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def track(self, video: torch.Tensor, queries_xy: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        """Track query points through a video chunk.

        Args:
            video: (T, 3, H, W) float tensor in [0, 1]; frame 0 is the query frame.
            queries_xy: (N, 2) float tensor of (x, y) query positions in frame 0.

        Returns:
            tracks: (T, N, 2) tracked (x, y) positions per frame.
            visibility: (T, N) float visibility in [0, 1].
        """
        video_batch = video[None].to(self.device) * 255.0
        queries = torch.cat([torch.zeros_like(queries_xy[:, :1]), queries_xy],
                            dim=1)[None].to(self.device).float()
        tracks, visibility = self.model(video_batch, queries=queries)
        return tracks[0], visibility[0].float()
