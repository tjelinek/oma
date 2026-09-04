"""VGGT TrackHead point-tracking adapter.

Exposes the VGGT-1B tracking head (VGGSfM-style iterative tracker trained inside VGGT)
behind the same track() interface as CoTrackerAdapter, so
PointTrackingMatchingProvider can switch trackers via BaseTrackingConfig.tracker.
Model loading is delegated to adapters.vggt_adapter.load_vggt_model (the sole VGGT
import location); the unused camera/depth/point heads are dropped after loading so a
tracking forward pays only for the aggregator + track head.
"""
import os

import torch
import torch.nn.functional as F


class VGGTTrackAdapter:
    # VGGT patch size is 14; 518 = 37 * 14 is the model's native inference resolution.
    RESOLUTION = 518
    # The track head builds per-query correlation volumes — chunk queries so their
    # memory stays bounded (2048 queries at once allocate >15 GB on T=10 chunks).
    QUERY_CHUNK = 512

    def __init__(self, device: str, custom_weights_path: str | None = None):
        from adapters.vggt_adapter import load_vggt_model
        self.device = device
        self.model = load_vggt_model(device, custom_weights_path)
        for head in ('camera_head', 'point_head', 'depth_head'):
            setattr(self.model, head, None)

    @torch.no_grad()
    def track(self, video: torch.Tensor, queries_xy: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        """Track query points through a video chunk (same contract as CoTrackerAdapter).

        Args:
            video: (T, 3, H, W) float tensor in [0, 1]; frame 0 is the query frame.
            queries_xy: (N, 2) float tensor of (x, y) query positions in frame 0.

        Returns:
            tracks: (T, N, 2) tracked (x, y) positions per frame, in input pixels.
            certainty: (T, N) float in [0, 1] — geometric mean of the head's sigmoided
                visibility and confidence (their product is systematically below
                CoTracker-calibrated thresholds).
        """
        torch.cuda.empty_cache()
        _, _, h, w = video.shape
        r = self.RESOLUTION
        frames = F.interpolate(video.to(self.device), size=(r, r), mode='bilinear',
                               align_corners=False)
        scale = torch.tensor([r / w, r / h], device=self.device)
        queries = queries_xy.float().to(self.device) * scale
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            aggregated_tokens_list, patch_start_idx = self.model.aggregator(frames[None])
        track_parts, cert_parts = [], []
        vis_parts, conf_parts = [], []
        for q0 in range(0, len(queries), self.QUERY_CHUNK):
            query_chunk = queries[q0:q0 + self.QUERY_CHUNK]
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                track_list, vis, conf = self.model.track_head(
                    aggregated_tokens_list, images=frames[None],
                    patch_start_idx=patch_start_idx, query_points=query_chunk[None])
            track_parts.append(track_list[-1][0].float())
            # Gate on visibility alone (the convention of VGGT's own demos); the conf
            # head is systematically pessimistic on out-of-domain footage and would
            # starve every edge if multiplied in.
            cert_parts.append(vis[0].float())
            if os.environ.get('GLOPOSE_TRACK_DEBUG'):
                vis_parts.append(vis[0, -1].float())
                conf_parts.append(conf[0, -1].float())
        del aggregated_tokens_list
        torch.cuda.empty_cache()
        tracks = torch.cat(track_parts, dim=1) / scale
        certainty = torch.cat(cert_parts, dim=1)
        if os.environ.get('GLOPOSE_TRACK_DEBUG'):
            q = torch.tensor([0.1, 0.5, 0.9], device=self.device)
            v = torch.cat(vis_parts)
            c = torch.cat(conf_parts)
            print(f'[VGGT-track-debug] endpoint vis q10/50/90: '
                  f'{[round(x, 3) for x in torch.quantile(v, q).tolist()]} '
                  f'| conf q10/50/90: {[round(x, 3) for x in torch.quantile(c, q).tolist()]} '
                  f'| drift p50: {(tracks[-1] - tracks[0]).norm(dim=1).median().item():.1f}px',
                  flush=True)
        return tracks, certainty
