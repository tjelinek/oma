"""CoWTracker dense point-tracking adapter — the sole location for CoWTracker imports.

CoWTracker (Lai et al., CVPR 2026, "Tracking by Warping instead of Correlation",
github.com/facebookresearch/cowtracker) is a DENSE tracker: one forward pass returns a
track field for every pixel of frame 0 ([B, T, H, W, 2] with visibility and confidence
maps), with no sparse-query interface. This adapter runs the dense forward once per
video chunk and reads the field out at the requested query pixels, so it plugs into
PointTrackingMatchingProvider's sparse track() contract at zero extra cost per query.

Not to be confused with adapters/cotracker_adapter.py (Meta's CoTracker3).

Notes:
    - The model is resolution-hungry (stride-2 features); inference runs at the demo
      resolution 336x560 and coordinates are mapped back to the input frame linearly.
    - Returned "visibility" is vis*conf, matching the reference demo's gating
      (visconf > 0.1); set BaseTrackingConfig.visibility_threshold accordingly.
    - Importing cowtracker pulls in its VENDORED vggt (cowtracker.thirdparty sets up
      the path), which may shadow repositories/vggt in this process. Do not combine
      with the VGGT TrackHead tracker in one run.
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

DEFAULT_WEIGHTS = os.environ.get('COWTRACKER_WEIGHTS', 'weights/cowtracker_model.pth')
INFERENCE_HW = (336, 560)  # demo resolution; H, W


class CoWTrackerAdapter:

    def __init__(self, device: str, custom_weights_path: str | None = None):
        repo = Path.home() / 'repositories' / 'cowtracker'
        if repo.exists() and str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from cowtracker.models.cowtracker import CoWTracker
        self.device = device
        self.model = CoWTracker.from_checkpoint(
            checkpoint_path=custom_weights_path or DEFAULT_WEIGHTS,
            device=device, dtype=torch.float16)

    @torch.no_grad()
    def track(self, video: torch.Tensor, queries_xy: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        """Track query points through a video chunk via the dense track field.

        Args:
            video: (T, 3, H, W) float tensor in [0, 1]; frame 0 is the query frame.
            queries_xy: (N, 2) float tensor of (x, y) query positions in frame 0.

        Returns:
            tracks: (T, N, 2) tracked (x, y) positions per frame, input pixel coords.
            visibility: (T, N) float vis*conf in [0, 1].
        """
        T, _, H, W = video.shape
        h, w = INFERENCE_HW
        vid = F.interpolate(video, size=(h, w), mode='bilinear', align_corners=False)
        vid = (vid * 255.0).to(self.device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            pred = self.model.forward(video=vid, queries=None)
        track = pred['track'][0].float()      # (T, h, w, 2) in inference pixel coords
        visconf = (pred['vis'][0] * pred['conf'][0]).float()  # (T, h, w)

        sx, sy = w / W, h / H
        qx = (queries_xy[:, 0].to(self.device) * sx).round().long().clamp(0, w - 1)
        qy = (queries_xy[:, 1].to(self.device) * sy).round().long().clamp(0, h - 1)

        tracks_q = track[:, qy, qx, :].clone()           # (T, N, 2)
        tracks_q[..., 0] /= sx
        tracks_q[..., 1] /= sy
        visibility_q = visconf[:, qy, qx]                # (T, N)
        return tracks_q, visibility_q
