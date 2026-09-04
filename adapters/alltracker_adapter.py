"""Adapter for AllTracker dense point tracking.

Sole location importing AllTracker internals. AllTracker (Harley et al.,
ICCV 2025) estimates dense flow fields from a query frame to every other
frame in a window — i.e. it tracks ALL pixels, not sparse queries. We run it
on the chunk (query frame first, matching this provider's chunk layout) and
bilinearly sample the resulting trajectory + visibility/confidence maps at
the query points, conforming to the same track() contract as CoTrackerAdapter.

Import note: the repo uses top-level `nets` and `utils` packages. GloPose's
`utils` is a regular package (has __init__.py), which would shadow the repo's
namespace-package `utils` — so we extend GloPose utils.__path__ with the
repo's utils directory instead of relying on sys.path order.
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ALLTRACKER_REPO = Path(__file__).resolve().parent.parent / 'repositories' / 'alltracker'
# HF release checkpoint, mirrored locally (compute nodes run HF_HUB_OFFLINE=1).
DEFAULT_WEIGHTS = os.environ.get('ALLTRACKER_WEIGHTS', 'weights/alltracker.pth')


def _ensure_alltracker_on_path():
    repo = str(ALLTRACKER_REPO)
    if repo not in sys.path:
        sys.path.append(repo)
    import utils as glopose_utils
    repo_utils = str(ALLTRACKER_REPO / 'utils')
    if repo_utils not in glopose_utils.__path__:
        glopose_utils.__path__.append(repo_utils)


class AllTrackerAdapter:
    # Flow iterations at inference (repo demo default).
    ITERS = 8

    def __init__(self, device: str, weights_path: str | None = None):
        _ensure_alltracker_on_path()
        from nets.alltracker import Net

        self.device = device
        self.model = Net(seqlen=16)
        sd = torch.load(weights_path or DEFAULT_WEIGHTS, map_location='cpu')
        self.model.load_state_dict(sd['model'] if 'model' in sd else sd, strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(device)

    @torch.no_grad()
    def track(self, video: torch.Tensor, queries_xy: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        """Track query points through a video chunk (same contract as CoTrackerAdapter).

        Args:
            video: (T, 3, H, W) float tensor in [0, 1]; frame 0 is the query frame.
            queries_xy: (N, 2) float tensor of (x, y) query positions in frame 0.

        Returns:
            tracks: (T, N, 2) tracked (x, y) positions per frame, in input pixels.
            certainty: (T, N) float in [0, 1] — visibility * confidence of the
                dense maps sampled at the query points.
        """
        torch.cuda.empty_cache()
        T, _, H, W = video.shape
        rgbs = (video * 255.0).float().to(self.device)[None]  # (1, T, 3, H, W)

        flows, visconfs, _, _ = self.model.forward_sliding(
            rgbs, iters=self.ITERS, sw=None, is_training=False)
        if flows.dim() == 4:
            # T==2 path of forward_sliding returns a single query->frame1 map
            # (B, 2, H, W) without the time dim; rebuild the (B, T, 2, H, W)
            # layout with an identity (zero-flow, fully visible) frame-0 slot.
            flows = torch.stack([torch.zeros_like(flows), flows], dim=1)
            visconfs = torch.stack([torch.ones_like(visconfs), visconfs], dim=1)
        flows = flows.to(self.device).float()        # (1, T, 2, H, W) query->frame flow
        visconfs = visconfs.to(self.device).float()  # (1, T, 2, H, W) [vis, conf]

        q = queries_xy.float().to(self.device)
        grid = torch.stack([q[:, 0] * 2 / (W - 1) - 1, q[:, 1] * 2 / (H - 1) - 1], dim=-1)
        grid = grid[None, None].expand(T, 1, -1, -1)  # (T, 1, N, 2)

        flow_q = F.grid_sample(flows[0], grid, mode='bilinear',
                               align_corners=True)[:, :, 0]      # (T, 2, N)
        vc_q = F.grid_sample(visconfs[0], grid, mode='bilinear',
                             align_corners=True)[:, :, 0]        # (T, 2, N)

        tracks = q[None] + flow_q.permute(0, 2, 1)               # (T, N, 2)
        certainty = (vc_q[:, 0] * vc_q[:, 1]).clamp(0, 1)        # (T, N)
        if os.environ.get('GLOPOSE_TRACK_DEBUG'):
            qs = torch.tensor([0.1, 0.5, 0.9], device=self.device)
            print(f'[AllTracker-debug] endpoint vis q10/50/90: '
                  f'{[round(x, 3) for x in torch.quantile(vc_q[-1, 0], qs).tolist()]} '
                  f'| conf q10/50/90: {[round(x, 3) for x in torch.quantile(vc_q[-1, 1], qs).tolist()]} '
                  f'| drift p50: {(tracks[-1] - tracks[0]).norm(dim=1).median().item():.1f}px',
                  flush=True)
        torch.cuda.empty_cache()
        return tracks, certainty
