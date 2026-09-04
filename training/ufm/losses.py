"""
Losses and metrics for UFM fine-tuning.

Two supervised signals, matching what the deployed `UFMMatchingProvider` consumes
(`result.flow.flow_output` and `result.covisibility`):

  1. Endpoint error (EPE) on the predicted flow, averaged over covisible object
     pixels. For the Refine model this supervises the regression flow (the
     residual's gradient is cancelled by construction -- see ufm.py:994).
  2. Binary cross-entropy on the covisibility/occlusion head's pre-sigmoid logits.

`UFMLoss` returns the weighted total plus a dict of detached scalar metrics for
logging (epe, covis BCE, covis accuracy/IoU).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def endpoint_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
                   valid: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-pixel L2 endpoint error, returned as (B, H, W). `valid` not applied."""
    return torch.sqrt(((pred_flow - gt_flow) ** 2).sum(dim=1) + eps)


def epe_loss(pred_flow: torch.Tensor, gt_flow: torch.Tensor,
             valid: torch.Tensor, eps: float = 1e-8,
             clamp: float | None = None) -> torch.Tensor:
    """Mean endpoint error over valid pixels (scalar). `valid`: (B, H, W) in {0,1}.

    `clamp`: if set, cap per-pixel EPE at this value before averaging. This robustifies
    the loss against catastrophic-flow pixels/pairs (the 80-140px hard-pair spikes) that
    otherwise dominate the gradient and push the matcher toward overconfident-but-wrong
    correspondences that break downstream SfM.
    """
    epe = endpoint_error(pred_flow, gt_flow, valid, eps)        # (B, H, W)
    if clamp is not None:
        epe = epe.clamp_max(clamp)
    denom = valid.sum().clamp_min(1.0)
    return (epe * valid).sum() / denom


def covisibility_bce(logits: torch.Tensor, target: torch.Tensor,
                     region: torch.Tensor | None = None,
                     pos_weight: float | None = None) -> torch.Tensor:
    """BCE-with-logits for the occlusion/covisibility head.

    logits : (B, 1, H, W) or (B, H, W) pre-sigmoid.
    target : (B, H, W) in {0, 1}.
    region : optional (B, H, W) mask of pixels to supervise (default: all).
    """
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    pw = torch.tensor(pos_weight, device=logits.device) if pos_weight is not None else None
    per_px = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
    if region is None:
        return per_px.mean()
    denom = region.sum().clamp_min(1.0)
    return (per_px * region).sum() / denom


@torch.no_grad()
def covisibility_metrics(logits: torch.Tensor, target: torch.Tensor,
                         thresh: float = 0.5) -> dict[str, float]:
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    pred = (torch.sigmoid(logits) > thresh).float()
    correct = (pred == target).float().mean().item()
    inter = (pred * target).sum().item()
    union = ((pred + target) > 0).float().sum().item()
    iou = inter / union if union > 0 else 0.0
    return {"covis_acc": correct, "covis_iou": iou}


class UFMLoss:
    """Weighted EPE + covisibility BCE."""

    def __init__(self, flow_weight: float = 1.0, covis_weight: float = 1.0,
                 covis_pos_weight: float | None = None, epe_clamp: float | None = None):
        self.flow_weight = flow_weight
        self.covis_weight = covis_weight
        self.covis_pos_weight = covis_pos_weight
        self.epe_clamp = epe_clamp

    def __call__(self, result, batch) -> tuple[torch.Tensor, dict[str, float]]:
        pred_flow = result.flow.flow_output                    # (B, 2, H, W)
        gt_flow = batch["gt_flow"]
        valid = batch["flow_valid"]

        loss_epe = epe_loss(pred_flow, gt_flow, valid, clamp=self.epe_clamp)
        # Always log the unclamped mean EPE so the metric is comparable across runs.
        with torch.no_grad():
            epe_raw = epe_loss(pred_flow, gt_flow, valid)
        logs = {"epe": epe_raw.item(), "n_valid": float(valid.sum().item())}

        total = self.flow_weight * loss_epe

        if result.covisibility is not None and result.covisibility.logits is not None:
            logits = result.covisibility.logits
            target = batch["covisibility"]
            loss_covis = covisibility_bce(logits, target, region=None,
                                          pos_weight=self.covis_pos_weight)
            total = total + self.covis_weight * loss_covis
            logs["covis_bce"] = loss_covis.item()
            logs.update(covisibility_metrics(logits, target))
        else:
            logs["covis_bce"] = float("nan")

        logs["total"] = total.item()
        return total, logs
