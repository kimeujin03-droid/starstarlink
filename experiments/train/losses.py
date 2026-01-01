from __future__ import annotations

import torch
import torch.nn.functional as F


def w_mask_schedule(epoch: int, warmup_epochs: int = 8, w0: float = 50.0, w1: float = 150.0) -> float:
    return float(w0 if epoch < warmup_epochs else w1)


def heavy_residual_l1(pred: torch.Tensor, gt: torch.Tensor,
                     streak_mask: torch.Tensor, other_mask: torch.Tensor,
                     w_mask: float = 100.0) -> torch.Tensor:
    """L1 with heavier weight inside artifact masks (streak ∪ other)."""
    union = torch.clamp(streak_mask + (other_mask > 0).float(), 0, 1)
    w = 1.0 + w_mask * union
    return torch.mean(w * torch.abs(pred - gt))


def bce_logits(logits: torch.Tensor, targets01: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets01.float())


def outside_oversub_loss(pred: torch.Tensor, gt: torch.Tensor,
                         streak_mask: torch.Tensor, other_mask: torch.Tensor,
                         weight: float = 5.0) -> torch.Tensor:
    """Penalize over-subtraction (gt > pred) OUTSIDE artifact masks.

    This directly reduces negative flux-bias tails due to accidental removal of stars/background.
    """
    union = torch.clamp(streak_mask + (other_mask > 0).float(), 0, 1)
    out = 1.0 - union
    return weight * torch.mean(out * F.relu(gt - pred))


def star_preservation_loss(pred: torch.Tensor, gt: torch.Tensor,
                           q: float = 0.99, weight: float = 50.0) -> torch.Tensor:
    """Penalize over-subtraction on brightest pixels (proxy for stellar cores)."""
    thr = torch.quantile(gt.detach(), q)
    m = (gt > thr).float()
    return weight * torch.mean(m * F.relu(gt - pred))
