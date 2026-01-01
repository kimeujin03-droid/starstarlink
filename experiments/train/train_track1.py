"""Track‑1 training: streak‑centric, photometry‑safe.

Model: simple U‑Net (image output only).
Loss:
  - HeavyResidualL1 (weights inside union mask)
  - outside_oversub_loss (prevents deleting stars/background)
  - star_preservation_loss (bright cores)

This is a standalone trainer that works with the simulator FITS datasets.

Usage:
  python experiments/train/train_track1.py --data dataset_track1 --out runs/track1 --epochs 60 --batch 16 --lr 1e-3
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.train.train_utils import FitsDataset, SplitConfig
from experiments.train.models import SimpleUNet
from experiments.train.losses import heavy_residual_l1, outside_oversub_loss, star_preservation_loss, w_mask_schedule


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainCfg:
    epochs: int = 60
    batch: int = 16
    lr: float = 1e-3
    num_workers: int = 2
    warmup_epochs: int = 8
    w0: float = 50.0
    w1: float = 150.0
    w_out: float = 5.0
    w_star: float = 0.5
    star_weight: float = 50.0


def train_one_epoch(model, loader, opt, device, epoch, cfg: TrainCfg):
    model.train()
    total = 0.0
    n = 0
    w_mask = w_mask_schedule(epoch, warmup_epochs=cfg.warmup_epochs, w0=cfg.w0, w1=cfg.w1)

    for batch in loader:
        obs = batch["obs"].to(device)
        cln = batch["clean"].to(device)
        stk = batch["streak"].to(device)
        oth = batch["other"].to(device)

        pred = model(obs)

        L_img = heavy_residual_l1(pred, cln, stk, oth, w_mask=w_mask)
        L_out = outside_oversub_loss(pred, cln, stk, oth, weight=cfg.w_out)
        L_star = star_preservation_loss(pred, cln, q=0.99, weight=cfg.star_weight)

        loss = L_img + L_out + cfg.w_star * L_star

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total += float(loss.item())
        n += 1

    return total / max(1, n)


@torch.no_grad()
def eval_one_epoch(model, loader, device, epoch, cfg: TrainCfg):
    model.eval()
    total = 0.0
    n = 0
    w_mask = w_mask_schedule(epoch, warmup_epochs=cfg.warmup_epochs, w0=cfg.w0, w1=cfg.w1)

    for batch in loader:
        obs = batch["obs"].to(device)
        cln = batch["clean"].to(device)
        stk = batch["streak"].to(device)
        oth = batch["other"].to(device)

        pred = model(obs)

        L_img = heavy_residual_l1(pred, cln, stk, oth, w_mask=w_mask)
        L_out = outside_oversub_loss(pred, cln, stk, oth, weight=cfg.w_out)
        L_star = star_preservation_loss(pred, cln, q=0.99, weight=cfg.star_weight)

        loss = L_img + L_out + cfg.w_star * L_star
        total += float(loss.item())
        n += 1

    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset directory containing obs_*.fits etc")
    ap.add_argument("--out", required=True, help="output run directory")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--base", type=int, default=32, help="UNet base channels")
    ap.add_argument("--w_mask0", type=float, default=50.0)
    ap.add_argument("--w_mask1", type=float, default=150.0)
    ap.add_argument("--warmup_epochs", type=int, default=8)
    ap.add_argument("--w_out", type=float, default=5.0)
    ap.add_argument("--w_star", type=float, default=0.5)
    ap.add_argument("--star_weight", type=float, default=50.0)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    split = SplitConfig(val_frac=args.val_frac, seed=args.seed)
    ds_tr = FitsDataset(args.data, split=split, split_role="train")
    ds_va = FitsDataset(args.data, split=split, split_role="val")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = SimpleUNet(in_ch=1, base=args.base, out_ch=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    cfg = TrainCfg(
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs,
        w0=args.w_mask0,
        w1=args.w_mask1,
        w_out=args.w_out,
        w_star=args.w_star,
        star_weight=args.star_weight,
    )

    best = 1e9
    for epoch in range(cfg.epochs):
        tr_loss = train_one_epoch(model, dl_tr, opt, device, epoch, cfg)
        va_loss = eval_one_epoch(model, dl_va, device, epoch, cfg)

        print(f"[E{epoch:03d}] train={tr_loss:.6f} val={va_loss:.6f}")

        # save last
        torch.save({"model": model.state_dict(), "epoch": epoch, "val": va_loss, "cfg": vars(args)}, os.path.join(args.out, "last.pt"))

        if va_loss < best:
            best = va_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": va_loss, "cfg": vars(args)}, os.path.join(args.out, "best.pt"))


if __name__ == "__main__":
    main()
