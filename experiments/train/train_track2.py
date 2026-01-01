"""Track‑2 training: smear/bloom rescue (multi‑head).

Model: U‑Net with two heads:
  - image head (clean)
  - mask head (3 channels: streak, bloom, smear) 

Loss:
  - HeavyResidualL1 for image
  - outside_oversub_loss for photometry safety
  - BCEWithLogits for masks

Dataset requirement: same FITS layout + optional `blm_*.fits` and `smr_*.fits` (produced by experiments/gen_track2.py)

Usage:
  python experiments/train/train_track2.py --data dataset_track2 --out runs/track2 --epochs 40 --batch 12 --lr 1e-3
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.train.train_utils import FitsDataset, SplitConfig
from experiments.train.models import UNetWithHeads
from experiments.train.losses import heavy_residual_l1, outside_oversub_loss, w_mask_schedule


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mask_bce_loss(logits, y01):
    return F.binary_cross_entropy_with_logits(logits, y01.float())


def train_one_epoch(model, loader, opt, device, epoch, warmup_epochs, w0, w1, w_out, w_maskloss):
    model.train()
    total = 0.0
    n = 0
    w_mask = w_mask_schedule(epoch, warmup_epochs=warmup_epochs, w0=w0, w1=w1)

    for batch in loader:
        obs = batch["obs"].to(device)
        cln = batch["clean"].to(device)
        stk = batch["streak"].to(device)
        oth = batch["other"].to(device)

        # split from other: bloom=1, smear=2
        blm = (oth == 1).float()
        smr = (oth == 2).float()

        pred_img, logits = model(obs)

        L_img = heavy_residual_l1(pred_img, cln, stk, oth, w_mask=w_mask)
        L_out = outside_oversub_loss(pred_img, cln, stk, oth, weight=w_out)

        y = torch.cat([stk.float(), blm, smr], dim=1)
        L_m = mask_bce_loss(logits, y)

        loss = L_img + L_out + w_maskloss * L_m

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total += float(loss.item())
        n += 1

    return total / max(1, n)


@torch.no_grad()
def eval_one_epoch(model, loader, device, epoch, warmup_epochs, w0, w1, w_out, w_maskloss):
    model.eval()
    total = 0.0
    n = 0
    w_mask = w_mask_schedule(epoch, warmup_epochs=warmup_epochs, w0=w0, w1=w1)

    for batch in loader:
        obs = batch["obs"].to(device)
        cln = batch["clean"].to(device)
        stk = batch["streak"].to(device)
        oth = batch["other"].to(device)

        blm = (oth == 1).float()
        smr = (oth == 2).float()

        pred_img, logits = model(obs)

        L_img = heavy_residual_l1(pred_img, cln, stk, oth, w_mask=w_mask)
        L_out = outside_oversub_loss(pred_img, cln, stk, oth, weight=w_out)

        y = torch.cat([stk.float(), blm, smr], dim=1)
        L_m = mask_bce_loss(logits, y)

        loss = L_img + L_out + w_maskloss * L_m
        total += float(loss.item())
        n += 1

    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--warmup_epochs", type=int, default=6)
    ap.add_argument("--w_mask0", type=float, default=50.0)
    ap.add_argument("--w_mask1", type=float, default=150.0)
    ap.add_argument("--w_out", type=float, default=5.0)
    ap.add_argument("--w_maskloss", type=float, default=0.5)
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

    model = UNetWithHeads(in_ch=1, base=args.base, out_ch_img=1, out_ch_mask=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = 1e9
    for epoch in range(args.epochs):
        tr_loss = train_one_epoch(model, dl_tr, opt, device, epoch, args.warmup_epochs, args.w_mask0, args.w_mask1, args.w_out, args.w_maskloss)
        va_loss = eval_one_epoch(model, dl_va, device, epoch, args.warmup_epochs, args.w_mask0, args.w_mask1, args.w_out, args.w_maskloss)

        print(f"[E{epoch:03d}] train={tr_loss:.6f} val={va_loss:.6f}")

        torch.save({"model": model.state_dict(), "epoch": epoch, "val": va_loss, "cfg": vars(args)}, os.path.join(args.out, "last.pt"))
        if va_loss < best:
            best = va_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": va_loss, "cfg": vars(args)}, os.path.join(args.out, "best.pt"))


if __name__ == "__main__":
    main()
