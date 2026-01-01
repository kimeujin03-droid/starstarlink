"""Evaluate per-image metrics + produce best/worst grids.

Works with:
  - dataset folders created by make_dataset.py or experiments/gen_track*.py
  - either precomputed predictions (`--pred_dir`) OR a checkpoint (`--ckpt`).

Outputs:
  - per_image_metrics.csv
  - summary.txt
  - grid_best.png / grid_worst.png (optional)

Examples:
  python experiments/eval/run_metrics.py --data dataset_track1 --pred_dir preds/track1 --out eval/track1
  python experiments/eval/run_metrics.py --data dataset_track1 --ckpt runs/track1/best.pt --track 1 --out eval/track1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass

import numpy as np

from astropy.io import fits


def read_fits(path: str) -> np.ndarray:
    return fits.getdata(path).astype(np.float32)


def psnr(pred: np.ndarray, gt: np.ndarray, data_range=None) -> float:
    if data_range is None:
        data_range = float(np.max(gt) - np.min(gt) + 1e-8)
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


def ssim_safe(pred: np.ndarray, gt: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(gt, pred, data_range=float(np.max(gt) - np.min(gt) + 1e-8)))
    except Exception:
        return float("nan")


def flux_bias(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps=1e-8) -> float:
    """Relative flux bias inside mask."""
    m = mask.astype(bool)
    num = float(np.sum(pred[m]) - np.sum(gt[m]))
    den = float(np.sum(gt[m]) + eps)
    return num / den


def oversub_tail(pred: np.ndarray, gt: np.ndarray, mask_out: np.ndarray, q=0.99) -> float:
    """How much we delete bright sources outside artifact masks.

    Computes mean ReLU(gt - pred) on bright pixels outside union masks.
    """
    out = mask_out.astype(bool)
    thr = np.quantile(gt[out], q) if np.any(out) else np.quantile(gt, q)
    star = (gt > thr)
    m = star & out
    if not np.any(m):
        return 0.0
    return float(np.mean(np.maximum(gt[m] - pred[m], 0.0)))


def build_grid(fig_path: str, rows: list[dict], n_show: int = 8):
    """Save a grid similar to your debug figures."""
    import matplotlib.pyplot as plt

    n = min(n_show, len(rows))
    if n == 0:
        return

    cols = ["Observed", "Pred Clean", "Target Clean", "Pred-Target", "StreakMask", "OtherMask"]
    fig, axes = plt.subplots(n, len(cols), figsize=(len(cols) * 3.1, n * 3.0))
    if n == 1:
        axes = np.array([axes])

    for i in range(n):
        r = rows[i]
        obs = r["obs"]
        prd = r["pred"]
        gt = r["gt"]
        stk = r["stk"]
        oth = r["oth"]
        diff = prd - gt

        ims = [obs, prd, gt, diff, stk, oth]
        for j, (ax, im, title) in enumerate(zip(axes[i], ims, cols)):
            ax.imshow(im)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def list_indices(data_dir: str):
    paths = sorted(glob.glob(os.path.join(data_dir, "obs_*.fits")))
    idx = [int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in paths]
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset folder with obs/cln/stk/oth FITS")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred_dir", default=None, help="folder containing prd_*.fits predictions")
    ap.add_argument("--ckpt", default=None, help="checkpoint path to run inference")
    ap.add_argument("--track", type=int, default=1, choices=[1, 2], help="model type if using --ckpt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_images", type=int, default=0, help="0=all")
    ap.add_argument("--grid_n", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Optional inference
    model = None
    if args.pred_dir is None and args.ckpt is None:
        raise SystemExit("Provide either --pred_dir or --ckpt")

    if args.pred_dir is None:
        import torch
        from experiments.train.models import UNet, UNetWithHeads
        from experiments.utils.ckpt import load_ckpt_strip_orig_mod

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if args.track == 1:
            model = UNet(in_ch=1, base=32, out_ch=1)
        else:
            model = UNetWithHeads(in_ch=1, base=32, out_ch_img=1, out_ch_mask=3)
        model = load_ckpt_strip_orig_mod(model, args.ckpt, map_location=device, strict=False)
        model.to(device)
        model.eval()

    idxs = list_indices(args.data)
    if args.max_images and args.max_images > 0:
        idxs = idxs[: args.max_images]

    rows_out = []
    cache_for_grids = []

    for i, k in enumerate(idxs):
        obs = read_fits(os.path.join(args.data, f"obs_{k:07d}.fits"))
        gt = read_fits(os.path.join(args.data, f"cln_{k:07d}.fits"))
        stk = read_fits(os.path.join(args.data, f"stk_{k:07d}.fits"))
        oth = read_fits(os.path.join(args.data, f"oth_{k:07d}.fits"))

        if args.pred_dir is not None:
            pred = read_fits(os.path.join(args.pred_dir, f"prd_{k:07d}.fits"))
        else:
            import torch
            x = torch.from_numpy(obs[None, None]).float().to(model.parameters().__next__().device)
            with torch.no_grad():
                y = model(x)
            pred = (y[0] if isinstance(y, (tuple, list)) else y).squeeze().detach().cpu().numpy().astype(np.float32)

        union = np.clip((stk > 0).astype(np.uint8) + (oth > 0).astype(np.uint8), 0, 1)
        out = 1 - union
        bloom = (oth == 1).astype(np.uint8)
        smear = (oth == 2).astype(np.uint8)

        # Metrics
        l1 = float(np.mean(np.abs(pred - gt)))
        l1_union = float(np.mean(np.abs(pred - gt)[union.astype(bool)])) if np.any(union) else 0.0
        l1_out = float(np.mean(np.abs(pred - gt)[out.astype(bool)]))
        l1_stk = float(np.mean(np.abs(pred - gt)[(stk > 0)])) if np.any(stk > 0) else 0.0
        l1_other = float(np.mean(np.abs(pred - gt)[(oth > 0)])) if np.any(oth > 0) else 0.0

        fb_union = flux_bias(pred, gt, union)
        fb_stk = flux_bias(pred, gt, (stk > 0).astype(np.uint8)) if np.any(stk > 0) else 0.0
        fb_bloom = flux_bias(pred, gt, bloom) if np.any(bloom) else 0.0
        fb_smear = flux_bias(pred, gt, smear) if np.any(smear) else 0.0

        over = oversub_tail(pred, gt, out)

        rows_out.append({
            "idx": k,
            "l1": l1,
            "l1_union": l1_union,
            "l1_out": l1_out,
            "l1_streak": l1_stk,
            "l1_other": l1_other,
            "psnr": psnr(pred, gt),
            "ssim": ssim_safe(pred, gt),
            "artifact_frac": float(union.mean()),
            "bloom_frac": float(bloom.mean()),
            "smear_frac": float(smear.mean()),
            "flux_bias_union": fb_union,
            "flux_bias_streak": fb_stk,
            "flux_bias_bloom": fb_bloom,
            "flux_bias_smear": fb_smear,
            "oversub_tail": over,
        })

        # keep raw arrays for grids later
        cache_for_grids.append({
            "idx": k,
            "obs": obs,
            "pred": pred,
            "gt": gt,
            "stk": (stk > 0).astype(np.uint8),
            "oth": oth,
        })

        if (i + 1) % 200 == 0:
            print(f"[metrics] {i+1}/{len(idxs)}")

    # Save CSV
    import pandas as pd
    df = pd.DataFrame(rows_out).sort_values("idx")
    csv_path = os.path.join(args.out, "per_image_metrics.csv")
    df.to_csv(csv_path, index=False)

    # Summary
    def q(v, p):
        return float(np.quantile(v, p))

    summary = {
        "n": int(len(df)),
        "l1_mean": float(df["l1"].mean()),
        "l1_p50": q(df["l1"].values, 0.50),
        "l1_p90": q(df["l1"].values, 0.90),
        "flux_bias_union_p05": q(df["flux_bias_union"].values, 0.05),
        "flux_bias_union_p95": q(df["flux_bias_union"].values, 0.95),
        "oversub_tail_p95": q(df["oversub_tail"].values, 0.95),
        "artifact_frac_mean": float(df["artifact_frac"].mean()),
        "bloom_frac_mean": float(df["bloom_frac"].mean()),
        "smear_frac_mean": float(df["smear_frac"].mean()),
    }

    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # best/worst by l1_union (hard cases) then l1
    df_sort = df.sort_values(["l1_union", "l1"], ascending=False)
    worst_idx = set(df_sort.head(args.grid_n)["idx"].tolist())
    best_idx = set(df.sort_values(["l1_union", "l1"], ascending=True).head(args.grid_n)["idx"].tolist())

    worst_rows = [r for r in cache_for_grids if r["idx"] in worst_idx]
    best_rows = [r for r in cache_for_grids if r["idx"] in best_idx]

    build_grid(os.path.join(args.out, "grid_worst.png"), worst_rows, n_show=args.grid_n)
    build_grid(os.path.join(args.out, "grid_best.png"), best_rows, n_show=args.grid_n)

    print("[DONE]", csv_path)


if __name__ == "__main__":
    main()
