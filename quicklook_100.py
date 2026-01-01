#!/usr/bin/env python
"""
Quicklook mosaic for ZTF-synth dataset.

Examples:
  # 100 random frames (deterministic) with mask overlays
  python quicklook_100.py --dir /content/dataset_5000 --n 100 --seed 0 --out /content/quicklook_100.png --overlay

  # show CLEAN only
  python quicklook_100.py --dir /content/dataset_5000 --n 64 --seed 1 --show clean --out /content/ql_clean.png
"""

import os
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.visualization import ZScaleInterval


def _load(path):
    return fits.getdata(path, memmap=False).astype(np.float32)


def _parse_idx_from_obs(path: str) -> int:
    m = re.search(r"sim_(\d+)_obs\.fits$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _pick_indices(dir_path: str, n: int, seed: int):
    obs_paths = sorted(glob.glob(os.path.join(dir_path, "sim_*_obs.fits")))
    idxs = [_parse_idx_from_obs(p) for p in obs_paths]
    idxs = [i for i in idxs if i >= 0]
    idxs.sort()
    if len(idxs) == 0:
        raise FileNotFoundError(f"No sim_*_obs.fits under: {dir_path}")

    rng = np.random.default_rng(int(seed))
    if n >= len(idxs):
        return idxs
    pick = rng.choice(idxs, size=int(n), replace=False)
    pick = np.asarray(pick, dtype=int)
    pick.sort()
    return pick.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dataset directory")
    ap.add_argument("--n", type=int, default=100, help="number of frames")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (for reproducibility)")
    ap.add_argument("--out", type=str, default=None, help="output png path")
    ap.add_argument("--overlay", action="store_true", help="overlay streak/other masks on top of image")
    ap.add_argument("--fill", action="store_true", help="fill overlays (semi-transparent) instead of contour-only")
    ap.add_argument("--show", type=str, default="obs", choices=["obs","clean","resid","streakmask","othermask"],
                    help="which layer to display as base")
    ap.add_argument("--cols", type=int, default=10, help="mosaic columns")
    ap.add_argument("--max", type=float, default=None, help="optional fixed vmax (otherwise zscale per tile)")
    args = ap.parse_args()

    idxs = _pick_indices(args.dir, args.n, args.seed)

    cols = max(1, int(args.cols))
    rows = int(np.ceil(len(idxs) / cols))
    fig = plt.figure(figsize=(cols * 1.8, rows * 1.8))

    zs = ZScaleInterval()

    for k, idx in enumerate(idxs):
        stem = os.path.join(args.dir, f"sim_{idx:010d}")
        obs_f = stem + "_obs.fits"
        cln_f = stem + "_clean.fits"
        sm_f  = stem + "_streakmask.fits"
        om_f  = stem + "_othermask.fits"

        obs = _load(obs_f)
        cln = _load(cln_f)
        sm  = fits.getdata(sm_f, memmap=False).astype(np.uint8) if os.path.exists(sm_f) else None
        om  = fits.getdata(om_f, memmap=False).astype(np.uint8) if os.path.exists(om_f) else None

        if args.show == "obs":
            base = obs
        elif args.show == "clean":
            base = cln
        elif args.show == "resid":
            base = obs - cln
        elif args.show == "streakmask":
            base = (sm.astype(np.float32) if sm is not None else np.zeros_like(obs, dtype=np.float32))
        else:  # othermask
            base = (om.astype(np.float32) if om is not None else np.zeros_like(obs, dtype=np.float32))

        ax = plt.subplot(rows, cols, k + 1)

        if args.show == "resid":
            # symmetric scale for residual
            med = np.median(base)
            mad = np.median(np.abs(base - med))
            sig = 1.4826 * mad
            lim = (8.0 * sig) if sig > 0 else np.max(np.abs(base))
            vmin, vmax = -lim, lim
            ax.imshow(base, origin="lower", vmin=vmin, vmax=vmax, cmap="gray", interpolation="nearest")
        else:
            if args.max is None:
                vmin, vmax = zs.get_limits(base)
            else:
                vmin, vmax = float(np.min(base)), float(args.max)
            ax.imshow(base, origin="lower", vmin=vmin, vmax=vmax, cmap="gray", interpolation="nearest")

        if args.overlay and (sm is not None or om is not None) and args.show in ("obs","clean","resid"):
            if args.fill:
                # Filled RGBA overlays (easier to judge true mask thickness than contour-only)
                h, w = base.shape
                alpha = 0.35

                if sm is not None and sm.sum() > 0:
                    rgba = np.zeros((h, w, 4), dtype=np.float32)
                    rgba[..., 0] = 1.0  # R
                    rgba[..., 2] = 1.0  # B (magenta)
                    rgba[..., 3] = alpha * (sm > 0).astype(np.float32)
                    ax.imshow(rgba, origin="lower", interpolation="nearest")

                if om is not None and om.sum() > 0:
                    bloom = (om == 1)
                    smear = (om == 2)

                    if np.any(bloom):
                        rgba = np.zeros((h, w, 4), dtype=np.float32)
                        rgba[..., 1] = 1.0  # G
                        rgba[..., 2] = 1.0  # B (cyan)
                        rgba[..., 3] = alpha * bloom.astype(np.float32)
                        ax.imshow(rgba, origin="lower", interpolation="nearest")

                    if np.any(smear):
                        rgba = np.zeros((h, w, 4), dtype=np.float32)
                        rgba[..., 0] = 1.0  # R
                        rgba[..., 1] = 1.0  # G (yellow)
                        rgba[..., 3] = alpha * smear.astype(np.float32)
                        ax.imshow(rgba, origin="lower", interpolation="nearest")
            else:
                # Contour-only overlays (lighter / less cluttered)
                if sm is not None and sm.sum() > 0:
                    ax.contour(sm > 0, levels=[0.5], colors="magenta", linewidths=0.6)
                if om is not None and om.sum() > 0:
                    # bloom=1 (cyan), smear=2 (yellow)
                    ax.contour(om == 1, levels=[0.5], colors="cyan", linewidths=0.5)
                    ax.contour(om == 2, levels=[0.5], colors="yellow", linewidths=0.5)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{idx}", fontsize=7)

    plt.tight_layout(pad=0.2)

    if args.out:
        fig.savefig(args.out, bbox_inches="tight", dpi=150)
        print(f"saved: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
