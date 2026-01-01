#!/usr/bin/env python
"""
ZTF-like synthetic dataset generator (FITS) with resume/chunk support.

Outputs per index i (float32 unless mask):
  sim_{i:010d}_obs.fits        : observed image (stars+sky+noise + streak + optional bloom/smear)
  sim_{i:010d}_clean.fits      : clean image (stars+sky+noise), no streak/bloom/smear (noise retained)
  sim_{i:010d}_streakmask.fits : uint8 mask, 1 where streak photons were added
  sim_{i:010d}_othermask.fits  : uint8 mask, 1 bloom pixels, 2 smear pixels (0 elsewhere)

Key flags:
  --resume     : skip indices that already exist on disk
  --start/--count : generate a contiguous chunk [start, start+count)
"""

import os
import glob
import argparse

from ztfsynth.config import SimConfig
from ztfsynth.ztfsim import simulate_one, write_fits


def _exists_bundle(out_dir: str, idx: int) -> bool:
    stem = os.path.join(out_dir, f"sim_{idx:010d}")
    return (
        os.path.exists(stem + "_obs.fits") and
        os.path.exists(stem + "_clean.fits") and
        os.path.exists(stem + "_streakmask.fits") and
        os.path.exists(stem + "_othermask.fits")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="output directory (recommend Google Drive path)")
    ap.add_argument("--n", type=int, default=5000, help="total number of frames (used if --count not set)")
    ap.add_argument("--size", type=int, default=512, help="image size (pixels)")
    ap.add_argument("--seed0", type=int, default=123, help="base seed; per-frame seed = seed0 + idx")
    ap.add_argument("--start", type=int, default=0, help="start index (for chunked generation)")
    ap.add_argument("--count", type=int, default=None, help="number of frames to generate (chunk size). default: n-start")
    ap.add_argument("--resume", action="store_true", help="skip indices already present on disk")
    ap.add_argument("--print_every", type=int, default=50, help="progress print interval")

    # Performance/realism toggles
    ap.add_argument("--use_grf", action="store_true", help="enable structured GRF background (slow if overused)")
    ap.add_argument("--grf_prob", type=float, default=None, help="P(apply GRF | frame) when --use_grf")
    ap.add_argument("--grf_amp_e", type=float, default=None, help="GRF amplitude in e-/pix")
    ap.add_argument("--grf_pool", type=int, default=None, help="cached GRF pool size")
    ap.add_argument("--grf_mode", type=str, default=None, choices=["fft","gauss"], help="GRF mode: fft=1/f^beta via FFT, gauss=fast smoothed noise")
    ap.add_argument("--grf_sigma", type=float, default=None, help="Gaussian smoothing sigma (pixels) when --grf_mode gauss")
    ap.add_argument("--fast_bg", action="store_true", help="shortcut: enable GRF and set --grf_mode gauss")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Resolve range
    start = int(args.start)
    if args.count is None:
        count = int(args.n - start)
    else:
        count = int(args.count)
    end = start + max(0, count)

    if end <= start:
        raise ValueError(f"Invalid range: start={start}, count={count}")

    # Configure simulation
    cfg = SimConfig()
    cfg.size = int(args.size)

    if args.fast_bg:
        cfg.use_grf_background = True
        cfg.grf_mode = "gauss"

    if args.use_grf:
        cfg.use_grf_background = True

    if getattr(cfg, "use_grf_background", False):
        if args.grf_prob is not None:
            cfg.grf_prob = float(args.grf_prob)
        if args.grf_amp_e is not None:
            cfg.grf_amp_e = float(args.grf_amp_e)
        if args.grf_pool is not None:
            cfg.grf_pool_size = int(args.grf_pool)
        if args.grf_mode is not None:
            cfg.grf_mode = str(args.grf_mode)
        if args.grf_sigma is not None:
            cfg.grf_gauss_sigma_pix = float(args.grf_sigma)

    # If resuming, show how many exist already in range
    if args.resume:
        exist = 0
        for i in range(start, end):
            if _exists_bundle(args.out, i):
                exist += 1
        print(f"[resume] existing bundles in range [{start},{end}): {exist}/{end-start}")

    saved = 0
    skipped = 0

    for idx in range(start, end):
        if args.resume and _exists_bundle(args.out, idx):
            skipped += 1
            continue

        seed = int(args.seed0) + int(idx)
        obs, clean, smsk, omsk, meta, hdr = simulate_one(seed, cfg)

        stem = f"sim_{idx:010d}"
        write_fits(os.path.join(args.out, stem + "_obs.fits"), obs, hdr)
        write_fits(os.path.join(args.out, stem + "_clean.fits"), clean, hdr)
        write_fits(os.path.join(args.out, stem + "_streakmask.fits"), smsk.astype("uint8"), hdr)
        write_fits(os.path.join(args.out, stem + "_othermask.fits"), omsk.astype("uint8"), hdr)

        saved += 1
        if (saved % args.print_every) == 0:
            print(f"[saved {saved:5d}] idx={idx} -> {stem} (skipped={skipped})")

    print(f"Done. saved={saved} skipped={skipped} out={args.out}")


if __name__ == "__main__":
    main()
