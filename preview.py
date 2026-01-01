#!/usr/bin/env python
"""
Single FITS viewer (zscale) for Colab.

Example:
  python preview.py --fits /content/dataset_5000/sim_0000000002_obs.fits
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", type=str, required=True, help="path to fits")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--out", type=str, default=None, help="save png instead of showing")
    args = ap.parse_args()

    data = fits.getdata(args.fits, memmap=False).astype(np.float32)
    vmin, vmax = ZScaleInterval().get_limits(data)

    plt.figure(figsize=(6,6))
    plt.imshow(data, origin="lower", vmin=vmin, vmax=vmax, cmap="gray", interpolation="nearest")
    plt.colorbar(label="ADU")
    plt.title(args.title if args.title else args.fits)
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"saved: {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
