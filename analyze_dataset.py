import os
import glob
import argparse
import numpy as np
import csv

from astropy.io import fits


def _safe_get(hdr, key, default=None):
    try:
        return hdr.get(key, default)
    except Exception:
        return default


def analyze(out_dir: str, max_files: int | None = None):
    obs_paths = sorted(glob.glob(os.path.join(out_dir, "*_obs.fits")))
    if max_files is not None:
        obs_paths = obs_paths[:max_files]
    if not obs_paths:
        raise FileNotFoundError(f"No *_obs.fits found under: {out_dir}")

    rows = []

    for p in obs_paths:
        stem = os.path.basename(p).replace("_obs.fits", "")
        sm_path = os.path.join(out_dir, stem + "_streakmask.fits")
        om_path = os.path.join(out_dir, stem + "_othermask.fits")

        with fits.open(p, memmap=False) as hdul:
            hdr = hdul[0].header

        sm = fits.getdata(sm_path, memmap=False).astype(np.uint8) if os.path.exists(sm_path) else None
        om = fits.getdata(om_path, memmap=False).astype(np.uint8) if os.path.exists(om_path) else None

        nstrk = int(_safe_get(hdr, "NSTRK", _safe_get(hdr, "HIERARCH SIM NSTREAK", -1)))
        sat = bool(_safe_get(hdr, "SATUR", False))
        bloom = bool(_safe_get(hdr, "BLOOM", False))
        smear = bool(_safe_get(hdr, "SMEAR", False))

        streak_area = int(sm.sum()) if sm is not None else -1
        bloom_area = int((om == 1).sum()) if om is not None else -1
        smear_area = int((om == 2).sum()) if om is not None else -1

        rows.append(
            {
                "file": os.path.basename(p),
                "NSTRK": nstrk,
                "SATUR": int(sat),
                "BLOOM": int(bloom),
                "SMEAR": int(smear),
                "streak_area": streak_area,
                "bloom_area": bloom_area,
                "smear_area": smear_area,
            }
        )

    # summary (no pandas dependency)
    total = len(rows)
    nstrk_counts = {}
    sat_vals, bloom_vals, smear_vals, streak_areas = [], [], [], []
    for r in rows:
        nstrk_counts[r["NSTRK"]] = nstrk_counts.get(r["NSTRK"], 0) + 1
        sat_vals.append(r["SATUR"])
        bloom_vals.append(r["BLOOM"])
        smear_vals.append(r["SMEAR"])
        if r["streak_area"] >= 0:
            streak_areas.append(r["streak_area"])

    sat_rate = float(np.mean(sat_vals)) if total else 0.0
    bloom_rate = float(np.mean(bloom_vals)) if total else 0.0
    smear_rate = float(np.mean(smear_vals)) if total else 0.0

    sat_idx = [i for i, v in enumerate(sat_vals) if v == 1]
    p_bloom_given_sat = float(np.mean([bloom_vals[i] for i in sat_idx])) if len(sat_idx) else 0.0
    p_smear_given_sat = float(np.mean([smear_vals[i] for i in sat_idx])) if len(sat_idx) else 0.0

    def _q(arr, q):
        if arr is None or len(arr) == 0:
            return float("nan")
        return float(np.quantile(np.asarray(arr, dtype=np.float32), q))

    sa = np.asarray(streak_areas, dtype=np.float32)
    print("=" * 70)
    print("DATASET STATS SUMMARY")
    print("=" * 70)
    print(f"dir: {out_dir}")
    print(f"frames: {total}")
    print("\nNSTRK distribution:")
    for k in sorted(nstrk_counts.keys()):
        v = nstrk_counts[k]
        print(f"  {k}: {int(v)} ({100.0 * v / total:.1f}%)")
    print("\nEvent rates:")
    print(f"  SATUR: {100.0 * sat_rate:.1f}%")
    print(f"  BLOOM: {100.0 * bloom_rate:.1f}% | P(BLOOM|SATUR): {100.0 * p_bloom_given_sat:.1f}%")
    print(f"  SMEAR: {100.0 * smear_rate:.1f}% | P(SMEAR|SATUR): {100.0 * p_smear_given_sat:.1f}%")
    print("\nStreak area (mask pixels):")
    print(f"  mean={float(np.mean(sa)):.1f} std={float(np.std(sa)):.1f}")
    print(f"  p50={_q(sa,0.5):.1f} p90={_q(sa,0.9):.1f} p99={_q(sa,0.99):.1f}")
    print("=" * 70)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dataset directory (contains *_obs.fits, *_streakmask.fits, *_othermask.fits)")
    ap.add_argument("--max", type=int, default=None, help="analyze only first N frames")
    ap.add_argument("--out_csv", type=str, default=None, help="write per-frame stats CSV")
    args = ap.parse_args()

    rows = analyze(args.dir, args.max)
    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"saved: {args.out_csv}")


if __name__ == "__main__":
    main()
