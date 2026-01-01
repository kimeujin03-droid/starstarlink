"""Generate a Track‑2 dataset (smear/bloom-heavy subset).

Track‑2 goal: rescue model for smear/bloom overlap failures.
We encourage high smear rate by rejecting non-smear frames until targets are met.

Also outputs split masks:
  - blm_*.fits : (oth==1)
  - smr_*.fits : (oth==2)

Usage:
  python experiments/gen_track2.py --out dataset_track2 --n_total 2000
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
from astropy.io import fits

from ztfsynth.config import SimConfig
from ztfsynth.ztfsim import simulate_one


def write_fits(path: str, data: np.ndarray, hdr=None):
    hdu = fits.PrimaryHDU(data.astype(np.float32), header=hdr)
    hdu.writeto(path, overwrite=True)


def patch_cfg_for_track2(cfg: SimConfig, smear_prob=0.4, smear_strength=0.003,
                         bloom_prob=0.3, bloom_strength=0.07,
                         sat_star_prob=0.2,
                         streak_probs=None):
    cfg.smear_conditional_prob = float(smear_prob)
    cfg.smear_strength = float(smear_strength)
    cfg.bloom_conditional_prob = float(bloom_prob)
    cfg.bloom_strength = float(bloom_strength)
    cfg.sat_star_frame_prob = float(sat_star_prob)
    if streak_probs is not None:
        cfg.streak_probs = tuple(float(x) for x in streak_probs)
    return cfg


@dataclass
class Track2Targets:
    n_total: int = 2000
    min_smear: int = 1400   # aim >=70% smear-on
    min_bloom: int = 500    # ensure bloom coverage
    seed0: int = 24681357


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dataset folder")
    ap.add_argument("--n_total", type=int, default=2000)
    ap.add_argument("--min_smear", type=int, default=1400)
    ap.add_argument("--min_bloom", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=24681357)

    ap.add_argument("--smear_prob", type=float, default=0.4)
    ap.add_argument("--smear_strength", type=float, default=0.003)
    ap.add_argument("--bloom_prob", type=float, default=0.3)
    ap.add_argument("--bloom_strength", type=float, default=0.07)
    ap.add_argument("--sat_star_prob", type=float, default=0.2)

    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    t2 = Track2Targets(
        n_total=args.n_total,
        min_smear=args.min_smear,
        min_bloom=args.min_bloom,
        seed0=args.seed0,
    )

    cfg = SimConfig()
    patch_cfg_for_track2(
        cfg,
        smear_prob=args.smear_prob,
        smear_strength=args.smear_strength,
        bloom_prob=args.bloom_prob,
        bloom_strength=args.bloom_strength,
        sat_star_prob=args.sat_star_prob,
    )

    accepted = 0
    tries = 0
    smear_cnt = 0
    bloom_cnt = 0

    meta_rows = []

    while accepted < t2.n_total:
        seed = int(t2.seed0 + tries)
        tries += 1

        obs, cln, stk, oth, meta, hdr = simulate_one(seed=seed, cfg=cfg)

        did_bloom = bool(meta.get("bloom_on", False))
        did_smear = bool(meta.get("smear_on", False))

        remaining_slots = t2.n_total - accepted
        miss_smear = max(0, t2.min_smear - smear_cnt)
        miss_bloom = max(0, t2.min_bloom - bloom_cnt)

        force_smear = (miss_smear > 0 and remaining_slots <= miss_smear)
        force_bloom = (miss_bloom > 0 and remaining_slots <= miss_bloom)

        if force_smear and (not did_smear):
            continue
        if force_bloom and (not did_bloom):
            continue

        idx = accepted

        write_fits(os.path.join(out_dir, f"obs_{idx:07d}.fits"), obs, hdr)
        write_fits(os.path.join(out_dir, f"cln_{idx:07d}.fits"), cln, hdr)
        write_fits(os.path.join(out_dir, f"stk_{idx:07d}.fits"), stk.astype(np.uint8), hdr)
        write_fits(os.path.join(out_dir, f"oth_{idx:07d}.fits"), oth.astype(np.uint8), hdr)

        # split other_mask
        blm = (oth == 1).astype(np.uint8)
        smr = (oth == 2).astype(np.uint8)
        write_fits(os.path.join(out_dir, f"blm_{idx:07d}.fits"), blm, hdr)
        write_fits(os.path.join(out_dir, f"smr_{idx:07d}.fits"), smr, hdr)

        accepted += 1
        bloom_cnt += int(did_bloom)
        smear_cnt += int(did_smear)

        meta_rows.append({
            "idx": idx,
            "seed": seed,
            "bloom": int(did_bloom),
            "smear": int(did_smear),
            "hard": int(bool(meta.get("hard", False))),
            "nstars": int(meta.get("nstars", -1)),
            "nstreaks": int(meta.get("nstreaks", -1)),
        })

        if accepted % 100 == 0:
            print(f"[Track2] saved={accepted}/{t2.n_total} | bloom={bloom_cnt} | smear={smear_cnt} | tries={tries}")

    summary = {
        "track": "track2",
        "config": asdict(t2),
        "sim_cfg_overrides": {
            "smear_conditional_prob": cfg.smear_conditional_prob,
            "smear_strength": cfg.smear_strength,
            "bloom_conditional_prob": cfg.bloom_conditional_prob,
            "bloom_strength": cfg.bloom_strength,
            "sat_star_frame_prob": cfg.sat_star_frame_prob,
            "streak_probs": cfg.streak_probs,
        },
        "result": {
            "accepted": accepted,
            "tries": tries,
            "bloom_cnt": bloom_cnt,
            "smear_cnt": smear_cnt,
            "bloom_frac": bloom_cnt / max(1, accepted),
            "smear_frac": smear_cnt / max(1, accepted),
        },
    }

    with open(os.path.join(out_dir, "track2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "track2_meta_rows.jsonl"), "w", encoding="utf-8") as f:
        for r in meta_rows:
            f.write(json.dumps(r) + "\n")

    print("[DONE] Track2 generation")
    print(json.dumps(summary["result"], indent=2))


if __name__ == "__main__":
    main()
