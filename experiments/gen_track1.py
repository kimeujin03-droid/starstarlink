"""Generate a Track‑1 dataset with **min-count guarantees**.

Track‑1 goal: photometry‑safe baseline (streak-centric), smear rare (0–3%).
We generate samples until we accept N images while enforcing:
  - minimum number of bloom-on frames
  - minimum number of smear-on frames (optional)

Outputs (FITS): obs_*, cln_*, stk_*, oth_*
Also writes a summary json.

Usage:
  python experiments/gen_track1.py --out dataset_track1 --n 5000 --min_bloom 600 --min_smear 50
"""

import os
import json
import argparse
from dataclasses import dataclass, asdict

import numpy as np

from ztfsynth.config import SimConfig
from ztfsynth.ztfsim import simulate_one, write_fits


@dataclass
class Track1Config:
    # dataset sizes
    n_total: int = 5000
    min_bloom: int = 600
    min_smear: int = 50

    # track‑1 physics knobs (overrides)
    smear_conditional_prob: float = 0.02
    smear_strength: float = 0.001

    bloom_conditional_prob: float = 0.15
    bloom_strength: float = 0.07

    sat_star_frame_prob: float = 0.07

    # streak count distribution (0..5)
    streak_probs: tuple = (0.12, 0.40, 0.25, 0.15, 0.06, 0.02)

    # misc
    seed0: int = 12345
    artifact_mode: str = "physical"  # physical | hard | both


def build_cfg(t1: Track1Config) -> SimConfig:
    cfg = SimConfig()
    cfg.artifact_mode = t1.artifact_mode

    cfg.smear_conditional_prob = float(t1.smear_conditional_prob)
    cfg.smear_strength = float(t1.smear_strength)

    cfg.bloom_conditional_prob = float(t1.bloom_conditional_prob)
    cfg.bloom_strength = float(t1.bloom_strength)

    cfg.sat_star_frame_prob = float(t1.sat_star_frame_prob)

    cfg.streak_probs = tuple(float(x) for x in t1.streak_probs)

    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dataset dir")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--min_bloom", type=int, default=600)
    ap.add_argument("--min_smear", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=12345)

    ap.add_argument("--smear_prob", type=float, default=0.02)
    ap.add_argument("--smear_strength", type=float, default=0.001)

    ap.add_argument("--bloom_prob", type=float, default=0.15)
    ap.add_argument("--bloom_strength", type=float, default=0.07)

    ap.add_argument("--sat_star_prob", type=float, default=0.07)
    ap.add_argument("--artifact_mode", type=str, default="physical", choices=["physical","hard","both"])

    args = ap.parse_args()

    t1 = Track1Config(
        n_total=args.n,
        min_bloom=args.min_bloom,
        min_smear=args.min_smear,
        seed0=args.seed0,
        smear_conditional_prob=args.smear_prob,
        smear_strength=args.smear_strength,
        bloom_conditional_prob=args.bloom_prob,
        bloom_strength=args.bloom_strength,
        sat_star_frame_prob=args.sat_star_prob,
        artifact_mode=args.artifact_mode,
    )

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    cfg = build_cfg(t1)

    accepted = 0
    tries = 0
    bloom_cnt = 0
    smear_cnt = 0

    meta_rows = []

    while accepted < t1.n_total:
        seed = int(t1.seed0 + tries)
        tries += 1

        obs, cln, stk, oth, meta, hdr = simulate_one(seed=seed, cfg=cfg)

        # ztfsim meta keys
        did_bloom = bool(meta.get("bloom_on", False))
        did_smear = bool(meta.get("smear_on", False))

        # ---- strict min-count acceptance policy ----
        # We must end with exactly N accepted samples and satisfy min_bloom/min_smear.
        # If we're running out of slots to satisfy a requirement, we start rejecting non-matching samples.
        remaining_slots = t1.n_total - accepted
        miss_bloom = max(0, t1.min_bloom - bloom_cnt)
        miss_smear = max(0, t1.min_smear - smear_cnt)

        # If a requirement must be met in the remaining slots, reject samples that don't help.
        forced_bloom_only = (miss_bloom > 0 and remaining_slots <= miss_bloom)
        forced_smear_only = (miss_smear > 0 and remaining_slots <= miss_smear)

        if forced_bloom_only and (not did_bloom):
            continue
        if forced_smear_only and (not did_smear):
            continue

        # Otherwise, accept all samples (keeps distribution close to configured probabilities).

        idx = accepted
        write_fits(os.path.join(out_dir, f"obs_{idx:07d}.fits"), obs, hdr)
        write_fits(os.path.join(out_dir, f"cln_{idx:07d}.fits"), cln, hdr)
        write_fits(os.path.join(out_dir, f"stk_{idx:07d}.fits"), stk.astype(np.uint8), hdr)
        write_fits(os.path.join(out_dir, f"oth_{idx:07d}.fits"), oth.astype(np.uint8), hdr)

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
            print(f"[Track1] saved={accepted}/{t1.n_total} | bloom={bloom_cnt} | smear={smear_cnt} | tries={tries}")

    summary = {
        "track": "track1",
        "config": asdict(t1),
        "result": {
            "accepted": accepted,
            "tries": tries,
            "bloom_cnt": bloom_cnt,
            "smear_cnt": smear_cnt,
            "bloom_frac": bloom_cnt / max(1, accepted),
            "smear_frac": smear_cnt / max(1, accepted),
        },
        "notes": "If bloom/smear counts are below min targets, increase N or adjust *_prob/strength.",
    }

    with open(os.path.join(out_dir, "track1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "track1_meta_rows.jsonl"), "w", encoding="utf-8") as f:
        for r in meta_rows:
            f.write(json.dumps(r) + "\n")

    print("[DONE] Track1 generation")
    print(json.dumps(summary["result"], indent=2))


if __name__ == "__main__":
    main()
