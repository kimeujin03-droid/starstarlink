#!/usr/bin/env bash
set -euo pipefail

# Example end-to-end commands (edit paths as needed).
# Run from repository root:
#   bash experiments/run_all.sh

# ---- Track 1 ----
python experiments/gen_track1.py --out dataset_track1 --n_total 5000 --min_bloom 600 --min_smear 80
python experiments/train/train_track1.py --data dataset_track1 --out runs/track1 --epochs 60 --batch 16
python experiments/eval/run_metrics.py --data dataset_track1 --ckpt runs/track1/best.pt --track 1 --out eval/track1

# ---- Track 2 ----
python experiments/gen_track2.py --out dataset_track2 --n_total 2000 --min_smear_frac 0.8
python experiments/train/train_track2.py --data dataset_track2 --out runs/track2 --epochs 40 --batch 12
python experiments/eval/run_metrics.py --data dataset_track2 --ckpt runs/track2/best.pt --track 2 --out eval/track2

echo "All experiments completed."
