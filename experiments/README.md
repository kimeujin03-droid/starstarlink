# Experiments Pack (Track‑1 / Track‑2 + Metrics)

This folder adds **reproducible experiment scripts** on top of the existing simulator.
It assumes the dataset format produced by `make_dataset.py`:

- `Observed`  : `obs_*.fits`
- `Target`    : `cln_*.fits`
- `StreakMask`: `stk_*.fits` (0/1)
- `OtherMask` : `oth_*.fits` (0=none, **1=bloom**, **2=smear**)

## 0) Environment

Typical Colab / local env:

```bash
pip install numpy torch astropy matplotlib pandas scikit-learn
# optional but recommended for SSIM:
pip install scikit-image
```

From repo root:

```bash
export PYTHONPATH=$PWD
```

## 1) Track‑1: photometry‑safe baseline (streak‑centric, rare smear)

### 1.1 Generate dataset with minimum coverage

This generator *enforces* minimum counts for bloom/smear cases while keeping the final dataset size fixed.

```bash
python experiments/gen_track1_dataset.py \
  --out dataset_track1 \
  --n 5000 \
  --min-bloom 600 \
  --min-smear 80 \
  --seed 123
```

It writes:
- `dataset_track1/obs_*.fits`, `cln_*.fits`, `stk_*.fits`, `oth_*.fits`
- `dataset_track1/track1_summary.json` (rates + config snapshot)

### 1.2 Train baseline U‑Net (includes anti‑over‑subtraction losses)

```bash
python experiments/train/train_track1_unet.py \
  --data dataset_track1 \
  --out runs/track1 \
  --epochs 60 \
  --batch 16 \
  --lr 1e-3
```

Key features:
- **w_mask schedule** (warmup → stronger artifact focus)
- **outside oversubtraction penalty** (reduces negative flux‑bias tail)
- **bright‑star preservation penalty** (reduces bloom overlap failures)

## 2) Track‑2: rescue model (smear/bloom heavy)

### 2.1 Generate smear‑heavy subset

```bash
python experiments/gen_track2_dataset.py \
  --out dataset_track2 \
  --n 2000 \
  --min-smear 1400 \
  --min-bloom 600 \
  --seed 456
```

### 2.2 Train multi‑head model (3 masks: streak / bloom / smear)

```bash
python experiments/train/train_track2_multihead.py \
  --data dataset_track2 \
  --out runs/track2 \
  --epochs 60 \
  --batch 16 \
  --lr 1e-3
```

## 3) Metrics (PSNR / SSIM / L1 splits / flux bias)

Evaluate any run (model checkpoint optional). If you already have `pred_*.fits` saved, set `--pred-dir`.

### 3.1 If you have a trained checkpoint

```bash
python experiments/eval/eval_metrics.py \
  --data dataset_track1 \
  --ckpt runs/track1/best.pt \
  --out eval/track1
```

### 3.2 If you already have predictions on disk

```bash
python experiments/eval/eval_metrics.py \
  --data dataset_track1 \
  --pred-dir my_predictions \
  --out eval/track1
```

Outputs:
- `per_image_metrics.csv`
- `summary.txt`
- plots: histograms/scatters + worst/best grid

## 4) Gating (smear score → decide Track‑2)

Rule‑based smear score:

```bash
python experiments/eval/sweep_smear_threshold.py --data dataset_track2
```

If your FITS headers contain smear labels, it will compute the best threshold (recall‑biased).

---

If you get checkpoint load errors with `_orig_mod.` keys (from `torch.compile`), use:

```python
from experiments.utils.ckpt import load_ckpt_strip_orig_mod
model = load_ckpt_strip_orig_mod(model, "path/to/ckpt.pt", map_location="cpu")
```
