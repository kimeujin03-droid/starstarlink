from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
from astropy.io import fits
from torch.utils.data import Dataset


def read_fits(path: str) -> np.ndarray:
    return fits.getdata(path).astype(np.float32)


def robust_norm(x: np.ndarray, eps=1e-6) -> np.ndarray:
    """Per-image robust normalization to ~N(0,1) scale (median/MAD)."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    s = 1.4826 * mad
    if s < eps:
        s = max(np.std(x), eps)
    return (x - med) / s


def to_torch(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x)


@dataclass
class DatasetPaths:
    obs_glob: str
    cln_glob: str
    stk_glob: str
    oth_glob: str
    blm_glob: str | None = None
    smr_glob: str | None = None


class ZTFSimFITS(Dataset):
    """Loads simulator FITS files.

    Returns:
      obs  : (1,H,W) float
      cln  : (1,H,W) float
      stk  : (1,H,W) float 0/1
      oth  : (1,H,W) float 0/1/2 (if present)
      blm/smr optional as separate 0/1 masks.
    """

    def __init__(self, root: str, split="train", val_ratio=0.1, seed=0,
                 normalize=True, use_split_masks=False):
        self.root = root
        self.normalize = normalize
        self.use_split_masks = use_split_masks

        obs = sorted(glob.glob(os.path.join(root, "obs_*.fits")))
        cln = sorted(glob.glob(os.path.join(root, "cln_*.fits")))
        stk = sorted(glob.glob(os.path.join(root, "stk_*.fits")))
        oth = sorted(glob.glob(os.path.join(root, "oth_*.fits")))
        assert len(obs) == len(cln) == len(stk) == len(oth) > 0, "FITS count mismatch"

        n = len(obs)
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = int(round(n * val_ratio))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        pick = tr_idx if split == "train" else val_idx
        self.obs = [obs[i] for i in pick]
        self.cln = [cln[i] for i in pick]
        self.stk = [stk[i] for i in pick]
        self.oth = [oth[i] for i in pick]

        if use_split_masks:
            self.blm = sorted(glob.glob(os.path.join(root, "blm_*.fits")))
            self.smr = sorted(glob.glob(os.path.join(root, "smr_*.fits")))
            if len(self.blm) != len(obs) or len(self.smr) != len(obs):
                raise RuntimeError("use_split_masks=True but blm_*.fits / smr_*.fits missing")
            self.blm = [self.blm[i] for i in pick]
            self.smr = [self.smr[i] for i in pick]
        else:
            self.blm = None
            self.smr = None

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, i: int):
        obs = read_fits(self.obs[i])
        cln = read_fits(self.cln[i])
        stk = read_fits(self.stk[i])
        oth = read_fits(self.oth[i])

        if self.normalize:
            obs_n = robust_norm(obs)
            cln_n = robust_norm(cln)
        else:
            obs_n = obs
            cln_n = cln

        # ensure mask is 0/1
        stk01 = (stk > 0.5).astype(np.float32)

        out = {
            "obs": to_torch(obs_n[None, ...]),
            "cln": to_torch(cln_n[None, ...]),
            "stk": to_torch(stk01[None, ...]),
            "oth": to_torch(oth[None, ...]),
        }

        if self.use_split_masks:
            blm = read_fits(self.blm[i])
            smr = read_fits(self.smr[i])
            out["blm"] = to_torch((blm > 0.5).astype(np.float32)[None, ...])
            out["smr"] = to_torch((smr > 0.5).astype(np.float32)[None, ...])

        return out
