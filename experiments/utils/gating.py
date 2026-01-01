import numpy as np


def smear_score_rule(obs: np.ndarray) -> float:
    """Fast smear score used for gating.

    Heuristic: column-wise median stripe energy / global std.
    obs: (H,W) float array
    """
    col_med = np.median(obs, axis=0)
    col_med = col_med - np.median(col_med)
    return float(np.std(col_med) / (np.std(obs) + 1e-8))


def pseudo_smear_mask(obs: np.ndarray, z_thr: float = 3.0) -> np.ndarray:
    """Create a rough smear mask from column stripes (fallback if GT smear mask unavailable)."""
    col = np.median(obs, axis=0)
    col = col - np.median(col)
    z = np.abs(col) / (np.std(col) + 1e-8)
    cols = (z > z_thr).astype(np.uint8)  # (W,)
    return np.tile(cols[None, :], (obs.shape[0], 1))
