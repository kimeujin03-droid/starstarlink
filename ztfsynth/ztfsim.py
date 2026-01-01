import os
import zlib
import numpy as np
from astropy.io import fits
import galsim

try:
    from .config import SimConfig
except ImportError:
    from config import SimConfig


# ============================================================
# RNG
# ============================================================
def rng_from_seed(seed: int):
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


# ============================================================
# Optional GRF background cache (cfg.use_grf_background)
# ============================================================
_GRF_CACHE = {}  # key -> list[np.ndarray]


def _stable_int_from_key(key_tuple) -> int:
    s = repr(key_tuple).encode("utf-8")
    return int(zlib.crc32(s) & 0xFFFFFFFF)


def _make_grf_fft(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian random field with power spectrum ~ 1/f^beta (normalized)."""
    kx = np.fft.fftfreq(n)[:, None]
    ky = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(kx * kx + ky * ky)
    k[0, 0] = 1.0

    P = 1.0 / (k ** beta)

    re = rng.normal(size=(n, n))
    im = rng.normal(size=(n, n))
    z = (re + 1j * im) * np.sqrt(P)

    field = np.fft.ifft2(z).real
    field -= field.mean()
    std = field.std()
    if std > 0:
        field /= std
    return field.astype(np.float32)


def _make_grf_gauss(n: int, sigma_pix: float, rng: np.random.Generator) -> np.ndarray:
    """Approx GRF: white noise smoothed by Gaussian blur in Fourier domain (normalized)."""
    white = rng.normal(size=(n, n)).astype(np.float32)

    kx = np.fft.fftfreq(n)[:, None]
    ky = np.fft.fftfreq(n)[None, :]
    k2 = kx * kx + ky * ky

    H = np.exp(-2.0 * (np.pi ** 2) * (sigma_pix ** 2) * k2).astype(np.float32)
    field = np.fft.ifft2(np.fft.fft2(white) * H).real.astype(np.float32)

    field -= field.mean()
    std = field.std()
    if std > 0:
        field /= std
    return field


def _get_grf_pool(cfg: SimConfig):
    key = (
        int(cfg.size),
        float(cfg.grf_beta),
        str(cfg.grf_mode),
        float(cfg.grf_gauss_sigma_pix),
        int(cfg.grf_pool_size),
        float(cfg.grf_amp_e),
    )
    if key in _GRF_CACHE:
        return _GRF_CACHE[key]

    seed0 = _stable_int_from_key(key)
    rng = rng_from_seed(seed0)

    pool = []
    for _ in range(int(cfg.grf_pool_size)):
        if str(cfg.grf_mode).lower() == "gauss":
            f = _make_grf_gauss(int(cfg.size), float(cfg.grf_gauss_sigma_pix), rng)
        else:
            f = _make_grf_fft(int(cfg.size), float(cfg.grf_beta), rng)
        pool.append(f)

    _GRF_CACHE[key] = pool
    return pool

def segment_through_bbox(cx: float, cy: float, ang: float, n: int):
    """
    Line through (cx,cy) with direction (cos ang, sin ang).
    Return endpoints on the image bbox [1,n]x[1,n], plus midpoint and length.
    """
    dx = float(np.cos(ang))
    dy = float(np.sin(ang))
    eps = 1e-12

    pts = []

    # x = 1, x = n
    if abs(dx) > eps:
        t = (1.0 - cx) / dx
        y = cy + t * dy
        if 1.0 <= y <= n:
            pts.append((1.0, float(y)))

        t = (float(n) - cx) / dx
        y = cy + t * dy
        if 1.0 <= y <= n:
            pts.append((float(n), float(y)))

    # y = 1, y = n
    if abs(dy) > eps:
        t = (1.0 - cy) / dy
        x = cx + t * dx
        if 1.0 <= x <= n:
            pts.append((float(x), 1.0))

        t = (float(n) - cy) / dy
        x = cx + t * dx
        if 1.0 <= x <= n:
            pts.append((float(x), float(n)))

    # Fallback (거의 안 뜸): 충분히 긴 선분
    if len(pts) < 2:
        L = 2.8 * n
        x1 = cx - 0.5 * L * dx
        y1 = cy - 0.5 * L * dy
        x2 = cx + 0.5 * L * dx
        y2 = cy + 0.5 * L * dy
    else:
        # choose the farthest pair
        best = None
        best_d2 = -1.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                x1_, y1_ = pts[i]
                x2_, y2_ = pts[j]
                d2 = (x2_ - x1_) ** 2 + (y2_ - y1_) ** 2
                if d2 > best_d2:
                    best_d2 = d2
                    best = (x1_, y1_, x2_, y2_)
        x1, y1, x2, y2 = best

    mx = 0.5 * (x1 + x2)
    my = 0.5 * (y1 + y2)
    L = float(np.hypot(x2 - x1, y2 - y1))

    # 경계 픽셀까지 “확실히” 닿게 살짝 늘림
    L = L + 4.0

    return (float(x1), float(y1), float(x2), float(y2), float(mx), float(my), float(L))



# ============================================================
# Streak sampling (0..5 per cfg.streak_probs) + altitude brightness proxy
# ============================================================
def sample_streaks(rng: np.random.Generator, cfg: SimConfig, bright_star_coords):
    probs = np.array(cfg.streak_probs, dtype=np.float64)
    probs = probs / probs.sum()
    k = int(rng.choice(np.arange(len(probs)), p=probs))

    # “스트릭 너무 안 나옴” 대응: 0개 뽑혔으면 일정 확률로 1개로 강제
    force_p = float(getattr(cfg, "force_streak_prob", 0.85))
    if k == 0 and rng.uniform() < force_p:
        k = 1

    streaks = []
    n = int(cfg.size)

    for _ in range(k):
        if bright_star_coords and rng.uniform() < 0.7:
            cx, cy = bright_star_coords[rng.integers(len(bright_star_coords))]
        else:
            cx, cy = float(rng.uniform(1.0, n)), float(rng.uniform(1.0, n))

        angle = float(rng.uniform(0.0, np.pi))

        alt_km = float(rng.uniform(cfg.alt_km_min, cfg.alt_km_max))
        alt_factor = (float(cfg.alt_km_min) / alt_km) ** float(cfg.alt_bright_gamma)

        peak_e = float(rng.uniform(cfg.streak_peak_e_min, cfg.streak_peak_e_max) * alt_factor)
        peak_e *= float(getattr(cfg, "streak_boost", 1.0))  # 필요하면 1.5~3.0 올려

        fwhm_pix = float(rng.uniform(cfg.streak_fwhm_pix_min, cfg.streak_fwhm_pix_max))

        # 핵심: bbox 끝까지 관통하는 선분으로 재정의
        x1, y1, x2, y2, mx, my, length_pix = segment_through_bbox(cx, cy, angle, n)

        streaks.append(
            dict(
                cx=mx, cy=my, angle=angle,
                x1=x1, y1=y1, x2=x2, y2=y2,
                length_pix=length_pix, fwhm_pix=fwhm_pix,
                peak_e=peak_e, alt_km=alt_km
            )
        )

    return streaks


def streak_mask_from_geometry(cfg: SimConfig, streaks):
    """Mask includes pixels within k*sigma from the line segment. (cfg.streak_mask_k_sigma)"""
    n = int(cfg.size)
    ksig = float(cfg.streak_mask_k_sigma)

    xs = (np.arange(n, dtype=np.float32) + 1.0)[None, :]
    ys = (np.arange(n, dtype=np.float32) + 1.0)[:, None]

    mask = np.zeros((n, n), dtype=np.uint8)

    for st in streaks:
        cx = float(st["cx"])
        cy = float(st["cy"])
        ang = float(st["angle"])
        L = float(st["length_pix"])

        sigma_pix = float(st["fwhm_pix"]) / 2.35482
        halfL = 0.5 * L
        halfW = ksig * sigma_pix

        ca = np.cos(ang)
        sa = np.sin(ang)

        dx = xs - cx
        dy = ys - cy

        u = dx * ca + dy * sa
        v = -dx * sa + dy * ca

        m = (np.abs(u) <= halfL) & (np.abs(v) <= halfW)
        mask[m] = 1

    return mask


# ============================================================
# Physical bloom/smear (conditional on saturation) -> other_mask
# other_mask: 0 none, 1 bloom, 2 smear, 3 both
# ============================================================
def apply_bloom_smear_physical(obs_e: np.ndarray, sat_mask: np.ndarray,
                              rng: np.random.Generator, cfg: SimConfig):
    n = int(cfg.size)
    other_mask = np.zeros((n, n), dtype=np.uint8)

    if not sat_mask.any():
        return obs_e, other_mask, {"bloom": False, "smear": False}

    fullwell = float(cfg.fullwell_e)

    # ---------- Bloom ----------
    did_bloom = False
    if rng.uniform() < float(cfg.bloom_conditional_prob):
        did_bloom = True
        strength = float(cfg.bloom_strength)       # fraction of excess redistributed
        decay = float(cfg.bloom_decay_pix)
        thresh = float(cfg.bloom_mask_thresh_e)

        ys, xs = np.where(sat_mask)
        add = np.zeros_like(obs_e, dtype=np.float32)

        excess = np.maximum(obs_e[ys, xs] - fullwell, 0.0).astype(np.float32)
        inj = excess * strength

        rmax = int(min(n, max(8, 4 * decay)))
        for y0, x0, q in zip(ys, xs, inj):
            if q <= 0:
                continue
            for dy in range(-rmax, rmax + 1):
                if dy == 0:
                    continue
                y = y0 + dy
                if 0 <= y < n:
                    w = np.exp(-abs(dy) / max(1e-6, decay))
                    add[y, x0] += q * w

        obs_e = obs_e + add
        other_mask[add > thresh] |= 1

    # ---------- Smear ----------
    did_smear = False
    if rng.uniform() < float(cfg.smear_conditional_prob):
        did_smear = True
        strength = float(cfg.smear_strength)
        decay = float(cfg.smear_decay_pix)
        thresh = float(cfg.smear_mask_thresh_e)

        cols = np.unique(np.where(sat_mask)[1])
        if cols.size > 0:
            rng.shuffle(cols)
            cols = cols[: int(cfg.smear_max_cols)]

            add = np.zeros_like(obs_e, dtype=np.float32)

            for x0 in cols:
                ys = np.where(sat_mask[:, x0])[0]
                if ys.size == 0:
                    continue
                y_anchor = float(np.mean(ys))
                amp = float(fullwell) * strength

                y = np.arange(n, dtype=np.float32)
                prof = amp * np.exp(-np.abs(y - y_anchor) / max(1e-6, decay))
                add[:, x0] += prof

            obs_e = obs_e + add
            other_mask[add > thresh] |= 2

    return obs_e, other_mask, {"bloom": did_bloom, "smear": did_smear}


# ============================================================
# Hard destruction (catastrophic info loss) -> other_mask
# Uses defaults if cfg lacks hard_* fields.
# ============================================================
def apply_destruction_hard(obs_e: np.ndarray, sat_mask: np.ndarray,
                           rng: np.random.Generator, cfg: SimConfig):
    n = int(cfg.size)
    other_mask = np.zeros((n, n), dtype=np.uint8)

    if not sat_mask.any():
        return obs_e, other_mask, {"hard": False}

    p = float(getattr(cfg, "hard_conditional_prob", 0.35))
    if rng.uniform() >= p:
        return obs_e, other_mask, {"hard": False}

    fullwell = float(cfg.fullwell_e)

    bl_min = int(getattr(cfg, "hard_bloom_len_min", 15))
    bl_max = int(getattr(cfg, "hard_bloom_len_max", 30))
    bw_min = int(getattr(cfg, "hard_bloom_w_min", 1))
    bw_max = int(getattr(cfg, "hard_bloom_w_max", 3))

    smear_lv_min = float(getattr(cfg, "hard_smear_level_min", 0.5))
    smear_lv_max = float(getattr(cfg, "hard_smear_level_max", 0.8))

    cols = np.unique(np.where(sat_mask)[1])
    if cols.size == 0:
        return obs_e, other_mask, {"hard": False}

    did_any = False

    for x0 in cols:
        if rng.uniform() > 0.4:
            continue
        did_any = True

        ys = np.where(sat_mask[:, x0])[0]
        # Hard bloom columns: rectangular vertical pillars around each saturated row
        for y0 in ys:
            bl = int(rng.integers(bl_min, bl_max + 1))
            bw = int(rng.integers(bw_min, bw_max + 1))
            y_s, y_e = max(0, y0 - bl), min(n, y0 + bl)

            for dx in range(-bw, bw + 1):
                xx = x0 + dx
                if 0 <= xx < n:
                    obs_e[y_s:y_e, xx] = fullwell * 1.1  # destroy info
                    other_mask[y_s:y_e, xx] |= 1

        # Hard smear: full-column overwrite
        if rng.uniform() < 0.5:
            lvl = float(rng.uniform(smear_lv_min, smear_lv_max))
            obs_e[:, x0] = fullwell * lvl
            other_mask[:, x0] |= 2

    return obs_e, other_mask, {"hard": did_any}


# ============================================================
# Core simulation
# ============================================================
def simulate_one(seed: int, cfg: SimConfig):
    rng = rng_from_seed(seed)
    n = int(cfg.size)
    scale = float(cfg.pixscale_arcsec)

    # ---- 1) Base image: sky + read noise (kept in clean) ----
    base = galsim.ImageF(n, n, scale=scale)
    base += float(cfg.sky_e_mean)

    gs_rng = galsim.BaseDeviate(int(seed) & 0x7FFFFFFF)
    base.addNoise(galsim.GaussianNoise(rng=gs_rng, sigma=float(cfg.rdnoise_e)))

    # Optional GRF background
    if bool(cfg.use_grf_background) and (rng.uniform() < float(cfg.grf_prob)):
        pool = _get_grf_pool(cfg)
        field = pool[int(rng.integers(0, len(pool)))]
        base.array[:] = base.array + field * float(cfg.grf_amp_e)

    # PSF
    psf = galsim.Gaussian(fwhm=float(cfg.psf_fwhm_arcsec))

    # ---- 2) Star field ----
    nstars = int(rng.integers(int(cfg.nstars_min), int(cfg.nstars_max) + 1))

    fluxes = (rng.uniform(0.1, 1.0, nstars) ** -1.7) * float(cfg.star_flux_min_e)
    fluxes = np.clip(fluxes, 0.0, float(cfg.star_flux_max_e)).astype(np.float32)

    bright_coords = []
    for f in fluxes:
        x = float(rng.uniform(1.0, n))
        y = float(rng.uniform(1.0, n))
        psf.withFlux(float(f)).drawImage(
            image=base, center=galsim.PositionD(x, y), method="auto", add_to_image=True
        )
        if f > float(cfg.fullwell_e) * 0.4:
            bright_coords.append((x, y))

    # Saturation-driven injected bright stars (to ensure rare SAT/BLOOM/SMEAR)
    did_sat_inject = False
    if rng.uniform() < float(cfg.sat_star_frame_prob):
        did_sat_inject = True

        sigma_arcsec = float(cfg.psf_fwhm_arcsec) / 2.35482
        sigma_pix = sigma_arcsec / scale
        denom = max(1e-6, 2.0 * np.pi * (sigma_pix ** 2))

        ksat = int(rng.integers(1, int(cfg.sat_star_max_count) + 1))
        for _ in range(ksat):
            x = float(rng.uniform(1.0, n))
            y = float(rng.uniform(1.0, n))
            peak_factor = float(rng.uniform(cfg.sat_star_peak_factor_min, cfg.sat_star_peak_factor_max))
            target_peak = peak_factor * float(cfg.fullwell_e)
            flux = target_peak * denom

            psf.withFlux(float(flux)).drawImage(
                image=base, center=galsim.PositionD(x, y), method="auto", add_to_image=True
            )
            bright_coords.append((x, y))

    clean_e = base.array.copy()  # clean retains noise/background/stars

    # ---- 3) Streak synthesis + streak mask ----
    streaks = sample_streaks(rng, cfg, bright_coords)
    streak_mask = streak_mask_from_geometry(cfg, streaks)

    streak_img = galsim.ImageF(n, n, scale=scale)

    for st in streaks:
        length_pix = float(st["length_pix"])
        fwhm_pix = float(st["fwhm_pix"])
        width_pix = max(1.0, fwhm_pix)

        area_pix = length_pix * width_pix
        total_flux_e = float(st["peak_e"]) * area_pix

        width_arcsec = length_pix * scale
        height_arcsec = width_pix * scale

        line = galsim.Box(width=width_arcsec, height=height_arcsec, flux=total_flux_e)
        line = line.rotate(float(st["angle"]) * galsim.radians)
        
        line = galsim.Convolve([line, psf])

        tmp = galsim.ImageF(n, n, scale=scale)
        line.drawImage(
            image=tmp,
            center=galsim.PositionD(float(st["cx"]), float(st["cy"])),
            method="auto",
            add_to_image=True,
        )
        streak_img += tmp

    obs_e = clean_e + streak_img.array

    # ---- 4) Saturation mask (pre-artifact) ----
    fullwell = float(cfg.fullwell_e)
    sat_mask = obs_e >= fullwell

    # ---- 5) Artifacts mode selection ----
    mode = str(getattr(cfg, "artifact_mode", "physical")).lower()  # "physical" | "hard" | "both"
    other_mask = np.zeros((n, n), dtype=np.uint8)

    did_bloom = did_smear = did_hard = False

    if mode in ("physical", "both"):
        obs_e, om_phys, meta_phys = apply_bloom_smear_physical(obs_e, sat_mask, rng, cfg)
        other_mask |= om_phys
        did_bloom = bool(meta_phys.get("bloom", False))
        did_smear = bool(meta_phys.get("smear", False))

    # re-evaluate saturation after physical injection (can expand sat regions)
    sat_mask = obs_e >= fullwell

    if mode in ("hard", "both"):
        obs_e, om_hard, meta_hard = apply_destruction_hard(obs_e, sat_mask, rng, cfg)
        other_mask |= om_hard
        did_hard = bool(meta_hard.get("hard", False))

    # ---- 6) Clip to fullwell (readout) ----
    obs_e = np.clip(obs_e, 0.0, fullwell).astype(np.float32)
    clean_e = np.clip(clean_e, 0.0, fullwell).astype(np.float32)

    # ---- 7) ADU conversion ----
    gain = float(cfg.gain_e_per_adu)
    bias = float(cfg.bias_adu)
    obs_adu = (obs_e / gain + bias).astype(np.float32)
    clean_adu = (clean_e / gain + bias).astype(np.float32)

    # ---- 8) Header + meta ----
    hdr = fits.Header()
    hdr["SEED"] = int(seed)
    hdr["BUNIT"] = str(getattr(cfg, "bunit", "ADU"))
    hdr["SATUR"] = bool((obs_e >= fullwell).any())
    hdr["NSTAR"] = int(nstars)
    hdr["NSTRK"] = int(len(streaks))
    hdr["GRF"] = bool(cfg.use_grf_background)
    hdr["SATINJ"] = bool(did_sat_inject)
    hdr["ARTMODE"] = mode
    hdr["BLOOM"] = bool(did_bloom)
    hdr["SMEAR"] = bool(did_smear)
    hdr["HARD"] = bool(did_hard)

    meta = {
        "satur": bool((obs_e >= fullwell).any()),
        "sat_injected": bool(did_sat_inject),
        "artifact_mode": mode,
        "bloom": bool(did_bloom),
        "smear": bool(did_smear),
        "hard": bool(did_hard),
        "nstars": int(nstars),
        "nstreaks": int(len(streaks)),
    }

    return obs_adu, clean_adu, streak_mask, other_mask, meta, hdr


# ============================================================
# FITS writer
# ============================================================
def write_fits(path, data, hdr):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)
