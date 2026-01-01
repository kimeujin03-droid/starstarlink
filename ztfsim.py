import numpy as np
from astropy.io import fits
from .config import SimConfig

def rng_from_seed(seed: int):
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)

def gaussian_kernel_stamp_shifted(sigma_pix: float, half_size: int, shift_x: float, shift_y: float):
    """Gaussian PSF stamp with a sub-pixel center offset.

    This avoids blocky / aliased stars that occur when all sources are rounded
    to the nearest pixel center.
    """
    ax = np.arange(-half_size, half_size + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax - shift_x, ax - shift_y, indexing="xy")
    rr2 = xx * xx + yy * yy
    ker = np.exp(-0.5 * rr2 / (sigma_pix * sigma_pix)).astype(np.float32)
    s = ker.sum()
    if s > 0:
        ker /= s
    return ker


def build_psf_kernel_bank(sigma_pix: float, half_size: int, grid: int = 4):
    """Precompute a small bank of PSF kernels for sub-pixel offsets."""
    bank = {}
    steps = np.linspace(0.0, 1.0, grid, endpoint=False)
    for ix, sx in enumerate(steps):
        for iy, sy in enumerate(steps):
            bank[(ix, iy)] = gaussian_kernel_stamp_shifted(sigma_pix, half_size, sx, sy)
    return bank, steps

def add_stamp_with_kernel(img: np.ndarray, xi: int, yi: int, flux_e: float, ker: np.ndarray):
    h = (ker.shape[0] - 1) // 2
    x0, x1 = xi - h, xi + h + 1
    y0, y1 = yi - h, yi + h + 1
    if x1 <= 0 or y1 <= 0 or x0 >= img.shape[1] or y0 >= img.shape[0]:
        return
    xs0 = max(0, x0); ys0 = max(0, y0)
    xs1 = min(img.shape[1], x1); ys1 = min(img.shape[0], y1)
    kx0 = xs0 - x0; ky0 = ys0 - y0
    kx1 = kx0 + (xs1 - xs0); ky1 = ky0 + (ys1 - ys0)
    img[ys0:ys1, xs0:xs1] += flux_e * ker[ky0:ky1, kx0:kx1]


def add_psf_precomputed(img_mean_e: np.ndarray, xi: int, yi: int, flux_total_e: float, ker: np.ndarray):
    """Compatibility helper.

    Older versions of this repo referenced `add_psf_precomputed` during saturated-star
    injection. The intended behavior is simply: add (flux_total_e * ker) centered at
    integer pixel (xi, yi), cropped at image bounds.
    """
    add_stamp_with_kernel(img_mean_e, int(xi), int(yi), float(flux_total_e), ker)


# -----------------------------------------------------------------------------
# Cached GRF pool (performance): generate a small pool once per run and reuse.
# We add random flips/rolls per-frame to avoid identical backgrounds.
# -----------------------------------------------------------------------------
_GRF_POOL = {}


def _get_grf_from_pool(
    rng: np.random.Generator,
    n: int,
    beta: float,
    amp: float,
    pool_size: int,
    mode: str = "fft",
    gauss_sigma_pix: float = 28.0,
):
    """Return one GRF background realization from a small cached pool.

    Pooling amortizes expensive FFT background generation; random flips/rolls add diversity.
    """
    key = (int(n), float(beta), float(amp), int(pool_size), str(mode), float(gauss_sigma_pix))
    if key not in _GRF_POOL:
        # fixed RNG so the pool itself is deterministic across runs
        r = np.random.default_rng(12345)
        pool = [
            gaussian_random_field(r, n, beta=float(beta), amp=float(amp), mode=str(mode), gauss_sigma_pix=float(gauss_sigma_pix))
            for _ in range(int(pool_size))
        ]
        _GRF_POOL[key] = pool

    pool = _GRF_POOL[key]
    field = pool[int(rng.integers(0, len(pool)))].copy()

    # random flips
    if rng.uniform() < 0.5:
        field = field[::-1, :]
    if rng.uniform() < 0.5:
        field = field[:, ::-1]

    # random roll (wrap)
    ry = int(rng.integers(0, int(n)))
    rx = int(rng.integers(0, int(n)))
    field = np.roll(np.roll(field, ry, axis=0), rx, axis=1)
    return field
# -----------------------------------------------------------------------------
# GRF background synthesis (fast + cached)
# -----------------------------------------------------------------------------
_GRF_RFFT_CACHE: dict[tuple[int, float], np.ndarray] = {}

def _grf_sqrt_ps_rfft(n: int, beta: float) -> np.ndarray:
    """Return sqrt(power spectrum) on the rfft2 grid, cached by (n, beta)."""
    key = (int(n), float(beta))
    v = _GRF_RFFT_CACHE.get(key)
    if v is not None:
        return v
    kx = np.fft.rfftfreq(n).reshape(1, -1)         # (1, n//2+1)
    ky = np.fft.fftfreq(n).reshape(-1, 1)          # (n, 1)
    kk = np.sqrt(kx * kx + ky * ky)
    kk[0, 0] = 1.0
    ps = 1.0 / (kk ** float(beta))
    sqrt_ps = np.sqrt(ps).astype(np.float32)
    _GRF_RFFT_CACHE[key] = sqrt_ps
    return sqrt_ps

def gaussian_random_field(
    rng: np.random.Generator,
    n: int,
    beta: float = 3.0,
    amp: float = 100.0,
    mode: str = "fft",
    gauss_sigma_pix: float = 28.0,
) -> np.ndarray:
    """Generate a normalized random background field and scale by `amp` (electrons).

    mode:
      - 'fft'  : 1/f^beta GRF via rfft2/irfft2 (cached spectrum)
      - 'gauss': fast smoothed noise (requires scipy; falls back to 'fft' if unavailable)
    """
    n = int(n)

    if mode == "gauss":
        try:
            from scipy.ndimage import gaussian_filter  # type: ignore
            field = rng.normal(size=(n, n)).astype(np.float32)
            field = gaussian_filter(field, float(gauss_sigma_pix), mode="reflect").astype(np.float32)
        except Exception:
            mode = "fft"

    if mode != "gauss":
        sqrt_ps = _grf_sqrt_ps_rfft(n, float(beta))
        shape = sqrt_ps.shape  # (n, n//2+1)
        mag = rng.normal(size=shape).astype(np.float32) + 1j * rng.normal(size=shape).astype(np.float32)
        ft = mag * sqrt_ps
        field = np.fft.irfft2(ft, s=(n, n)).astype(np.float32)

    field -= float(field.mean())
    field = field / (float(field.std()) + 1e-6)
    return (field * float(amp)).astype(np.float32)

def pick_streak_count(rng, probs):
    u = rng.uniform()
    c = 0.0
    for k, p in enumerate(probs):
        c += p
        if u <= c:
            return k
    return len(probs) - 1

def altitude_to_brightness_scale(alt_km, cfg: SimConfig):
    a0, a1 = cfg.alt_km_min, cfg.alt_km_max
    alt = np.clip(alt_km, a0, a1)
    t = (a1 - alt) / (a1 - a0)  # 0 high alt, 1 low alt
    return np.power(t, cfg.alt_bright_gamma)

def render_streak_addition_and_mask(n, angle_rad, anchor_xy, peak_e, fwhm_pix, k_sigma: float):
    """Render a long streak as an (effectively) infinite line with a *finite* mask.

    Important: a pure Gaussian is >0 everywhere, so using (streak_e > 0) would
    erroneously mark the entire frame as streak. We therefore apply a geometric
    cutoff dist <= k_sigma*sigma to define a physically meaningful mask width.
    """
    vx, vy = np.cos(angle_rad), np.sin(angle_rad)
    px, py = anchor_xy
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    dx = xx - px
    dy = yy - py
    dist = np.abs(vx*dy - vy*dx)
    sigma = (fwhm_pix / 2.35482)
    w = (k_sigma * sigma)
    m = (dist <= w)
    # Gaussian profile within the cutoff
    prof = np.zeros((n, n), dtype=np.float32)
    if sigma > 0:
        prof[m] = np.exp(-0.5 * (dist[m]/sigma)**2).astype(np.float32)
    return (peak_e * prof).astype(np.float32), m.astype(np.uint8)

def make_streak_bundle(rng, cfg: SimConfig, nstreak: int):
    n = cfg.size
    streaks = []
    if nstreak <= 0:
        return streaks

    base_angle = rng.uniform(0, 2*np.pi)
    base_px = rng.uniform(0, n)
    base_py = rng.uniform(0, n)
    # perpendicular direction (for bundle separation)
    ux, uy = -np.sin(base_angle), np.cos(base_angle)

    for i in range(nstreak):
        if nstreak >= 3:
            # For 3+ streaks: make a rail-like bundle that is clearly separated.
            # Too-small separations look like a thick single streak.
            ang = base_angle + rng.normal(0, np.deg2rad(0.4))
            sep = rng.uniform(18.0, 42.0)
            off = (i - (nstreak - 1) / 2.0) * sep
            px = base_px + off * ux
            py = base_py + off * uy
            # mild along-track jitter only
            px += rng.normal(0, 3.0) * np.cos(base_angle)
            py += rng.normal(0, 3.0) * np.sin(base_angle)
        else:
            # For 1~2 streaks: fully independent (prevents "stuck" streaks)
            ang = rng.uniform(0, 2*np.pi)
            px = rng.uniform(0, n)
            py = rng.uniform(0, n)

        alt_km = rng.uniform(cfg.alt_km_min, cfg.alt_km_max)
        scale = altitude_to_brightness_scale(alt_km, cfg)
        peak = cfg.streak_peak_e_min + scale * (cfg.streak_peak_e_max - cfg.streak_peak_e_min)
        peak *= rng.uniform(0.8, 1.2)
        fwhm = rng.uniform(cfg.streak_fwhm_pix_min, cfg.streak_fwhm_pix_max)

        streaks.append((float(ang), (float(px), float(py)), float(alt_km), float(peak), float(fwhm)))
    return streaks

def _pick_nstars_mixture(rng, cfg: SimConfig) -> int:
    """Mixture over star density so fields aren't all the same."""
    u = rng.uniform()
    if u < 0.15:
        return int(rng.integers(120, 520))
    if u < 0.35:
        return int(rng.integers(1200, 2600))
    return int(rng.integers(cfg.nstars_min, cfg.nstars_max + 1))

def _sample_star_fluxes(rng, nstars: int, cfg: SimConfig):
    """Power-law mixture for fluxes (faint-dominated with a small bright tail).

    Returns fluxes in electrons (per exposure) for each star stamp.
    """
    # slight variation in slope per frame
    alpha = float(rng.uniform(1.45, 1.85))
    fmin = float(cfg.star_flux_min_e)
    # keep most stars below saturation; allow a small bright tail
    fmax_main = float(min(cfg.star_flux_max_e, 0.75 * cfg.fullwell_e))
    fmax_tail = float(min(1.10 * cfg.fullwell_e, 2.0 * fmax_main))

    u = rng.uniform(size=nstars).astype(np.float32)
    pow_ = (1.0 - alpha)
    fminp = fmin**pow_
    fmaxp = fmax_main**pow_
    flux = (u*(fmaxp - fminp) + fminp) ** (1.0/pow_)

    # bright tail: upgrade a small random subset
    ntail = max(1, int(0.03 * nstars))
    idx = rng.choice(nstars, size=ntail, replace=False)
    u2 = rng.uniform(size=ntail).astype(np.float32)
    fmaxp2 = fmax_tail**pow_
    flux[idx] = (u2*(fmaxp2 - fminp) + fminp) ** (1.0/pow_)
    return flux.astype(np.float32)

def apply_blooming_from_saturation(obs_e, sat_mask, rng, cfg: SimConfig):
    """Simple blooming: redistribute a fraction of excess charge along the column."""
    n = cfg.size
    bloom_e = np.zeros((n, n), dtype=np.float32)
    bloom_mask = np.zeros((n, n), dtype=np.uint8)

    if (not sat_mask.any()) or (rng.uniform() > cfg.bloom_conditional_prob):
        return bloom_e, bloom_mask, False

    ys, xs = np.where(sat_mask)
    # limit number of seeds for speed
    if ys.size > 60:
        idx = rng.choice(ys.size, size=60, replace=False)
        ys = ys[idx]; xs = xs[idx]

    decay = float(cfg.bloom_decay_pix)
    y = np.arange(n, dtype=np.float32)
    for y0, x0 in zip(ys, xs):
        excess = float(max(obs_e[y0, x0] - cfg.fullwell_e, 0.0))
        if excess <= 0:
            continue
        amp = cfg.bloom_strength * excess
        trail = amp * np.exp(-np.abs(y - float(y0)) / decay)
        bloom_e[:, x0] += trail.astype(np.float32)

    bloom_mask[bloom_e >= float(cfg.bloom_mask_thresh_e)] = 1
    return bloom_e, bloom_mask, True

def apply_smear_from_saturated_columns(obs_e, sat_mask, rng, cfg: SimConfig):
    """Readout smear: conditional on saturation; creates long trails along saturated columns."""
    n = cfg.size
    smear_e = np.zeros((n, n), dtype=np.float32)
    smear_mask = np.zeros((n, n), dtype=np.uint8)

    if (not sat_mask.any()) or (rng.uniform() > cfg.smear_conditional_prob):
        return smear_e, smear_mask, False

    cols = np.unique(np.where(sat_mask)[1])
    if cols.size == 0:
        return smear_e, smear_mask, False

    # cap number of columns
    max_cols = int(getattr(cfg, 'smear_max_cols', 20))
    if cols.size > max_cols:
        cols = rng.choice(cols, size=max_cols, replace=False)

    decay = float(cfg.smear_decay_pix)
    y = np.arange(n, dtype=np.float32)
    for x0 in cols:
        ys = np.where(sat_mask[:, x0])[0]
        y0 = float(np.median(ys)) if ys.size else float(rng.uniform(0, n))
        excess_col = float(np.maximum(obs_e[:, x0] - float(cfg.fullwell_e), 0.0).sum())
        if excess_col <= 0:
            continue
        amp = float(cfg.smear_strength) * excess_col
        # symmetric exponential trail (readout smear proxy)
        trail = amp * np.exp(-np.abs(y - y0) / decay)
        smear_e[:, x0] += trail.astype(np.float32)

    smear_mask[smear_e >= float(cfg.smear_mask_thresh_e)] = 2
    return smear_e, smear_mask, True

def apply_smear_from_bright_pixels(clean_mean_e, rng, cfg: SimConfig):
    # legacy API (kept for backward-compat in older notebooks)
    return apply_smear_from_saturated_columns(clean_mean_e, clean_mean_e >= cfg.fullwell_e, rng, cfg)

def simulate_one(seed: int, cfg: SimConfig):
    rng = rng_from_seed(seed)
    n = cfg.size

    sigma_pix = (cfg.psf_fwhm_arcsec / cfg.pixscale_arcsec) / 2.35482
    ker_bank, steps = build_psf_kernel_bank(float(sigma_pix), half_size=6, grid=4)

    sky = np.full((n,n), cfg.sky_e_mean, dtype=np.float32)
    if bool(getattr(cfg, "use_grf_background", False)) and (rng.uniform() < float(getattr(cfg, "grf_prob", 1.0))):
        sky += _get_grf_from_pool(
            rng,
            n,
            beta=float(getattr(cfg, "grf_beta", 3.0)),
            amp=float(getattr(cfg, "grf_amp_e", 0.0)),
            pool_size=int(getattr(cfg, "grf_pool_size", 8)),
            mode=str(getattr(cfg, "grf_mode", "fft")),
            gauss_sigma_pix=float(getattr(cfg, "grf_gauss_sigma_pix", 28.0)),
        )

    nstars = _pick_nstars_mixture(rng, cfg)
    fluxes = _sample_star_fluxes(rng, nstars, cfg)

    xs = rng.uniform(0, n, size=nstars).astype(np.float32)
    ys = rng.uniform(0, n, size=nstars).astype(np.float32)

    stars_mean = np.zeros((n,n), dtype=np.float32)
    grid = len(steps)
    for x, y, f in zip(xs, ys, fluxes):
        # subpixel-aware PSF placement (quantized)
        xi = int(np.floor(x)); yi = int(np.floor(y))
        fx = float(x - xi); fy = float(y - yi)
        ix = int(min(max(int(np.floor(fx * grid)), 0), grid - 1))
        iy = int(min(max(int(np.floor(fy * grid)), 0), grid - 1))
        ker = ker_bank[(ix, iy)]
        add_stamp_with_kernel(stars_mean, xi, yi, float(f), ker)

    # Optional: inject rare saturated stars so that SATUR/BLOOM/SMEAR events are not stuck at 0%.
    # We deliberately set the *peak pixel* to exceed FULLWELL using the selected PSF kernel's maximum.
    nsatstars = inject_saturated_stars(stars_mean, rng, cfg, ker_bank, steps)

    clean_mean_e = np.clip(sky + stars_mean, 0.0, None).astype(np.float32)

    nstreak = pick_streak_count(rng, cfg.streak_probs)
    streaks = make_streak_bundle(rng, cfg, nstreak)

    streak_e = np.zeros((n,n), dtype=np.float32)
    streak_mask = np.zeros((n, n), dtype=np.uint8)
    for ang, (px, py), alt_km, peak, fwhm in streaks:
        add, m = render_streak_addition_and_mask(
            n,
            ang,
            (px, py),
            peak,
            fwhm,
            k_sigma=float(cfg.streak_mask_k_sigma),
        )
        streak_e += add
        streak_mask = np.maximum(streak_mask, m)

    artifact_mean_e = streak_e

    # Shared base noise
    base_e = rng.poisson(lam=np.clip(clean_mean_e, 0, None)).astype(np.float32)
    base_e += rng.normal(0.0, cfg.rdnoise_e, size=(n,n)).astype(np.float32)

    clean_e = base_e.copy()
    obs_e = base_e.copy()

    if artifact_mean_e.max() > 0:
        art_e = rng.poisson(lam=np.clip(artifact_mean_e, 0, None)).astype(np.float32)
        obs_e += art_e

    # --- Saturation triggers (based on OBS, so bloom/smear correlate with artifacts) ---
    sat_mask = (obs_e >= cfg.fullwell_e)
    sat_on = bool(sat_mask.any())

    # bloom and smear artifacts
    bloom_e, bloom_mask, bloom_on = apply_blooming_from_saturation(obs_e, sat_mask, rng, cfg)
    obs_e = obs_e + bloom_e

    smear_e, smear_mask, smear_on = apply_smear_from_saturated_columns(obs_e, sat_mask, rng, cfg)
    obs_e = obs_e + smear_e

    other_mask = np.maximum(bloom_mask, smear_mask).astype(np.uint8)

    # clip to fullwell for stability
    clean_e = np.minimum(clean_e, cfg.fullwell_e).astype(np.float32)
    obs_e = np.minimum(obs_e, cfg.fullwell_e).astype(np.float32)

    sat_adu = (cfg.fullwell_e / cfg.gain_e_per_adu + cfg.bias_adu)
    clean_adu = (clean_e / cfg.gain_e_per_adu + cfg.bias_adu).astype(np.float32)
    obs_adu = (obs_e / cfg.gain_e_per_adu + cfg.bias_adu).astype(np.float32)
    clean_adu = np.minimum(clean_adu, sat_adu).astype(np.float32)
    obs_adu = np.minimum(obs_adu, sat_adu).astype(np.float32)

    meta = {
        "seed": int(seed),
        "nstars": int(nstars),
        "nsatstars": int(nsatstars),
        "nstreak": int(nstreak),
        "satur": bool(sat_on),
        "bloom": bool(bloom_on),
        "smear": bool(smear_on),
        "streak_area": int(streak_mask.sum()),
        "streaks": [
            {"angle_deg": float(np.rad2deg(s[0])),
             "alt_km": float(s[2]),
             "peak_e": float(s[3]),
             "fwhm_pix": float(s[4])}
            for s in streaks
        ],
    }
    hdr = build_header(cfg, meta)
    return obs_adu, clean_adu, streak_mask, other_mask, meta, hdr

def build_header(cfg: SimConfig, meta: dict):
    hdr = fits.Header()
    hdr["BUNIT"] = (cfg.bunit, "data units")
    hdr["GAIN"] = (float(cfg.gain_e_per_adu), "e-/ADU")
    hdr["RDNOISE"] = (float(cfg.rdnoise_e), "read noise (e- rms)")
    hdr["EXPTIME"] = (float(cfg.exptime_s), "exposure time (s)")
    hdr["FILTER"] = (cfg.filt, "filter")
    hdr["PIXSCALE"] = (float(cfg.pixscale_arcsec), "arcsec/pixel")
    hdr["PSFFWHM"] = (float(cfg.psf_fwhm_arcsec), "PSF FWHM (arcsec)")
    hdr["SKYMEAN"] = (float(cfg.sky_e_mean), "sky mean (e-/pix)")
    hdr["FULLWELL"] = (float(cfg.fullwell_e), "full well capacity (e-)")
    hdr["SATADU"] = (float(cfg.fullwell_e / cfg.gain_e_per_adu + cfg.bias_adu), "saturation level (ADU)")

    # compact, easy-to-parse keys
    hdr["SEED"] = int(meta.get("seed", -1))
    hdr["NSTARS"] = int(meta.get("nstars", -1))
    hdr["NSTRK"] = int(meta.get("nstreak", -1))
    hdr["SATUR"] = bool(meta.get("satur", False))
    hdr["BLOOM"] = bool(meta.get("bloom", False))
    hdr["SMEAR"] = bool(meta.get("smear", False))
    hdr["STRKAREA"] = int(meta.get("streak_area", -1))

    # extended metadata (standardized)
    hdr["HIERARCH SIM VERSION"] = "v11_2"
    hdr["HIERARCH SIM STREAK_PROBS"] = str(tuple(cfg.streak_probs))
    hdr["HIERARCH SIM STREAK_MASK_KSIG"] = float(cfg.streak_mask_k_sigma)
    hdr["HIERARCH SIM SMEAR_P"] = float(cfg.smear_conditional_prob)
    hdr["HIERARCH SIM BLOOM_P"] = float(cfg.bloom_conditional_prob)
    hdr["HIERARCH SIM SATSTAR_P"] = float(getattr(cfg, "sat_star_frame_prob", 0.0))
    hdr["HIERARCH SIM SMEAR_MAXC"] = int(getattr(cfg, "smear_max_cols", 0))
    hdr["HIERARCH SIM SMEAR_MTH"] = float(getattr(cfg, "smear_mask_thresh_e", 0.0))
    hdr["HIERARCH SIM BLOOM_MTH"] = float(getattr(cfg, "bloom_mask_thresh_e", 0.0))

    # clean HISTORY/COMMENT (avoid long cards)
    hdr.add_history("Synthetic ZTF-like frame (NumPy) for U-Net artifact removal.")
    hdr.add_history("Primary: satellite streak(s) + optional saturation-driven bloom/smear.")
    hdr.add_comment("Ghost/halo/large galaxies are excluded (out-of-scope in this dataset).")
    return hdr

def write_fits(path, data, hdr: fits.Header):
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True, output_verify="silentfix")

def inject_saturated_stars(img_mean_e: np.ndarray, rng, cfg: SimConfig, ker_bank: np.ndarray, steps: np.ndarray):
    """Inject 1–2 extremely bright stars such that the *peak pixel* can exceed FULLWELL.

    Purpose: allow rare but non-zero SATUR/BLOOM/SMEAR events in a controlled way.
    This models occasional saturated stars in real ZTF frames without introducing halo/ghost.
    """
    if rng.uniform() > float(cfg.sat_star_frame_prob):
        return 0

    n = cfg.size
    nsat = int(rng.integers(1, int(cfg.sat_star_max_count) + 1))
    grid = len(steps)

    for _ in range(nsat):
        x = float(rng.uniform(0.0, n))
        y = float(rng.uniform(0.0, n))
        xi = int(np.floor(x)); yi = int(np.floor(y))
        fx = float(x - xi); fy = float(y - yi)
        ix = int(min(max(int(np.floor(fx * grid)), 0), grid - 1))
        iy = int(min(max(int(np.floor(fy * grid)), 0), grid - 1))

        # ker_bank is a dict keyed by (ix, iy)
        ker = ker_bank.get((ix, iy), None)
        ker_max = float(np.max(ker)) if ker is not None else 0.0
        if ker_max <= 0:
            continue

        peak_factor = float(rng.uniform(cfg.sat_star_peak_factor_min, cfg.sat_star_peak_factor_max))
        flux_total = float(cfg.fullwell_e) * peak_factor / ker_max  # ensures peak >= FULLWELL

        # add star using the selected PSF kernel
        add_psf_precomputed(img_mean_e, xi, yi, flux_total, ker)

    return nsat    # number of injected saturated stars

