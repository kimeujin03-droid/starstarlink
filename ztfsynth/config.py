from dataclasses import dataclass

@dataclass
class SimConfig:
    # image geometry
    size: int = 512
    pixscale_arcsec: float = 1.0

    # exposure / instrument (ZTF-ish)
    exptime_s: float = 30.0
    # Use ZTF r-band by default (30s)
    filt: str = "r"
    gain_e_per_adu: float = 1.5
    rdnoise_e: float = 10.0
    bias_adu: float = 1000.0

    # detector saturation / full well (approx ZTF-ish; used for bloom/smear triggers)
    fullwell_e: float = 9.0e4

    # PSF (Gaussian approx)
    psf_fwhm_arcsec: float = 2.0

    # sky background (mean electrons per pixel over exptime)
    sky_e_mean: float = 1550.0

    # star field
    # baseline (actual nstars is sampled from a mixture to allow sparse/dense fields)
    nstars_min: int = 300
    nstars_max: int = 1400
    # flux distribution: power-law in flux (many faint, few bright)
    star_flux_min_e: float = 25.0
    star_flux_max_e: float = 5.0e4

    # mild extended background structure (optional)
    # NOTE: A full GRF via FFT per-frame is expensive in Colab.
    # Default: OFF. If you enable it, use a cached GRF pool and apply it
    # only to a small fraction of frames.
    use_grf_background: bool = False
    grf_prob: float = 0.05        # P(apply GRF | frame)
    grf_pool_size: int = 8        # number of cached GRF fields per run
    grf_amp_e: float = 60.0       # amplitude (e-/pix)
    grf_beta: float = 3.0         # power spectrum ~ 1/f^beta

    
    # Background synthesis mode: 'fft' (default, GRF via FFT) or 'gauss' (fast, smoothed noise)
    grf_mode: str = "fft"
    grf_gauss_sigma_pix: float = 28.0
# streak count distribution (k=0..5)
    # User target: 0:30%, 1:30%, remaining 40% split mostly into 2/3, rarely 4/5.
    # Streak count distribution (user requirement):
    # 0:30%, 1:30%, remaining 40% mostly 2/3 with rare 4/5.
    streak_probs = (0.30, 0.30, 0.25, 0.10, 0.04, 0.01)

    # streak photometry/geometry (electrons added along path)
    streak_fwhm_pix_min: float = 1.2
    streak_fwhm_pix_max: float = 3.5
    streak_peak_e_min: float = 300.0
    streak_peak_e_max: float = 6500.0

    # streak mask cut (finite width) - mask includes pixels within k*sigma from the line
    streak_mask_k_sigma: float = 2.0

    # altitude->brightness proxy for variability
    alt_km_min: float = 350.0
    alt_km_max: float = 1200.0
    alt_bright_gamma: float = 1.7

    # blooming (conditional on saturation)
    bloom_conditional_prob: float = 0.50
    bloom_strength: float = 0.15   # fraction of excess charge redistributed
    bloom_decay_pix: float = 12.0

    # smear (conditional on saturation) - physically: readout smear along saturated columns
    smear_conditional_prob: float = 0.30
    smear_strength: float = 0.0025
    smear_decay_pix: float = 80.0

    # saturation-driven injected bright stars (to ensure rare but non-zero SATUR/BLOOM/SMEAR events)
    sat_star_frame_prob: float = 0.08  # fraction of frames that contain 1–2 saturated stars
    sat_star_max_count: int = 2
    sat_star_peak_factor_min: float = 1.05  # peak pixel >= 1.05*FULLWELL
    sat_star_peak_factor_max: float = 1.80

    # bloom/smear mask thresholds (avoid marking the entire column due to tiny non-zero tails)
    bloom_mask_thresh_e: float = 120.0
    smear_mask_thresh_e: float = 80.0
    smear_max_cols: int = 3

    # output
    bunit: str = "ADU"

    artifact_mode: str = "physical"

    # hard-destruction (conditional on saturation)
    hard_conditional_prob: float = 0.35
    hard_bloom_len_min: int = 15
    hard_bloom_len_max: int = 30
    hard_bloom_w_min: int = 1
    hard_bloom_w_max: int = 3
    hard_smear_level_min: float = 0.5
    hard_smear_level_max: float = 0.8
