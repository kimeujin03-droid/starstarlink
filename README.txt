ZTF-like Synthetic Dataset Generator (NumPy + FITS)
===================================================

Outputs per sample (float32 FITS):
- *_obs.fits        : observed (stars+sky+noise + streak + optional bloom/smear)
- *_clean.fits      : clean (stars+sky+noise), NO streak/bloom/smear (noise retained)
- *_streakmask.fits : uint8 mask (1 where streak photons added)
- *_othermask.fits  : uint8 mask (1=bloom, 2=smear, 0=none)

Other mask convention (uint8):
- 1: blooming (saturation-driven)
- 2: smear (saturation-driven)

Design choices:
- Primary task: streak removal + underlying star photometry recovery.
- No saturation/halo/ghost modeling (out-of-scope).
- Bloom/smear are conditional on an OBS saturation event (physically motivated).
  P(bloom|sat)=0.50, P(smear|sat)=0.30 by default.
- Streak count distribution: 0:30%, 1:30%, remaining 40% mostly 2/3 (rare 4/5).
- For >=3 streaks: roughly-parallel bundle (small angle jitter + perpendicular offsets),
  but avoids overly stacked identical lines.

Colab usage:
  !python make_dataset.py --out /content/dataset_5000 --n 5000 --size 512 --seed 123
  !python preview.py --fits /content/dataset_5000/sim_0000000123_obs.fits
  # quick grid view (first 100)
  !python quicklook_100.py --dir /content/dataset_5000 --n 100 --out /content/quicklook_100.png --overlay
  # stats aggregation
  !python analyze_dataset.py --dir /content/dataset_5000 --out_csv /content/frame_stats.csv

Requirements:
  numpy, astropy, matplotlib
Colab / Google Drive (recommended)
---------------------------------
Colab runtime resets delete /content. Write outputs to Google Drive:

  from google.colab import drive
  drive.mount('/content/drive')
  OUT = "/content/drive/MyDrive/ztf_synth/dataset_5000"

Resume / Chunked generation
---------------------------
Generate (or resume) a full 5000-frame dataset:

  python make_dataset.py --out "$OUT" --n 5000 --seed0 123 --resume

Generate in chunks (e.g., 500 x 10), safe against runtime resets:

  python make_dataset.py --out "$OUT" --start 0    --count 500 --seed0 123 --resume
  python make_dataset.py --out "$OUT" --start 500  --count 500 --seed0 123 --resume
  ...
  python make_dataset.py --out "$OUT" --start 4500 --count 500 --seed0 123 --resume

Quicklook mosaic
----------------
Deterministic 100-frame quicklook with overlays:

  python quicklook_100.py --dir "$OUT" --n 100 --seed 0 --out quicklook_100.png --overlay

Optional: mild structured background (GRF)
----------------------------------------
Real ZTF frames sometimes have faint extended structure. Generating a full 512×512 GRF via FFT
per-frame is slow, so this repo uses a cached pool and applies it only on a subset of frames.

By default GRF is OFF. Enable like this:

  python make_dataset.py --out "$OUT" --n 5000 --seed0 123 --resume \
    --use_grf --grf_prob 0.05 --grf_amp_e 20 --grf_pool 8


New in v11.2.4 (patch)
----------------------
1) Faster GRF backgrounds:
   - Added Config.grf_mode = "fft" | "gauss"
   - Added Config.grf_gauss_sigma_pix
   - `make_dataset.py` flags:
       --fast_bg            (shortcut: enables GRF + uses grf_mode=gauss)
       --grf_mode fft|gauss (if GRF enabled)
       --grf_sigma <pix>    (only for grf_mode=gauss)

   If you were hitting long stalls / KeyboardInterrupt while generating many frames,
   try:
       python make_dataset.py --out dataset --num 5000 --use_grf --fast_bg

2) Quicklook overlay clarity:
   - Added `quicklook_100.py --fill` to render filled semi-transparent mask overlays.
     (Contour-only overlays can make thick streak masks *look* like just outlines.)

