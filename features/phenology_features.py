"""
phenology_features.py
=====================
.. deprecated:: 2.0
    This module is DEPRECATED. Use ``phenology_features_heading.py`` instead.
    This file contains legacy perennial-crop phenology features.
    The cabbage detection pipeline uses ``HeadingPhenologyExtractor``
    from ``phenology_features_heading.py`` (BBCH-scale heading features).
Extract phenological shape features from vegetation index time series.

These features are region-invariant because they describe the *shape* of the
growing season curve rather than absolute reflectance values, making them
robust to domain shift across Indian agro-climatic zones.

Features computed
-----------------
For each vegetation index (NDVI, EVI, LSWI):
  auc              — Area Under the Curve (trapezoidal integration)
  peak_value       — Maximum value in the time series
  peak_month       — Month index of peak value (1–24 for 2-year series)
  greenup_half_month — Month when greenness reaches 50% of peak (green-up half)
  max_greenup_slope  — Maximum slope between consecutive months (green-up rate)
  max_senescence_slope — Maximum negative slope (senescence rate)
  season_length    — Number of months above NDVI threshold
  n_growing_seasons — Number of distinct growing seasons detected
  amplitude        — peak_value - baseline (10th percentile)
  asymmetry        — (peak_month - greenup_half_month) / season_length
  smoothness       — 1 / (std of second derivative) — measures curve smoothness

For SAR (VV, VH, RVI):
  sar_auc          — AUC of SAR time series
  sar_peak_value   — Peak SAR value
  sar_season_length — Months above SAR threshold

Usage
-----
    from features.phenology_features import PhenologyExtractor

    extractor = PhenologyExtractor(config_path="config.yaml")
    pheno_df = extractor.compute(df_wide, time_tags=["2022_01", ...])
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid

logger = logging.getLogger(__name__)

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Core phenology functions (operate on 1D time series arrays)
# ---------------------------------------------------------------------------

def smooth_series(ts: np.ndarray, window: int = 3, polyorder: int = 2) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to a time series.
    Handles NaN by forward-filling before smoothing.
    """
    ts = ts.copy().astype(np.float32)
    # Forward-fill NaN (do not interpolate across long gaps)
    for i in range(1, len(ts)):
        if np.isnan(ts[i]):
            ts[i] = ts[i - 1]
    # Backward-fill leading NaN
    for i in range(len(ts) - 2, -1, -1):
        if np.isnan(ts[i]):
            ts[i] = ts[i + 1]

    if np.all(np.isnan(ts)):
        return ts

    # Ensure window is odd and <= series length
    window = min(window, len(ts))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return ts

    try:
        return savgol_filter(ts, window_length=window, polyorder=min(polyorder, window - 1))
    except Exception:
        return ts


def compute_auc(ts: np.ndarray, months: Optional[np.ndarray] = None) -> float:
    """Area Under the Curve using trapezoidal integration."""
    valid = ~np.isnan(ts)
    if valid.sum() < 2:
        return np.nan
    x = months[valid] if months is not None else np.where(valid)[0]
    y = ts[valid]
    return float(trapezoid(y, x))


def compute_peak(ts: np.ndarray) -> Tuple[float, int]:
    """Return (peak_value, peak_index)."""
    valid = ~np.isnan(ts)
    if not valid.any():
        return np.nan, -1
    idx = int(np.nanargmax(ts))
    return float(ts[idx]), idx


def compute_greenup_half(ts: np.ndarray, peak_idx: int) -> int:
    """
    Find the month index where the series first reaches 50% of peak value
    (on the ascending limb before the peak).
    """
    if peak_idx <= 0 or np.isnan(ts[peak_idx]):
        return -1
    half_peak = ts[peak_idx] * 0.5
    ascending = ts[:peak_idx + 1]
    candidates = np.where(ascending >= half_peak)[0]
    return int(candidates[0]) if len(candidates) > 0 else -1


def compute_max_slope(ts: np.ndarray) -> Tuple[float, float]:
    """
    Return (max_greenup_slope, max_senescence_slope).
    Greenup slope = max positive diff between consecutive months.
    Senescence slope = max negative diff (most negative).
    """
    valid_ts = ts.copy()
    # Replace NaN with previous valid value for slope computation
    for i in range(1, len(valid_ts)):
        if np.isnan(valid_ts[i]):
            valid_ts[i] = valid_ts[i - 1]

    diffs = np.diff(valid_ts)
    max_greenup = float(np.nanmax(diffs)) if len(diffs) > 0 else np.nan
    max_senescence = float(np.nanmin(diffs)) if len(diffs) > 0 else np.nan
    return max_greenup, max_senescence


def compute_season_length(ts: np.ndarray, threshold: float = 0.3) -> int:
    """Number of months where ts > threshold."""
    return int(np.sum(ts > threshold))


def count_growing_seasons(ts: np.ndarray, threshold: float = 0.3, min_gap: int = 2) -> int:
    """
    Count distinct growing seasons (contiguous runs above threshold separated
    by at least min_gap months below threshold).
    """
    above = (ts > threshold).astype(int)
    seasons = 0
    in_season = False
    gap_count = 0

    for val in above:
        if val == 1:
            if not in_season:
                seasons += 1
                in_season = True
            gap_count = 0
        else:
            gap_count += 1
            if gap_count >= min_gap:
                in_season = False

    return seasons


def compute_smoothness(ts: np.ndarray) -> float:
    """
    Smoothness = 1 / (std of second derivative).
    Higher value = smoother curve.
    """
    valid = ts[~np.isnan(ts)]
    if len(valid) < 3:
        return np.nan
    d2 = np.diff(valid, n=2)
    std_d2 = np.std(d2)
    return float(1.0 / (std_d2 + _EPS))


# ---------------------------------------------------------------------------
# DataFrame-level extractor
# ---------------------------------------------------------------------------

class PhenologyExtractor:
    """
    Extract phenological features from a wide-format pixel DataFrame.

    Parameters
    ----------
    config_path    : path to config.yaml
    vi_bands       : vegetation index bands to process
    sar_bands      : SAR bands to process
    ndvi_threshold : threshold for growing season detection
    smooth_window  : Savitzky-Golay window size
    """

    DEFAULT_VI_BANDS = ["NDVI", "EVI", "LSWI"]
    DEFAULT_SAR_BANDS = ["VV", "VH", "RVI"]

    def __init__(
        self,
        config_path: str = "config.yaml",
        vi_bands: Optional[List[str]] = None,
        sar_bands: Optional[List[str]] = None,
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        pheno_cfg = self.cfg["features"].get("phenology", {})
        self.ndvi_threshold = pheno_cfg.get("ndvi_threshold", 0.3)
        self.smooth_window = pheno_cfg.get("smooth_window", 3)

        self.vi_bands = vi_bands or self.DEFAULT_VI_BANDS
        self.sar_bands = sar_bands or self.DEFAULT_SAR_BANDS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        df: pd.DataFrame,
        time_tags: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute phenological features for all pixels.

        Parameters
        ----------
        df        : wide-format DataFrame
        time_tags : list of "YYYY_MM" strings; if None, auto-detected

        Returns
        -------
        DataFrame of shape (n_pixels, n_pheno_features)
        """
        if time_tags is None:
            time_tags = self._detect_time_tags(df)

        n_pixels = len(df)
        n_timesteps = len(time_tags)
        months_arr = np.arange(n_timesteps, dtype=np.float32)

        logger.info(
            f"Computing phenology features: {n_pixels} pixels × "
            f"{n_timesteps} time steps | VIs={self.vi_bands}"
        )

        all_features = {}

        # ---- Vegetation index phenology ----
        for band in self.vi_bands:
            band_cols = [f"{band}_{tag}" for tag in time_tags]
            available_cols = [c for c in band_cols if c in df.columns]

            if not available_cols:
                logger.warning(f"Band {band} not found in DataFrame, skipping.")
                continue

            # Build (n_pixels, n_timesteps) array, NaN where column missing
            arr = np.full((n_pixels, n_timesteps), np.nan, dtype=np.float32)
            for t_idx, col in enumerate(band_cols):
                if col in df.columns:
                    arr[:, t_idx] = df[col].values.astype(np.float32)

            # Compute features per pixel
            feats = self._compute_vi_phenology(arr, months_arr, band)
            all_features.update(feats)

        # ---- SAR phenology ----
        for band in self.sar_bands:
            band_cols = [f"{band}_{tag}" for tag in time_tags]
            available_cols = [c for c in band_cols if c in df.columns]

            if not available_cols:
                continue

            arr = np.full((n_pixels, n_timesteps), np.nan, dtype=np.float32)
            for t_idx, col in enumerate(band_cols):
                if col in df.columns:
                    arr[:, t_idx] = df[col].values.astype(np.float32)

            feats = self._compute_sar_phenology(arr, months_arr, band)
            all_features.update(feats)

        # ---- Cross-sensor phenology coherence ----
        if "NDVI" in self.vi_bands and "VV" in self.sar_bands:
            ndvi_arr = np.full((n_pixels, n_timesteps), np.nan, dtype=np.float32)
            vv_arr = np.full((n_pixels, n_timesteps), np.nan, dtype=np.float32)
            for t_idx, tag in enumerate(time_tags):
                if f"NDVI_{tag}" in df.columns:
                    ndvi_arr[:, t_idx] = df[f"NDVI_{tag}"].values.astype(np.float32)
                if f"VV_{tag}" in df.columns:
                    vv_arr[:, t_idx] = df[f"VV_{tag}"].values.astype(np.float32)

            coherence = self._compute_sar_optical_coherence(ndvi_arr, vv_arr)
            all_features.update(coherence)

        result = pd.DataFrame(all_features)
        logger.info(f"Phenology features shape: {result.shape}")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_vi_phenology(
        self,
        arr: np.ndarray,
        months: np.ndarray,
        band: str,
    ) -> dict:
        """Compute VI phenology features for all pixels (vectorised where possible)."""
        n_pixels = arr.shape[0]
        feats = {
            f"{band}_auc": np.full(n_pixels, np.nan, dtype=np.float32),
            f"{band}_peak_value": np.full(n_pixels, np.nan, dtype=np.float32),
            f"{band}_peak_month": np.full(n_pixels, np.nan, dtype=np.float32),
            f"{band}_greenup_half_month": np.full(n_pixels, np.nan, dtype=np.float32),
            f"{band}_max_greenup_slope": np.full(n_pixels, np.nan, dtype=np.float32),
            f"{band}_max_senescence_slope": np.full(n_pixels, np.nan, dtype=np.float32),
            f"{band}_season_length": np.full(n_pixels, np.nan, dtype=np.float32),
            f"{band}_n_growing_seasons": np.full(n_pixels, np.nan, dtype=np.float32),
            f"{band}_amplitude": np.full(n_pixels, np.nan, dtype=np.float32),
            f"{band}_asymmetry": np.full(n_pixels, np.nan, dtype=np.float32),
            f"{band}_smoothness": np.full(n_pixels, np.nan, dtype=np.float32),
        }

        for i in range(n_pixels):
            ts_raw = arr[i]
            ts = smooth_series(ts_raw, window=self.smooth_window)

            if np.all(np.isnan(ts)):
                continue

            # AUC
            feats[f"{band}_auc"][i] = compute_auc(ts, months)

            # Peak
            peak_val, peak_idx = compute_peak(ts)
            feats[f"{band}_peak_value"][i] = peak_val
            feats[f"{band}_peak_month"][i] = float(peak_idx)

            # Green-up half
            gh = compute_greenup_half(ts, peak_idx)
            feats[f"{band}_greenup_half_month"][i] = float(gh)

            # Slopes
            max_up, max_down = compute_max_slope(ts)
            feats[f"{band}_max_greenup_slope"][i] = max_up
            feats[f"{band}_max_senescence_slope"][i] = max_down

            # Season length
            sl = compute_season_length(ts, self.ndvi_threshold)
            feats[f"{band}_season_length"][i] = float(sl)

            # Number of growing seasons
            feats[f"{band}_n_growing_seasons"][i] = float(
                count_growing_seasons(ts, self.ndvi_threshold)
            )

            # Amplitude
            baseline = float(np.nanpercentile(ts, 10))
            feats[f"{band}_amplitude"][i] = peak_val - baseline

            # Asymmetry
            if sl > 0 and gh >= 0 and peak_idx >= 0:
                feats[f"{band}_asymmetry"][i] = (peak_idx - gh) / (sl + _EPS)

            # Smoothness
            feats[f"{band}_smoothness"][i] = compute_smoothness(ts)

        return feats

    def _compute_sar_phenology(
        self,
        arr: np.ndarray,
        months: np.ndarray,
        band: str,
    ) -> dict:
        """Compute SAR phenology features (simpler set than VI)."""
        n_pixels = arr.shape[0]
        # SAR threshold: -15 dB is a reasonable vegetation threshold
        sar_threshold = -15.0 if np.nanmean(arr) < 0 else 0.1

        feats = {
            f"{band}_sar_auc": np.full(n_pixels, np.nan, dtype=np.float32),
            f"{band}_sar_peak_value": np.full(n_pixels, np.nan, dtype=np.float32),
            f"{band}_sar_season_length": np.full(n_pixels, np.nan, dtype=np.float32),
            f"{band}_sar_std": np.full(n_pixels, np.nan, dtype=np.float32),
        }

        for i in range(n_pixels):
            ts = arr[i]
            if np.all(np.isnan(ts)):
                continue
            feats[f"{band}_sar_auc"][i] = compute_auc(ts, months)
            peak_val, _ = compute_peak(ts)
            feats[f"{band}_sar_peak_value"][i] = peak_val
            feats[f"{band}_sar_season_length"][i] = float(
                compute_season_length(ts, sar_threshold)
            )
            feats[f"{band}_sar_std"][i] = float(np.nanstd(ts))

        return feats

    def _compute_sar_optical_coherence(
        self,
        ndvi_arr: np.ndarray,
        vv_arr: np.ndarray,
    ) -> dict:
        """
        Compute temporal correlation between NDVI and VV SAR.
        High correlation indicates vegetation-driven SAR response.
        """
        n_pixels = ndvi_arr.shape[0]
        correlations = np.full(n_pixels, np.nan, dtype=np.float32)

        for i in range(n_pixels):
            ndvi_ts = ndvi_arr[i]
            vv_ts = vv_arr[i]
            valid = ~np.isnan(ndvi_ts) & ~np.isnan(vv_ts)
            if valid.sum() >= 4:
                try:
                    corr = np.corrcoef(ndvi_ts[valid], vv_ts[valid])[0, 1]
                    correlations[i] = float(corr)
                except Exception:
                    pass

        return {"NDVI_VV_temporal_correlation": correlations}

    @staticmethod
    def _detect_time_tags(df: pd.DataFrame) -> List[str]:
        """Auto-detect time tags from column names."""
        tags = set()
        for col in df.columns:
            parts = col.split("_")
            if len(parts) >= 3:
                try:
                    year = int(parts[-2])
                    month = int(parts[-1])
                    if 2000 <= year <= 2100 and 1 <= month <= 12:
                        tags.add(f"{year}_{month:02d}")
                except ValueError:
                    pass
        return sorted(tags)
