"""
phenology_features_heading.py
=============================
Phenological feature extraction for heading leafy vegetables
(cabbage, lettuce, endive) following the BBCH scale.

BBCH Growth Stages for Cabbage (Brassica oleracea var. capitata):
- Stage 0: Germination (nursery — not visible from space)
- Stage 1: Leaf development (transplant → 30 days) — low NDVI
- Stage 4: Head formation (30-60 days) — rapid NDVI increase
  - 41: Heads begin to form (two youngest leaves do not unfold)
  - 43: 30% of expected head size
  - 45: 50% of expected head size
  - 49: Typical size, form, and firmness reached
- Stage 9: Senescence — leaf discoloration, harvest

Features computed per vegetation index (NDVI, NDRE, EVI, LSWI, CCCI):
  heading_onset_week     — Week when derivative exceeds threshold (BBCH Stage 4)
  heading_duration       — Consecutive time steps with index > 0.7
  plateau_stability      — Std during heading phase (lower = healthier crop)
  greenup_rate           — Max positive slope (rapid leaf expansion)
  harvest_drop           — Binary: sharp NDVI drop detected (1/0)
  harvest_drop_magnitude — Largest single-step decline
  decline_rate           — Rate of decline from peak to end
  time_to_peak           — Normalized position of peak (0-1)
  auc                    — Area under curve (cumulative productivity)
  pre_post_heading_ratio — Mean index ratio before/after heading onset

Cross-index features:
  ndvi_ndre_heading_corr — NDVI-NDRE correlation during heading
  ndvi_vv_correlation    — NDVI-SAR temporal correlation

References:
- BBCH-scale (leafy vegetables forming heads)
- Ryu et al. (2024): NDRE > NDVI for cabbage growth status assessment
- Besand & Katroschan (2022): NDRE, CCCI for cabbage N status

Usage
-----
    from features.phenology_features_heading import HeadingPhenologyExtractor

    extractor = HeadingPhenologyExtractor(config_path="config.yaml")
    pheno_df = extractor.compute(df_wide, time_tags=["2024_10", "2024_11", ...])
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid

logger = logging.getLogger(__name__)

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Core heading phenology functions (operate on 2D arrays: n_pixels × n_time)
# ---------------------------------------------------------------------------

def _detect_heading_onset(ts_matrix: np.ndarray, derivative_threshold: float = 0.05,
                          consecutive_steps: int = 2) -> np.ndarray:
    """
    Detect the time step where derivative exceeds threshold for consecutive
    steps. Marks transition to head formation (BBCH Stage 4).

    Returns: array of onset indices (-1 if not detected).
    """
    n_pixels, n_timesteps = ts_matrix.shape
    onset = np.full(n_pixels, -1, dtype=np.int32)

    for i in range(n_pixels):
        ts = ts_matrix[i].copy()
        # Forward-fill NaN
        for t in range(1, len(ts)):
            if np.isnan(ts[t]):
                ts[t] = ts[t - 1]

        diffs = np.diff(ts)
        for t in range(len(diffs) - consecutive_steps + 1):
            if np.all(diffs[t:t + consecutive_steps] > derivative_threshold):
                onset[i] = t
                break

    return onset


def _count_high_index_periods(ts_matrix: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    """Count time steps where index exceeds threshold (heading duration)."""
    return np.nansum(ts_matrix > threshold, axis=1).astype(np.float32)


def _compute_plateau_stability(ts_matrix: np.ndarray, onset_weeks: np.ndarray) -> np.ndarray:
    """
    Std of index values from heading onset to end.
    Lower std = more stable heading = healthier cabbage.
    """
    n_pixels = ts_matrix.shape[0]
    stability = np.full(n_pixels, np.nan, dtype=np.float32)

    for i in range(n_pixels):
        onset = onset_weeks[i]
        if onset >= 0 and onset < ts_matrix.shape[1] - 1:
            plateau = ts_matrix[i, onset:]
            valid = plateau[~np.isnan(plateau)]
            if len(valid) > 1:
                stability[i] = np.std(valid)

    return stability


def _compute_greenup_rate(ts_matrix: np.ndarray, window: int = 2) -> np.ndarray:
    """Maximum positive slope over a sliding window (green-up rate)."""
    n_pixels, n_timesteps = ts_matrix.shape
    max_slopes = np.full(n_pixels, np.nan, dtype=np.float32)

    for i in range(n_pixels):
        ts = ts_matrix[i].copy()
        for t in range(1, len(ts)):
            if np.isnan(ts[t]):
                ts[t] = ts[t - 1]

        slopes = []
        for t in range(n_timesteps - window):
            dy = ts[t + window] - ts[t]
            if not np.isnan(dy) and dy > 0:
                slopes.append(dy / window)
        if slopes:
            max_slopes[i] = np.max(slopes)

    return max_slopes


def _detect_harvest_drop(ts_matrix: np.ndarray, drop_threshold: float = 0.2,
                         window: int = 2) -> np.ndarray:
    """
    Detect sharp NDVI decline indicating harvest.
    Returns: 1 if harvest signature detected, 0 otherwise.
    """
    n_pixels, n_timesteps = ts_matrix.shape
    detected = np.zeros(n_pixels, dtype=np.float32)

    for i in range(n_pixels):
        for t in range(n_timesteps - window):
            drop = ts_matrix[i, t] - ts_matrix[i, t + window]
            if not np.isnan(drop) and drop > drop_threshold:
                detected[i] = 1.0
                break

    return detected


def _compute_max_drop_magnitude(ts_matrix: np.ndarray) -> np.ndarray:
    """Maximum single-step drop magnitude (most negative diff)."""
    n_pixels = ts_matrix.shape[0]
    max_drops = np.full(n_pixels, np.nan, dtype=np.float32)

    for i in range(n_pixels):
        ts = ts_matrix[i].copy()
        for t in range(1, len(ts)):
            if np.isnan(ts[t]):
                ts[t] = ts[t - 1]

        diffs = np.diff(ts)
        valid = diffs[~np.isnan(diffs)]
        if len(valid) > 0:
            max_drops[i] = np.min(valid)  # most negative = largest drop

    return max_drops


def _compute_decline_rate(ts_matrix: np.ndarray) -> np.ndarray:
    """Rate of decline from peak to end of series."""
    n_pixels, n_timesteps = ts_matrix.shape
    rates = np.full(n_pixels, np.nan, dtype=np.float32)

    for i in range(n_pixels):
        valid = ~np.isnan(ts_matrix[i])
        if np.sum(valid) < 2:
            continue
        valid_idx = np.where(valid)[0]
        peak_idx = valid_idx[np.argmax(ts_matrix[i, valid])]
        final_idx = valid_idx[-1]
        if final_idx > peak_idx:
            decline = ts_matrix[i, peak_idx] - ts_matrix[i, final_idx]
            rates[i] = decline / (final_idx - peak_idx)

    return rates


def _compute_time_to_peak(ts_matrix: np.ndarray) -> np.ndarray:
    """Normalized time step of peak (0.0=early, 1.0=late)."""
    n_pixels, n_timesteps = ts_matrix.shape
    ttp = np.full(n_pixels, np.nan, dtype=np.float32)

    for i in range(n_pixels):
        valid = ~np.isnan(ts_matrix[i])
        if np.sum(valid) < 2:
            continue
        valid_idx = np.where(valid)[0]
        peak_pos = valid_idx[np.argmax(ts_matrix[i, valid])]
        ttp[i] = peak_pos / max(n_timesteps - 1, 1)

    return ttp


def _compute_pre_post_ratio(ts_matrix: np.ndarray, onset_weeks: np.ndarray) -> np.ndarray:
    """Ratio of mean index before vs after heading onset."""
    n_pixels = ts_matrix.shape[0]
    ratios = np.full(n_pixels, np.nan, dtype=np.float32)

    for i in range(n_pixels):
        onset = onset_weeks[i]
        if onset > 0 and onset < ts_matrix.shape[1] - 1:
            pre_mean = np.nanmean(ts_matrix[i, :onset])
            post_mean = np.nanmean(ts_matrix[i, onset:])
            if pre_mean > 0:
                ratios[i] = post_mean / pre_mean

    return ratios


def _compute_cross_correlation(ts_a: np.ndarray, ts_b: np.ndarray) -> np.ndarray:
    """Per-pixel Pearson correlation between two index time series."""
    n_pixels = ts_a.shape[0]
    corrs = np.full(n_pixels, np.nan, dtype=np.float32)

    for i in range(n_pixels):
        valid = ~(np.isnan(ts_a[i]) | np.isnan(ts_b[i]))
        if np.sum(valid) >= 3:
            try:
                r, _ = stats.pearsonr(ts_a[i, valid], ts_b[i, valid])
                corrs[i] = float(r)
            except Exception:
                pass

    return corrs


def smooth_series(ts: np.ndarray, window: int = 3, polyorder: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay smoothing. Handles NaN by forward-fill."""
    ts = ts.copy().astype(np.float32)
    for i in range(1, len(ts)):
        if np.isnan(ts[i]):
            ts[i] = ts[i - 1]
    for i in range(len(ts) - 2, -1, -1):
        if np.isnan(ts[i]):
            ts[i] = ts[i + 1]

    if np.all(np.isnan(ts)):
        return ts

    window = min(window, len(ts))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return ts

    try:
        return savgol_filter(ts, window_length=window, polyorder=min(polyorder, window - 1))
    except Exception:
        return ts


# ---------------------------------------------------------------------------
# DataFrame-level extractor
# ---------------------------------------------------------------------------

class HeadingPhenologyExtractor:
    """
    Extract heading-vegetable phenological features from a wide-format
    pixel DataFrame. Designed for cabbage (Brassica oleracea var. capitata).

    Parameters
    ----------
    config_path    : path to config.yaml
    vi_bands       : vegetation indices to process
    sar_bands      : SAR bands to process
    """

    DEFAULT_VI_BANDS = ["NDVI", "NDRE", "EVI", "LSWI", "CCCI"]
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

        self.vi_bands = vi_bands or pheno_cfg.get(
            "key_indices", self.DEFAULT_VI_BANDS
        )
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
        Compute heading phenology features for all pixels.

        Returns DataFrame of shape (n_pixels, n_heading_features)
        """
        if time_tags is None:
            time_tags = self._detect_time_tags(df)

        n_pixels = len(df)
        n_timesteps = len(time_tags)
        months_arr = np.arange(n_timesteps, dtype=np.float32)

        logger.info(
            f"Computing HEADING phenology: {n_pixels} pixels × "
            f"{n_timesteps} time steps | VIs={self.vi_bands}"
        )

        all_features = {}

        # ---- Heading phenology per VI band ----
        for band in self.vi_bands:
            band_cols = [f"{band}_{tag}" for tag in time_tags]
            available_cols = [c for c in band_cols if c in df.columns]

            if len(available_cols) < 3:
                logger.warning(f"Band {band}: only {len(available_cols)} time steps, skipping.")
                continue

            arr = np.full((n_pixels, n_timesteps), np.nan, dtype=np.float32)
            for t_idx, col in enumerate(band_cols):
                if col in df.columns:
                    arr[:, t_idx] = df[col].values.astype(np.float32)

            # Smooth each pixel's time series
            for i in range(n_pixels):
                arr[i] = smooth_series(arr[i], window=self.smooth_window)

            feats = self._compute_heading_features(arr, months_arr, band)
            all_features.update(feats)

        # ---- SAR phenology (simplified) ----
        for band in self.sar_bands:
            band_cols = [f"{band}_{tag}" for tag in time_tags]
            available_cols = [c for c in band_cols if c in df.columns]
            if not available_cols:
                continue

            arr = np.full((n_pixels, n_timesteps), np.nan, dtype=np.float32)
            for t_idx, col in enumerate(band_cols):
                if col in df.columns:
                    arr[:, t_idx] = df[col].values.astype(np.float32)

            feats = self._compute_sar_features(arr, months_arr, band)
            all_features.update(feats)

        # ---- Cross-index correlations ----
        ndvi_arr = self._get_band_array(df, "NDVI", time_tags, n_pixels, n_timesteps)
        ndre_arr = self._get_band_array(df, "NDRE", time_tags, n_pixels, n_timesteps)
        vv_arr = self._get_band_array(df, "VV", time_tags, n_pixels, n_timesteps)

        if ndvi_arr is not None and ndre_arr is not None:
            all_features["NDVI_NDRE_heading_corr"] = _compute_cross_correlation(ndvi_arr, ndre_arr)

        if ndvi_arr is not None and vv_arr is not None:
            all_features["NDVI_VV_temporal_correlation"] = _compute_cross_correlation(ndvi_arr, vv_arr)

        result = pd.DataFrame(all_features)
        logger.info(f"Heading phenology features shape: {result.shape}")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_heading_features(self, arr: np.ndarray, months: np.ndarray,
                                   band: str) -> dict:
        """Compute all 10 heading-specific features for a VI band."""
        n_pixels = arr.shape[0]

        # 1. Heading onset detection
        onset = _detect_heading_onset(arr)

        feats = {
            f"{band}_heading_onset_week": onset.astype(np.float32),
            f"{band}_heading_duration": _count_high_index_periods(arr, threshold=0.7),
            f"{band}_plateau_stability": _compute_plateau_stability(arr, onset),
            f"{band}_greenup_rate": _compute_greenup_rate(arr),
            f"{band}_harvest_drop": _detect_harvest_drop(arr),
            f"{band}_harvest_drop_magnitude": _compute_max_drop_magnitude(arr),
            f"{band}_decline_rate": _compute_decline_rate(arr),
            f"{band}_time_to_peak": _compute_time_to_peak(arr),
            f"{band}_auc": np.array([trapezoid(arr[i][~np.isnan(arr[i])]) if np.sum(~np.isnan(arr[i])) >= 2 else np.nan for i in range(n_pixels)], dtype=np.float32),
            f"{band}_pre_post_heading_ratio": _compute_pre_post_ratio(arr, onset),
        }

        # Also compute peak value + peak month (useful for any crop)
        for i in range(n_pixels):
            valid = ~np.isnan(arr[i])
            if valid.any():
                peak_idx = np.nanargmax(arr[i])
                feats.setdefault(f"{band}_peak_value", np.full(n_pixels, np.nan, dtype=np.float32))
                feats.setdefault(f"{band}_peak_month", np.full(n_pixels, np.nan, dtype=np.float32))
                feats[f"{band}_peak_value"][i] = arr[i, peak_idx]
                feats[f"{band}_peak_month"][i] = float(peak_idx)

        return feats

    def _compute_sar_features(self, arr: np.ndarray, months: np.ndarray,
                               band: str) -> dict:
        """Compute SAR phenology features (simpler set)."""
        n_pixels = arr.shape[0]
        sar_threshold = -15.0 if np.nanmean(arr) < 0 else 0.1

        return {
            f"{band}_sar_auc": np.array([
                trapezoid(arr[i][~np.isnan(arr[i])]) if np.sum(~np.isnan(arr[i])) >= 2 else np.nan
                for i in range(n_pixels)
            ], dtype=np.float32),
            f"{band}_sar_peak_value": np.nanmax(arr, axis=1).astype(np.float32),
            f"{band}_sar_season_length": np.nansum(arr > sar_threshold, axis=1).astype(np.float32),
            f"{band}_sar_std": np.nanstd(arr, axis=1).astype(np.float32),
        }

    def _get_band_array(self, df, band, time_tags, n_pixels, n_timesteps):
        """Get band time series as 2D array, or None if not available."""
        band_cols = [f"{band}_{tag}" for tag in time_tags]
        available = [c for c in band_cols if c in df.columns]
        if len(available) < 3:
            return None
        arr = np.full((n_pixels, n_timesteps), np.nan, dtype=np.float32)
        for t_idx, col in enumerate(band_cols):
            if col in df.columns:
                arr[:, t_idx] = df[col].values.astype(np.float32)
        return arr

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
