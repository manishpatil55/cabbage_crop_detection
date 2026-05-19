"""
spectral_indices.py
===================
Compute spectral vegetation and water indices from Sentinel-2 and Sentinel-1
band values stored in a pandas DataFrame (wide format).

Supported indices
-----------------
  Sentinel-2 derived:
    NDVI  — Normalised Difference Vegetation Index
    EVI   — Enhanced Vegetation Index
    NDWI  — Normalised Difference Water Index (Gao)
    LSWI  — Land Surface Water Index
    SAVI  — Soil-Adjusted Vegetation Index
    MSAVI — Modified SAVI
    NBR   — Normalised Burn Ratio
    NDRE  — Normalised Difference Red Edge

  Sentinel-1 derived:
    RVI   — Radar Vegetation Index
    RFDI  — Radar Forest Degradation Index
    CR    — Cross-Ratio (VH/VV)

Usage
-----
    from features.spectral_indices import SpectralIndexCalculator

    calc = SpectralIndexCalculator()
    df_with_indices = calc.compute_all(df, time_tags=["2022_01", "2022_02", ...])
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Small epsilon to avoid division by zero
_EPS = 1e-8


# ---------------------------------------------------------------------------
# Pure-numpy index functions (operate on arrays)
# ---------------------------------------------------------------------------

def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - RED) / (NIR + RED)"""
    return (nir - red) / (nir + red + _EPS)


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)"""
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + _EPS)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDWI (Gao) = (GREEN - NIR) / (GREEN + NIR)"""
    return (green - nir) / (green + nir + _EPS)


def lswi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """LSWI = (NIR - SWIR1) / (NIR + SWIR1)"""
    return (nir - swir1) / (nir + swir1 + _EPS)


def savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
    """SAVI = (NIR - RED) / (NIR + RED + L) * (1 + L)"""
    return (nir - red) / (nir + red + L + _EPS) * (1 + L)


def msavi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """MSAVI = (2*NIR + 1 - sqrt((2*NIR+1)^2 - 8*(NIR-RED))) / 2"""
    inner = np.maximum(0, (2 * nir + 1) ** 2 - 8 * (nir - red))
    return (2 * nir + 1 - np.sqrt(inner)) / 2


def nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """NBR = (NIR - SWIR2) / (NIR + SWIR2)"""
    return (nir - swir2) / (nir + swir2 + _EPS)


def ndre(re1: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDRE = (RE1 - RED) / (RE1 + RED)  — Red Edge NDVI"""
    return (re1 - red) / (re1 + red + _EPS)


def rvi_sar(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """
    Radar Vegetation Index (SAR).
    RVI = 4*VH / (VV + VH)
    Input: linear power scale (not dB).
    """
    # Convert dB to linear if values look like dB (typically < 0)
    if np.nanmean(vv) < 0:
        vv = 10 ** (vv / 10)
        vh = 10 ** (vh / 10)
    return 4 * vh / (vv + vh + _EPS)


def rfdi(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """
    Radar Forest Degradation Index.
    RFDI = (VV - VH) / (VV + VH)
    """
    if np.nanmean(vv) < 0:
        vv = 10 ** (vv / 10)
        vh = 10 ** (vh / 10)
    return (vv - vh) / (vv + vh + _EPS)


def cross_ratio(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """CR = VH / VV  (linear scale)"""
    if np.nanmean(vv) < 0:
        vv = 10 ** (vv / 10)
        vh = 10 ** (vh / 10)
    return vh / (vv + _EPS)


# ---------------------------------------------------------------------------
# DataFrame-level calculator
# ---------------------------------------------------------------------------

# Mapping from Sentinel-2 band name → column prefix in wide DataFrame
_S2_BAND_MAP = {
    "blue": "B2",
    "green": "B3",
    "red": "B4",
    "re1": "B5",
    "re2": "B6",
    "re3": "B7",
    "nir": "B8",
    "nir2": "B8A",
    "swir1": "B11",
    "swir2": "B12",
}

_S1_BAND_MAP = {
    "vv": "VV",
    "vh": "VH",
}


class SpectralIndexCalculator:
    """
    Compute spectral indices for each time step in a wide-format DataFrame.

    The DataFrame is expected to have columns named:
        <BAND>_<YYYY_MM>   e.g. B8_2022_01, B4_2022_01, VV_2022_01

    After calling compute_all(), new columns are added:
        NDVI_2022_01, EVI_2022_01, NDWI_2022_01, ...
    """

    # Indices to compute by default
    DEFAULT_S2_INDICES = ["NDVI", "EVI", "NDWI", "LSWI", "SAVI", "MSAVI", "NBR", "NDRE"]
    DEFAULT_S1_INDICES = ["RVI", "RFDI", "CR"]

    def __init__(
        self,
        s2_indices: Optional[List[str]] = None,
        s1_indices: Optional[List[str]] = None,
    ):
        self.s2_indices = s2_indices or self.DEFAULT_S2_INDICES
        self.s1_indices = s1_indices or self.DEFAULT_S1_INDICES

    def compute_all(
        self,
        df: pd.DataFrame,
        time_tags: Optional[List[str]] = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Compute all configured indices for every time step.

        Parameters
        ----------
        df        : wide-format DataFrame with band columns
        time_tags : list of "YYYY_MM" strings; if None, auto-detected
        inplace   : if True, modify df in place; else return a copy

        Returns
        -------
        DataFrame with additional index columns
        """
        if not inplace:
            df = df.copy()

        if time_tags is None:
            time_tags = self._detect_time_tags(df)

        logger.info(f"Computing indices for {len(time_tags)} time steps...")

        for tag in time_tags:
            self._compute_s2_indices_for_tag(df, tag)
            self._compute_s1_indices_for_tag(df, tag)

        return df

    def _detect_time_tags(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect time tags from column names like B8_2022_01."""
        tags = set()
        for col in df.columns:
            parts = col.split("_")
            if len(parts) >= 3 and parts[0] in _S2_BAND_MAP.values():
                tag = "_".join(parts[1:])
                tags.add(tag)
        return sorted(tags)

    def _get_band(self, df: pd.DataFrame, band: str, tag: str) -> np.ndarray:
        """Safely retrieve a band array, returning NaN array if missing."""
        col = f"{band}_{tag}"
        if col in df.columns:
            return df[col].values.astype(np.float32)
        return np.full(len(df), np.nan, dtype=np.float32)

    def _compute_s2_indices_for_tag(self, df: pd.DataFrame, tag: str):
        """Compute all S2 indices for a single time tag."""
        nir = self._get_band(df, "B8", tag)
        red = self._get_band(df, "B4", tag)
        green = self._get_band(df, "B3", tag)
        blue = self._get_band(df, "B2", tag)
        swir1 = self._get_band(df, "B11", tag)
        swir2 = self._get_band(df, "B12", tag)
        re1 = self._get_band(df, "B5", tag)

        index_map = {
            "NDVI": lambda: ndvi(nir, red),
            "EVI": lambda: evi(nir, red, blue),
            "NDWI": lambda: ndwi(green, nir),
            "LSWI": lambda: lswi(nir, swir1),
            "SAVI": lambda: savi(nir, red),
            "MSAVI": lambda: msavi(nir, red),
            "NBR": lambda: nbr(nir, swir2),
            "NDRE": lambda: ndre(re1, red),
        }

        for idx_name in self.s2_indices:
            if idx_name in index_map:
                col = f"{idx_name}_{tag}"
                if col not in df.columns:  # Don't overwrite GEE-computed indices
                    df[col] = index_map[idx_name]()

    def _compute_s1_indices_for_tag(self, df: pd.DataFrame, tag: str):
        """Compute all S1 indices for a single time tag."""
        vv = self._get_band(df, "VV", tag)
        vh = self._get_band(df, "VH", tag)

        index_map = {
            "RVI": lambda: rvi_sar(vv, vh),
            "RFDI": lambda: rfdi(vv, vh),
            "CR": lambda: cross_ratio(vv, vh),
        }

        for idx_name in self.s1_indices:
            if idx_name in index_map:
                col = f"{idx_name}_{tag}"
                if col not in df.columns:
                    df[col] = index_map[idx_name]()

    # ------------------------------------------------------------------
    # Utility: compute indices on raw arrays (for GEE-independent use)
    # ------------------------------------------------------------------

    @staticmethod
    def from_arrays(
        nir: np.ndarray,
        red: np.ndarray,
        green: np.ndarray,
        blue: np.ndarray,
        swir1: np.ndarray,
        swir2: Optional[np.ndarray] = None,
        re1: Optional[np.ndarray] = None,
        vv: Optional[np.ndarray] = None,
        vh: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compute all indices from raw band arrays.

        Returns
        -------
        dict mapping index_name → np.ndarray
        """
        result = {
            "NDVI": ndvi(nir, red),
            "EVI": evi(nir, red, blue),
            "NDWI": ndwi(green, nir),
            "LSWI": lswi(nir, swir1),
            "SAVI": savi(nir, red),
            "MSAVI": msavi(nir, red),
        }
        if swir2 is not None:
            result["NBR"] = nbr(nir, swir2)
        if re1 is not None:
            result["NDRE"] = ndre(re1, red)
        if vv is not None and vh is not None:
            result["RVI"] = rvi_sar(vv, vh)
            result["RFDI"] = rfdi(vv, vh)
            result["CR"] = cross_ratio(vv, vh)
        return result
