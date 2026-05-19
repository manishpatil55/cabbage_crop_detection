"""
utils.py
========
Shared utility functions used across the banana detection pipeline.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def detect_time_tags(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect YYYY_MM time tags from DataFrame column names.

    Scans all column names for patterns like <PREFIX>_<YYYY>_<MM>
    and returns sorted unique time tags.

    Parameters
    ----------
    df : DataFrame with columns named <BAND>_<YYYY>_<MM>

    Returns
    -------
    Sorted list of "YYYY_MM" strings, e.g. ["2022_05", "2022_06", ...]
    """
    tags = set()
    for col in df.columns:
        parts = col.split("_")
        if len(parts) >= 3:
            try:
                y, m = int(parts[-2]), int(parts[-1])
                if 2000 <= y <= 2100 and 1 <= m <= 12:
                    tags.add(f"{y}_{m:02d}")
            except ValueError:
                pass
    return sorted(tags)
