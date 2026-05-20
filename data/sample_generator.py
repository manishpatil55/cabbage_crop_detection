"""
sample_generator.py
===================
Generate balanced training/testing pixel samples from KML plots.

For each KML polygon:
  - Class 1 (Cabbage)     : all pixels inside the polygon
  - Class 0 (Non-Cabbage) : equal number of random pixels in a buffer ring
                           (500 mâ€“1 km outside the polygon boundary)

Jalgaon-specific design notes
------------------------------
When all positive KMLs come from a single region (e.g., Jalgaon, Maharashtra),
standard random CV will leak spatial autocorrelation â€” nearby pixels from the
same plot appear in both train and val, inflating metrics.

This module therefore implements:

  1. **Negative sample diversity** â€” negative (non-Cabbage) points are drawn
     from a configurable list of external regions/states so the model learns
     a globally discriminative boundary, not just "not-Jalgaon".

  2. **Leave-One-Plot-Out (LOPO) CV split** â€” splits are grouped by plot_id
     so every pixel from a held-out plot is unseen during training.
     `split_leave_one_plot_out()` yields (train_idx, val_idx) pairs.

  3. **Monsoon hold-out plot** â€” `reserve_monsoon_test_plot()` selects the
     plot with the highest cloud-gap fraction (most monsoon-affected) and
     reserves it as a dedicated test set for monsoon robustness evaluation.

Outputs
-------
  1. A labeled CSV for 2D models (RF, XGBoost):
       columns = [longitude, latitude, state, label, plot_id,
                  <band>_<YYYY_MM>, ..., <stat_feature>, ...]

  2. A 3D NumPy array for BiLSTM:
       shape = (n_samples, n_timesteps, n_features)
       saved as .npy alongside a metadata CSV

Usage
-----
    from data.sample_generator import SampleGenerator

    gen = SampleGenerator(config_path="config.yaml")
    df_2d, arr_3d, meta_df = gen.generate(
        gdf=parsed_kml_gdf,
        start_date="2022-01-01",
        end_date="2024-01-01",
        external_neg_regions=["Tamil Nadu", "Gujarat"],  # diverse negatives
    )
    gen.save(df_2d, arr_3d, meta_df, out_dir="data/processed")

    # Leave-one-plot-out splits (prevents spatial overfitting)
    for fold, (tr_idx, val_idx, held_plot) in enumerate(
        SampleGenerator.split_leave_one_plot_out(meta_df)
    ):
        ...
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import Point
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_points_in_ring(
    polygon,
    n: int,
    inner_buffer_m: float = 500.0,
    outer_buffer_m: float = 1000.0,
    seed: int = 42,
    crs_utm: str = "EPSG:32643",
) -> list:
    """
    Generate n random points in the ring between inner_buffer and outer_buffer
    around a polygon (in metres).

    Uses a UTM projection for accurate buffering, then reprojects to WGS84.
    """
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    rng = np.random.default_rng(seed)

    # Project to UTM for metric buffering
    to_utm = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
    to_wgs = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)

    poly_utm = shapely_transform(to_utm.transform, polygon)
    inner = poly_utm.buffer(inner_buffer_m)
    outer = poly_utm.buffer(outer_buffer_m)
    ring = outer.difference(inner)

    minx, miny, maxx, maxy = ring.bounds
    points = []
    attempts = 0
    max_attempts = n * 50

    while len(points) < n and attempts < max_attempts:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        pt = Point(x, y)
        if ring.contains(pt):
            # Back to WGS84
            lon, lat = to_wgs.transform(x, y)
            points.append(Point(lon, lat))
        attempts += 1

    if len(points) < n:
        logger.warning(
            f"Only generated {len(points)}/{n} negative samples "
            f"(ring too small or too many attempts)."
        )
    return points


def _utm_zone_for_india(lon: float) -> str:
    """Return the appropriate UTM EPSG code for a longitude in India."""
    if lon < 78:
        return "EPSG:32643"   # UTM 43N
    elif lon < 84:
        return "EPSG:32644"   # UTM 44N
    else:
        return "EPSG:32645"   # UTM 45N


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SampleGenerator:
    """
    Generate balanced pixel samples for cabbage crop detection training.

    Parameters
    ----------
    config_path : path to config.yaml
    downloader  : optional pre-initialised GEEDownloader instance
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        downloader=None,
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self._config_path = config_path
        self.buffer_inner = self.cfg["sampling"]["buffer_m"] * 0.5
        self.buffer_outer = self.cfg["sampling"]["buffer_m"]
        self.max_pixels = self.cfg["sampling"]["max_pixels_per_plot"]
        self.neg_ratio = self.cfg["sampling"]["negative_ratio"]
        self.seed = self.cfg["sampling"]["random_seed"]

        self._downloader = downloader

    @property
    def downloader(self):
        if self._downloader is None:
            from data.gee_downloader import GEEDownloader
            self._downloader = GEEDownloader(config_path="config.yaml")
        return self._downloader

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        gdf: gpd.GeoDataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        scale: int = 10,
        external_neg_gdf: Optional[gpd.GeoDataFrame] = None,
        external_neg_regions: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """
        Generate labeled pixel samples for all plots in gdf.

        Anchor-date-driven windows
        --------------------------
        If the GeoDataFrame has ``date_start`` / ``date_end`` columns (set by
        KMLParser from filenames like "1_3aug2023.kml"), each plot uses its own
        Â±12-month window centred on the confirmed cabbage date.

        ``start_date`` / ``end_date`` are used as a **fallback** for plots that
        have no anchor date in their filename.  If neither is provided and a
        plot has no anchor date, that plot is skipped with a warning.

        Jalgaon scenario
        ----------------
        When all positive KMLs are from one region, negative samples from the
        local buffer ring alone will be geographically narrow and the model
        will overfit to Jalgaon-specific background.

        Pass ``external_neg_gdf`` (a GeoDataFrame of non-cabbage polygons from
        other states) OR ``external_neg_regions`` (state names already present
        in gdf with label=0) to inject geographically diverse negatives.

        Parameters
        ----------
        gdf                  : GeoDataFrame from KMLParser (geometry, state,
                               plot_id, label, anchor_date, date_start, date_end)
        start_date           : fallback GEE start date "YYYY-MM-DD"
        end_date             : fallback GEE end date "YYYY-MM-DD"
        scale                : pixel resolution in metres
        external_neg_gdf     : optional GeoDataFrame of external non-cabbage
                               polygons (label=0) from other regions/states
        external_neg_regions : optional list of state names already in gdf
                               whose label=0 rows should be treated as the
                               external negative pool

        Returns
        -------
        df_2d   : pd.DataFrame  â€” wide-format features for RF/XGBoost
        arr_3d  : np.ndarray    â€” (n_samples, n_timesteps, n_features) for BiLSTM
        meta_df : pd.DataFrame  â€” per-sample metadata (lon, lat, state, label,
                                  plot_id, cloud_gap_fraction, anchor_date,
                                  date_start, date_end)
        """
        import ee
        from data.gee_downloader import SatelliteDownloader

        # Ensure GEE is initialised before any ee.Geometry calls
        try:
            import yaml as _yaml
            _cfg = _yaml.safe_load(open(self._config_path))
            _proj = _cfg["gee"]["project_id"]
            ee.Initialize(project=_proj)
        except Exception:
            pass  # Already initialised, or non-GEE backend

        self.downloader.initialize()

        # Separate positive (Cabbage) and external-negative plots
        pos_gdf = gdf[gdf["label"] == 1].copy()
        ext_neg_gdf = gdf[gdf["label"] == 0].copy()

        # Merge any explicitly supplied external negatives
        if external_neg_gdf is not None:
            ext_neg_gdf = pd.concat(
                [ext_neg_gdf, external_neg_gdf], ignore_index=True
            )

        # Filter to requested external regions if specified
        if external_neg_regions and not ext_neg_gdf.empty:
            ext_neg_gdf = ext_neg_gdf[
                ext_neg_gdf["state"].isin(external_neg_regions)
            ]

        all_rows: List[pd.DataFrame] = []

        # ----------------------------------------------------------------
        # 1. Positive samples â€” each plot uses its own anchor-date window
        # ----------------------------------------------------------------
        for _, plot in pos_gdf.iterrows():
            plot_id = plot["plot_id"]
            state = plot["state"]
            geom = plot.geometry

            # Resolve date window: per-plot anchor takes priority
            p_start = plot.get("date_start") or start_date
            p_end = plot.get("date_end") or end_date
            anchor = plot.get("anchor_date")

            if not p_start or not p_end:
                logger.warning(
                    f"[SKIP] plot={plot_id}: no date window available. "
                    f"Rename the KML to include a date (e.g. '1_3aug2023.kml') "
                    f"or pass start_date/end_date to generate()."
                )
                continue

            logger.info(
                f"[POS] plot={plot_id} | state={state} | "
                f"anchor={anchor} | window=[{p_start} -> {p_end}]"
            )
            pos_geom = geom.__geo_interface__   # plain GeoJSON dict

            try:
                pos_df = self.downloader.extract_pixel_timeseries_wide(
                    geometry_geojson=pos_geom,
                    start_date=p_start,
                    end_date=p_end,
                    scale=scale,
                    max_pixels=self.max_pixels,
                )
                pos_df["label"] = 1
                pos_df["plot_id"] = plot_id
                pos_df["state"] = state
                pos_df["anchor_date"] = str(anchor) if anchor else ""
                pos_df["date_start"] = p_start
                pos_df["date_end"] = p_end
                pos_df["cloud_gap_fraction"] = self._cloud_gap_fraction(pos_df)
                all_rows.append(pos_df)
                logger.info(
                    f"  -> {len(pos_df)} positive pixels | "
                    f"cloud_gap={pos_df['cloud_gap_fraction'].mean():.2f}"
                )
            except Exception as exc:
                logger.warning(f"  Failed positive sampling for {plot_id}: {exc}")
                continue

            # ---- Local buffer-ring negatives (always generated) ----
            n_local_neg = max(1, int(len(pos_df) * self.neg_ratio * 0.5))
            centroid = geom.centroid
            utm_crs = _utm_zone_for_india(centroid.x)

            neg_points = _random_points_in_ring(
                geom,
                n=n_local_neg,
                inner_buffer_m=self.buffer_inner,
                outer_buffer_m=self.buffer_outer,
                seed=self.seed,
                crs_utm=utm_crs,
            )

            if neg_points:
                neg_coords = [[pt.x, pt.y] for pt in neg_points]
                # Build a MultiPoint GeoJSON dict for the buffer-ring negatives
                neg_geojson = {"type": "MultiPoint", "coordinates": neg_coords}
                try:
                    neg_df = self.downloader.extract_pixel_timeseries_wide(
                        geometry_geojson=neg_geojson,
                        start_date=p_start,
                        end_date=p_end,
                        scale=scale,
                        max_pixels=n_local_neg,
                    )
                    neg_df["label"] = 0
                    neg_df["plot_id"] = f"{plot_id}_local_neg"
                    neg_df["state"] = state
                    neg_df["anchor_date"] = str(anchor) if anchor else ""
                    neg_df["date_start"] = p_start
                    neg_df["date_end"] = p_end
                    neg_df["cloud_gap_fraction"] = self._cloud_gap_fraction(neg_df)
                    all_rows.append(neg_df)
                    logger.info(f"  -> {len(neg_df)} local-buffer negative pixels")
                except Exception as exc:
                    logger.warning(f"  Failed local-neg sampling for {plot_id}: {exc}")

        # ----------------------------------------------------------------
        # 2. External negative samples â€” diverse regions
        # ----------------------------------------------------------------
        if not ext_neg_gdf.empty:
            logger.info(
                f"Sampling external negatives from "
                f"{ext_neg_gdf['state'].unique().tolist()} ..."
            )
            total_pos = sum(
                len(r) for r in all_rows if r["label"].iloc[0] == 1
            )
            n_ext_per_plot = max(
                5, total_pos // max(1, len(ext_neg_gdf)),
            )

            # Collect all cabbage date windows so we can assign one to
            # non-cabbage plots that have no date in their filename.
            # We spread the windows across non-cabbage plots so the model
            # sees non-cabbage pixels from many different seasons.
            cabbage_windows = []
            for r in all_rows:
                if r["label"].iloc[0] == 1:
                    ds = r["date_start"].iloc[0]
                    de = r["date_end"].iloc[0]
                    if ds and de and (ds, de) not in cabbage_windows:
                        cabbage_windows.append((ds, de))

            # Fallback: if no cabbage windows collected yet, use a safe default
            if not cabbage_windows:
                cabbage_windows = [("2022-06-01", "2023-06-01")]

            for neg_idx, (_, plot) in enumerate(ext_neg_gdf.iterrows()):
                plot_id = plot["plot_id"]
                state = plot["state"]
                geom = plot.geometry

                # Priority: plot's own date_start/end -> rotate through cabbage windows
                p_start = plot.get("date_start") or start_date
                p_end   = plot.get("date_end")   or end_date

                # non-cabbage KMLs with no date: assign a cabbage window
                # Round-robin so different non-cabbage plots cover different seasons
                if not p_start or not p_end:
                    p_start, p_end = cabbage_windows[neg_idx % len(cabbage_windows)]
                    logger.info(
                        f"[EXT-NEG] plot={plot_id} | state={state} | "
                        f"auto-window=[{p_start} -> {p_end}]"
                    )
                else:
                    logger.info(f"[EXT-NEG] plot={plot_id} | state={state}")

                ext_geojson = geom.__geo_interface__   # plain GeoJSON dict

                try:
                    ext_df = self.downloader.extract_pixel_timeseries_wide(
                        geometry_geojson=ext_geojson,
                        start_date=p_start,
                        end_date=p_end,
                        scale=scale,
                        max_pixels=min(n_ext_per_plot, self.max_pixels),
                    )
                    ext_df["label"] = 0
                    ext_df["plot_id"] = f"{plot_id}_ext_neg"
                    ext_df["state"] = state
                    ext_df["anchor_date"] = ""
                    ext_df["date_start"] = p_start
                    ext_df["date_end"] = p_end
                    ext_df["cloud_gap_fraction"] = self._cloud_gap_fraction(ext_df)
                    all_rows.append(ext_df)
                    logger.info(f"  -> {len(ext_df)} external negative pixels")
                except Exception as exc:
                    logger.warning(f"  Failed external-neg sampling for {plot_id}: {exc}")

        if not all_rows:
            raise RuntimeError(
                "No samples generated. Check GEE connectivity and KML validity."
            )

        # ----------------------------------------------------------------
        # 3. Combine and balance
        # ----------------------------------------------------------------
        df_combined = pd.concat(all_rows, ignore_index=True)

        n_pos = (df_combined["label"] == 1).sum()
        n_neg = (df_combined["label"] == 0).sum()
        logger.info(
            f"Raw totals: {n_pos} cabbage | {n_neg} non-cabbage | "
            f"ratio={n_neg/max(n_pos,1):.2f}:1"
        )

        max_neg = int(n_pos * self.neg_ratio)
        if n_neg > max_neg:
            neg_idx = df_combined[df_combined["label"] == 0].index
            rng = np.random.default_rng(self.seed)
            drop_idx = rng.choice(neg_idx, size=n_neg - max_neg, replace=False)
            df_combined = df_combined.drop(index=drop_idx).reset_index(drop=True)
            logger.info(
                f"Balanced to {(df_combined['label']==1).sum()} cabbage | "
                f"{(df_combined['label']==0).sum()} non-cabbage"
            )

        # ----------------------------------------------------------------
        # 4. Metadata and 3D array
        # ----------------------------------------------------------------
        meta_cols = [
            "longitude", "latitude", "state", "label",
            "plot_id", "cloud_gap_fraction", "anchor_date",
            "date_start", "date_end",
        ]
        meta_cols = [c for c in meta_cols if c in df_combined.columns]
        meta_df = df_combined[meta_cols].copy()

        arr_3d = self._build_3d_array(df_combined)

        return df_combined, arr_3d, meta_df

    # ------------------------------------------------------------------
    # Cloud-gap fraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _cloud_gap_fraction(df: pd.DataFrame) -> pd.Series:
        """
        Compute per-pixel fraction of months with missing optical data.
        Uses optical_mask_YYYY_MM columns (0 = cloudy/missing, 1 = valid).
        Returns a scalar (mean across pixels) for logging, stored per-pixel.
        """
        mask_cols = [c for c in df.columns if c.startswith("optical_mask_")]
        if not mask_cols:
            return pd.Series(0.0, index=df.index)
        # 0 = missing, 1 = valid -> gap fraction = mean of (1 - mask)
        gap = 1.0 - df[mask_cols].fillna(0).mean(axis=1)
        return gap

    def _build_3d_array(self, df: pd.DataFrame) -> np.ndarray:
        """
        Reshape wide-format DataFrame into 3D array for BiLSTM.

        Shape: (n_samples, n_timesteps, n_features)

        Features per timestep (in order):
          S2 bands (10) + S2 indices (4) + S1 bands (2) + RVI (1) = 17 features
        Missing optical months are filled with NaN (handled by Masking layer).

        Time tags are auto-detected from column names (BAND_YYYY_MM format).
        """
        from data.gee_downloader import S2_BANDS, S1_BANDS

        feature_bands = S2_BANDS + ["NDVI", "EVI", "NDWI", "LSWI"] + S1_BANDS + ["RVI"]
        n_features = len(feature_bands)

        # Discover time tags from column names
        time_tags = sorted(set(
            "_".join(c.split("_")[1:])
            for c in df.columns
            if c.startswith("B2_") or c.startswith("NDVI_")
        ))
        n_timesteps = len(time_tags)

        if n_timesteps == 0:
            raise ValueError("No time-tagged band columns found in DataFrame.")

        n_samples = len(df)
        arr = np.full((n_samples, n_timesteps, n_features), np.nan, dtype=np.float32)

        for t_idx, tag in enumerate(time_tags):
            for f_idx, band in enumerate(feature_bands):
                col = f"{band}_{tag}"
                if col in df.columns:
                    arr[:, t_idx, f_idx] = df[col].values.astype(np.float32)

        logger.info(f"3D array shape: {arr.shape} | NaN fraction: {np.isnan(arr).mean():.3f}")
        return arr

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        df_2d: pd.DataFrame,
        arr_3d: np.ndarray,
        meta_df: pd.DataFrame,
        out_dir: str = "data/processed",
        prefix: str = "samples",
    ):
        """
        Save generated samples to disk.

        Files created
        -------------
        <out_dir>/<prefix>_2d.csv       â€” wide-format feature table
        <out_dir>/<prefix>_3d.npy       â€” 3D array for BiLSTM
        <out_dir>/<prefix>_meta.csv     â€” per-sample metadata
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / f"{prefix}_2d.csv"
        npy_path = out_dir / f"{prefix}_3d.npy"
        meta_path = out_dir / f"{prefix}_meta.csv"

        df_2d.to_csv(csv_path, index=False)
        np.save(str(npy_path), arr_3d)
        meta_df.to_csv(meta_path, index=False)

        logger.info(f"Saved 2D CSV  : {csv_path}  ({df_2d.shape})")
        logger.info(f"Saved 3D array: {npy_path}  ({arr_3d.shape})")
        logger.info(f"Saved metadata: {meta_path} ({meta_df.shape})")

    @staticmethod
    def load(
        out_dir: str = "data/processed",
        prefix: str = "samples",
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """Load previously saved samples."""
        out_dir = Path(out_dir)
        df_2d = pd.read_csv(out_dir / f"{prefix}_2d.csv")
        arr_3d = np.load(str(out_dir / f"{prefix}_3d.npy"), allow_pickle=False)
        meta_df = pd.read_csv(out_dir / f"{prefix}_meta.csv")
        logger.info(f"Loaded: 2D={df_2d.shape}, 3D={arr_3d.shape}, meta={meta_df.shape}")
        return df_2d, arr_3d, meta_df

    # ------------------------------------------------------------------
    # Train / test split â€” Leave-One-Plot-Out (LOPO)
    # ------------------------------------------------------------------

    @staticmethod
    def split_leave_one_plot_out(
        meta_df: pd.DataFrame,
        positive_only_rotation: bool = True,
        seed: int = 42,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, str], None, None]:
        """
        Leave-One-Plot-Out cross-validation grouped by plot_id.

        Each fold holds out ALL pixels from one cabbage plot (and its paired
        local-buffer negatives) as the validation set.  This prevents spatial
        autocorrelation leakage that would occur with random pixel-level splits
        when all positives come from the same geographic region (Jalgaon).

        Parameters
        ----------
        meta_df               : metadata DataFrame with plot_id and label columns
        positive_only_rotation: if True, only rotate over cabbage (label=1) plots;
                                external negatives always stay in training.
                                Set False to also rotate over negative plots.
        seed                  : random seed for shuffling fold order

        Yields
        ------
        (train_idx, val_idx, held_plot_id)
          train_idx    : np.ndarray of integer row indices for training
          val_idx      : np.ndarray of integer row indices for validation
          held_plot_id : the plot_id that was held out in this fold
        """
        rng = np.random.default_rng(seed)

        # Identify which plot_ids to rotate over
        if positive_only_rotation:
            # Only rotate cabbage plots; external negatives (_ext_neg suffix)
            # always stay in training
            rotate_plots = meta_df.loc[
                (meta_df["label"] == 1),
                "plot_id",
            ].unique().tolist()
        else:
            rotate_plots = meta_df["plot_id"].unique().tolist()

        rng.shuffle(rotate_plots)

        for held_plot in rotate_plots:
            # Val: all pixels from the held plot AND its paired local negatives
            # (plot_id == f"{held_plot}_local_neg")
            val_mask = (
                (meta_df["plot_id"] == held_plot) |
                (meta_df["plot_id"] == f"{held_plot}_local_neg")
            )
            val_idx = np.where(val_mask)[0]
            train_idx = np.where(~val_mask)[0]

            if len(val_idx) == 0:
                logger.warning(f"LOPO: no samples found for plot {held_plot}, skipping.")
                continue

            n_val_pos = int((meta_df.iloc[val_idx]["label"] == 1).sum())
            n_val_neg = int((meta_df.iloc[val_idx]["label"] == 0).sum())
            logger.info(
                f"LOPO fold | held={held_plot} | "
                f"val: {n_val_pos} cabbage + {n_val_neg} non-cabbage | "
                f"train: {len(train_idx)} samples"
            )
            yield train_idx, val_idx, held_plot

    @staticmethod
    def reserve_monsoon_test_plot(
        meta_df: pd.DataFrame,
        n_reserve: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Reserve the plot(s) with the highest cloud-gap fraction as a dedicated
        monsoon-robustness test set.

        This ensures the BiLSTM validation strategy always includes at least
        one complete Jalgaon plot where optical data is heavily missing during
        the monsoon (Juneâ€“September), forcing the model to rely on SAR features
        for those months.

        Parameters
        ----------
        meta_df   : metadata DataFrame (must have plot_id, label,
                    cloud_gap_fraction columns)
        n_reserve : number of plots to reserve (default 1)

        Returns
        -------
        train_idx      : np.ndarray â€” indices for training
        monsoon_idx    : np.ndarray â€” indices for monsoon test set
        reserved_plots : list of reserved plot_ids
        """
        if "cloud_gap_fraction" not in meta_df.columns:
            raise ValueError(
                "meta_df must have a 'cloud_gap_fraction' column. "
                "Re-run generate() with the updated SampleGenerator."
            )

        # Compute mean cloud-gap fraction per cabbage plot
        cabbage_meta = meta_df[meta_df["label"] == 1].copy()
        plot_cloud_means = (
            cabbage_meta.groupby("plot_id")["cloud_gap_fraction"]
            .mean()
            .sort_values(ascending=False)
        )

        if len(plot_cloud_means) == 0:
            raise ValueError("No cabbage plots found in meta_df.")

        reserved_plots = plot_cloud_means.head(n_reserve).index.tolist()
        logger.info(
            f"Monsoon test plots (highest cloud-gap): "
            + ", ".join(
                f"{p} (gap={plot_cloud_means[p]:.2f})" for p in reserved_plots
            )
        )

        # Reserve the plot AND its paired local negatives
        monsoon_mask = meta_df["plot_id"].isin(reserved_plots) | meta_df[
            "plot_id"
        ].isin([f"{p}_local_neg" for p in reserved_plots])

        monsoon_idx = np.where(monsoon_mask)[0]
        train_idx = np.where(~monsoon_mask)[0]

        logger.info(
            f"Monsoon test set: {len(monsoon_idx)} pixels | "
            f"Training set: {len(train_idx)} pixels"
        )
        return train_idx, monsoon_idx, reserved_plots

    @staticmethod
    def train_test_split_by_state(
        df_2d: pd.DataFrame,
        arr_3d: np.ndarray,
        meta_df: pd.DataFrame,
        test_states: list,
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> dict:
        """
        Split data into train / val / test sets by geographic state.

        Test set = all samples from test_states (geographic hold-out).
        Val set  = val_fraction of remaining training samples (random).

        Note: for the Jalgaon single-region scenario, prefer
        ``split_leave_one_plot_out()`` over this method to avoid spatial
        autocorrelation leakage.

        Returns
        -------
        dict with keys: X_train_2d, X_val_2d, X_test_2d,
                        X_train_3d, X_val_3d, X_test_3d,
                        y_train, y_val, y_test,
                        meta_train, meta_val, meta_test
        """
        rng = np.random.default_rng(seed)

        test_mask = meta_df["state"].isin(test_states)
        train_val_mask = ~test_mask

        idx_test = np.where(test_mask)[0]
        idx_train_val = np.where(train_val_mask)[0]

        rng.shuffle(idx_train_val)
        n_val = max(1, int(len(idx_train_val) * val_fraction))
        idx_val = idx_train_val[:n_val]
        idx_train = idx_train_val[n_val:]

        def _split(arr, idx):
            if isinstance(arr, pd.DataFrame):
                return arr.iloc[idx].reset_index(drop=True)
            return arr[idx]

        labels = meta_df["label"].values

        return {
            "X_train_2d": _split(df_2d, idx_train),
            "X_val_2d": _split(df_2d, idx_val),
            "X_test_2d": _split(df_2d, idx_test),
            "X_train_3d": _split(arr_3d, idx_train),
            "X_val_3d": _split(arr_3d, idx_val),
            "X_test_3d": _split(arr_3d, idx_test),
            "y_train": labels[idx_train],
            "y_val": labels[idx_val],
            "y_test": labels[idx_test],
            "meta_train": _split(meta_df, idx_train),
            "meta_val": _split(meta_df, idx_val),
            "meta_test": _split(meta_df, idx_test),
        }

    @staticmethod
    def lopo_split_to_arrays(
        df_2d: pd.DataFrame,
        arr_3d: np.ndarray,
        meta_df: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> dict:
        """
        Convert LOPO index arrays into train/val data dicts.

        Convenience wrapper so LOPO folds can be used identically to
        ``train_test_split_by_state`` output.

        Returns
        -------
        dict with keys: X_train_2d, X_val_2d, X_train_3d, X_val_3d,
                        y_train, y_val, meta_train, meta_val
        """
        labels = meta_df["label"].values

        def _s(arr, idx):
            if isinstance(arr, pd.DataFrame):
                return arr.iloc[idx].reset_index(drop=True)
            return arr[idx]

        return {
            "X_train_2d": _s(df_2d, train_idx),
            "X_val_2d": _s(df_2d, val_idx),
            "X_train_3d": _s(arr_3d, train_idx),
            "X_val_3d": _s(arr_3d, val_idx),
            "y_train": labels[train_idx],
            "y_val": labels[val_idx],
            "meta_train": _s(meta_df, train_idx),
            "meta_val": _s(meta_df, val_idx),
        }
