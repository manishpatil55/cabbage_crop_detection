"""
predictor.py
============
End-to-end cabbage crop detection inference pipeline.

Design principles
-----------------
1. CROP-BASED, NOT SOIL-BASED
   The model learns cabbage's temporal spectral signature â€” the shape of how
   NDVI/EVI/SAR change across seasons â€” not absolute reflectance values.

2. SINGLE-DATE INTERFACE
   Your testing team confirms a crop on a specific date.  You provide:
       kml_path  â€” the polygon
       crop_date â€” the date the crop was confirmed present
   The system internally computes the optimal GEE download window
   (crop_date âˆ’ 6 months â†’ crop_date + 6 months, capped to available data)
   and downloads exactly what is needed.  No date range required.

3. AUTOMATIC GENERALISATION
   Works across all of India without any labels from the target region:
   - Temporal statistics (min/max/mean/std of NDVI across months) are
     region-independent â€” they describe the *shape* of the crop cycle.
   - SAR features (VV, VH, RVI) penetrate clouds and are soil-independent.
   - Phenology features (season length, peak NDVI timing) are invariant.

Usage â€” testing team workflow
------------------------------
    from inference.predictor import CabbagePredictor

    predictor = CabbagePredictor()

    result = predictor.predict(
        kml_path="plots/field_42.kml",
        crop_date="2024-08-15",        # date crop was confirmed
    )

    print(f"Is cabbage: {result['is_cabbage']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"Probability map: {result['probability_map_path']}")

Command-line usage
------------------
    python inference/predictor.py plots/field_42.kml 2024-08-15
    python inference/predictor.py plots/field_42.kml 2024-08-15 --state Assam
    python inference/predictor.py plots/field_42.kml 2024-08-15 --output outputs/
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Window computation â€” the only date logic the user never needs to touch
# ---------------------------------------------------------------------------

def _crop_date_to_window(crop_date: str, months_before: int = 3, months_after: int = 3) -> Tuple[str, str]:
    """
    Convert a single confirmed crop date to a GEE download window.

    Why Â±3 months (default)?
    ------------------------
    Cabbage lifecycle is 60-120 days, but India has extreme cloud variability.
    In monsoon regions (NE, hills) optical data can be missing for 2-3 months.
    Â±3 months = 6 monthly composites, ensuring at least 4-5 usable observations
    even in the cloudiest regions.  SAR (Sentinel-1) fills remaining gaps.

    This makes the model work across ALL Indian states and seasons:
    - Rabi (Oct-Mar): Northern plains, Eastern, Western
    - Kharif (Jun-Oct): Hills, parts of Southern India
    - Year-round: Some hill stations, polytunnel cultivation

    Parameters
    ----------
    crop_date    : "YYYY-MM-DD" â€” the date the crop was confirmed present
    months_before: months before crop_date to include (default 3)
    months_after : months after crop_date to include (default 3)

    Returns
    -------
    (start_date, end_date) as "YYYY-MM-DD" strings
    """
    try:
        from dateutil.relativedelta import relativedelta
        anchor = datetime.strptime(crop_date, "%Y-%m-%d").date()
        start = anchor - relativedelta(months=months_before)
        end   = anchor + relativedelta(months=months_after)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    except ImportError:
        # Fallback without dateutil
        anchor = datetime.strptime(crop_date, "%Y-%m-%d")
        from datetime import timedelta
        start = anchor - timedelta(days=30 * months_before)
        end   = anchor + timedelta(days=30 * months_after)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Main predictor
# ---------------------------------------------------------------------------

class CabbagePredictor:
    """
    cabbage crop detection predictor.

    Loads the best saved model (Stacking Ensemble / XGBoost / RF) from
    best_model.pkl, which contains the model object, scaler, feature columns,
    train medians for imputation, and optimal threshold â€” all saved by train.py.

    Single-date interface: provide a KML and the date the crop was confirmed.
    The system handles everything else internally.

    Parameters
    ----------
    model_dir   : directory containing saved model files
    config_path : path to config.yaml
    """

    def __init__(
        self,
        model_dir: str = "models/saved",
        config_path: str = "config.yaml",
    ):
        self.model_dir = Path(model_dir)
        self.config_path = config_path

        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.model = None
        self.model_name = None
        self.scaler = None
        self.feature_cols = []
        self.train_medians = None
        self.optimal_threshold = self.cfg.get("inference", {}).get("probability_threshold", 0.5)
        self._models_loaded = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self):
        """Load the best trained model from best_model.pkl.

        best_model.pkl is saved by train.py and contains:
          - model        : the trained classifier (Stacking / XGBoost / RF)
          - scaler       : fitted StandardScaler
          - feature_cols : list of feature column names
          - train_medians: per-feature medians for NaN imputation
          - optimal_threshold : calibrated classification threshold
          - model_type   : string identifier ("stack", "xgb", "rf")
        """
        if self._models_loaded:
            return
        import pickle

        best_path = self.model_dir / "best_model.pkl"
        if not best_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {best_path}. Run train.py first."
            )

        with open(best_path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data.get("scaler")
        self.feature_cols = data.get("feature_cols", [])
        self.train_medians = data.get("train_medians")
        self.optimal_threshold = data.get("optimal_threshold", 0.5)
        self.model_name = data.get("model_type", "unknown")

        self._models_loaded = True
        logger.info(
            f"Loaded model: {self.model_name} | "
            f"threshold: {self.optimal_threshold:.3f} | "
            f"features: {len(self.feature_cols)}"
        )


    # ------------------------------------------------------------------
    # PRIMARY ENTRY POINT â€” single date, no date range needed
    # ------------------------------------------------------------------

    def predict(
        self,
        kml_path: str,
        crop_date: str,
        target_state: Optional[str] = None,
        output_dir: str = "outputs",
        scale: int = 10,
        probability_threshold: Optional[float] = None,
        months_before: int = 2,
        months_after: int = 2,
    ) -> Dict:
        """
        Detect cabbage crop in a KML polygon on a specific confirmed date.

        Parameters
        ----------
        kml_path    : path to .kml or .kmz file
        crop_date   : "YYYY-MM-DD" â€” the date the crop was confirmed present.
                      This is the ONLY date you need to provide.
                      The system automatically downloads Â±2 months of satellite
                      data centred on this date.
        target_state: Indian state name (optional â€” used for logging only)
        output_dir  : folder where GeoTIFF outputs are saved
        scale       : pixel resolution in metres (default 10 = Sentinel native)
        probability_threshold : override the 0.5 default if needed
        months_before : months of history before crop_date (default 2)
        months_after  : months of future after crop_date (default 2)

        Returns
        -------
        dict with keys:
          is_cabbage            : bool   â€” True if majority of pixels are cabbage
          confidence           : float  â€” mean cabbage probability (0â€“1)
          cabbage_fraction      : float  â€” fraction of pixels classified as cabbage
          n_pixels             : int    â€” total pixels analysed
          probability_map_path : str    â€” path to probability GeoTIFF
          binary_map_path      : str    â€” path to binary classification GeoTIFF
          crop_date            : str    â€” the input crop date
          window_start         : str    â€” actual GEE download start date
          window_end           : str    â€” actual GEE download end date
          state                : str    â€” detected or provided state name
        """
        self.load_models()

        threshold = probability_threshold or self.optimal_threshold
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # â”€â”€ Step 1: Parse KML â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        logger.info(f"[1/5] Parsing KML: {kml_path}")
        gdf = self._parse_kml(kml_path, target_state)
        plot_id = gdf["plot_id"].iloc[0]
        state = target_state or gdf["state"].iloc[0]

        # â”€â”€ Step 2: Compute GEE window from single crop date â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        window_start, window_end = _crop_date_to_window(
            crop_date, months_before, months_after
        )
        logger.info(
            f"[2/5] Crop date: {crop_date} â†’ "
            f"GEE window: {window_start} â†’ {window_end} "
            f"({months_before + months_after} months)"
        )

        # â”€â”€ Step 3: Download satellite data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        logger.info(f"[3/5] Downloading Sentinel-1 + Sentinel-2 from GEE...")
        df_wide = self._download_gee_data(gdf, window_start, window_end, scale)

        # â”€â”€ Step 4: Feature engineering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        logger.info("[4/5] Computing crop features...")
        df_2d, time_tags = self._compute_features(df_wide)

        # â”€â”€ Step 5: Model inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        logger.info(f"[5/5] Running {self.model_name} inference...")

        # Align features to training columns
        for col in self.feature_cols:
            if col not in df_2d.columns:
                df_2d[col] = np.nan

        arr = df_2d[self.feature_cols].values.astype(np.float32)

        # Impute NaN using training medians (not inference-time medians)
        if self.train_medians is not None:
            nan_mask = np.isnan(arr)
            if nan_mask.any():
                arr[nan_mask] = np.take(self.train_medians, np.where(nan_mask)[1])
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale using the training scaler
        if self.scaler is not None:
            arr = self.scaler.transform(arr)

        probs = self.model.predict_proba(arr)[:, 1]

        logger.info(f"  Mean probability: {probs.mean():.3f}")

        # â”€â”€ Export maps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        prob_path, binary_path = self._export_maps(
            df_wide, probs, threshold, output_dir, plot_id, crop_date
        )

        # â”€â”€ Build result â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        cabbage_fraction = float((probs >= threshold).mean())
        result = {
            "is_cabbage":            cabbage_fraction >= 0.5,
            "confidence":           float(probs.mean()),
            "cabbage_fraction":      cabbage_fraction,
            "n_pixels":             len(probs),
            "probability_map_path": str(prob_path),
            "binary_map_path":      str(binary_path),
            "crop_date":            crop_date,
            "window_start":         window_start,
            "window_end":           window_end,
            "state":                state,
            "plot_id":              plot_id,
        }

        logger.info(
            f"\n{'='*55}\n"
            f"  RESULT  |  {plot_id}  |  {crop_date}\n"
            f"  Is cabbage      : {'YES âœ“' if result['is_cabbage'] else 'NO âœ—'}\n"
            f"  Confidence     : {result['confidence']*100:.1f}%\n"
            f"  Cabbage pixels  : {cabbage_fraction*100:.1f}% of plot\n"
            f"{'='*55}"
        )
        return result

    # ------------------------------------------------------------------
    # Batch prediction â€” list of (kml_path, crop_date) pairs
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        jobs: List[Dict],
        output_dir: str = "outputs",
    ) -> List[Dict]:
        """
        Run prediction on multiple plots.

        Parameters
        ----------
        jobs : list of dicts, each with keys:
                 kml_path  (required)
                 crop_date (required)
                 state     (optional)
        output_dir : output folder

        Returns
        -------
        list of result dicts

        Example
        -------
            results = predictor.predict_batch([
                {"kml_path": "plots/field1.kml", "crop_date": "2024-08-15"},
                {"kml_path": "plots/field2.kml", "crop_date": "2024-03-10",
                 "state": "Tamil Nadu"},
            ])
        """
        results = []
        for i, job in enumerate(jobs):
            logger.info(f"\n[Batch {i+1}/{len(jobs)}] {job['kml_path']}")
            try:
                result = self.predict(
                    kml_path=job["kml_path"],
                    crop_date=job["crop_date"],
                    target_state=job.get("state"),
                    output_dir=output_dir,
                )
                results.append(result)
            except Exception as exc:
                logger.error(f"  Failed: {exc}")
                results.append({
                    "kml_path": job["kml_path"],
                    "crop_date": job.get("crop_date"),
                    "error": str(exc),
                    "is_cabbage": None,
                })
        return results

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _parse_kml(self, kml_path: str, state: Optional[str] = None):
        from data.kml_parser import KMLParser
        parser = KMLParser()
        gdf = parser.parse_file(kml_path, state=state)
        logger.info(f"  {len(gdf)} polygon(s) | area={gdf['area_ha'].sum():.2f} ha | state={gdf['state'].iloc[0]}")
        return gdf

    def _download_gee_data(
        self,
        gdf,
        start_date: str,
        end_date: str,
        scale: int,
    ) -> pd.DataFrame:
        from data.gee_downloader import GEEDownloader
        from shapely.ops import unary_union

        dl = GEEDownloader(config_path=self.config_path)
        dl.initialize()

        union_geom = unary_union(gdf.geometry.values)
        geojson_dict = union_geom.__geo_interface__

        max_pixels = self.cfg["sampling"]["max_pixels_per_plot"]
        df_wide = dl.extract_pixel_timeseries_wide(
            geometry_geojson=geojson_dict,
            start_date=start_date,
            end_date=end_date,
            scale=scale,
            max_pixels=max_pixels,
        )
        logger.info(f"  {df_wide.shape[0]} pixels Ã— {df_wide.shape[1]} columns downloaded")
        return df_wide

    def _compute_features(
        self, df_wide: pd.DataFrame
    ) -> Tuple[pd.DataFrame, list]:
        from features.spectral_indices import SpectralIndexCalculator
        from features.temporal_stats import TemporalStatsExtractor
        from features.phenology_features_heading import HeadingPhenologyExtractor

        time_tags = self._detect_time_tags(df_wide)
        logger.info(f"  {len(time_tags)} monthly composites: {time_tags[0]} -> {time_tags[-1]}")

        # Spectral indices (NDVI, NDRE, CCCI, EVI, LSWI, etc.)
        calc = SpectralIndexCalculator()
        df_wide = calc.compute_all(df_wide, time_tags=time_tags)

        # Temporal statistics (2D table for XGBoost)
        stats_ex = TemporalStatsExtractor(self.config_path)
        df_stats = stats_ex.compute(df_wide, time_tags=time_tags)

        # Heading-vegetable phenology features (BBCH-scale, cabbage-specific)
        pheno_ex = HeadingPhenologyExtractor(self.config_path)
        df_pheno = pheno_ex.compute(df_wide, time_tags=time_tags)

        meta_cols = [c for c in ["longitude", "latitude", "state", "label", "plot_id"]
                     if c in df_stats.columns]
        feat_cols = [c for c in df_stats.columns if c not in meta_cols]
        df_2d = pd.concat([
            df_stats[meta_cols + feat_cols].reset_index(drop=True),
            df_pheno.reset_index(drop=True),
        ], axis=1)

        logger.info(f"  2D features: {df_2d.shape}")
        return df_2d, time_tags

    def _export_maps(
        self,
        df_wide: pd.DataFrame,
        probs: np.ndarray,
        threshold: float,
        output_dir: Path,
        plot_id: str,
        crop_date: str,
    ) -> Tuple[Path, Path]:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        lons = df_wide["longitude"].values
        lats = df_wide["latitude"].values
        binary = (probs >= threshold).astype(np.uint8)

        min_lon, max_lon = lons.min(), lons.max()
        min_lat, max_lat = lats.min(), lats.max()
        pixel_size = self._estimate_pixel_size(lons, lats)
        buf = pixel_size * 2
        min_lon -= buf; max_lon += buf
        min_lat -= buf; max_lat += buf

        width  = max(1, int((max_lon - min_lon) / pixel_size))
        height = max(1, int((max_lat - min_lat) / pixel_size))
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
        crs = CRS.from_epsg(4326)

        prob_raster   = np.full((height, width), np.nan, dtype=np.float32)
        binary_raster = np.zeros((height, width), dtype=np.uint8)

        for lon, lat, prob, bval in zip(lons, lats, probs, binary):
            col = np.clip(int((lon - min_lon) / pixel_size), 0, width - 1)
            row = np.clip(int((max_lat - lat) / pixel_size), 0, height - 1)
            prob_raster[row, col]   = prob
            binary_raster[row, col] = bval

        date_tag = crop_date.replace("-", "")
        prob_path   = output_dir / f"{plot_id}_{date_tag}_probability.tif"
        binary_path = output_dir / f"{plot_id}_{date_tag}_binary.tif"

        _write_tif(prob_path,   prob_raster,   np.float32, crs, transform, nodata=np.nan,
                   tags={"description": "cabbage probability (0-1)", "crop_date": crop_date})
        _write_tif(binary_path, binary_raster, np.uint8,   crs, transform, nodata=255,
                   tags={"description": "cabbage binary (1=Yes, 0=No)", "crop_date": crop_date,
                         "threshold": str(threshold)})

        logger.info(f"  Probability map : {prob_path}")
        logger.info(f"  Binary map      : {binary_path}")
        return prob_path, binary_path

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_time_tags(df: pd.DataFrame) -> list:
        tags = set()
        for col in df.columns:
            parts = col.split("_")
            if len(parts) >= 3:
                try:
                    year  = int(parts[-2])
                    month = int(parts[-1])
                    if 2000 <= year <= 2100 and 1 <= month <= 12:
                        tags.add(f"{year}_{month:02d}")
                except ValueError:
                    pass
        return sorted(tags)

    @staticmethod
    def _estimate_pixel_size(lons: np.ndarray, lats: np.ndarray) -> float:
        if len(lons) < 2:
            return 0.0001
        sorted_lons = np.sort(np.unique(lons))
        if len(sorted_lons) > 1:
            diffs = np.diff(sorted_lons)
            pos = diffs[diffs > 0]
            return float(np.median(pos)) if len(pos) > 0 else 0.0001
        return 0.0001


# ---------------------------------------------------------------------------
# GeoTIFF writer helper
# ---------------------------------------------------------------------------

def _write_tif(path, data, dtype, crs, transform, nodata, tags):
    import rasterio
    with rasterio.open(
        str(path), "w",
        driver="GTiff",
        height=data.shape[0], width=data.shape[1],
        count=1, dtype=dtype,
        crs=crs, transform=transform,
        nodata=nodata, compress="lzw",
    ) as dst:
        dst.write(data, 1)
        dst.update_tags(**tags)


# ---------------------------------------------------------------------------
# CLI â€” the simplest possible interface for the testing team
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="cabbage crop detection â€” provide a KML and a confirmed crop date."
    )
    parser.add_argument("kml_path",  help="Path to .kml or .kmz file")
    parser.add_argument("crop_date", help="Confirmed crop date: YYYY-MM-DD")
    parser.add_argument("--state",      default=None,       help="Target state name (optional)")
    parser.add_argument("--output",     default="outputs",  help="Output directory")
    parser.add_argument("--model_dir",  default="models/saved")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--threshold",  type=float, default=None)
    args = parser.parse_args()

    predictor = CabbagePredictor(model_dir=args.model_dir, config_path=args.config)

    result = predictor.predict(
        kml_path=args.kml_path,
        crop_date=args.crop_date,
        target_state=args.state,
        output_dir=args.output,
        probability_threshold=args.threshold,
    )

    print("\n" + "="*55)
    print("cabbage detection RESULT")
    print("="*55)
    print(f"  KML file       : {args.kml_path}")
    print(f"  Crop date      : {result['crop_date']}")
    print(f"  State          : {result['state']}")
    print(f"  Is cabbage      : {'YES' if result['is_cabbage'] else 'NO'}")
    print(f"  Confidence     : {result['confidence']*100:.1f}%")
    print(f"  Cabbage pixels  : {result['cabbage_fraction']*100:.1f}%")
    print(f"  Pixels total   : {result['n_pixels']}")
    print(f"  Probability map: {result['probability_map_path']}")
    print(f"  Binary map     : {result['binary_map_path']}")
    print("="*55)
