"""
Cabbage Detection API
====================
FastAPI server with Swagger UI for cabbage crop detection.

Run:  python api.py
Swagger UI: http://localhost:8009/docs
"""

import logging
import os
import pickle
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cabbage Crop Detection API",
    description="""
## Pan-India Cabbage Detection from Satellite Imagery

Upload a KML file with a farm boundary polygon and a date when the crop was observed.
The API will:
1. Download Sentinel-1 (SAR) + Sentinel-2 (optical) data from Google Earth Engine
2. Compute spectral indices (NDVI, NDRE, CCCI, EVI, LSWI, etc.)
3. Extract temporal + heading phenological features
4. Run the trained ML model (XGBoost / RF / Stacking ensemble)
5. Return probability of cabbage presence

### How it works
- **Input**: KML file + observation date
- **Output**: Cabbage probability (0-100%), classification, per-pixel stats
- **Model**: Trained on cabbage farms across India (Rabi season)
- **Satellites**: Sentinel-1 (radar, cloud-free) + Sentinel-2 (optical)
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response models ───────────────────────────────────────────────────────

class DetectionResult(BaseModel):
    # ── 1. VERDICT (top — what testers look at first) ──────────────
    verdict: str = Field(description="Human-readable verdict with confidence level")
    classification: str = Field(description="'CABBAGE' or 'NON-CABBAGE'")
    label: int = Field(description="1 = Cabbage, 0 = Non-Cabbage")
    confidence: str = Field(description="Confidence level: HIGH / MODERATE / LOW")
    cabbage_percentage: float = Field(description="% of area classified as cabbage")

    # ── 2. LOCATION INFO ──────────────────────────────────────────
    area_hectares: Optional[float] = Field(default=None, description="Total farm area in hectares")
    centroid_latitude: Optional[float] = Field(default=None, description="Farm centroid latitude")
    centroid_longitude: Optional[float] = Field(default=None, description="Farm centroid longitude")
    bounding_box: Optional[dict] = Field(default=None, description="Farm bounding box {north, south, east, west}")
    state: Optional[str] = Field(default=None, description="Indian state (if detected from KML filename)")

    # ── 3. PIXEL ANALYSIS ─────────────────────────────────────────
    total_pixels: int = Field(description="Total pixels analysed")
    cabbage_pixels: int = Field(description="Pixels classified as cabbage")
    non_cabbage_pixels: int = Field(description="Pixels classified as non-cabbage")
    mean_probability: float = Field(description="Mean cabbage probability (0-1)")
    max_probability: float = Field(description="Max cabbage probability")
    min_probability: float = Field(description="Min cabbage probability")

    # ── 4. CLOUD COVER & DATA QUALITY ─────────────────────────────
    cloud_free_percentage: Optional[float] = Field(default=None, description="% of months with clear optical data (0-100)")
    cloudy_months: Optional[int] = Field(default=None, description="Number of months affected by cloud cover")
    total_months: Optional[int] = Field(default=None, description="Total months in satellite window")
    cloud_details: Optional[dict] = Field(default=None, description="Per-month cloud status {YYYY_MM: 'clear'|'cloudy'}")
    data_quality: Optional[str] = Field(default=None, description="EXCELLENT / GOOD / FAIR / POOR based on cloud coverage")

    # ── 5. MODEL & SATELLITE INFO ─────────────────────────────────
    status: str = Field(description="'success' or 'error'")
    model_name: str = Field(description="Model used: Stacking Ensemble (RF + XGBoost)")
    threshold: float = Field(description="Classification threshold used")
    satellites_used: str = Field(default="Sentinel-1 (SAR) + Sentinel-2 (Optical)", description="Satellites used")
    date_checked: Optional[str] = Field(default=None, description="Crop date provided by user")
    satellite_window: Optional[str] = Field(default=None, description="Satellite data window used")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    accuracy: Optional[float]
    threshold: Optional[float]


# ── Model loading ─────────────────────────────────────────────────────────

MODEL_DIR = Path("models/saved")
model_data = None


def load_model():
    global model_data
    if model_data is not None:
        return

    best_path = MODEL_DIR / "best_model.pkl"
    if best_path.exists():
        with open(best_path, "rb") as f:
            model_data = pickle.load(f)
        logger.info(f"Loaded best model: {model_data.get('model_type', 'unknown')}")
        logger.info(f"  Threshold: {model_data.get('optimal_threshold', 0.5)}")
        logger.info(f"  Features: {len(model_data.get('feature_cols', []))}")
        logger.info(f"  Metrics: {model_data.get('metrics', {})}")
        return

    # Fallback: try loading XGBoost model
    xgb_path = MODEL_DIR / "xgb_model.pkl"
    if xgb_path.exists():
        with open(xgb_path, "rb") as f:
            data = pickle.load(f)
        model_data = {
            "model": data["model"],
            "scaler": data["scaler"],
            "feature_cols": data["feature_cols"],
            "train_medians": data.get("train_medians"),
            "optimal_threshold": 0.5,
            "model_type": "xgb",
            "metrics": {},
        }
        logger.info("Loaded XGBoost model (fallback)")
        return

    logger.warning("No trained model found in models/saved/. Run train.py first.")


# ── Prediction logic ──────────────────────────────────────────────────────

def predict_from_kml(kml_path: str, crop_date: str) -> dict:
    """Run the full prediction pipeline on a KML file."""
    import ee
    try:
        ee.Initialize(project="crop-detection-494609")
    except Exception:
        pass

    from data.kml_parser import KMLParser
    from data.sample_generator import SampleGenerator
    from features.spectral_indices import SpectralIndexCalculator
    from features.temporal_stats import TemporalStatsExtractor
    from features.phenology_features_heading import HeadingPhenologyExtractor

    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_cols = model_data["feature_cols"]
    train_medians = model_data.get("train_medians")
    threshold = model_data.get("optimal_threshold", 0.5)

    # 1. Parse KML
    logger.info(f"[1/5] Parsing KML: {kml_path}")
    parser = KMLParser()
    gdf = parser.parse_file(kml_path, anchor_date_override=crop_date)
    if gdf.empty:
        raise HTTPException(status_code=400, detail="No valid polygons found in KML file.")

    total_area = gdf.geometry.to_crs(epsg=32643).area.sum() / 10000  # hectares

    # Extract location info from the KML geometry
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    centroid = gdf.geometry.unary_union.centroid
    detected_state = gdf["state"].iloc[0] if "state" in gdf.columns else None
    if detected_state in (None, "Unknown", ""):
        detected_state = None

    # Compute satellite window from the crop date (read from config for consistency)
    import yaml as _yaml
    _cfg = _yaml.safe_load(open("config.yaml"))
    _months_before = _cfg.get("compositing", {}).get("months_before", 2)
    _months_after = _cfg.get("compositing", {}).get("months_after", 2)

    from dateutil.relativedelta import relativedelta
    from datetime import datetime as _dt
    try:
        cd = _dt.strptime(crop_date, "%Y-%m-%d")
        win_start = (cd - relativedelta(months=_months_before)).strftime("%Y-%m-%d")
        win_end = (cd + relativedelta(months=_months_after)).strftime("%Y-%m-%d")
        sat_window = f"{win_start} to {win_end}"
    except Exception:
        sat_window = None

    # 2. Download satellite data
    logger.info("[2/5] Downloading satellite data from GEE...")
    gen = SampleGenerator(config_path="config.yaml")
    df_wide, _, meta_df = gen.generate(gdf=gdf, external_neg_gdf=None)

    if df_wide.empty:
        raise HTTPException(status_code=400, detail="No satellite data returned for this area/date.")

    # ── Cloud cover analysis (from optical_mask columns) ──────────
    mask_cols = sorted([c for c in df_wide.columns if c.startswith("optical_mask_")])
    total_months_count = len(mask_cols)
    cloud_details = {}
    cloudy_month_count = 0
    for mc_col in mask_cols:
        tag = mc_col.replace("optical_mask_", "")
        # optical_mask: 1 = valid/clear, 0 = cloudy/missing
        clear_frac = df_wide[mc_col].replace(-999, 0).mean()
        if clear_frac >= 0.5:
            cloud_details[tag] = "clear"
        else:
            cloud_details[tag] = "cloudy"
            cloudy_month_count += 1
    cloud_free_pct = round(
        ((total_months_count - cloudy_month_count) / max(total_months_count, 1)) * 100, 1
    )
    if cloud_free_pct >= 90:
        data_quality = "EXCELLENT"
    elif cloud_free_pct >= 70:
        data_quality = "GOOD"
    elif cloud_free_pct >= 50:
        data_quality = "FAIR — SAR features compensating for cloud gaps"
    else:
        data_quality = "POOR — heavy cloud cover, relying mostly on SAR radar data"

    # 3. Compute features
    logger.info("[3/5] Computing spectral indices...")
    tags = set()
    for col in df_wide.columns:
        parts = col.split("_")
        if len(parts) >= 3:
            try:
                y, m = int(parts[-2]), int(parts[-1])
                if 2000 <= y <= 2100 and 1 <= m <= 12:
                    tags.add(f"{y}_{m:02d}")
            except ValueError:
                pass
    time_tags = sorted(tags)

    calc = SpectralIndexCalculator()
    df_wide = calc.compute_all(df_wide, time_tags=time_tags)

    logger.info("[4/5] Computing temporal & phenology features...")
    stats_ext = TemporalStatsExtractor("config.yaml")
    df_stats = stats_ext.compute(df_wide, time_tags=time_tags)

    pheno_ext = HeadingPhenologyExtractor("config.yaml")
    df_pheno = pheno_ext.compute(df_wide, time_tags=time_tags)

    meta_cols_list = ["longitude", "latitude", "state", "label", "plot_id",
                      "anchor_date", "date_start", "date_end", "cloud_gap_fraction"]
    mc = [c for c in meta_cols_list if c in df_stats.columns]
    fc = [c for c in df_stats.columns if c not in mc]
    df_2d = pd.concat([
        df_stats[mc + fc].reset_index(drop=True),
        df_pheno.reset_index(drop=True),
    ], axis=1)

    # 4. Align features to training columns
    for col in feature_cols:
        if col not in df_2d.columns:
            df_2d[col] = np.nan

    arr = df_2d[feature_cols].values.astype(np.float32)

    # Impute NaN using training medians
    if train_medians is not None:
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            arr[nan_mask] = np.take(train_medians, np.where(nan_mask)[1])
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale
    arr = scaler.transform(arr)

    # 5. Predict
    logger.info("[5/5] Running inference...")
    probs = model.predict_proba(arr)[:, 1]

    cabbage_mask = probs >= threshold
    cabbage_count = int(cabbage_mask.sum())
    total = len(probs)
    cabbage_pct = round((cabbage_count / total * 100) if total > 0 else 0, 1)

    # Classification, label, confidence, verdict
    if cabbage_pct >= 70:
        classification = "CABBAGE"
        label = 1
        confidence = "HIGH"
        verdict = (
            f"✅ HIGH CONFIDENCE — Cabbage crop detected. "
            f"{cabbage_pct}% of pixels ({cabbage_count}/{total}) classified as cabbage. "
            f"Mean probability: {probs.mean():.1%}. "
            f"Area: {total_area:.2f} hectares."
        )
    elif cabbage_pct >= 40:
        classification = "CABBAGE"
        label = 1
        confidence = "MODERATE"
        verdict = (
            f"⚠️ MODERATE CONFIDENCE — Partial cabbage presence detected. "
            f"{cabbage_pct}% of pixels ({cabbage_count}/{total}) show cabbage signatures. "
            f"Could be mixed cropping or early-stage growth."
        )
    elif cabbage_pct >= 10:
        classification = "NON-CABBAGE"
        label = 0
        confidence = "LOW"
        verdict = (
            f"❌ LOW CONFIDENCE — Unlikely cabbage. "
            f"Only {cabbage_pct}% of pixels ({cabbage_count}/{total}) show weak cabbage signatures. "
            f"This area is most likely NOT a cabbage field."
        )
    else:
        classification = "NON-CABBAGE"
        label = 0
        confidence = "HIGH"
        verdict = (
            f"❌ NOT CABBAGE — No cabbage crop detected. "
            f"Only {cabbage_pct}% of pixels ({cabbage_count}/{total}) matched. "
            f"This area does not contain cabbage crops."
        )

    return {
        # 1. VERDICT (top)
        "verdict": verdict,
        "classification": classification,
        "label": label,
        "confidence": confidence,
        "cabbage_percentage": cabbage_pct,

        # 2. LOCATION
        "area_hectares": round(total_area, 2) if total_area else None,
        "centroid_latitude": round(centroid.y, 6),
        "centroid_longitude": round(centroid.x, 6),
        "bounding_box": {
            "north": round(bounds[3], 6),
            "south": round(bounds[1], 6),
            "east": round(bounds[2], 6),
            "west": round(bounds[0], 6),
        },
        "state": detected_state,

        # 3. PIXEL ANALYSIS
        "total_pixels": total,
        "cabbage_pixels": cabbage_count,
        "non_cabbage_pixels": total - cabbage_count,
        "mean_probability": round(float(probs.mean()), 4),
        "max_probability": round(float(probs.max()), 4),
        "min_probability": round(float(probs.min()), 4),

        # 4. CLOUD COVER
        "cloud_free_percentage": cloud_free_pct,
        "cloudy_months": cloudy_month_count,
        "total_months": total_months_count,
        "cloud_details": cloud_details,
        "data_quality": data_quality,

        # 5. MODEL INFO
        "status": "success",
        "model_name": "Stacking Ensemble (Random Forest + XGBoost)",
        "threshold": round(threshold, 3),
        "satellites_used": "Sentinel-1 (SAR) + Sentinel-2 (Optical)",
        "date_checked": crop_date,
        "satellite_window": sat_window,
    }


# ── API Endpoints ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check if the API and model are healthy."""
    if model_data is None:
        return HealthResponse(
            status="error",
            model_loaded=False,
            model_name="none",
            accuracy=None,
            threshold=None,
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=model_data.get("model_type", "unknown"),
        accuracy=model_data.get("metrics", {}).get("accuracy"),
        threshold=model_data.get("optimal_threshold", 0.5),
    )


@app.post("/detect", response_model=DetectionResult, tags=["Detection"])
async def detect_cabbage(
    kml_file: UploadFile = File(..., description="KML file with farm boundary polygon"),
    crop_date: str = Form(..., description="Date when cabbage presence is to be checked (YYYY-MM-DD format)"),
):
    """
    ## Detect Cabbage Crop

    Upload a KML file containing a farm boundary polygon and specify the date
    you want to check for cabbage presence.

    The API will automatically download satellite data, compute vegetation indices,
    and classify the area as cabbage or non-cabbage using our stacked ensemble.

    ### Parameters:
    - **kml_file**: A `.kml` file with one or more polygon boundaries
    - **crop_date**: Date in `YYYY-MM-DD` format

    ### Returns:
    - Cabbage probability per pixel
    - Overall classification
    - Human-readable verdict
    """
    # Validate date
    try:
        from datetime import datetime
        parsed = datetime.strptime(crop_date, "%Y-%m-%d")
        crop_date_str = parsed.strftime("%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD (e.g. 2025-10-12)")

    # Save uploaded KML to temp file
    suffix = Path(kml_file.filename or "upload.kml").suffix or ".kml"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=".") as tmp:
        content = await kml_file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = predict_from_kml(tmp_path, crop_date_str)
        return DetectionResult(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Run server ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Cabbage Detection API")
    print("  Swagger UI: http://localhost:8009/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8009, log_level="info")

