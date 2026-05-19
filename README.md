# 🍌 Banana Crop Detection System

Pan-India banana crop detection using **Sentinel-1 SAR + Sentinel-2 optical** satellite data with a **Stacking Ensemble** (Random Forest + XGBoost + Logistic Regression meta-learner).

**Accuracy: 88.84%** | **F1: 0.902** | **AUC-ROC: 0.951** (at calibrated threshold 0.220)

---

## Quick Start

### 1. Install

```bash
cd banana_detection
pip install -r requirements.txt
earthengine authenticate       # One-time GEE setup
```

### 2. Train

```bash
# Place KML files in data/kml/banana/ and data/kml/non_banana/
python train.py
```

### 3. Run API Server

```bash
python api.py
# Server: http://localhost:8008
# Swagger UI: http://localhost:8008/docs
```

### 4. Detect Banana (API)

```bash
curl -X POST http://localhost:8008/detect \
  -F "kml_file=@farm_boundary.kml" \
  -F "crop_date=2025-10-12"
```

### 5. Detect Banana (Python)

```python
from inference.predictor import BananaPredictor

predictor = BananaPredictor(model_dir="models/saved", config_path="config.yaml")
result = predictor.predict(
    kml_path="path/to/farm.kml",
    crop_date="2025-10-12",        # Date banana was confirmed present
)
print(f"Is Banana: {result['is_banana']}")
print(f"Confidence: {result['confidence']:.1%}")
```

---

## Architecture

```
KML File + Crop Date
   │
   ▼
┌───────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                    │
│  kml_parser.py → gee_downloader.py → sample_generator.py      │
│  (Parse KML)     (S1+S2 monthly       (Balanced pixel         │
│                   composites via GEE)   sampling ±3 months)    │
└───────────────────────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING (436 features → 80 selected)             │
│                                                                │
│  spectral_indices.py   temporal_stats.py  phenology_features.py│
│  8 Optical indices:    Per-band stats:    Growth curve shape:  │
│  NDVI,EVI,NDWI,LSWI   min/max/mean/std   AUC, peak, slopes   │
│  SAVI,MSAVI,NBR,NDRE   p10–p90, CV       season length        │
│  3 SAR indices:        Monsoon vs dry     green-up timing      │
│  RVI, RFDI, CR         SAR-optical ratios NDVI-VV correlation  │
└───────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────┐  ┌──────────────┐
│ Random Forest│  │   XGBoost    │
│  (500 trees) │  │ (early stop) │
└──────┬───────┘  └──────┬───────┘
       │  5-fold OOF      │
       └────────┬─────────┘
                ▼
     ┌─────────────────────┐
     │  Logistic Regression │
     │  Meta-Learner        │
     └──────────┬──────────┘
                │
                ▼
     Threshold: 0.220 (calibrated)
     Accuracy: 88.84% | F1: 0.902
                │
                ▼
     ┌─────────────────────┐
     │  GeoTIFF Output     │
     │  + JSON API Response │
     └─────────────────────┘
```

---

## Project Structure

```
banana_detection/
├── config.yaml                  # All configuration
├── train.py                     # Training pipeline (run this)
├── api.py                       # FastAPI REST server
├── utils.py                     # Shared utility functions
├── requirements.txt             # Python dependencies
│
├── data/
│   ├── kml_parser.py            # KML/KMZ → GeoDataFrame
│   ├── gee_downloader.py        # Multi-backend satellite downloader (GEE/PC/SH)
│   ├── sample_generator.py      # Positive/negative pixel sampling
│   ├── kml/
│   │   ├── banana/              # Banana farm KML files
│   │   └── non_banana/          # Non-banana plot KML files
│   └── processed/               # Cached satellite data
│
├── features/
│   ├── spectral_indices.py      # 11 indices (8 optical + 3 SAR)
│   ├── temporal_stats.py        # Temporal statistics + seasonal contrast
│   └── phenology_features.py    # Phenological shape features
│
├── models/
│   ├── base_models.py           # RF and XGBoost model wrappers
│   └── saved/                   # Trained model files
│       ├── best_model.pkl       # Stacking ensemble (15 MB)
│       ├── rf_model.pkl         # Standalone RF
│       └── xgb_model.pkl        # Standalone XGBoost
│
├── inference/
│   └── predictor.py             # BananaPredictor class (CLI inference)
│
└── outputs/                     # GeoTIFF probability + binary maps
```

---

## Satellites Used

Both **Sentinel-1** and **Sentinel-2** are downloaded every month:

| Satellite | Type | Bands | Resolution | Cloud-proof? |
|:---|:---|:---|:---:|:---:|
| **Sentinel-2** | Optical | B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 | 10–20m | ❌ Masked by clouds |
| **Sentinel-1** | SAR Radar | VV, VH | 10m | ✅ Penetrates clouds |

### During cloudy months:
- **Sentinel-2** → cloud-masked → NaN (imputed with training medians)
- **Sentinel-1** → always valid → SAR features carry the classification

---

## Spectral Indices (11 total)

### Optical (Sentinel-2) — 8 indices

| Index | Formula | Purpose |
|:---|:---|:---|
| **NDVI** | (B8−B4)/(B8+B4) | Vegetation greenness |
| **EVI** | 2.5×(B8−B4)/(B8+6×B4−7.5×B2+1) | Enhanced vegetation (soil-corrected) |
| **NDWI** | (B3−B8)/(B3+B8) | Water/moisture content |
| **LSWI** | (B8−B11)/(B8+B11) | Leaf water content |
| **SAVI** | (B8−B4)/(B8+B4+0.5)×1.5 | Soil-adjusted vegetation |
| **MSAVI** | (2×B8+1−√((2×B8+1)²−8×(B8−B4)))/2 | Modified soil adjustment |
| **NBR** | (B8−B12)/(B8+B12) | Crop residue/moisture |
| **NDRE** | (B5−B4)/(B5+B4) | Chlorophyll (red edge) |

### SAR (Sentinel-1) — 3 indices

| Index | Formula | Purpose |
|:---|:---|:---|
| **RVI** | 4×VH/(VV+VH) | Biomass estimation |
| **RFDI** | (VV−VH)/(VV+VH) | Canopy density |
| **CR** | VH/VV | Volume scattering |

---

## API Response Format

```json
{
  "verdict": "✅ HIGH CONFIDENCE — Banana plantation detected. 85% of pixels classified as banana.",
  "classification": "BANANA",
  "label": 1,
  "confidence": "HIGH",
  "banana_percentage": 85.0,

  "area_hectares": 0.74,
  "centroid_latitude": 21.226072,
  "centroid_longitude": 75.623744,
  "bounding_box": { "north": 21.226, "south": 21.225, "east": 75.624, "west": 75.623 },
  "state": "Maharashtra",

  "total_pixels": 120,
  "banana_pixels": 102,
  "non_banana_pixels": 18,
  "mean_probability": 0.72,
  "max_probability": 0.96,
  "min_probability": 0.11,

  "cloud_free_percentage": 100.0,
  "cloudy_months": 0,
  "total_months": 7,
  "cloud_details": { "2023_08": "clear", "2023_09": "clear", "...": "..." },
  "data_quality": "EXCELLENT",

  "model_name": "Stacking Ensemble (Random Forest + XGBoost)",
  "threshold": 0.220,
  "satellites_used": "Sentinel-1 (SAR) + Sentinel-2 (Optical)",
  "date_checked": "2023-11-15",
  "satellite_window": "2023-08-15 to 2024-02-15"
}
```

---

## Classification Logic

| Banana % | Classification | Label | Confidence |
|:---:|:---|:---:|:---|
| ≥ 70% | BANANA | 1 | HIGH |
| 40–69% | BANANA | 1 | MODERATE |
| 10–39% | NON-BANANA | 0 | LOW |
| < 10% | NON-BANANA | 0 | HIGH |

---

## Model Performance

| Model | Accuracy | F1 | AUC-ROC |
|:---|:---:|:---:|:---:|
| Random Forest | 77.39% | 0.759 | 0.9486 |
| XGBoost | 81.04% | 0.807 | 0.9476 |
| **Stacking (t=0.220)** | **88.84%** | **0.902** | **0.9513** |

### Training Data
- **100 KML plots** (50 banana + 50 non-banana)
- **14,542 pixel samples** (7,367 banana + 7,175 non-banana)
- **Regions**: Maharashtra (Jalgaon) + Andhra Pradesh
- **Spatial CV**: GroupShuffleSplit by `plot_id` (no data leakage)

---

## Configuration (`config.yaml`)

| Key | Value | Description |
|:---|:---|:---|
| `data_backend` | `"gee"` | GEE / planetary_computer / sentinel_hub |
| `gee.project_id` | `"crop-detection-494609"` | Your GEE project ID |
| `compositing.months_before` | 3 | Training: ±3 months from anchor date |
| `inference.probability_threshold` | **0.220** | Calibrated classification threshold |
| `sampling.buffer_m` | 750 | Negative sample buffer (meters) |

---

## Adding New Training Data

1. Place new KML files in `data/kml/banana/` or `data/kml/non_banana/`
2. Name format: `<number>_<date>.kml` (e.g., `51_15nov2025.kml`)
3. Run `python train.py` — retrains from scratch with all data
4. Restart API: `python api.py`

---

## Troubleshooting

**GEE authentication error:**
```bash
earthengine authenticate --auth_mode=notebook
```

**Port 8008 already in use:**
```bash
# Windows
netstat -aon | findstr :8008
taskkill /PID <pid> /F
python api.py
```

**No satellite data returned:**
- Check your KML has valid polygon geometry
- Check the crop_date is in YYYY-MM-DD format
- GEE may be slow — try again after a few minutes

**Low confidence on new regions:**
Add a few KML plots from the new state to the training data and retrain.

---

## Tech Stack

- **Python 3.9+** — Core language
- **scikit-learn + XGBoost** — ML models
- **Google Earth Engine** — Primary satellite backend
- **FastAPI + Uvicorn** — REST API server
- **Sentinel-1 + Sentinel-2** — Satellite imagery
- **rasterio** — GeoTIFF export
- **geopandas + fiona** — KML parsing

---

> **Full technical reference:** See [ARCHITECTURE_ANALYSIS.md](ARCHITECTURE_ANALYSIS.md) for complete documentation of all 11 spectral indices, 436 features, model internals, cloud cover behavior, and bug fix history.
