# 🥬 Cabbage Crop Detection System

## Pan-India Cabbage Detection from Satellite Imagery

A production-grade machine learning pipeline that detects **cabbage (Brassica oleracea var. capitata)** crops from satellite imagery across India using a **Stacking Ensemble** of Random Forest + XGBoost with Sentinel-1 (SAR) and Sentinel-2 (optical) data fusion.

---

## 🏗️ Architecture

```
KML Polygon → GEE Download → Spectral Indices → Temporal Stats → Heading Phenology → Stacking Ensemble → Classification
                  ↓                  ↓                 ↓               ↓
          Sentinel-1 (SAR)    12 indices         Monthly stats      10 BBCH heading
          Sentinel-2 (Opt)    (NDVI, NDRE,       per band           features per VI
                               CCCI, EVI,                           (onset, plateau,
                               LSWI, etc.)                           harvest drop, etc.)
```

### Key Technical Features
- **Satellite Fusion**: Sentinel-1 SAR (cloud-free radar) + Sentinel-2 optical (10m resolution)
- **12 Spectral Indices**: NDVI, EVI, NDWI, LSWI, SAVI, MSAVI, NBR, NDRE, **CCCI**, RVI, RFDI, CR
- **BBCH Heading Phenology**: Features tailored for heading vegetables (Stage 4: head formation)
- **Monthly Compositing**: ±3 month window captures cabbage's full lifecycle + cloud buffer
- **State-Aware Calendar**: India-specific seasonal windows for 5 agro-climatic zones
- **Spatial Cross-Validation**: GroupShuffleSplit by plot_id prevents spatial data leakage

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Authenticate Google Earth Engine
```bash
earthengine authenticate
```

### 3. Add Training Data
Place KML polygon files in:
```
data/kml/cabbage/         # Cabbage field boundaries (currently 50 KMLs)
data/kml/non_cabbage/     # Non-cabbage fields (currently 50 KMLs)
```

**Naming convention**: `{plot_id}_{date}.kml`
Examples: `1_feb2024.kml`, `15_jun2022.kml`, `31_mar2026.kml`

The anchor date is extracted automatically from the filename and used to centre
the ±3 month satellite download window on the confirmed crop presence date.

### 4. Train the Model
```bash
python train.py
```

### 5. Run the API Server
```bash
python api.py
```
Then open **http://localhost:8009/docs** for Swagger UI.

### 6. Make Predictions
Upload a KML file to the `/detect` endpoint with a crop date.

---

## 📁 Project Structure

```
cabbage_detection/
├── api.py                              # FastAPI REST server (port 8009)
├── train.py                            # Training pipeline (v2, production-optimised)
├── config.yaml                         # Configuration (cabbage-specific)
├── requirements.txt                    # Python dependencies
├── utils.py                            # Shared utilities
├── ARCHITECTURE_ANALYSIS.md            # Deep scientific & architectural analysis
├── data/
│   ├── gee_downloader.py               # Multi-backend satellite downloader
│   │                                     (GEE / Planetary Computer / Sentinel Hub)
│   ├── kml_parser.py                   # KML/KMZ polygon parser with anchor dates
│   ├── sample_generator.py             # Pixel sampling with buffer zones
│   └── kml/
│       ├── cabbage/                    # Cabbage KML training data (50 plots)
│       └── non_cabbage/               # Non-cabbage KML training data (50 plots)
├── features/
│   ├── spectral_indices.py             # 12 spectral indices (incl. CCCI, NDRE)
│   ├── temporal_stats.py               # Temporal statistics per band (11 stats)
│   └── phenology_features_heading.py   # BBCH heading-vegetable phenology (cabbage)
├── models/
│   ├── base_models.py                  # RF + XGBoost wrappers
│   └── saved/                          # Trained model artifacts
├── inference/
│   └── predictor.py                    # CabbagePredictor inference pipeline
└── outputs/                            # Probability + binary GeoTIFF maps
```

> **Important**: Both `train.py` and `inference/predictor.py` use
> `HeadingPhenologyExtractor` from `phenology_features_heading.py`.

---

## 🌱 Cabbage Detection Science

### Why Cabbage is Different from Other Crops

| Dimension | Cabbage Signature |
|:---|:---|
| **Lifecycle** | Annual, 60-120 days transplant → harvest |
| **Season** | Rabi (cool-season): Oct-Mar across India |
| **Canopy** | Low (30-60cm), rosette → compact head |
| **Key spectral** | NDRE + CCCI best for growth status |
| **Red-edge advantage** | Red-edge (Band 5) penetrates deeper into cabbage canopy than red (Band 4) |
| **SAR** | Weak-moderate backscatter (low canopy) |
| **Field size** | Small: 0.1-1 ha (fragmented) |
| **Confusion risk** | Cauliflower, broccoli (same Brassica family) |

### Heading Phenology Features (BBCH-Scale)

The `HeadingPhenologyExtractor` computes 10 features per vegetation index,
plus cross-index correlations, following the BBCH scale for heading vegetables:

1. **Heading onset detection** — transition from vegetative to head formation (BBCH Stage 4)
2. **Heading duration** — consecutive periods with high VI values
3. **Plateau stability** — std during heading phase (lower = healthier)
4. **Green-up rate** — rapid leaf expansion slope
5. **Harvest drop** — sharp NDVI decline detected (binary 1/0)
6. **Harvest drop magnitude** — largest single-step decline
7. **Decline rate** — peak-to-end rate of change
8. **Time to peak** — normalized peak timing
9. **AUC** — area under curve (cumulative productivity)
10. **Pre/post heading ratio** — mean index ratio before/after onset

**Cross-index features:**
- NDVI-NDRE heading correlation (distinguishes cabbage from other greens)
- NDVI-VV SAR temporal correlation (canopy structure vs radar response)

### Why NDRE > NDVI for Cabbage

Research (Ryu et al. 2024) shows that **NDRE outperforms NDVI** for cabbage
growth assessment because:
- Red-edge light (Band 5, ~705 nm) penetrates deeper into the compact cabbage canopy
- NDVI saturates in dense canopies, while NDRE continues to differentiate growth stages
- The **CCCI** (Canopy Chlorophyll Content Index) normalises NDRE and correlates
  strongly with nitrogen status in heading vegetables

---

## 🌍 India Seasonal Calendar

| Region | States | Transplant | Harvest |
|:---|:---|:---|:---|
| Northern Plains | UP, Bihar, Punjab, Haryana | Sep-Oct | Dec-Feb |
| Eastern | West Bengal, Odisha, Assam | Oct-Dec | Jan-Mar |
| Western/Central | Gujarat, MP, Maharashtra | Nov-Dec | Feb-Mar |
| Southern | Karnataka, Tamil Nadu | Jun-Nov | Sep-Apr |
| Hills | Himachal, Uttarakhand, NE | Apr-Sep | Jul-Nov |

---

## 📊 API Response Format

```json
{
  "verdict": "✅ HIGH CONFIDENCE — Cabbage crop detected. 82% of pixels classified as cabbage.",
  "classification": "CABBAGE",
  "label": 1,
  "confidence": "HIGH",
  "cabbage_percentage": 82.0,
  "area_hectares": 0.45,
  "centroid_latitude": 22.5723,
  "centroid_longitude": 88.3639,
  "state": "West Bengal",
  "total_pixels": 50,
  "cabbage_pixels": 41,
  "non_cabbage_pixels": 9,
  "mean_probability": 0.7823,
  "cloud_free_percentage": 95.0,
  "data_quality": "EXCELLENT",
  "model_name": "Stacking Ensemble (Random Forest + XGBoost)",
  "satellites_used": "Sentinel-1 (SAR) + Sentinel-2 (Optical)"
}
```

---

## 📋 Configuration Reference

Key settings in `config.yaml`:

| Setting | Value | Why |
|:---|:---|:---|
| `composite_frequency` | monthly | Monthly composites for 60-120 day crop |
| `months_before/after` | 3 | ±3 months captures full lifecycle + cloud buffer for pan-India |
| `buffer_m` | 250 | Sized for fragmented 0.1-1 ha fields |
| `minimum_field_area_ha` | 0.10 | 10 Sentinel pixels minimum |
| `probability_threshold` | 0.200 | Calibrated after training (Youden/F1 optimal) |
| `key_indices` | NDVI, NDRE, EVI, LSWI, CCCI | Red-edge emphasis for heading vegetables |
| `function_set` | heading_vegetable | BBCH-scale phenology features |

---

## 🔬 Training Data

### Current Dataset (v1)

| Category | Count | Details |
|:---|:---|:---|
| Cabbage KMLs | 50 plots | Dates: Feb 2024, Jun 2022, Apr 2024, May 2022, Mar 2026 |
| Non-cabbage KMLs | 50 plots | Diverse non-cabbage backgrounds |

### Recommended for Production (v2+)

| Category | Minimum | Recommended |
|:---|:---|:---|
| Cabbage KMLs | 60 plots | 80+ plots |
| Non-cabbage KMLs | 80 plots | 100+ plots |
| States covered | 3 | 5+ (must include West Bengal, Odisha) |
| Seasons | 1 Rabi | 2+ seasons |

Non-cabbage data **must include explicit brassica negatives** (cauliflower, broccoli)
to prevent confusion within the same family.

---

## 🚀 Production Deployment

### Train-Inference Consistency

The pipeline ensures feature consistency between training and inference:

| Component | Training (`train.py`) | Inference (`predictor.py` / `api.py`) |
|:---|:---|:---|
| Phenology extractor | `HeadingPhenologyExtractor` | `HeadingPhenologyExtractor` ✅ |
| Spectral indices | 12 indices (NDVI → CR) | Same 12 indices ✅ |
| Window | ±3 months (from config) | ±3 months (from config) ✅ |
| Scaler | Fit on training data | Applied from saved scaler ✅ |
| NaN imputation | Training medians | Same training medians ✅ |

### Running the API

```bash
python api.py
# → Swagger UI: http://localhost:8009/docs
# → Health check: http://localhost:8009/health
```

### CLI Inference

```bash
python inference/predictor.py plots/field_42.kml 2024-12-15
python inference/predictor.py plots/field_42.kml 2024-12-15 --state "West Bengal"
```

---

## 📚 References

- BBCH-scale for leafy vegetables forming heads (Meier et al.)
- Ryu et al. (2024): NDRE for cabbage growth assessment
- Besand & Katroschan (2022): CCCI for cabbage N status (IHC 2022)
- India cabbage production: West Bengal ~25%, Odisha ~11%, Gujarat ~8%
- Sentinel-2 Red-Edge bands for heading vegetable discrimination
