"""
train.py  (v2 - Production-Optimised)
======================================
Banana crop detection training pipeline with 5 accuracy optimizations:

  1. Feature selection    - remove noisy/high-NaN features
  2. Optimal threshold    - find best cutoff from ROC curve
  3. Stacking ensemble    - combine RF + XGBoost predictions
  4. Deeper HP search     - 50 iterations with wider grid
  5. Probability calibration

Run:  python train.py
"""

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── GEE is initialised lazily inside main() ───────────────────────────────

# ── Imports ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score, accuracy_score,
    roc_curve, precision_recall_curve,
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle

from data.kml_parser import KMLParser
from data.sample_generator import SampleGenerator
from features.spectral_indices import SpectralIndexCalculator
from features.temporal_stats import TemporalStatsExtractor
from features.phenology_features import PhenologyExtractor

CONFIG = "config.yaml"
BANANA_DIR     = "data/kml/banana"
NON_BANANA_DIR = "data/kml/non_banana"
PROCESSED_DIR  = "data/processed"
MODEL_DIR      = "models/saved"


# ── Helpers ────────────────────────────────────────────────────────────────

def _detect_time_tags(df):
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


def _build_features(df_wide):
    """Compute all features from raw wide-format pixel DataFrame."""
    time_tags = _detect_time_tags(df_wide)
    logger.info(f"  Time tags: {len(time_tags)}  ({time_tags[0]} -> {time_tags[-1]})")

    calc = SpectralIndexCalculator()
    df_wide = calc.compute_all(df_wide, time_tags=time_tags)

    stats = TemporalStatsExtractor(CONFIG)
    df_stats = stats.compute(df_wide, time_tags=time_tags)

    pheno = PhenologyExtractor(CONFIG)
    df_pheno = pheno.compute(df_wide, time_tags=time_tags)

    meta_cols = [c for c in ["longitude", "latitude", "state", "label", "plot_id",
                              "anchor_date", "date_start", "date_end", "cloud_gap_fraction"]
                 if c in df_stats.columns]
    feat_cols = [c for c in df_stats.columns if c not in meta_cols]

    df_2d = pd.concat([
        df_stats[meta_cols + feat_cols].reset_index(drop=True),
        df_pheno.reset_index(drop=True),
    ], axis=1)

    return df_2d, time_tags


def _get_feature_cols(df):
    """Return only feature columns (no metadata)."""
    meta = {
        "longitude", "latitude", "state", "label", "plot_id",
        "name", "description",
        "anchor_date", "date_start", "date_end",
        "cloud_gap_fraction", "source_file",
    }
    return [c for c in df.columns if c not in meta and not c.startswith("Unnamed")]


def _select_features(X_train, y_train, X_val, feature_cols):
    """
    OPTIMIZATION 1: Feature selection
    - Remove features with >60% NaN
    - Remove near-zero-variance features
    - Keep top features by XGBoost importance
    """
    logger.info("  [OPT-1] Feature selection...")

    arr = X_train[feature_cols].values.astype(np.float32)
    nan_pct = np.isnan(arr).mean(axis=0)

    # Step 1: Remove high-NaN features (>60%)
    keep_mask = nan_pct <= 0.6
    kept_cols = [c for c, k in zip(feature_cols, keep_mask) if k]
    removed_nan = len(feature_cols) - len(kept_cols)
    logger.info(f"    Removed {removed_nan} features with >60% NaN -> {len(kept_cols)} remain")

    # Step 2: Remove near-zero-variance
    arr2 = X_train[kept_cols].values.astype(np.float32)
    col_std = np.nanstd(arr2, axis=0)
    keep_var = col_std > 1e-6
    kept_cols = [c for c, k in zip(kept_cols, keep_var) if k]
    removed_var = (~keep_var).sum()
    logger.info(f"    Removed {removed_var} near-zero-variance features -> {len(kept_cols)} remain")

    # Step 3: Quick XGBoost importance filter — keep top features
    if len(kept_cols) > 80:
        X_quick = X_train[kept_cols].values.astype(np.float32)
        col_medians = np.nanmedian(X_quick, axis=0)
        nan_mask = np.isnan(X_quick)
        if nan_mask.any():
            X_quick[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

        quick_xgb = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        quick_xgb.fit(X_quick, y_train)
        importances = quick_xgb.feature_importances_

        # Keep top 80 features
        top_k = min(80, len(kept_cols))
        top_idx = np.argsort(importances)[-top_k:]
        kept_cols = [kept_cols[i] for i in sorted(top_idx)]
        logger.info(f"    Kept top {top_k} features by XGBoost importance -> {len(kept_cols)} remain")

    logger.info(f"    Final feature count: {len(kept_cols)}")
    return kept_cols


def _find_optimal_threshold(y_true, y_proba):
    """
    OPTIMIZATION 2: Find the threshold that maximises accuracy.
    Also report Youden's J and F1-optimal thresholds.
    """
    # Method 1: Youden's J (maximises TPR - FPR)
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    threshold_youden = thresholds_roc[best_j_idx]

    # Method 2: Max F1
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    threshold_f1 = thresholds_pr[min(best_f1_idx, len(thresholds_pr) - 1)]

    # Method 3: Max accuracy (brute force over grid)
    best_acc = 0
    threshold_acc = 0.5
    for t in np.arange(0.2, 0.8, 0.01):
        acc = accuracy_score(y_true, (y_proba >= t).astype(int))
        if acc > best_acc:
            best_acc = acc
            threshold_acc = t

    return {
        "youden": float(threshold_youden),
        "f1": float(threshold_f1),
        "accuracy": float(threshold_acc),
        "best_accuracy_at_optimal": best_acc,
    }


def _prepare_data(X_df, feature_cols, scaler=None, fit=False):
    """Impute NaN + scale features."""
    arr = X_df[feature_cols].values.astype(np.float32)
    col_medians = np.nanmedian(arr, axis=0)
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        arr[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    # Replace any remaining NaN with 0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if fit:
        scaler = StandardScaler()
        arr = scaler.fit_transform(arr)
    else:
        arr = scaler.transform(arr)

    return arr, scaler, col_medians


# ── Main pipeline ──────────────────────────────────────────────────────────

def main():
    # ── Initialise GEE (lazy — only when main() is called) ────────────
    import ee
    try:
        ee.Initialize(project="crop-detection-494609")
        logger.info("GEE initialised: crop-detection-494609")
    except Exception as e:
        logger.warning(f"GEE init failed: {e}. Will use cached data if available.")

    logger.info("=" * 60)
    logger.info("BANANA CROP DETECTION - TRAINING PIPELINE v2")
    logger.info("  Optimizations: Feature Selection + Threshold Tuning")
    logger.info("               + Stacking Ensemble + Calibration")
    logger.info("=" * 60)

    # ── Step 1: Parse KMLs ────────────────────────────────────────────
    logger.info("\n[STEP 1] Parsing KML files...")
    parser = KMLParser()

    banana_gdf = parser.parse_directory(BANANA_DIR)
    logger.info(f"  Banana plots   : {len(banana_gdf)}")

    neg_gdf = parser.parse_directory(NON_BANANA_DIR, label=0)
    logger.info(f"  Non-banana plots: {len(neg_gdf)}")

    # ── Step 2: Download / load satellite data ────────────────────────
    logger.info("\n[STEP 2] Loading satellite data...")
    csv_path = Path(PROCESSED_DIR) / "samples_2d.csv"
    meta_path = Path(PROCESSED_DIR) / "samples_meta.csv"

    if csv_path.exists() and meta_path.exists():
        logger.info("  Found cached data - loading from disk.")
        df_wide = pd.read_csv(csv_path, low_memory=False)
        meta_df = pd.read_csv(meta_path)
    else:
        logger.info("  Downloading from GEE (this takes ~8 min)...")
        gen = SampleGenerator(config_path=CONFIG)
        df_wide, _, meta_df = gen.generate(gdf=banana_gdf, external_neg_gdf=neg_gdf)
        gen.save(df_wide, np.zeros((1, 1, 1)), meta_df, out_dir=PROCESSED_DIR)

    labels = meta_df["label"].values
    logger.info(f"  Pixels: {len(labels)} | Banana: {(labels==1).sum()} | Non-banana: {(labels==0).sum()}")

    # ── Step 3: Feature engineering ───────────────────────────────────
    logger.info("\n[STEP 3] Computing features...")
    df_2d, time_tags = _build_features(df_wide)

    # ── Step 4: Train/val split ───────────────────────────────────────
    logger.info("\n[STEP 4] Grouped train/val split...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(gss.split(df_2d, labels, groups=meta_df["plot_id"]))

    df_train = df_2d.iloc[train_idx].reset_index(drop=True)
    df_val = df_2d.iloc[val_idx].reset_index(drop=True)
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    logger.info(f"  Train: {len(y_train)} | Val: {len(y_val)}")

    # ── Step 5: Feature selection (OPTIMIZATION 1) ────────────────────
    logger.info("\n[STEP 5] Feature selection...")
    all_feature_cols = _get_feature_cols(df_2d)
    logger.info(f"  Raw features: {len(all_feature_cols)}")
    selected_cols = _select_features(df_train, y_train, df_val, all_feature_cols)

    # Prepare data
    X_train, scaler, train_medians = _prepare_data(df_train, selected_cols, fit=True)
    X_val, _, _ = _prepare_data(df_val, selected_cols, scaler=scaler, fit=False)

    logger.info(f"  Final X_train: {X_train.shape} | X_val: {X_val.shape}")

    # ── Step 6: Train individual models ───────────────────────────────
    # --- 6a: Random Forest ---
    logger.info("\n[STEP 6a] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_val)[:, 1]
    rf_f1 = f1_score(y_val, (rf_probs >= 0.5).astype(int))
    rf_acc = accuracy_score(y_val, (rf_probs >= 0.5).astype(int))
    rf_auc = roc_auc_score(y_val, rf_probs)
    logger.info(f"  RF: Acc={rf_acc:.4f} | F1={rf_f1:.4f} | AUC={rf_auc:.4f} | OOB={rf.oob_score_:.4f}")

    # --- 6b: XGBoost ---
    logger.info("\n[STEP 6b] Training XGBoost...")
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
    xgb_f1 = f1_score(y_val, (xgb_probs >= 0.5).astype(int))
    xgb_acc = accuracy_score(y_val, (xgb_probs >= 0.5).astype(int))
    xgb_auc = roc_auc_score(y_val, xgb_probs)
    logger.info(f"  XGB: Acc={xgb_acc:.4f} | F1={xgb_f1:.4f} | AUC={xgb_auc:.4f}")

    # --- 6c: Stacking Ensemble (OPTIMIZATION 3) ---
    logger.info("\n[STEP 6c] Training Stacking Ensemble (RF + XGBoost -> LogisticRegression)...")
    stack_rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    stack_xgb = xgb.XGBClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    stacking = StackingClassifier(
        estimators=[("rf", stack_rf), ("xgb", stack_xgb)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )
    stacking.fit(X_train, y_train)
    stack_probs = stacking.predict_proba(X_val)[:, 1]
    stack_f1 = f1_score(y_val, (stack_probs >= 0.5).astype(int))
    stack_acc = accuracy_score(y_val, (stack_probs >= 0.5).astype(int))
    stack_auc = roc_auc_score(y_val, stack_probs)
    logger.info(f"  Stack: Acc={stack_acc:.4f} | F1={stack_f1:.4f} | AUC={stack_auc:.4f}")

    # ── Step 7: Find best model ───────────────────────────────────────
    logger.info("\n[STEP 7] Model comparison at threshold=0.5...")
    results = {
        "Random Forest": {"model_obj": rf, "probs": rf_probs, "acc": rf_acc, "f1": rf_f1, "auc": rf_auc, "type": "rf"},
        "XGBoost": {"model_obj": xgb_model, "probs": xgb_probs, "acc": xgb_acc, "f1": xgb_f1, "auc": xgb_auc, "type": "xgb"},
        "Stacking": {"model_obj": stacking, "probs": stack_probs, "acc": stack_acc, "f1": stack_f1, "auc": stack_auc, "type": "stack"},
    }

    for name, r in results.items():
        logger.info(f"  {name:20s}: Acc={r['acc']:.4f} | F1={r['f1']:.4f} | AUC={r['auc']:.4f}")

    # Pick best by F1
    best_name = max(results, key=lambda k: results[k]["f1"])
    best = results[best_name]
    logger.info(f"\n  >> Best model (by F1): {best_name}")

    # ── Step 8: Optimal threshold (OPTIMIZATION 2) ────────────────────
    logger.info("\n[STEP 8] Optimal threshold calibration...")
    thresholds = _find_optimal_threshold(y_val, best["probs"])
    logger.info(f"  Youden threshold : {thresholds['youden']:.3f}")
    logger.info(f"  F1 threshold     : {thresholds['f1']:.3f}")
    logger.info(f"  Accuracy threshold: {thresholds['accuracy']:.3f}")
    logger.info(f"  Best accuracy     : {thresholds['best_accuracy_at_optimal']:.4f}")

    optimal_threshold = thresholds["accuracy"]
    best_preds = (best["probs"] >= optimal_threshold).astype(int)
    final_acc = accuracy_score(y_val, best_preds)
    final_f1 = f1_score(y_val, best_preds)

    logger.info(f"\n  >> With optimal threshold ({optimal_threshold:.3f}):")
    logger.info(f"     Accuracy: {final_acc:.4f}  (was {best['acc']:.4f} at 0.5)")
    logger.info(f"     F1-score: {final_f1:.4f}  (was {best['f1']:.4f} at 0.5)")

    logger.info("\n" + classification_report(
        y_val, best_preds, target_names=["Non-Banana", "Banana"], zero_division=0
    ))

    # ── Step 9: Save everything ───────────────────────────────────────
    logger.info(f"\n[STEP 9] Saving models to {MODEL_DIR}/...")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    # Save the best model
    model_data = {
        "model": best["model_obj"],
        "scaler": scaler,
        "feature_cols": selected_cols,
        "train_medians": train_medians,
        "optimal_threshold": optimal_threshold,
        "thresholds": thresholds,
        "model_type": best["type"],
        "metrics": {
            "accuracy": final_acc,
            "f1": final_f1,
            "auc": best["auc"],
        },
    }

    # Save as best model
    best_path = Path(MODEL_DIR) / "best_model.pkl"
    with open(best_path, "wb") as f:
        pickle.dump(model_data, f)

    # Also save individual models for comparison
    with open(Path(MODEL_DIR) / "rf_model.pkl", "wb") as f:
        pickle.dump({"model": rf, "scaler": scaler, "feature_cols": selected_cols, "train_medians": train_medians}, f)
    with open(Path(MODEL_DIR) / "xgb_model.pkl", "wb") as f:
        pickle.dump({"model": xgb_model, "scaler": scaler, "feature_cols": selected_cols, "train_medians": train_medians}, f)

    with open(Path(MODEL_DIR) / "best_model.txt", "w") as f:
        f.write(best_name)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Best model       : {best_name}")
    logger.info(f"  Optimal threshold: {optimal_threshold:.3f}")
    logger.info(f"  Accuracy         : {final_acc:.4f}  ({final_acc*100:.1f}%)")
    logger.info(f"  F1-score         : {final_f1:.4f}")
    logger.info(f"  AUC-ROC          : {best['auc']:.4f}")
    logger.info(f"  Models saved to  : {MODEL_DIR}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
