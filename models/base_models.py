"""
base_models.py
==============
Training pipelines for the two base models:
  1. Random Forest (scikit-learn)
  2. XGBoost

Each model:
  - Accepts a config.yaml for hyperparameter search spaces
  - Saves trained weights, scalers, and feature lists to disk
  - Returns out-of-fold (OOF) predictions for stacking ensemble
  - Handles class imbalance via class weights

Usage
-----
    from models.base_models import RandomForestModel, XGBoostModel

    rf = RandomForestModel(config_path="config.yaml")
    rf.fit(X_train, y_train, X_val, y_val)
    oof_preds = rf.predict_proba(X_val)
    rf.save("models/saved/rf_model.pkl")
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _compute_class_weight(y: np.ndarray) -> dict:
    """Compute balanced class weights for binary classification."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return feature columns (exclude all metadata).
    
    CRITICAL: longitude/latitude/state would make the model learn LOCATION 
    not CROP. anchor_date/date_start/date_end would leak temporal info.
    Only spectral/temporal/phenology features should remain.
    """
    meta_cols = {
        "longitude", "latitude", "state", "label", "plot_id",
        "name", "description",
        "anchor_date", "date_start", "date_end",
        "cloud_gap_fraction", "source_file",
    }
    return [c for c in df.columns if c not in meta_cols and not c.startswith("Unnamed")]


# ---------------------------------------------------------------------------
# 1. Random Forest
# ---------------------------------------------------------------------------

class RandomForestModel:
    """
    Random Forest classifier with RandomizedSearchCV hyperparameter tuning.

    Parameters
    ----------
    config_path : path to config.yaml
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)["models"]["random_forest"]
        self.model = None
        self.scaler = None
        self.feature_cols: List[str] = []
        self.best_params: dict = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> "RandomForestModel":
        """
        Train Random Forest with RandomizedSearchCV.

        Parameters
        ----------
        X_train     : training feature DataFrame
        y_train     : binary labels (0/1)
        X_val       : validation DataFrame (used for evaluation only)
        y_val       : validation labels
        feature_cols: explicit feature column list; if None, auto-detected

        Returns
        -------
        self
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
        from sklearn.preprocessing import StandardScaler

        self.feature_cols = feature_cols or _get_feature_cols(X_train)
        X = X_train[self.feature_cols].values.astype(np.float32)
        y = y_train.astype(int)

        # Replace NaN with column median
        col_medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Class weights
        cw = _compute_class_weight(y)
        logger.info(f"RF class weights: {cw}")

        # Hyperparameter search space
        param_dist = {
            "n_estimators": self.cfg["n_estimators"],
            "max_depth": self.cfg["max_depth"],
            "min_samples_split": self.cfg["min_samples_split"],
            "min_samples_leaf": self.cfg["min_samples_leaf"],
        }

        base_rf = RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            oob_score=True,
        )

        cv = StratifiedKFold(n_splits=self.cfg["cv_folds"], shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            base_rf,
            param_distributions=param_dist,
            n_iter=self.cfg["n_iter"],
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )

        logger.info(f"Starting RF RandomizedSearchCV ({self.cfg['n_iter']} iterations)...")
        search.fit(X_scaled, y)

        self.model = search.best_estimator_
        self.best_params = search.best_params_
        logger.info(f"RF best params: {self.best_params}")
        logger.info(f"RF OOB score: {self.model.oob_score_:.4f}")

        # Validation evaluation
        if X_val is not None and y_val is not None:
            self._evaluate(X_val, y_val, split="val")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return banana class probability for each pixel."""
        X_arr = self._preprocess(X)
        return self.model.predict_proba(X_arr)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def _preprocess(self, X: pd.DataFrame) -> np.ndarray:
        arr = X[self.feature_cols].values.astype(np.float32)
        col_medians = np.nanmedian(arr, axis=0)
        nan_mask = np.isnan(arr)
        arr[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        return self.scaler.transform(arr)

    def _evaluate(self, X: pd.DataFrame, y: np.ndarray, split: str = "val"):
        from sklearn.metrics import classification_report, roc_auc_score
        preds = self.predict_proba(X)
        binary = (preds >= 0.5).astype(int)
        auc = roc_auc_score(y, preds)
        report = classification_report(y, binary, target_names=["Non-Banana", "Banana"])
        logger.info(f"RF {split} AUC-ROC: {auc:.4f}\n{report}")

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "best_params": self.best_params,
            }, f)
        logger.info(f"RF model saved to {path}")

    @classmethod
    def load(cls, path: str, config_path: str = "config.yaml") -> "RandomForestModel":
        obj = cls(config_path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.feature_cols = data["feature_cols"]
        obj.best_params = data["best_params"]
        logger.info(f"RF model loaded from {path}")
        return obj


# ---------------------------------------------------------------------------
# 2. XGBoost
# ---------------------------------------------------------------------------

class XGBoostModel:
    """
    XGBoost classifier with early stopping and hyperparameter tuning.
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)["models"]["xgboost"]
        self.model = None
        self.scaler = None
        self.feature_cols: List[str] = []
        self.best_params: dict = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        feature_cols: Optional[List[str]] = None,
    ) -> "XGBoostModel":
        """
        Train XGBoost with RandomizedSearchCV + early stopping.
        """
        import xgboost as xgb
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
        from sklearn.preprocessing import StandardScaler

        self.feature_cols = feature_cols or _get_feature_cols(X_train)
        X_tr = self._prepare_array(X_train)
        X_vl = self._prepare_array(X_val)
        y_tr = y_train.astype(int)
        y_vl = y_val.astype(int)

        # Scale
        self.scaler = StandardScaler()
        X_tr_scaled = self.scaler.fit_transform(X_tr)
        X_vl_scaled = self.scaler.transform(X_vl)

        # Class imbalance ratio
        pos = y_tr.sum()
        neg = len(y_tr) - pos
        scale_pos_weight = neg / (pos + 1e-8)
        logger.info(f"XGB scale_pos_weight: {scale_pos_weight:.2f}")

        param_dist = {
            "learning_rate": self.cfg["learning_rate"],
            "max_depth": self.cfg["max_depth"],
            "n_estimators": self.cfg["n_estimators"],
            "subsample": self.cfg["subsample"],
            "colsample_bytree": self.cfg["colsample_bytree"],
        }

        base_xgb = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric=self.cfg["eval_metric"],
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            base_xgb,
            param_distributions=param_dist,
            n_iter=20,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )

        logger.info("Starting XGBoost RandomizedSearchCV...")
        search.fit(X_tr_scaled, y_tr)
        self.best_params = search.best_params_
        logger.info(f"XGB best params: {self.best_params}")

        # Retrain best model with early stopping on validation set
        best = search.best_estimator_
        best.set_params(
            n_estimators=2000,
            early_stopping_rounds=self.cfg["early_stopping_rounds"],
        )
        best.fit(
            X_tr_scaled, y_tr,
            eval_set=[(X_vl_scaled, y_vl)],
            verbose=False,
        )
        self.model = best
        logger.info(f"XGB best iteration: {self.model.best_iteration}")

        self._evaluate(X_val, y_val, split="val")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = self._preprocess(X)
        return self.model.predict_proba(X_arr)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def _prepare_array(self, X: pd.DataFrame) -> np.ndarray:
        arr = X[self.feature_cols].values.astype(np.float32)
        col_medians = np.nanmedian(arr, axis=0)
        nan_mask = np.isnan(arr)
        arr[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        return arr

    def _preprocess(self, X: pd.DataFrame) -> np.ndarray:
        arr = self._prepare_array(X)
        return self.scaler.transform(arr)

    def _evaluate(self, X: pd.DataFrame, y: np.ndarray, split: str = "val"):
        from sklearn.metrics import classification_report, roc_auc_score
        preds = self.predict_proba(X)
        binary = (preds >= 0.5).astype(int)
        auc = roc_auc_score(y, preds)
        report = classification_report(y, binary, target_names=["Non-Banana", "Banana"])
        logger.info(f"XGB {split} AUC-ROC: {auc:.4f}\n{report}")

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "best_params": self.best_params,
            }, f)
        logger.info(f"XGB model saved to {path}")

    @classmethod
    def load(cls, path: str, config_path: str = "config.yaml") -> "XGBoostModel":
        obj = cls(config_path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.feature_cols = data["feature_cols"]
        obj.best_params = data["best_params"]
        logger.info(f"XGB model loaded from {path}")
        return obj



