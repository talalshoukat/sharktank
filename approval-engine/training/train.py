"""
Training pipeline.
Loads feature data, trains a statsmodels Logit model, saves artifacts.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def train_model(
    df_features: pd.DataFrame,
    model_path: str = "artifacts/model_v1.pkl",
    scaler_path: str = "artifacts/scaler_v1.pkl",
    metrics_path: str = "artifacts/metrics_v1.json",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Train a logistic regression model on the feature dataframe.

    Primary feature: kashif_score (validated by EDA).
    Additional features added here as they are validated.
    """
    # ── Prepare data ────────────────────────────────────────────────────────
    feature_cols = ["kashif_score"]  # extend as more features are validated
    target_col = "is_approved"

    df_model = df_features.dropna(subset=[target_col] + feature_cols).copy()
    if len(df_model) < 30:
        raise ValueError(f"Insufficient training data: {len(df_model)} rows after dropping NaNs.")

    logger.info("Training on %d rows, %d features: %s", len(df_model), len(feature_cols), feature_cols)

    X = df_model[feature_cols].values.astype(float)
    y = df_model[target_col].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ── Scale ────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Train statsmodels Logit ──────────────────────────────────────────────
    X_train_const = sm.add_constant(X_train_scaled, has_constant="add")
    model = sm.Logit(y_train, X_train_const).fit(disp=False)

    print(model.summary())
    logger.info("Model trained. Pseudo R-squared: %.4f", model.prsquared)

    # ── Evaluate ────────────────────────────────────────────────────────────
    metrics = evaluate_model(model, scaler, X_test_scaled, y_test, feature_cols)
    logger.info("Test AUC: %.4f", metrics["auc_roc"])

    # ── Save artifacts ───────────────────────────────────────────────────────
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", model_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved to %s", scaler_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)


def evaluate_model(
    model,
    scaler: StandardScaler,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
) -> dict:
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

    X_const = sm.add_constant(X_test_scaled, has_constant="add")
    y_prob = model.predict(X_const)
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "n_test": int(len(y_test)),
        "features": feature_cols,
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "pseudo_r2": float(model.prsquared),
        "classification_report": report,
        "confusion_matrix": cm,
        "coefficients": {
            k: float(v) for k, v in zip(["const"] + feature_cols, model.params)
        },
        "p_values": {
            k: float(v) for k, v in zip(["const"] + feature_cols, model.pvalues)
        },
    }

    logger.info(
        "Metrics — AUC: %.4f | Precision(approved): %.4f | Recall(approved): %.4f",
        metrics["auc_roc"],
        report.get("1", {}).get("precision", 0),
        report.get("1", {}).get("recall", 0),
    )
    return metrics
