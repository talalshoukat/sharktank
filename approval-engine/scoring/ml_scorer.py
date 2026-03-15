"""
ML-based scoring layer (Phase 2 ready, used in blender if model exists).
Wraps the trained statsmodels Logit or sklearn model.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_scaler = None


def load_model(model_path: str, scaler_path: str | None = None) -> None:
    """Load model and optional scaler from disk. Call at app startup."""
    global _model, _scaler

    mp = Path(model_path)
    if mp.exists():
        with open(mp, "rb") as f:
            _model = pickle.load(f)
        logger.info("ML model loaded from %s", model_path)
    else:
        logger.warning("Model artifact not found at %s — ML scoring disabled", model_path)

    if scaler_path:
        sp = Path(scaler_path)
        if sp.exists():
            with open(sp, "rb") as f:
                _scaler = pickle.load(f)
            logger.info("Scaler loaded from %s", scaler_path)


def is_model_loaded() -> bool:
    return _model is not None


def predict_proba(features: dict) -> float | None:
    """
    Return approval probability (0.0–1.0) for a feature dict.
    Returns None if model is not loaded.

    Feature used: kashif_score (primary predictor from EDA).
    Extend this list as more features are validated.
    """
    if _model is None:
        return None

    kashif = features.get("kashif_score")
    if kashif is None:
        return None

    try:
        X = np.array([[1.0, float(kashif)]])  # intercept + kashif_score

        if _scaler is not None:
            X[:, 1:] = _scaler.transform(X[:, 1:])

        # statsmodels result object uses .predict()
        if hasattr(_model, "predict"):
            prob = float(_model.predict(X)[0])
        # sklearn model
        elif hasattr(_model, "predict_proba"):
            prob = float(_model.predict_proba(X)[0][1])
        else:
            return None

        return max(0.0, min(1.0, prob))

    except Exception as exc:
        logger.warning("ML prediction failed: %s", exc)
        return None
