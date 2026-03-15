"""
Blends rule-based points and ML probability into a single final score (0–100).
Weights are configurable from settings (RULE_WEIGHT + ML_WEIGHT = 1.0).
"""
from __future__ import annotations

from config.settings import settings


def blend(
    rule_points: int,
    ml_probability: float | None,
    rule_weight: float | None = None,
    ml_weight: float | None = None,
) -> float:
    """
    Compute final approval likelihood score (0–100).

    If ML probability is unavailable, falls back to 100% rule-based.

    Args:
        rule_points:    Raw points from rule engine (including base score).
        ml_probability: Approval probability from ML model (0.0–1.0), or None.
        rule_weight:    Override settings.rule_weight (useful for A/B testing).
        ml_weight:      Override settings.ml_weight.

    Returns:
        Final score clamped to [0, 100].
    """
    rw = rule_weight if rule_weight is not None else settings.rule_weight
    mw = ml_weight if ml_weight is not None else settings.ml_weight

    # Normalise rule points to 0–100 using configured max
    rule_normalised = (rule_points / settings.rule_score_max) * 100
    rule_normalised = max(0.0, min(100.0, rule_normalised))

    if ml_probability is None:
        # No ML model — fall back to rule score only
        return round(rule_normalised, 1)

    ml_normalised = ml_probability * 100
    final = (rw * rule_normalised) + (mw * ml_normalised)
    return round(max(0.0, min(100.0, final)), 1)
