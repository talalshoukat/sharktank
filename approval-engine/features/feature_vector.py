"""
Assembles a normalised feature vector from raw transaction data.
This module is the single shared boundary between data fetching and scoring.
Both the training loop and live inference call build_feature_vector().
"""
from __future__ import annotations

import math
from datetime import datetime, date


def build_feature_vector(raw: dict) -> dict:
    """
    Transform raw data dict (from data/loader.py) into a clean feature dict.

    Input keys expected (all nullable):
        transaction_trace_id, establishment_id, person_id,
        law_type, law_category, nin_present, joining_date,
        est_approval_rate, contributor_approval_rate,
        violation_count_per_month, kashif_score, status,
        employee_id, employee_id_type, unified_national_no  ← HRSD + insurance inputs
        engagement_start_date, engagement_end_date          ← insurance date range

    Returns:
        dict with all feature values required by rule_engine and ml_scorer.
    """
    return {
        "transaction_trace_id": raw.get("transaction_trace_id"),

        # Identity
        "establishment_id": raw.get("establishment_id"),
        "person_id": raw.get("person_id"),

        # Categorical features
        "law_category": raw.get("law_category", "UNKNOWN"),

        # Data completeness
        "nin_present": bool(raw.get("nin_present", False)),

        # Joining date — how many months back (for backdating scoring)
        "backdated_months": _compute_backdated_months(raw.get("joining_date")),

        # Historical rates (0-100 scale, None if no history)
        "est_approval_rate": _safe_float(raw.get("est_approval_rate")),
        "contributor_approval_rate": _safe_float(raw.get("contributor_approval_rate")),

        # Violation rate (violations per month)
        "violation_count_per_month": _safe_float(raw.get("violation_count_per_month"), default=0.0),

        # KASHIF fraud score (raw value, higher = more risk)
        "kashif_score": _safe_float(raw.get("kashif_score")),

        # ── HRSD + Insurance verification inputs ──────────────────────────
        # employee_id / employee_id_type: the employee's identifier from transaction params.
        # unified_national_no: employer's Unified National Number from the establishment record.
        "employee_id": raw.get("employee_id"),
        "employee_id_type": raw.get("employee_id_type", "National ID"),
        "unified_national_no": raw.get("unified_national_no"),

        # Engagement date range — used by the insurance checker.
        # Derived from the violation/engagement request dates.
        "engagement_start_date": raw.get("engagement_start_date"),
        "engagement_end_date": raw.get("engagement_end_date"),

        # Ground truth label (only present during training)
        "is_approved": _encode_status(raw.get("status")),
    }


def _compute_backdated_months(joining_date) -> float | None:
    """Return months between joining_date and today (positive = in the past)."""
    if joining_date is None:
        return None
    try:
        if isinstance(joining_date, (datetime, date)):
            jd = joining_date if isinstance(joining_date, date) else joining_date.date()
        else:
            # Try common Oracle date string formats
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y"):
                try:
                    jd = datetime.strptime(str(joining_date), fmt).date()
                    break
                except ValueError:
                    continue
            else:
                return None

        today = date.today()
        delta_days = (today - jd).days
        return delta_days / 30.44  # approximate months
    except Exception:
        return None


def _safe_float(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return default


def _encode_status(status: str | None) -> int | None:
    """Map status string to binary label for training."""
    if status is None:
        return None
    return 1 if status.strip().lower() == "approved" else 0
