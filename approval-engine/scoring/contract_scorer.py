"""
Contract verification scoring.

Calls HRSD with (employee_id, employee_id_type, unified_national_no),
maps the returned verification_score (0–100) to rule engine points,
and surfaces a reason string.

In mock mode the HRSD client returns a random verification_score.
The bands below map that to points so the full pipeline runs end-to-end.

When the real HRSD API is available, extend `_compare_contract_fields()`
for field-level matching instead of relying on HRSD's pre-computed score.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging

from services.hrsd_client import HRSDClient, HRSDError

logger = logging.getLogger(__name__)

_hrsd_client = HRSDClient()

# (score_low, score_high, points, reason)
VERIFICATION_BANDS: list[tuple[int, int, int, str | None]] = [
    (85, 100,  30, "Contract details fully verified with HRSD records (+30)"),
    (65,  84,  15, "Contract details mostly match HRSD records (+15)"),
    (40,  64,   0, None),
    (20,  39, -15, "Partial mismatch in HRSD contract details (-15)"),
    (0,   19, -30, "Significant discrepancy between submission and HRSD contract (-30)"),
]


def score_contract_verification(
    employee_id: str | None,
    employee_id_type: str | None,
    unified_national_no: str | None,
) -> tuple[int, str | None]:
    """
    Synchronous wrapper — calls async HRSD lookup and returns (points, reason).
    Returns (0, reason) when any required input is missing or the call fails,
    so a lookup failure never blocks the rest of scoring.
    """
    if not employee_id or not unified_national_no:
        return 0, "Insufficient identity info — HRSD contract verification skipped"

    id_type = employee_id_type or "National ID"

    try:
        contract = _run_async(
            _hrsd_client.get_contract_details(employee_id, id_type, unified_national_no)
        )
        return _map_verification_score(contract.get("verification_score"), contract)

    except HRSDError as exc:
        logger.warning("HRSD lookup failed: %s", exc)
        return 0, f"HRSD verification unavailable: {exc}"
    except Exception:
        logger.exception("Unexpected error in HRSD contract scoring")
        return 0, "HRSD verification error — skipped"


def _map_verification_score(
    verification_score: float | None,
    contract: dict,
) -> tuple[int, str | None]:
    if verification_score is None:
        return 0, "HRSD returned no verification score"

    score = float(verification_score)
    is_mock = contract.get("mock", False)

    for low, high, points, reason in VERIFICATION_BANDS:
        if low <= score <= high:
            if reason and is_mock:
                reason = f"[MOCK] {reason}"
            return points, reason

    return 0, None


def _compare_contract_fields(contract: dict, submitted: dict) -> float:
    """
    Field-level comparison for when the live HRSD API is available.
    Returns verification_score (0–100) based on weighted field matches.
    Extend as you learn which fields HRSD actually returns.
    """
    checks = [
        # (contract_field,              submitted_value,                    weight)
        ("employee_id",                 submitted.get("employee_id"),        30),
        ("establishment_unified_national_no", submitted.get("unified_national_no"), 25),
        ("nationality",                 submitted.get("nationality"),         15),
        ("job_title",                   submitted.get("job_title"),           10),
        ("contract_start_date",         submitted.get("start_date"),          10),
        ("gender",                      submitted.get("gender"),              10),
    ]

    total_weight = sum(w for _, _, w in checks)
    matched = 0.0

    for field, submitted_value, weight in checks:
        contract_value = contract.get(field)
        if contract_value and submitted_value:
            if str(contract_value).strip().lower() == str(submitted_value).strip().lower():
                matched += weight

    return (matched / total_weight) * 100 if total_weight else 0.0


def _run_async(coro):
    """Run an async coroutine from a synchronous context."""
    try:
        asyncio.get_running_loop()
        # Inside an async event loop (FastAPI) — offload to a thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=35)
    except RuntimeError:
        # No running loop (training pipeline, scripts)
        return asyncio.run(coro)
