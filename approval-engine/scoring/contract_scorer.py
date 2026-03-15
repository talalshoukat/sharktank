"""
Contract verification scoring.

Calls the HRSD API (or mock) with the contract_id from the transaction,
then compares the returned contract details against what was submitted.
Discrepancies reduce the score; strong matches increase it.

In mock mode the HRSD client returns a random verification_score (0–100).
The comparison bands below map that score to points + a reason string,
so the full scoring pipeline runs end-to-end even without live API access.

When the real HRSD API is available, replace the mock and extend
`_compare_contract_fields()` to do field-level matching.
"""
from __future__ import annotations

import asyncio
import logging

from services.hrsd_client import HRSDClient, HRSDError

logger = logging.getLogger(__name__)

# Singleton client — stateless, safe to share
_hrsd_client = HRSDClient()


# ── Score bands for HRSD verification_score → points ─────────────────────────
# verification_score comes from HRSD (0–100).
# 0–100 maps to contract_points contribution in the overall rule engine.
VERIFICATION_BANDS: list[tuple[int, int, int, str]] = [
    # (low, high, points, reason)
    (85, 100, 30, "Contract details fully verified with HRSD records (+30)"),
    (65,  84, 15, "Contract details mostly match HRSD records (+15)"),
    (40,  64,  0, None),   # neutral — partial match, no impact
    (20,  39, -15, "Partial mismatch in HRSD contract details (-15)"),
    (0,   19, -30, "Significant discrepancy between submission and HRSD contract (-30)"),
]


def score_contract_verification(contract_id: str | None) -> tuple[int, str | None]:
    """
    Synchronous wrapper — calls the async HRSD lookup and returns (points, reason).
    Safe to call from the synchronous rule engine and FastAPI path operations.

    Returns (0, None) if contract_id is missing or HRSD call fails,
    so a lookup failure never blocks the rest of scoring.
    """
    if not contract_id:
        return 0, "No contract ID present — HRSD verification skipped"

    try:
        # Run the async client call in a synchronous context
        try:
            loop = asyncio.get_running_loop()
            # We're inside an async context (e.g. FastAPI) — use a thread executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _hrsd_client.get_contract_details(contract_id))
                contract = future.result(timeout=35)
        except RuntimeError:
            # No running loop — we're in a plain sync context (e.g. training pipeline)
            contract = asyncio.run(_hrsd_client.get_contract_details(contract_id))

        return _map_verification_score(contract.get("verification_score"), contract)

    except HRSDError as exc:
        logger.warning("HRSD lookup failed for contract '%s': %s", contract_id, exc)
        return 0, f"HRSD verification unavailable: {exc}"
    except Exception as exc:
        logger.exception("Unexpected error in HRSD contract scoring for '%s'", contract_id)
        return 0, "HRSD verification error — skipped"


def _map_verification_score(
    verification_score: float | None,
    contract: dict,
) -> tuple[int, str | None]:
    """
    Map HRSD verification_score to rule engine points.

    When the live API is available, extend this to call
    `_compare_contract_fields(contract, submitted_data)` for
    field-level comparison rather than relying on HRSD's pre-computed score.
    """
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
    Field-level comparison for use when the live HRSD API is available.
    Returns a verification_score (0–100) based on how many key fields match.

    Extend this as you learn which fields HRSD actually returns.
    """
    checks = [
        # (contract_field, submitted_field, weight)
        ("id_number",              submitted.get("nin"),               30),
        ("establishment_unified_id", submitted.get("establishment_id"), 25),
        ("nationality",            submitted.get("nationality"),        15),
        ("job_title",              submitted.get("job_title"),          10),
        ("contract_start_date",    submitted.get("joining_date"),       10),
        ("gender",                 submitted.get("gender"),             10),
    ]

    total_weight = sum(w for _, _, w in checks)
    matched_weight = 0.0

    for contract_field, submitted_value, weight in checks:
        contract_value = contract.get(contract_field)
        if contract_value and submitted_value:
            if str(contract_value).strip().lower() == str(submitted_value).strip().lower():
                matched_weight += weight

    return (matched_weight / total_weight) * 100 if total_weight else 0.0
