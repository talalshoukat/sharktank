"""
Insurance verification scoring.

Calls the insurance API with (employee_id, employee_id_type,
unified_national_no, start_date, end_date) and maps the boolean
response to rule engine points.

Active insurance → positive contribution to approval score.
Inactive / no coverage → penalty (employee was not insured during
the claimed engagement period, which is a key rejection signal).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging

from services.insurance_client import InsuranceClient, InsuranceError

logger = logging.getLogger(__name__)

_insurance_client = InsuranceClient()

INSURANCE_ACTIVE_POINTS = 20
INSURANCE_INACTIVE_POINTS = -25


def score_insurance_verification(
    employee_id: str | None,
    employee_id_type: str | None,
    unified_national_no: str | None,
    start_date: str | None,
    end_date: str | None,
) -> tuple[int, str | None]:
    """
    Synchronous wrapper — calls the async insurance check and returns (points, reason).
    Returns (0, reason) when required inputs are missing or the call fails,
    so a lookup failure never blocks the rest of scoring.

    Args:
        employee_id:         Employee's ID number.
        employee_id_type:    "National ID" | "Iqama" | "Passport" | "GCC ID".
        unified_national_no: Employer's Unified National Number.
        start_date:          Engagement start date (YYYY-MM-DD).
        end_date:            Engagement end date (YYYY-MM-DD).

    Returns:
        (points, reason_string)
    """
    if not employee_id or not unified_national_no:
        return 0, "Insufficient identity info — insurance verification skipped"

    if not start_date or not end_date:
        return 0, "Engagement dates missing — insurance verification skipped"

    id_type = employee_id_type or "National ID"

    try:
        result = _run_async(
            _insurance_client.verify_insurance(
                employee_id, id_type, unified_national_no, start_date, end_date
            )
        )
        return _map_insurance_result(result)

    except InsuranceError as exc:
        logger.warning("Insurance verification failed: %s", exc)
        return 0, f"Insurance check unavailable: {exc}"
    except Exception:
        logger.exception("Unexpected error in insurance scoring")
        return 0, "Insurance verification error — skipped"


def _map_insurance_result(result: dict) -> tuple[int, str | None]:
    is_active: bool = result.get("insurance_active", False)
    is_mock: bool = result.get("mock", False)
    prefix = "[MOCK] " if is_mock else ""

    if is_active:
        return (
            INSURANCE_ACTIVE_POINTS,
            f"{prefix}Active insurance coverage confirmed for engagement period "
            f"({result.get('checked_from')} → {result.get('checked_to')}) "
            f"(+{INSURANCE_ACTIVE_POINTS})",
        )
    else:
        return (
            INSURANCE_INACTIVE_POINTS,
            f"{prefix}No active insurance coverage found for engagement period "
            f"({result.get('checked_from')} → {result.get('checked_to')}) "
            f"({INSURANCE_INACTIVE_POINTS})",
        )


def _run_async(coro):
    """Run an async coroutine from a synchronous context."""
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=35)
    except RuntimeError:
        return asyncio.run(coro)
