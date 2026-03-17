"""
Insurance verification client.

Given an employee's ID details, the employer's unified national number,
and an engagement date range, checks whether the employee had active
insurance coverage with that employer for the full duration.

When INSURANCE_MOCK_MODE=true (default), returns a randomly sampled
boolean so the scoring pipeline runs end-to-end in demos.

When INSURANCE_MOCK_MODE=false, calls the live insurance verification API.
Switch is purely via environment variables — no code changes needed.
"""
from __future__ import annotations

import logging
import os
import random
from datetime import date

import httpx

logger = logging.getLogger(__name__)


def _mock_insurance_response(
    employee_id: str,
    employee_id_type: str,
    unified_national_no: str,
    start_date: str,
    end_date: str,
) -> dict:
    """
    Realistic stub. Returns the same shape as the live insurance API.
    `insurance_active` is randomly True or False to simulate real-world variation.
    """
    is_active = random.choice([True, False])
    return {
        "insurance_active": is_active,
        "employee_id": employee_id,
        "employee_id_type": employee_id_type,
        "unified_national_no": unified_national_no,
        "checked_from": start_date,
        "checked_to": end_date,
        "mock": True,
        "note": (
            "DEMO DATA — insurance_active is randomly assigned. "
            "In production this reflects actual coverage records."
        ),
    }


class InsuranceClient:
    """
    Verifies whether an employee had active insurance at a given employer
    during a specified date range.

    Environment variables:
        INSURANCE_MOCK_MODE         "true" (default) or "false"
        INSURANCE_API_URL           Base URL of the live insurance API
        INSURANCE_API_KEY           Bearer token / API key
        INSURANCE_TIMEOUT_SECONDS   HTTP timeout (default 30)
    """

    def __init__(self):
        self.mock_mode: bool = os.getenv("INSURANCE_MOCK_MODE", "true").lower() != "false"
        self.api_url: str = os.getenv("INSURANCE_API_URL", "")
        self.api_key: str = os.getenv("INSURANCE_API_KEY", "")
        self.timeout: int = int(os.getenv("INSURANCE_TIMEOUT_SECONDS", "30"))

        if self.mock_mode:
            logger.info(
                "InsuranceClient: MOCK MODE active — "
                "set INSURANCE_MOCK_MODE=false with INSURANCE_API_KEY to use the live API."
            )

    async def verify_insurance(
        self,
        employee_id: str,
        employee_id_type: str,
        unified_national_no: str,
        start_date: str | date,
        end_date: str | date,
    ) -> dict:
        """
        Check whether the employee had active insurance coverage for the given period.

        Args:
            employee_id:          Employee's ID number (NIN, Iqama, Passport, GCC ID).
            employee_id_type:     Type of ID — "National ID" | "Iqama" | "Passport" | "GCC ID".
            unified_national_no:  Employer's Unified National Number.
            start_date:           Start of the engagement period (YYYY-MM-DD or date).
            end_date:             End of the engagement period (YYYY-MM-DD or date).

        Returns:
            dict containing:
                insurance_active (bool): True if coverage was active for the full period.
                employee_id, employee_id_type, unified_national_no: echoed inputs.
                checked_from, checked_to: echoed date range.
                mock (bool): True when running in demo mode.
        """
        start_str = start_date.isoformat() if isinstance(start_date, date) else str(start_date)
        end_str = end_date.isoformat() if isinstance(end_date, date) else str(end_date)

        if self.mock_mode:
            logger.info(
                "Insurance mock: checking coverage for employee_id='%s' "
                "(type=%s) at employer='%s' from %s to %s",
                employee_id, employee_id_type, unified_national_no, start_str, end_str,
            )
            return _mock_insurance_response(
                employee_id, employee_id_type, unified_national_no, start_str, end_str
            )

        return await self._call_live_api(
            employee_id, employee_id_type, unified_national_no, start_str, end_str
        )

    async def _call_live_api(
        self,
        employee_id: str,
        employee_id_type: str,
        unified_national_no: str,
        start_date: str,
        end_date: str,
    ) -> dict:
        if not self.api_key:
            raise InsuranceError("INSURANCE_API_KEY is not set.")
        if not self.api_url:
            raise InsuranceError("INSURANCE_API_URL is not set.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        payload = {
            "employeeId": employee_id,
            "employeeIdType": employee_id_type,
            "unifiedNationalNo": unified_national_no,
            "startDate": start_date,
            "endDate": end_date,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as http:
            try:
                response = await http.post(
                    f"{self.api_url}/insurance/verify",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                raise InsuranceError(
                    f"Insurance API HTTP {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except httpx.RequestError as exc:
                raise InsuranceError(f"Insurance API connection error: {exc}") from exc


class InsuranceError(Exception):
    pass
