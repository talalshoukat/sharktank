"""
HRSD (Human Resources and Social Development) contract lookup client.

Given an employee's ID details and the employer's unified national number,
calls the HRSD API and returns the full labour contract on record —
establishment info, employee info, and job/contract details.

When HRSD_MOCK_MODE=true (default), get_contract_details() returns a
realistic stub with a randomly sampled verification_score so the scoring
pipeline runs end-to-end in demos without live credentials.

When HRSD_MOCK_MODE=false, a real HTTP call is made to HRSD_API_URL.
Switch is purely via environment variables — no code changes needed.
"""
from __future__ import annotations

import logging
import os
import random

import httpx

logger = logging.getLogger(__name__)


def _mock_contract_details(
    employee_id: str,
    employee_id_type: str,
    unified_national_no: str,
) -> dict:
    """
    Realistic stub. Returns the same shape as the live HRSD API.
    `verification_score` is random (0–100) to simulate variation across
    contracts until real scoring logic is wired to real API responses.
    """
    return {
        # ── Establishment ─────────────────────────────────────────────────
        "establishment_unified_national_no": unified_national_no,
        "establishment_name_ar": "شركة التقنية المتقدمة",
        "establishment_name_en": "Advanced Technology Company",
        "establishment_email": "hr@advancedtech.com.sa",
        "establishment_phone": "+966112345678",
        "establishment_cr_number": "1010234567",
        "establishment_activity": "Information Technology",

        # ── Employee ──────────────────────────────────────────────────────
        "employee_id": employee_id,
        "employee_id_type": employee_id_type,
        "employee_name_ar": "محمد أحمد العمري",
        "employee_name_en": "Mohammed Ahmed Al-Omari",
        "nationality": "Saudi Arabian",
        "gender": "Male",
        "birth_date": "1990-05-14",

        # ── Job / Contract ────────────────────────────────────────────────
        "speciality": "Software Engineering",
        "education_level": "Bachelor's Degree",
        "field_of_study": "Computer Science",
        "occupation": "Software Developer",
        "job_title": "Senior Software Engineer",
        "contract_start_date": "2023-01-01",
        "contract_end_date": "2025-12-31",
        "contract_type": "Fixed-term",
        "work_location": "Riyadh",
        "basic_salary": 15000.00,
        "currency": "SAR",

        # ── Mock meta ─────────────────────────────────────────────────────
        "mock": True,
        # Random score simulates variation across real contracts.
        # Replace with real comparison logic once HRSD API is available.
        "verification_score": random.randint(0, 100),
    }


class HRSDClient:
    """
    Looks up a labour contract from the HRSD API by employee identity
    and employer unified national number.

    Environment variables:
        HRSD_MOCK_MODE          "true" (default) or "false"
        HRSD_API_URL            Base URL of the live HRSD API
        HRSD_API_KEY            Bearer token / API key
        HRSD_TIMEOUT_SECONDS    HTTP timeout (default 30)
    """

    def __init__(self):
        self.mock_mode: bool = os.getenv("HRSD_MOCK_MODE", "true").lower() != "false"
        self.api_url: str = os.getenv("HRSD_API_URL", "")
        self.api_key: str = os.getenv("HRSD_API_KEY", "")
        self.timeout: int = int(os.getenv("HRSD_TIMEOUT_SECONDS", "30"))

        if self.mock_mode:
            logger.info(
                "HRSDClient: MOCK MODE active — "
                "set HRSD_MOCK_MODE=false with HRSD_API_KEY to use the live API."
            )

    async def get_contract_details(
        self,
        employee_id: str,
        employee_id_type: str,
        unified_national_no: str,
    ) -> dict:
        """
        Fetch the labour contract from HRSD for a given employee at a given employer.

        Args:
            employee_id:        Employee's ID number (NIN, Iqama, Passport, GCC ID).
            employee_id_type:   Type of ID — "National ID" | "Iqama" | "Passport" | "GCC ID".
            unified_national_no: Employer's Unified National Number (الرقم الوطني الموحد).

        Returns:
            dict with establishment info, employee info, contract/job details,
            and a `verification_score` (0–100).
            In mock mode, verification_score is random to simulate variance.
        """
        if self.mock_mode:
            logger.info(
                "HRSD mock: returning stub contract for employee_id='%s' "
                "(type=%s) at employer='%s'",
                employee_id, employee_id_type, unified_national_no,
            )
            return _mock_contract_details(employee_id, employee_id_type, unified_national_no)

        return await self._call_live_api(employee_id, employee_id_type, unified_national_no)

    async def _call_live_api(
        self,
        employee_id: str,
        employee_id_type: str,
        unified_national_no: str,
    ) -> dict:
        if not self.api_key:
            raise HRSDError("HRSD_API_KEY is not set.")
        if not self.api_url:
            raise HRSDError("HRSD_API_URL is not set.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        payload = {
            "employeeId": employee_id,
            "employeeIdType": employee_id_type,
            "unifiedNationalNo": unified_national_no,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as http:
            try:
                response = await http.post(
                    f"{self.api_url}/contracts/lookup",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                raise HRSDError(
                    f"HRSD API HTTP {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except httpx.RequestError as exc:
                raise HRSDError(f"HRSD API connection error: {exc}") from exc


class HRSDError(Exception):
    pass
