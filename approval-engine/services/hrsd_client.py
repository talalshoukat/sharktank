"""
HRSD (Human Resources and Social Development) contract lookup client.

Given a contract_id, calls the HRSD API and returns the structured
contract details on file — establishment info, employee info, job details.

When HRSD_MOCK_MODE=true (default, used until real credentials are available),
get_contract_details() returns a mocked response with a randomly sampled
verification score so the scoring pipeline runs end-to-end in demos.

When HRSD_MOCK_MODE=false, a real HTTP call is made to HRSD_API_URL.
No code changes needed — the switch is purely via environment variables.
"""
from __future__ import annotations

import logging
import os
import random

import httpx

logger = logging.getLogger(__name__)


class HRSDContractDetails:
    """
    Contract details returned by HRSD for a given contract_id.
    Fields mirror what the HRSD Watheeq / contract-reading service provides.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self) -> dict:
        return self.__dict__


def _mock_contract_details(contract_id: str) -> dict:
    """
    Realistic stub. Returns the same shape as the live HRSD API.
    `verification_score` is random (0–100) to simulate variation across
    contracts until real scoring logic is wired to real API responses.
    """
    return {
        # ── Establishment ─────────────────────────────────────────────────
        "establishment_unified_id": "7000123456",
        "establishment_name_ar": "شركة التقنية المتقدمة",
        "establishment_name_en": "Advanced Technology Company",
        "establishment_email": "hr@advancedtech.com.sa",
        "establishment_phone": "+966112345678",
        "establishment_cr_number": "1010234567",
        "establishment_activity": "Information Technology",

        # ── Employee ──────────────────────────────────────────────────────
        "employee_name_ar": "محمد أحمد العمري",
        "employee_name_en": "Mohammed Ahmed Al-Omari",
        "nationality": "Saudi Arabian",
        "id_type": "National ID",
        "id_number": "1098765432",
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
        "contract_id": contract_id,
        "mock": True,
        # Random score simulates the variation you'd see across real contracts.
        # Replace with real comparison logic once HRSD API is available.
        "verification_score": random.randint(0, 100),
    }


class HRSDClient:
    """
    Looks up a labour contract by contract_id from the HRSD API.

    Environment variables:
        HRSD_MOCK_MODE      "true" (default) or "false"
        HRSD_API_URL        Base URL of the live HRSD API
        HRSD_API_KEY        Bearer token / API key
        HRSD_TIMEOUT_SECONDS  HTTP timeout (default 30)
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

    async def get_contract_details(self, contract_id: str) -> dict:
        """
        Fetch contract details from HRSD by contract_id.

        Returns a dict with establishment info, employee info, and
        a `verification_score` (0–100) that reflects how well the
        contract data matches what was submitted in the transaction.

        In mock mode, verification_score is randomised to simulate variance.
        In live mode, it comes from comparing transaction data against HRSD records.
        """
        if self.mock_mode:
            logger.info("HRSD mock: returning stub contract for contract_id='%s'", contract_id)
            return _mock_contract_details(contract_id)

        return await self._call_live_api(contract_id)

    async def _call_live_api(self, contract_id: str) -> dict:
        if not self.api_key:
            raise HRSDError("HRSD_API_KEY is not set.")
        if not self.api_url:
            raise HRSDError("HRSD_API_URL is not set.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as http:
            try:
                response = await http.get(
                    f"{self.api_url}/contracts/{contract_id}",
                    headers=headers,
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
