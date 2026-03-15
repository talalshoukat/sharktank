"""
Unit tests for the rule engine.
Tests every BRD scoring function against known inputs and boundary conditions.
No database connections needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import patch, MagicMock

# Mock settings before importing rule_engine (avoids .env dependency in CI)
mock_settings = MagicMock()
mock_settings.law_points = {
    "GOSI_GOV": 10,
    "GOSI_SEMI_GOV": 10,
    "GOSI_PRIVATE": 5,
    "PPA": 0,
}
mock_settings.joining_date_points = [(2, 4, 20), (4, 12, 10), (12, 24, 5)]
mock_settings.est_approval_rate_points = [
    (90, 100, 50), (80, 90, 30), (60, 80, 0), (40, 60, -10), (20, 40, -20), (0, 20, -30)
]
mock_settings.contributor_approval_rate_points = [
    (90, 100, 50), (80, 90, 30), (60, 80, 0), (40, 60, -10), (20, 40, -20), (0, 20, -30)
]
mock_settings.base_score = 50
mock_settings.rule_score_max = 180
mock_settings.violation_penalty_per_month = 5.0

with patch("config.settings.settings", mock_settings):
    from scoring.rule_engine import (
        score_establishment_law,
        score_joining_date,
        score_approval_rate_est,
        score_approval_rate_contributor,
        score_violation_history,
        compute_rule_score,
    )


# ── Establishment law ─────────────────────────────────────────────────────────

class TestScoreEstablishmentLaw:
    def test_gosi_gov(self):
        pts, _ = score_establishment_law("GOSI_GOV")
        assert pts == 10

    def test_gosi_semi_gov(self):
        pts, _ = score_establishment_law("GOSI_SEMI_GOV")
        assert pts == 10

    def test_gosi_private(self):
        pts, _ = score_establishment_law("GOSI_PRIVATE")
        assert pts == 5

    def test_ppa(self):
        pts, _ = score_establishment_law("PPA")
        assert pts == 0

    def test_unknown(self):
        pts, reason = score_establishment_law("UNKNOWN")
        assert pts == 0
        assert reason is not None

    def test_none(self):
        pts, _ = score_establishment_law(None)
        assert pts == 0

    def test_case_insensitive(self):
        pts, _ = score_establishment_law("gosi_gov")
        assert pts == 10


# ── Joining date ──────────────────────────────────────────────────────────────

class TestScoreJoiningDate:
    def test_2_to_4_months(self):
        pts, _ = score_joining_date(3.0)
        assert pts == 20

    def test_4_to_12_months(self):
        pts, _ = score_joining_date(6.0)
        assert pts == 10

    def test_1_to_2_years(self):
        pts, _ = score_joining_date(18.0)
        assert pts == 5

    def test_over_2_years(self):
        pts, _ = score_joining_date(30.0)
        assert pts == -10

    def test_future_date(self):
        pts, _ = score_joining_date(-1.0)
        assert pts == -20

    def test_none(self):
        pts, _ = score_joining_date(None)
        assert pts == 0

    def test_boundary_2_months(self):
        pts, _ = score_joining_date(2.0)
        assert pts == 20

    def test_boundary_4_months(self):
        pts, _ = score_joining_date(4.0)
        assert pts == 20  # 4 is within [2,4] inclusive


# ── Establishment approval rate ───────────────────────────────────────────────

class TestScoreApprovalRateEst:
    @pytest.mark.parametrize("rate,expected_pts", [
        (95.0, 50),
        (85.0, 30),
        (70.0, 0),
        (50.0, -10),
        (30.0, -20),
        (10.0, -30),
    ])
    def test_bands(self, rate, expected_pts):
        pts, _ = score_approval_rate_est(rate)
        assert pts == expected_pts

    def test_none_returns_zero(self):
        pts, reason = score_approval_rate_est(None)
        assert pts == 0
        assert reason is not None

    def test_100_pct(self):
        pts, _ = score_approval_rate_est(100.0)
        assert pts == 50

    def test_0_pct(self):
        pts, _ = score_approval_rate_est(0.0)
        assert pts == -30


# ── Violation history ─────────────────────────────────────────────────────────

class TestScoreViolationHistory:
    def test_no_violations(self):
        pts, reason = score_violation_history(0.0)
        assert pts == 0
        assert reason is None

    def test_2_per_month(self):
        pts, reason = score_violation_history(2.0)
        assert pts == -10
        assert "violation" in reason.lower()

    def test_fractional(self):
        pts, _ = score_violation_history(0.5)
        assert pts == -2  # round(0.5 * 5) = 2 or 3 depending on rounding


# ── Categories ────────────────────────────────────────────────────────────────

class TestCategories:
    def test_all_categories(self):
        from scoring.categories import get_category
        assert get_category(85)[0] == "High"
        assert get_category(80)[0] == "High"
        assert get_category(65)[0] == "Moderate"
        assert get_category(60)[0] == "Moderate"
        assert get_category(45)[0] == "Risky"
        assert get_category(40)[0] == "Risky"
        assert get_category(25)[0] == "High Rejection Risk"
        assert get_category(20)[0] == "High Rejection Risk"
        assert get_category(15)[0] == "Very Likely Rejection"
        assert get_category(0)[0] == "Very Likely Rejection"


# ── Blender ───────────────────────────────────────────────────────────────────

class TestBlender:
    def test_no_ml_falls_back_to_rules(self):
        from scoring.blender import blend
        with patch("scoring.blender.settings", mock_settings):
            mock_settings.rule_score_max = 180
            score = blend(rule_points=90, ml_probability=None)
        assert 0 <= score <= 100

    def test_clamps_to_zero(self):
        from scoring.blender import blend
        with patch("scoring.blender.settings", mock_settings):
            mock_settings.rule_score_max = 180
            mock_settings.rule_weight = 0.7
            mock_settings.ml_weight = 0.3
            score = blend(rule_points=-100, ml_probability=0.0)
        assert score == 0.0

    def test_clamps_to_hundred(self):
        from scoring.blender import blend
        with patch("scoring.blender.settings", mock_settings):
            mock_settings.rule_score_max = 180
            mock_settings.rule_weight = 0.7
            mock_settings.ml_weight = 0.3
            score = blend(rule_points=500, ml_probability=1.0)
        assert score == 100.0
