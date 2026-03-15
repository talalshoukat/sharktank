"""
Unit tests for feature vector assembly.
"""
import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.feature_vector import build_feature_vector, _compute_backdated_months, _encode_status


class TestBuildFeatureVector:
    def _raw(self, **overrides):
        base = {
            "transaction_trace_id": 1,
            "establishment_id": 1234,
            "person_id": 5678,
            "law_type": 1001,
            "law_category": "GOSI_PRIVATE",
            "nin_present": True,
            "joining_date": date.today() - timedelta(days=90),  # ~3 months
            "est_approval_rate": 92.0,
            "contributor_approval_rate": 85.0,
            "violation_count_per_month": 0.0,
            "kashif_score": 25.0,
            "status": "Approved",
        }
        base.update(overrides)
        return base

    def test_happy_path(self):
        fv = build_feature_vector(self._raw())
        assert fv["law_category"] == "GOSI_PRIVATE"
        assert fv["nin_present"] is True
        assert fv["est_approval_rate"] == 92.0
        assert fv["kashif_score"] == 25.0
        assert fv["is_approved"] == 1

    def test_rejected_label(self):
        fv = build_feature_vector(self._raw(status="Rejected"))
        assert fv["is_approved"] == 0

    def test_unknown_status_returns_zero(self):
        fv = build_feature_vector(self._raw(status="Pending"))
        assert fv["is_approved"] == 0

    def test_none_kashif(self):
        fv = build_feature_vector(self._raw(kashif_score=None))
        assert fv["kashif_score"] is None

    def test_backdated_months_approx_3(self):
        joining = date.today() - timedelta(days=90)
        fv = build_feature_vector(self._raw(joining_date=joining))
        assert 2.5 <= fv["backdated_months"] <= 3.5


class TestComputeBackdatedMonths:
    def test_3_months_back(self):
        d = date.today() - timedelta(days=91)
        result = _compute_backdated_months(d)
        assert 2.5 < result < 3.5

    def test_future_date_negative(self):
        d = date.today() + timedelta(days=30)
        result = _compute_backdated_months(d)
        assert result < 0

    def test_none_returns_none(self):
        assert _compute_backdated_months(None) is None

    def test_string_date(self):
        d = (date.today() - timedelta(days=60)).strftime("%Y-%m-%d")
        result = _compute_backdated_months(d)
        assert result is not None
        assert result > 0


class TestEncodeStatus:
    def test_approved(self):
        assert _encode_status("Approved") == 1
        assert _encode_status("APPROVED") == 1
        assert _encode_status("approved") == 1

    def test_rejected(self):
        assert _encode_status("Rejected") == 0

    def test_none(self):
        assert _encode_status(None) is None
