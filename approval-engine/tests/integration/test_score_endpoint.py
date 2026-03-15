"""
Integration tests for the /score endpoint.
Uses FastAPI TestClient with mocked DB connections and data loader.
No actual DB connection required.
"""
import sys
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from fastapi.testclient import TestClient

# Mock settings before app import
mock_settings = MagicMock()
mock_settings.ameen_usr = "test"
mock_settings.ameen_pass = "test"
mock_settings.ameen_dsn = "test"
mock_settings.fraud_usr = "test"
mock_settings.fraud_pass = "test"
mock_settings.fraud_dsn = "test"
mock_settings.ameen_pool_min = 1
mock_settings.ameen_pool_max = 2
mock_settings.fraud_pool_min = 1
mock_settings.fraud_pool_max = 2
mock_settings.model_path = "artifacts/model_v1.pkl"
mock_settings.scaler_path = "artifacts/scaler_v1.pkl"
mock_settings.rule_weight = 0.7
mock_settings.ml_weight = 0.3
mock_settings.base_score = 50
mock_settings.rule_score_max = 180
mock_settings.violation_penalty_per_month = 5.0
mock_settings.law_points = {"GOSI_GOV": 10, "GOSI_SEMI_GOV": 10, "GOSI_PRIVATE": 5, "PPA": 0}
mock_settings.joining_date_points = [(2, 4, 20), (4, 12, 10), (12, 24, 5)]
mock_settings.est_approval_rate_points = [
    (90, 100, 50), (80, 90, 30), (60, 80, 0), (40, 60, -10), (20, 40, -20), (0, 20, -30)
]
mock_settings.contributor_approval_rate_points = [
    (90, 100, 50), (80, 90, 30), (60, 80, 0), (40, 60, -10), (20, 40, -20), (0, 20, -30)
]
mock_settings.channel_filter = "taminaty"
mock_settings.transaction_id_filter = 101004
mock_settings.lawtype_filter = 1001
mock_settings.history_start_date = "01-03-2024"


MOCK_RAW = {
    "transaction_trace_id": 99999,
    "establishment_id": 1234,
    "person_id": 5678,
    "law_type": 1001,
    "law_category": "GOSI_PRIVATE",
    "nin_present": True,
    "joining_date": date.today() - timedelta(days=90),
    "est_approval_rate": 92.0,
    "contributor_approval_rate": 85.0,
    "violation_count_per_month": 0.0,
    "kashif_score": 20.0,
    "status": "Approved",
}


@pytest.fixture
def client():
    with (
        patch("config.settings.settings", mock_settings),
        patch("db.connections.init_pools"),
        patch("db.connections.close_pools"),
        patch("scoring.ml_scorer.load_model"),
        patch("scoring.ml_scorer.is_model_loaded", return_value=False),
        patch("api.routers.score.fetch_single_transaction", return_value=MOCK_RAW),
    ):
        from api.main import app
        with TestClient(app) as c:
            yield c


class TestScoreEndpoint:
    def test_score_returns_200(self, client):
        response = client.post("/api/v1/score", json={"transaction_trace_id": 99999})
        assert response.status_code == 200

    def test_response_structure(self, client):
        data = client.post("/api/v1/score", json={"transaction_trace_id": 99999}).json()
        assert "final_score" in data
        assert "category" in data
        assert "reasons" in data
        assert "rule_breakdown" in data
        assert "disclaimer" in data
        assert 0 <= data["final_score"] <= 100

    def test_category_in_valid_set(self, client):
        data = client.post("/api/v1/score", json={"transaction_trace_id": 99999}).json()
        valid_categories = {
            "High", "Moderate", "Risky", "High Rejection Risk", "Very Likely Rejection"
        }
        assert data["category"] in valid_categories

    def test_missing_transaction_returns_404(self, client):
        with patch(
            "api.routers.score.fetch_single_transaction",
            side_effect=ValueError("Transaction not found"),
        ):
            response = client.post("/api/v1/score", json={"transaction_trace_id": 0})
        assert response.status_code == 404

    def test_batch_score(self, client):
        response = client.post(
            "/api/v1/score/batch",
            json={"transaction_trace_ids": [99999, 99998]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "errors" in data
