from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── Ameen DB ──────────────────────────────────────────────────────────────
    ameen_usr: str = Field(..., env="AMEEN_USR")
    ameen_pass: str = Field(..., env="AMEEN_PASS")
    ameen_dsn: str = Field(..., env="AMEEN_DSN")
    ameen_pool_min: int = Field(2, env="AMEEN_POOL_MIN")
    ameen_pool_max: int = Field(10, env="AMEEN_POOL_MAX")

    # ── Fraud DB ──────────────────────────────────────────────────────────────
    fraud_usr: str = Field(..., env="FRAUD_USR")
    fraud_pass: str = Field(..., env="FRAUD_PASS")
    fraud_dsn: str = Field(..., env="FRAUD_DSN")
    fraud_pool_min: int = Field(2, env="FRAUD_POOL_MIN")
    fraud_pool_max: int = Field(10, env="FRAUD_POOL_MAX")

    # ── Model artifacts ───────────────────────────────────────────────────────
    model_path: str = Field("artifacts/model_v1.pkl", env="MODEL_PATH")
    scaler_path: str = Field("artifacts/scaler_v1.pkl", env="SCALER_PATH")

    # ── Scoring blender ───────────────────────────────────────────────────────
    rule_weight: float = Field(0.70, env="RULE_WEIGHT")
    ml_weight: float = Field(0.30, env="ML_WEIGHT")

    # ── MVP scope filters ─────────────────────────────────────────────────────
    transaction_id_filter: int = Field(101004, env="TRANSACTION_ID_FILTER")
    channel_filter: str = Field("taminaty", env="CHANNEL_FILTER")
    lawtype_filter: int = Field(1001, env="LAWTYPE_FILTER")
    history_start_date: str = Field("01-03-2024", env="HISTORY_START_DATE")

    # ── BRD scoring tables ────────────────────────────────────────────────────
    # Establishment law classification points
    law_points: dict = {
        "GOSI_GOV": 10,
        "GOSI_SEMI_GOV": 10,
        "GOSI_PRIVATE": 5,
        "PPA": 0,
    }

    # Joining date backdated months → points
    joining_date_points: list[tuple] = [
        (2, 4, 20),    # 2–4 months back → +20
        (4, 12, 10),   # 4–12 months back → +10
        (12, 24, 5),   # 1–2 years back → +5
    ]

    # Establishment historical approval rate → points
    est_approval_rate_points: list[tuple] = [
        (90, 100, 50),   # 90–100% → +50
        (80, 90, 30),    # 80–90% → +30
        (60, 80, 0),     # 60–80% → +0
        (40, 60, -10),   # 40–60% → -10
        (20, 40, -20),   # 20–40% → -20
        (0, 20, -30),    # 0–20% → -30
    ]

    # Contributor historical approval rate → points (same structure as est)
    contributor_approval_rate_points: list[tuple] = [
        (90, 100, 50),
        (80, 90, 30),
        (60, 80, 0),
        (40, 60, -10),
        (20, 40, -20),
        (0, 20, -30),
    ]

    # Base score before adjustments
    base_score: int = 50

    # Max possible points from rules (for normalisation)
    # law(10) + joining_date(20) + est_rate(50) + contributor_rate(50) = 130 above base of 50 → max=180
    # Penalty floor is open-ended, but we clamp at 0-100 anyway
    rule_score_max: int = 180

    # Violation history: penalty per violation per month
    violation_penalty_per_month: float = 5.0

    class Config:
        env_file = ".env"
        extra = "ignore"


# Singleton — import this everywhere
settings = Settings()
