from __future__ import annotations
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class RuleBreakdown(BaseModel):
    base: int
    establishment_law: int = 0
    joining_date: int = 0
    est_approval_rate: int = 0
    contributor_approval: int = 0
    violation_history: int = 0
    kashif_score: int = 0
    nin_presence: int = 0


class ScoreResponse(BaseModel):
    transaction_trace_id: int
    final_score: float = Field(description="Approval likelihood score 0–100")
    category: str = Field(description="High | Moderate | Risky | High Rejection Risk | Very Likely Rejection")
    category_colour: str
    rule_score_raw: int = Field(description="Raw rule engine points before normalisation")
    ml_probability: Optional[float] = Field(None, description="ML model approval probability 0–1, null if model not loaded")
    reasons: list[str] = Field(description="Key factors impacting the score")
    rule_breakdown: RuleBreakdown
    disclaimer: str = "Indicator only — final decision subject to validation"
    scored_at: datetime = Field(default_factory=datetime.utcnow)


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]
    errors: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    ameen_db: str
    fraud_db: str
    model_loaded: bool
