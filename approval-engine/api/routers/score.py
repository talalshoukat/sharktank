"""
Scoring endpoints.
POST /score        — score a single transaction
POST /score/batch  — score up to 100 transactions
"""
import logging

from fastapi import APIRouter, HTTPException

from api.schemas.request import ScoreRequest, BatchScoreRequest
from api.schemas.response import ScoreResponse, BatchScoreResponse, RuleBreakdown
from db.connections import get_ameen_conn, get_fraud_conn
from data.loader import fetch_single_transaction
from features.feature_vector import build_feature_vector
from scoring.rule_engine import compute_rule_score
from scoring.ml_scorer import predict_proba
from scoring.blender import blend
from scoring.categories import get_category

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/score", tags=["scoring"])


def _score_one(transaction_trace_id: int) -> ScoreResponse:
    with get_ameen_conn() as ameen_conn, get_fraud_conn() as fraud_conn:
        raw = fetch_single_transaction(transaction_trace_id, ameen_conn, fraud_conn)

    features = build_feature_vector(raw)
    rule_points, reasons, breakdown = compute_rule_score(features)
    ml_prob = predict_proba(features)
    final_score = blend(rule_points, ml_prob)
    category, colour = get_category(final_score)

    return ScoreResponse(
        transaction_trace_id=transaction_trace_id,
        final_score=final_score,
        category=category,
        category_colour=colour,
        rule_score_raw=rule_points,
        ml_probability=ml_prob,
        reasons=reasons,
        rule_breakdown=RuleBreakdown(**breakdown),
    )


@router.post("", response_model=ScoreResponse)
def score_transaction(request: ScoreRequest):
    """
    Score a single transaction and return approval likelihood.

    Disclaimer: indicator only — final decision subject to validation.
    """
    try:
        return _score_one(request.transaction_trace_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Scoring failed for transaction %d", request.transaction_trace_id)
        raise HTTPException(status_code=500, detail="Internal scoring error")


@router.post("/batch", response_model=BatchScoreResponse)
def score_batch(request: BatchScoreRequest):
    """Score up to 100 transactions in a single call."""
    results = []
    errors = []

    for txn_id in request.transaction_trace_ids:
        try:
            results.append(_score_one(txn_id))
        except ValueError as exc:
            errors.append({"transaction_trace_id": txn_id, "error": str(exc)})
        except Exception as exc:
            logger.exception("Batch scoring failed for transaction %d", txn_id)
            errors.append({"transaction_trace_id": txn_id, "error": "Internal scoring error"})

    return BatchScoreResponse(results=results, errors=errors)
