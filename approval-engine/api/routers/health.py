from fastapi import APIRouter
from api.schemas.response import HealthResponse
from db.connections import health_check
from scoring.ml_scorer import is_model_loaded

router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthResponse)
def health():
    """Check DB connectivity and model availability."""
    db_status = health_check()
    all_ok = all(v == "ok" for v in db_status.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        ameen_db=db_status.get("ameen_db", "unknown"),
        fraud_db=db_status.get("fraud_db", "unknown"),
        model_loaded=is_model_loaded(),
    )
