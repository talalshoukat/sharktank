"""
FastAPI application factory.
Initialises DB connection pools and loads model on startup.
"""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from config.settings import settings
from db.connections import init_pools, close_pools
from scoring.ml_scorer import load_model
from api.routers import score, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Initialising DB connection pools...")
    init_pools()

    logger.info("Loading ML model from %s", settings.model_path)
    load_model(settings.model_path, settings.scaler_path)

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Closing DB connection pools...")
    close_pools()


app = FastAPI(
    title="Predictive Approval & Rejection Engine",
    description=(
        "Provides real-time approval likelihood scoring for GOSI transactions. "
        "Indicator only — final decision subject to validation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(score.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")
