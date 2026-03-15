"""
Training pipeline entry point.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --model-path artifacts/model_v2.pkl

Steps:
    1. Connect to Ameen + Fraud databases
    2. Extract and enrich historical transactions
    3. Build feature vectors
    4. Train logistic regression model
    5. Save model, scaler, and metrics to artifacts/
"""
import argparse
import logging
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config.settings import settings
from db.connections import init_pools, get_ameen_conn, get_fraud_conn, close_pools
from training.extract import extract_training_dataset, build_feature_rows
from training.train import train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(model_path: str, scaler_path: str, metrics_path: str) -> None:
    logger.info("=== Predictive Engine — Training Pipeline ===")

    # 1. Connect
    logger.info("Initialising connection pools...")
    init_pools()

    try:
        with get_ameen_conn() as ameen_conn, get_fraud_conn() as fraud_conn:
            # 2. Extract
            logger.info("Extracting training data...")
            df_raw = extract_training_dataset(ameen_conn, fraud_conn)

        # 3. Feature engineering
        logger.info("Building feature vectors...")
        df_features = build_feature_rows(df_raw)

        label_counts = df_features["is_approved"].value_counts()
        logger.info("Label distribution: %s", label_counts.to_dict())

        # 4. Train + save
        logger.info("Training model...")
        train_model(
            df_features,
            model_path=model_path,
            scaler_path=scaler_path,
            metrics_path=metrics_path,
        )

        logger.info("=== Training complete ===")

    finally:
        close_pools()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train approval/rejection scoring model")
    parser.add_argument("--model-path", default=settings.model_path)
    parser.add_argument("--scaler-path", default=settings.scaler_path)
    parser.add_argument("--metrics-path", default="artifacts/metrics_v1.json")
    args = parser.parse_args()

    main(args.model_path, args.scaler_path, args.metrics_path)
