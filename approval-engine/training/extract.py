"""
Training data extraction.
Pulls historical completed transactions and enriches them with all features.
"""
from __future__ import annotations

import logging

import pandas as pd
import oracledb

from config.settings import settings
from data.loader import (
    fetch_historical_transactions,
    fetch_trace_params,
    fetch_violation_requests,
    fetch_establishments,
    fetch_enc_map,
    fetch_kashif_indicators,
    _map_law_category,
)
from db.queries import ameen_queries as aq

logger = logging.getLogger(__name__)


def extract_training_dataset(
    ameen_conn: oracledb.Connection,
    fraud_conn: oracledb.Connection,
) -> pd.DataFrame:
    """
    Full extraction: transactions → params → establishments → fraud scores.
    Returns a denormalised dataframe ready for feature engineering.
    """
    # 1. Base transaction status
    df_status = fetch_historical_transactions(ameen_conn)
    logger.info("Step 1: %d transactions fetched", len(df_status))

    if df_status.empty:
        raise RuntimeError("No historical transactions found. Check filters in settings.")

    # 2. Pivot transaction parameters
    df_params = fetch_trace_params(df_status["TRANSACTIONTRACEID"].unique(), ameen_conn)
    df_pivot = (
        df_params.pivot_table(
            index="TRANSACTIONTRACEID",
            columns="PARAMKEY",
            values="PARAMVALUE",
            aggfunc="first",
        )
        .reset_index()
    )
    df_pivot.columns.name = None
    logger.info("Step 2: trace params pivoted, %d columns", len(df_pivot.columns))

    # 3. Merge
    df_merged = df_status.merge(df_pivot, on="TRANSACTIONTRACEID", how="inner")
    df_merged["ESTABLISHMENTID"] = df_merged["ESTABLISHMENTID"].astype("Int64")

    # 4. Filter GOSI law only (MVP scope)
    # Keep records that have an ESTABLISHMENTID (needed for law type check)
    df_merged = df_merged[df_merged["ESTABLISHMENTID"].notna()].copy()

    # 5. Violation requests
    if "ContributorViolationRequestId" in df_merged.columns:
        viol_ids = df_merged["ContributorViolationRequestId"].dropna().unique()
        if len(viol_ids):
            df_viol = fetch_violation_requests(viol_ids, ameen_conn)
            df_merged = df_merged.merge(
                df_viol,
                left_on="ContributorViolationRequestId",
                right_on="CONTRIBUTORVIOLATIONREQID",
                how="left",
            )

    # 6. Establishment master + law type filter
    reg_numbers = df_merged.get("GOSIREGISTRATIONNUMBER", df_merged.get("RegistrationNo"))
    if reg_numbers is not None:
        df_est = fetch_establishments(reg_numbers.dropna().unique(), ameen_conn)
        df_merged = df_merged.merge(df_est, on="ESTABLISHMENTID", how="inner")

    # Filter to GOSI law only
    df_merged = df_merged[df_merged["LAWTYPE"] == settings.lawtype_filter].copy()
    logger.info("Step 3: %d records after GOSI law filter", len(df_merged))

    # 7. Establishment historical approval rates
    df_est_rates = pd.read_sql(
        aq.ESTABLISHMENT_APPROVAL_RATE_SQL,
        ameen_conn,
        params={
            "channel": settings.channel_filter,
            "transaction_id": settings.transaction_id_filter,
            "year": 2025,
        },
    )
    df_est_rates = df_est_rates.rename(columns={"APPROVAL_PCT": "EST_APPROVAL_RATE"})
    df_merged = df_merged.merge(
        df_est_rates[["ESTABLISHMENTID", "EST_APPROVAL_RATE"]],
        on="ESTABLISHMENTID",
        how="left",
    )

    # 8. KASHIF fraud scores
    df_enc = fetch_enc_map(df_merged["ESTABLISHMENTID"].unique(), fraud_conn)
    df_kashif = fetch_kashif_indicators(df_enc["ESTABLISHMENTID_ENC"].unique(), fraud_conn)

    # Join enc map then kashif
    df_merged = df_merged.merge(df_enc, on="ESTABLISHMENTID", how="left")
    df_merged = df_merged.merge(
        df_kashif.rename(columns={"ESTABLISHMENTID": "ESTABLISHMENTID_ENC"}),
        on="ESTABLISHMENTID_ENC",
        how="left",
    )

    logger.info("Extraction complete: %d rows", len(df_merged))
    return df_merged


def build_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw denormalised dataframe into a feature matrix.
    Returns a dataframe where each row = one feature vector + label.
    """
    from features.feature_vector import build_feature_vector, _compute_backdated_months, _encode_status

    rows = []
    for _, row in df.iterrows():
        raw = {
            "transaction_trace_id": row.get("TRANSACTIONTRACEID"),
            "establishment_id": row.get("ESTABLISHMENTID"),
            "person_id": row.get("PERSONID"),
            "law_type": row.get("LAWTYPE"),
            "law_category": _map_law_category(row.get("LAWTYPE")),
            "nin_present": pd.notna(row.get("NIN")),
            "joining_date": row.get("JOININGDATE") or row.get("STARTDATE"),
            "est_approval_rate": row.get("EST_APPROVAL_RATE"),
            "contributor_approval_rate": None,  # computed separately per contributor
            "violation_count_per_month": _calc_violation_rate(row),
            "kashif_score": row.get("COMPOUND_VALUE_1"),
            "status": row.get("STATUS_x") or row.get("STATUS"),
        }
        rows.append(build_feature_vector(raw))

    return pd.DataFrame(rows)


def _calc_violation_rate(row) -> float:
    viol_count = row.get("VIOLATIONSCOUNT", 0) or 0
    months_active = row.get("MONTHSACTIVE", 1) or 1
    return float(viol_count) / float(months_active)
