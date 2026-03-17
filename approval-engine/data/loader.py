"""
Data loading utilities — batch fetching and single-transaction enrichment.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
import oracledb

from db.queries import ameen_queries as aq
from db.queries import fraud_queries as fq
from config.settings import settings

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000


# ── Generic batch loader (from notebook) ─────────────────────────────────────

def _chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def load_data(ids: Any, sql_prefix: str, conn: oracledb.Connection) -> pd.DataFrame:
    """
    Execute a batched IN-clause query.
    `sql_prefix` must end with the column name, e.g.:
        "SELECT * FROM T_FOO WHERE COLUMN_ID"
    Bind params are appended as :1, :2, ...
    """
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    ids = [i for i in ids if i is not None]

    if not ids:
        return pd.DataFrame()

    results = []
    for batch in _chunks(ids, CHUNK_SIZE):
        placeholders = ",".join([f":{i + 1}" for i in range(len(batch))])
        query = f"{sql_prefix} IN ({placeholders})"
        results.append(pd.read_sql(query, conn, params=batch))

    return pd.concat(results, ignore_index=True)


# ── Historical data extraction (training) ─────────────────────────────────────

def fetch_historical_transactions(
    ameen_conn: oracledb.Connection,
) -> pd.DataFrame:
    """
    Pull all completed transactions matching MVP scope from Ameen DB.
    Returns status dataframe.
    """
    df = pd.read_sql(
        aq.TRANSACTION_STATUS_SQL,
        ameen_conn,
        params={
            "channel": settings.channel_filter,
            "transaction_id": settings.transaction_id_filter,
            "start_date": settings.history_start_date,
        },
    )
    logger.info("Fetched %d historical transactions", len(df))
    return df


def fetch_trace_params(
    transaction_trace_ids: Any, ameen_conn: oracledb.Connection
) -> pd.DataFrame:
    return load_data(transaction_trace_ids, aq.TRACE_PARAM_SQL, ameen_conn)


def fetch_violation_requests(
    violation_req_ids: Any, ameen_conn: oracledb.Connection
) -> pd.DataFrame:
    return load_data(violation_req_ids, aq.VIOLATION_REQ_SQL, ameen_conn)


def fetch_establishments(
    registration_numbers: Any, ameen_conn: oracledb.Connection
) -> pd.DataFrame:
    return load_data(registration_numbers, aq.ESTABLISHMENT_SQL, ameen_conn)


def fetch_enc_map(
    establishment_ids: Any, fraud_conn: oracledb.Connection
) -> pd.DataFrame:
    return load_data(establishment_ids, fq.EST_ENC_MAP_SQL, fraud_conn)


def fetch_kashif_indicators(
    enc_establishment_ids: Any, fraud_conn: oracledb.Connection
) -> pd.DataFrame:
    df = load_data(enc_establishment_ids, fq.KASHIF_INDICATORS_SQL, fraud_conn)
    if df.empty:
        return df
    # Keep only the most recent record per establishment
    df = (
        df.sort_values("CREATION_DATE", ascending=False)
        .drop_duplicates(subset="ESTABLISHMENTID", keep="first")
        .reset_index(drop=True)
    )
    return df


# ── Single-transaction enrichment (inference) ─────────────────────────────────

def fetch_single_transaction(
    transaction_trace_id: int,
    ameen_conn: oracledb.Connection,
    fraud_conn: oracledb.Connection,
) -> dict:
    """
    Fetch all data needed to score one transaction.
    Returns a flat dict ready for feature engineering.
    """
    # 1. Core transaction record
    df_txn = pd.read_sql(
        aq.SINGLE_TRANSACTION_STATUS_SQL,
        ameen_conn,
        params={"transaction_trace_id": transaction_trace_id},
    )
    if df_txn.empty:
        raise ValueError(f"Transaction {transaction_trace_id} not found.")

    txn = df_txn.iloc[0].to_dict()
    establishment_id = txn.get("ESTABLISHMENTID")
    person_id = txn.get("PERSONID")

    # 2. Transaction parameters (pivot to wide)
    df_params = pd.read_sql(
        aq.SINGLE_TRACE_PARAM_SQL,
        ameen_conn,
        params={"transaction_trace_id": transaction_trace_id},
    )
    params = (
        df_params.pivot_table(
            index="TRANSACTIONTRACEID",
            columns="PARAMKEY",
            values="PARAMVALUE",
            aggfunc="first",
        )
        .reset_index()
        .iloc[0]
        .to_dict()
        if not df_params.empty
        else {}
    )

    # 3. Establishment master
    est_record: dict = {}
    if establishment_id:
        df_est = pd.read_sql(
            aq.ESTABLISHMENT_BY_ID_SQL,
            ameen_conn,
            params={"establishment_id": int(establishment_id)},
        )
        est_record = df_est.iloc[0].to_dict() if not df_est.empty else {}

    # 4. Historical approval rate for this establishment
    est_approval_rate: float | None = None
    df_est_rate = pd.read_sql(
        aq.ESTABLISHMENT_APPROVAL_RATE_SQL,
        ameen_conn,
        params={
            "channel": settings.channel_filter,
            "transaction_id": settings.transaction_id_filter,
            "year": 2025,
        },
    )
    if not df_est_rate.empty and establishment_id:
        row = df_est_rate[df_est_rate["ESTABLISHMENTID"] == establishment_id]
        if not row.empty:
            est_approval_rate = float(row.iloc[0]["APPROVAL_PCT"])

    # 5. Contributor historical approval rate
    contributor_approval_rate: float | None = None
    if person_id:
        df_contrib_rate = pd.read_sql(
            aq.CONTRIBUTOR_APPROVAL_RATE_SQL,
            ameen_conn,
            params={
                "channel": settings.channel_filter,
                "transaction_id": settings.transaction_id_filter,
                "year": 2025,
                "person_id": int(person_id),
            },
        )
        if not df_contrib_rate.empty:
            contributor_approval_rate = float(df_contrib_rate.iloc[0]["APPROVAL_PCT"])

    # 6. Violation request
    violation_req_id = params.get("ContributorViolationRequestId")
    violation_count_per_month: float = 0.0
    joining_date = None

    if violation_req_id:
        df_viol = pd.read_sql(
            aq.SINGLE_VIOLATION_REQ_SQL,
            ameen_conn,
            params={"violation_req_id": int(violation_req_id)},
        )
        if not df_viol.empty:
            viol = df_viol.iloc[0]
            joining_date = viol.get("JOININGDATE") or viol.get("STARTDATE")
            # violations_count / months_active gives per-month rate
            viol_count = viol.get("VIOLATIONSCOUNT", 0) or 0
            months_active = viol.get("MONTHSACTIVE", 1) or 1
            violation_count_per_month = float(viol_count) / float(months_active)

    # 7. HRSD + Insurance identity fields
    # employee_id: the employee's ID number — try NIN first, then Iqama
    # Adjust PARAMKEY names below to match actual keys in T_TRANSACTIONTRACEPARAM.
    employee_id: str | None = (
        params.get("NIN")
        or params.get("Iqama")
        or params.get("EmployeeId")
        or str(person_id) if person_id else None
    )
    employee_id_type: str = _resolve_id_type(params)

    # unified_national_no: employer's Unified National Number from the establishment record
    unified_national_no: str | None = (
        est_record.get("UNIFIEDNATIONALID")
        or est_record.get("UNIFIDNATIONALID")
        or params.get("UnifiedNationalNo")
    )

    # Engagement dates — used by the insurance checker
    # Pulled from violation/engagement request; adjust column names to match your schema
    engagement_start_date: str | None = None
    engagement_end_date: str | None = None
    if violation_req_id:
        df_viol_dates = pd.read_sql(
            aq.SINGLE_VIOLATION_REQ_SQL,
            ameen_conn,
            params={"violation_req_id": int(violation_req_id)},
        )
        if not df_viol_dates.empty:
            vr = df_viol_dates.iloc[0]
            start = vr.get("JOININGDATE") or vr.get("STARTDATE") or vr.get("ENGAGEMENTSTARTDATE")
            end = vr.get("ENDDATE") or vr.get("ENGAGEMENTENDDATE")
            if start:
                engagement_start_date = str(start)[:10]
            if end:
                engagement_end_date = str(end)[:10]

    # 8. KASHIF fraud score
    kashif_score: float | None = None
    if establishment_id:
        df_enc = load_data([int(establishment_id)], fq.EST_ENC_MAP_SQL, fraud_conn)
        if not df_enc.empty:
            enc_id = df_enc.iloc[0]["ESTABLISHMENTID_ENC"]
            df_kashif = pd.read_sql(
                fq.KASHIF_SINGLE_SQL,
                fraud_conn,
                params={"enc_establishment_id": enc_id},
            )
            if not df_kashif.empty:
                kashif_score = float(df_kashif.iloc[0]["COMPOUND_VALUE_1"])

    return {
        "transaction_trace_id": transaction_trace_id,
        "establishment_id": establishment_id,
        "person_id": person_id,
        "law_type": est_record.get("LAWTYPE"),
        "law_category": _map_law_category(est_record.get("LAWTYPE")),
        "nin_present": bool(params.get("NIN")),
        "joining_date": joining_date,
        "est_approval_rate": est_approval_rate,
        "contributor_approval_rate": contributor_approval_rate,
        "violation_count_per_month": violation_count_per_month,
        "kashif_score": kashif_score,
        "employee_id": employee_id,
        "employee_id_type": employee_id_type,
        "unified_national_no": unified_national_no,
        "engagement_start_date": engagement_start_date,
        "engagement_end_date": engagement_end_date,
        "status": txn.get("STATUS"),
    }


def _resolve_id_type(params: dict) -> str:
    """
    Infer the employee ID type from transaction parameters.
    Adjust the PARAMKEY names to match what is actually stored in T_TRANSACTIONTRACEPARAM.
    """
    if params.get("NIN"):
        return "National ID"
    if params.get("Iqama"):
        return "Iqama"
    if params.get("PassportNo"):
        return "Passport"
    if params.get("GCCId"):
        return "GCC ID"
    # Fall back to an explicit IdType param if present
    return params.get("IdType") or params.get("EmployeeIdType") or "National ID"


def _map_law_category(law_type: int | None) -> str:
    mapping = {
        1001: "GOSI_PRIVATE",   # adjust to match actual DB codes
        1002: "GOSI_GOV",
        1003: "GOSI_SEMI_GOV",
        2001: "PPA",
    }
    return mapping.get(law_type, "UNKNOWN") if law_type else "UNKNOWN"
