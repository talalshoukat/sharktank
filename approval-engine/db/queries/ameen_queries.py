"""
All SQL queries for the Ameen USR database.
Parameterised with bind variables — never use string formatting on user input.
"""

# ── Transaction trace with latest step status ─────────────────────────────────
# Used to fetch a single transaction's status and core identifiers.
TRANSACTION_STATUS_SQL = """
WITH transaction AS (
    SELECT transactiontraceid,
           transactionid,
           establishmentid,
           personid
    FROM   T_TRANSACTIONTRACE
    WHERE  channel    = :channel
      AND  transactionid = :transaction_id
      AND  STATUS     = 'Completed'
      AND  CREATIONTIMESTAMP >= TO_DATE(:start_date, 'DD-MM-YYYY')
),
combined AS (
    SELECT step.STATUS,
           step.ACTIONDATE,
           trans.*
    FROM   transaction trans
    LEFT JOIN T_TRANSACTIONSTEPTRACE step
           ON trans.TRANSACTIONTRACEID = step.TRANSACTIONTRACEID
),
final_status AS (
    SELECT *
    FROM (
        SELECT t.*,
               ROW_NUMBER() OVER (
                   PARTITION BY TRANSACTIONTRACEID
                   ORDER BY ACTIONDATE DESC
               ) AS rn
        FROM combined t
    )
    WHERE rn = 1
)
SELECT *
FROM   final_status
"""

# Fetch the latest step status for a SINGLE known TRANSACTIONTRACEID.
SINGLE_TRANSACTION_STATUS_SQL = """
WITH combined AS (
    SELECT t.transactiontraceid,
           t.transactionid,
           t.establishmentid,
           t.personid,
           s.STATUS,
           s.ACTIONDATE
    FROM   T_TRANSACTIONTRACE t
    LEFT JOIN T_TRANSACTIONSTEPTRACE s
           ON t.TRANSACTIONTRACEID = s.TRANSACTIONTRACEID
    WHERE  t.TRANSACTIONTRACEID = :transaction_trace_id
)
SELECT *
FROM (
    SELECT c.*,
           ROW_NUMBER() OVER (
               PARTITION BY TRANSACTIONTRACEID
               ORDER BY ACTIONDATE DESC
           ) AS rn
    FROM combined c
)
WHERE rn = 1
"""

# ── Transaction parameters ─────────────────────────────────────────────────────
# Returns all PARAMKEY/PARAMVALUE pairs; caller pivots these.
TRACE_PARAM_SQL = """
SELECT p.PARAMKEY,
       p.PARAMVALUE,
       p.TRANSACTIONTRACEID
FROM   T_TRANSACTIONTRACEPARAM p
WHERE  p.TRANSACTIONTRACEID
"""

SINGLE_TRACE_PARAM_SQL = """
SELECT p.PARAMKEY,
       p.PARAMVALUE,
       p.TRANSACTIONTRACEID
FROM   T_TRANSACTIONTRACEPARAM p
WHERE  p.TRANSACTIONTRACEID = :transaction_trace_id
"""

# ── Violation requests ────────────────────────────────────────────────────────
VIOLATION_REQ_SQL = """
SELECT *
FROM   T_CONTRIBUTOR_VIOLATION_REQ
WHERE  CONTRIBUTORVIOLATIONREQID
"""

SINGLE_VIOLATION_REQ_SQL = """
SELECT *
FROM   T_CONTRIBUTOR_VIOLATION_REQ
WHERE  CONTRIBUTORVIOLATIONREQID = :violation_req_id
"""

# ── Establishment master data ─────────────────────────────────────────────────
ESTABLISHMENT_SQL = """
SELECT *
FROM   T_ESTABLISHMENT
WHERE  REGISTRATIONNUMBER
"""

ESTABLISHMENT_BY_ID_SQL = """
SELECT *
FROM   T_ESTABLISHMENT
WHERE  ESTABLISHMENTID = :establishment_id
"""

# ── Historical approval rates per establishment ────────────────────────────────
# For a set of establishment IDs, compute approval rate for the given year.
ESTABLISHMENT_APPROVAL_RATE_SQL = """
WITH trace AS (
    SELECT transactiontraceid,
           establishmentid,
           transactionid,
           CREATIONTIMESTAMP
    FROM   T_TRANSACTIONTRACE
    WHERE  channel         = :channel
      AND  transactionid   = :transaction_id
      AND  STATUS          = 'Completed'
      AND  EXTRACT(YEAR FROM CREATIONTIMESTAMP) = :year
),
step AS (
    SELECT s.TRANSACTIONTRACEID,
           s.STATUS AS final_status
    FROM (
        SELECT TRANSACTIONTRACEID,
               STATUS,
               ROW_NUMBER() OVER (
                   PARTITION BY TRANSACTIONTRACEID
                   ORDER BY ACTIONDATE DESC
               ) AS rn
        FROM T_TRANSACTIONSTEPTRACE
    ) s
    WHERE s.rn = 1
),
joined AS (
    SELECT t.ESTABLISHMENTID,
           s.final_status
    FROM   trace t
    JOIN   step s ON t.TRANSACTIONTRACEID = s.TRANSACTIONTRACEID
)
SELECT ESTABLISHMENTID,
       COUNT(*)                                                AS total_txn,
       SUM(CASE WHEN LOWER(final_status) = 'approved' THEN 1 ELSE 0 END) AS approved_txn,
       ROUND(
           SUM(CASE WHEN LOWER(final_status) = 'approved' THEN 1 ELSE 0 END)
           / NULLIF(COUNT(*), 0) * 100, 2
       )                                                       AS approval_pct
FROM   joined
GROUP BY ESTABLISHMENTID
"""

# ── Contributor historical approval rate ──────────────────────────────────────
CONTRIBUTOR_APPROVAL_RATE_SQL = """
WITH trace AS (
    SELECT transactiontraceid,
           personid,
           transactionid,
           CREATIONTIMESTAMP
    FROM   T_TRANSACTIONTRACE
    WHERE  channel         = :channel
      AND  transactionid   = :transaction_id
      AND  STATUS          = 'Completed'
      AND  EXTRACT(YEAR FROM CREATIONTIMESTAMP) = :year
      AND  personid = :person_id
),
step AS (
    SELECT s.TRANSACTIONTRACEID,
           s.STATUS AS final_status
    FROM (
        SELECT TRANSACTIONTRACEID,
               STATUS,
               ROW_NUMBER() OVER (
                   PARTITION BY TRANSACTIONTRACEID
                   ORDER BY ACTIONDATE DESC
               ) AS rn
        FROM T_TRANSACTIONSTEPTRACE
    ) s
    WHERE s.rn = 1
),
joined AS (
    SELECT t.PERSONID,
           s.final_status
    FROM   trace t
    JOIN   step s ON t.TRANSACTIONTRACEID = s.TRANSACTIONTRACEID
)
SELECT PERSONID,
       COUNT(*)                                                AS total_txn,
       SUM(CASE WHEN LOWER(final_status) = 'approved' THEN 1 ELSE 0 END) AS approved_txn,
       ROUND(
           SUM(CASE WHEN LOWER(final_status) = 'approved' THEN 1 ELSE 0 END)
           / NULLIF(COUNT(*), 0) * 100, 2
       )                                                       AS approval_pct
FROM   joined
GROUP BY PERSONID
"""
