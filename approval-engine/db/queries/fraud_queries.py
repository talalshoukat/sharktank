"""
All SQL queries for the Fraud database.
"""

# ── Establishment ID encryption mapping ──────────────────────────────────────
# Needed to join Ameen ESTABLISHMENTID with Fraud DB ESTABLISHMENTID_ENC.
EST_ENC_MAP_SQL = """
SELECT ESTABLISHMENTID,
       ESTABLISHMENTID_ENC
FROM   FRAUD_USR.T_EST_ENC_MAP
WHERE  ESTABLISHMENTID
"""

# ── KASHIF fraud indicators ───────────────────────────────────────────────────
# COMPOUND_VALUE_1 is the primary fraud risk score.
# We take the most recent record per establishment.
KASHIF_INDICATORS_SQL = """
SELECT ESTABLISHMENTID,
       CREATION_DATE,
       COMPOUND_VALUE_1
FROM   FRAUD_USR.KASHIF_INDICATORS
WHERE  ESTABLISHMENTID
"""

KASHIF_SINGLE_SQL = """
SELECT ESTABLISHMENTID,
       CREATION_DATE,
       COMPOUND_VALUE_1
FROM   FRAUD_USR.KASHIF_INDICATORS
WHERE  ESTABLISHMENTID = :enc_establishment_id
ORDER BY CREATION_DATE DESC
FETCH FIRST 1 ROW ONLY
"""
