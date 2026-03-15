"""
Singleton connection pools for Ameen USR and Fraud databases.
Call init_pools() once at application startup.
Use get_ameen_conn() / get_fraud_conn() as context managers per request.
"""
from contextlib import contextmanager
from typing import Generator

import oracledb

from config.settings import settings

_ameen_pool: oracledb.ConnectionPool | None = None
_fraud_pool: oracledb.ConnectionPool | None = None


def init_pools() -> None:
    global _ameen_pool, _fraud_pool

    _ameen_pool = oracledb.create_pool(
        user=settings.ameen_usr,
        password=settings.ameen_pass,
        dsn=settings.ameen_dsn,
        min=settings.ameen_pool_min,
        max=settings.ameen_pool_max,
        increment=1,
    )

    _fraud_pool = oracledb.create_pool(
        user=settings.fraud_usr,
        password=settings.fraud_pass,
        dsn=settings.fraud_dsn,
        min=settings.fraud_pool_min,
        max=settings.fraud_pool_max,
        increment=1,
    )


def close_pools() -> None:
    if _ameen_pool:
        _ameen_pool.close()
    if _fraud_pool:
        _fraud_pool.close()


@contextmanager
def get_ameen_conn() -> Generator[oracledb.Connection, None, None]:
    if _ameen_pool is None:
        raise RuntimeError("Ameen connection pool not initialised. Call init_pools() first.")
    conn = _ameen_pool.acquire()
    try:
        yield conn
    finally:
        _ameen_pool.release(conn)


@contextmanager
def get_fraud_conn() -> Generator[oracledb.Connection, None, None]:
    if _fraud_pool is None:
        raise RuntimeError("Fraud connection pool not initialised. Call init_pools() first.")
    conn = _fraud_pool.acquire()
    try:
        yield conn
    finally:
        _fraud_pool.release(conn)


def health_check() -> dict:
    """Lightweight DB ping for the /health endpoint."""
    results = {}
    for name, pool in [("ameen_db", _ameen_pool), ("fraud_db", _fraud_pool)]:
        try:
            conn = pool.acquire()
            conn.cursor().execute("SELECT 1 FROM DUAL")
            pool.release(conn)
            results[name] = "ok"
        except Exception as exc:
            results[name] = f"error: {exc}"
    return results
