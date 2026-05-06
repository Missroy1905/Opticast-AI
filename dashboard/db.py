"""
OptiCast AI — db.py
====================
Database connection pool and query helpers for the Streamlit dashboard.

All functions return DataFrames that exactly match the shape expected
by the dashboard's mock generators, so the UI never needs to change.

Fallback contract:
  Every public function catches ALL exceptions and returns None on failure.
  The dashboard loader functions detect None and silently use mock data.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import streamlit as st

log = logging.getLogger("opticast.db")

# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION POOL
# Using @st.cache_resource so a single psycopg2 connection is shared across
# all Streamlit reruns in the same session (not re-created every 15 s refresh).
# ─────────────────────────────────────────────────────────────────────────────

def _dsn() -> dict:
    """Read connection params from environment / .env file."""
    return dict(
        host     = os.getenv("DB_HOST",     "localhost"),
        port     = int(os.getenv("DB_PORT", "5432")),
        dbname   = os.getenv("DB_NAME",     "opticast_db"),
        user     = os.getenv("DB_USER",     "opticast"),
        password = os.getenv("DB_PASSWORD", "opticast_secret"),
        connect_timeout = 3,          # fail fast — don't stall the UI
        application_name = "opticast_dashboard",
    )


@st.cache_resource(show_spinner=False)
def _get_pool():
    """
    Returns a psycopg2 connection, or None if the DB is unreachable.
    Cached at the Streamlit resource level — created once per server process.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(**_dsn())
        conn.autocommit = True          # read-only dashboard, no transactions needed
        log.info("✅ TimescaleDB connection established.")
        return conn
    except Exception as exc:
        log.warning("⚠️  DB unavailable — dashboard will use mock data. (%s)", exc)
        return None


def _query(sql: str, params: tuple = ()) -> Optional[pd.DataFrame]:
    """
    Execute a SELECT and return a DataFrame, or None on any error.
    Automatically reconnects if the cached connection has gone stale.
    """
    conn = _get_pool()
    if conn is None:
        return None
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception as exc:
        log.warning("DB query failed: %s", exc)
        # Invalidate the cached connection so next call reconnects
        _get_pool.clear()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC QUERY FUNCTIONS
# Each function signature mirrors what the mock generator returns.
# ─────────────────────────────────────────────────────────────────────────────

def load_forecast_df(date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """
    Returns forecast ribbon data for `date` (defaults to today UTC).

    Returned DataFrame columns:
        timestamp : datetime64[ns, UTC]  — 15-min intervals, 96 rows
        P10       : float64              — MW
        P50       : float64              — MW
        P90       : float64              — MW

    Returns None if DB is unreachable.
    """
    if date is None:
        date = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    sql = """
        SELECT
            time                        AS timestamp,
            COALESCE(p10_mw, 0)         AS "P10",
            COALESCE(p50_mw, 0)         AS "P50",
            COALESCE(p90_mw, 0)         AS "P90"
        FROM forecasts
        WHERE
            plant_id = 'PAVAGADA_SOLAR'
            AND time >= %s
            AND time <  %s + INTERVAL '1 day'
        ORDER BY time
        LIMIT 200
    """
    df = _query(sql, (date, date))
    if df is None or df.empty:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ("P10", "P50", "P90"):
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)
    return df


def load_qca_df(date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """
    Returns the 96-block KERC QCA schedule for `date`.

    Returned DataFrame columns (match mock generator exactly):
        Block           : int
        Time Slot       : str   "HH:MM"
        Declared (MW)   : float
        Actual (MW)     : float
        Deviation (MW)  : float
        DSM Charge (₹L) : float  (rupees lakh)
        Status          : str

    Returns None if DB is unreachable.
    """
    if date is None:
        date = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    sql = """
        SELECT
            block_number                                    AS block,
            time,
            declared_mw,
            actual_mw,
            deviation_mw,
            dsm_charge_rs,
            deviation_pct,
            is_within_band
        FROM qca_compliance
        WHERE
            plant_id = 'PAVAGADA_SOLAR'
            AND time >= %s
            AND time <  %s + INTERVAL '1 day'
        ORDER BY block_number
        LIMIT 96
    """
    df = _query(sql, (date, date))
    if df is None or df.empty:
        return None

    # Reformat to match the dashboard's expected column names exactly
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["Time Slot"] = df["time"].dt.tz_convert("Asia/Kolkata").dt.strftime("%H:%M")

    def _status(row):
        t_hour = row["time"].tz_convert("Asia/Kolkata").hour + \
                 row["time"].tz_convert("Asia/Kolkata").minute / 60
        if 13.5 <= t_hour <= 14.5:
            return "⚠️ Ramp"
        if row["is_within_band"]:
            return "✅ Normal"
        return "🔴 Deviated"

    df["Status"] = df.apply(_status, axis=1)

    # DSM charge: convert Rs → Rs Lakh (1 L = 100,000)
    df["DSM Charge (₹L)"] = (df["dsm_charge_rs"] / 1e5).round(3)

    return df.rename(columns={
        "block":        "Block",
        "declared_mw":  "Declared (MW)",
        "actual_mw":    "Actual (MW)",
        "deviation_mw": "Deviation (MW)",
    })[["Block", "Time Slot", "Declared (MW)", "Actual (MW)",
        "Deviation (MW)", "DSM Charge (₹L)", "Status"]]


def load_kpi_metrics(date: Optional[datetime] = None) -> Optional[dict]:
    """
    Returns live KPI values for the top card row.

    Returns dict with keys:
        dsm_penalty_lakh   : float   — running DSM penalty today (₹ lakh)
        active_alerts      : int     — alert count in last 2 hours
        capacity_factor_pct: float   — today's avg CF % for Pavagada
        non_compliant_blocks: int    — blocks with |deviation| > 15 %
        compliance_score_pct: float  — (compliant / total) * 100
        total_declared_mwh : float
        total_actual_mwh   : float
        net_deviation_mwh  : float

    Returns None if DB is unreachable.
    """
    if date is None:
        date = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    sql = """
        SELECT
            COALESCE(SUM(dsm_charge_rs) / 1e5,      0)   AS dsm_penalty_lakh,
            COALESCE(COUNT(*),                        0)   AS total_blocks,
            COALESCE(SUM(CASE WHEN NOT is_within_band THEN 1 ELSE 0 END), 0)
                                                          AS non_compliant_blocks,
            COALESCE(SUM(declared_mw) * 0.25,        0)   AS total_declared_mwh,
            COALESCE(SUM(actual_mw)   * 0.25,        0)   AS total_actual_mwh,
            COALESCE(SUM(actual_mw - declared_mw) * 0.25, 0) AS net_deviation_mwh,
            COALESCE(AVG(actual_mw) / 2050.0 * 100,  0)   AS capacity_factor_pct
        FROM qca_compliance
        WHERE
            plant_id = 'PAVAGADA_SOLAR'
            AND time >= %s
            AND time <  %s + INTERVAL '1 day'
    """
    df = _query(sql, (date, date))
    if df is None or df.empty:
        return None

    row = df.iloc[0]
    total    = int(row["total_blocks"])
    non_comp = int(row["non_compliant_blocks"])
    comp_pct = round((total - non_comp) / total * 100, 1) if total > 0 else 0.0

    # Active alerts (last 2 hours)
    alert_sql = """
        SELECT COUNT(*) AS cnt
        FROM alerts
        WHERE
            plant_id  = 'PAVAGADA_SOLAR'
            AND time >= NOW() - INTERVAL '2 hours'
            AND resolved_at IS NULL
    """
    alert_df = _query(alert_sql)
    active_alerts = int(alert_df.iloc[0]["cnt"]) if alert_df is not None else 0

    return {
        "dsm_penalty_lakh":    round(float(row["dsm_penalty_lakh"]), 2),
        "active_alerts":       active_alerts,
        "capacity_factor_pct": round(float(row["capacity_factor_pct"]), 1),
        "non_compliant_blocks":non_comp,
        "compliance_score_pct":comp_pct,
        "total_declared_mwh":  round(float(row["total_declared_mwh"]), 0),
        "total_actual_mwh":    round(float(row["total_actual_mwh"]), 0),
        "net_deviation_mwh":   round(float(row["net_deviation_mwh"]), 1),
    }


def db_is_live() -> bool:
    """Quick health check — True if DB is reachable."""
    result = _query("SELECT 1 AS ok")
    return result is not None and not result.empty
