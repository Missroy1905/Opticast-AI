-- =============================================================
--  OptiCast AI — TimescaleDB Initialisation Script
--  Auto-executed by Docker on first container boot.
--  File: scripts/init.sql
-- =============================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================
-- TABLE 1: forecasts
--   Stores probabilistic solar/wind generation forecasts.
--   One row per (plant, timestamp, model_run).
--   Dashboard query returns P10 / P50 / P90 as JSON.
-- =============================================================

CREATE TABLE IF NOT EXISTS forecasts (
    time            TIMESTAMPTZ         NOT NULL,   -- forecast valid time (IST stored as UTC)
    plant_id        TEXT                NOT NULL,   -- e.g. 'PAVAGADA_SOLAR', 'CHITRADURGA_WIND'
    model_run       TIMESTAMPTZ         NOT NULL,   -- when the model was run
    horizon_hours   SMALLINT            NOT NULL,   -- lead time in hours (1–72)
    p10_mw          DOUBLE PRECISION    NOT NULL,   -- 10th percentile
    p50_mw          DOUBLE PRECISION    NOT NULL,   -- median (point forecast)
    p90_mw          DOUBLE PRECISION    NOT NULL,   -- 90th percentile
    ghi_wm2         DOUBLE PRECISION,               -- input GHI from ERA5 (W/m²)
    temp_c          DOUBLE PRECISION,               -- input temperature (°C)
    wind_speed_ms   DOUBLE PRECISION,               -- input wind speed (m/s)
    created_at      TIMESTAMPTZ         DEFAULT NOW()
);

-- Convert to hypertable partitioned by time (7-day chunks)
SELECT create_hypertable(
    'forecasts', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_forecasts_plant_time
    ON forecasts (plant_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_forecasts_model_run
    ON forecasts (model_run DESC);

-- =============================================================
-- TABLE 2: actuals
--   SCADA-sourced (or simulated) actual generation readings.
--   15-minute resolution → 96 blocks per day.
-- =============================================================

CREATE TABLE IF NOT EXISTS actuals (
    time        TIMESTAMPTZ         NOT NULL,
    plant_id    TEXT                NOT NULL,
    actual_mw   DOUBLE PRECISION    NOT NULL,
    data_source TEXT                DEFAULT 'SCADA_SIM',   -- 'SCADA_LIVE' or 'SCADA_SIM'
    created_at  TIMESTAMPTZ         DEFAULT NOW()
);

SELECT create_hypertable(
    'actuals', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_actuals_plant_time
    ON actuals (plant_id, time DESC);

-- =============================================================
-- TABLE 3: qca_compliance
--   96-block KERC DSM compliance metrics per day per plant.
--   Dashboard data contract:
--     {"block": int, "declared_mw": float, "actual_mw": float,
--      "deviation": float, "dsm_charge": float}
-- =============================================================

CREATE TABLE IF NOT EXISTS qca_compliance (
    time            TIMESTAMPTZ         NOT NULL,   -- block start time (UTC)
    plant_id        TEXT                NOT NULL,
    block_number    SMALLINT            NOT NULL,   -- 1–96 (KERC 15-min blocks)
    declared_mw     DOUBLE PRECISION    NOT NULL,   -- scheduled / declared generation
    actual_mw       DOUBLE PRECISION    NOT NULL,   -- metered actual
    deviation_mw    DOUBLE PRECISION    NOT NULL,   -- actual − declared
    deviation_pct   DOUBLE PRECISION    NOT NULL,   -- deviation as % of declared
    dsm_charge_rs   DOUBLE PRECISION    NOT NULL,   -- rupee charge/credit for this block
    frequency_hz    DOUBLE PRECISION,               -- grid frequency at block midpoint
    ui_rate_rs_kwh  DOUBLE PRECISION,               -- applicable UI rate (Rs/kWh)
    is_within_band  BOOLEAN             NOT NULL DEFAULT FALSE,  -- TRUE if |dev%| ≤ 15%
    created_at      TIMESTAMPTZ         DEFAULT NOW(),
    UNIQUE (plant_id, time, block_number)           -- prevent duplicate block entries
);

SELECT create_hypertable(
    'qca_compliance', 'time',
    chunk_time_interval => INTERVAL '1 day',        -- tighter chunks — daily reporting
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_qca_plant_time
    ON qca_compliance (plant_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_qca_block
    ON qca_compliance (block_number);

-- =============================================================
-- TABLE 4: alerts
--   SLDC ramp alerts and grid events for the alert panel.
-- =============================================================

CREATE TABLE IF NOT EXISTS alerts (
    time        TIMESTAMPTZ     NOT NULL,
    plant_id    TEXT            NOT NULL,
    alert_type  TEXT            NOT NULL,   -- 'RAMP_UP', 'RAMP_DOWN', 'CURTAILMENT', 'UNDER_FREQ'
    severity    TEXT            NOT NULL,   -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message     TEXT,
    resolved_at TIMESTAMPTZ,
    created_at  TIMESTAMPTZ     DEFAULT NOW()
);

SELECT create_hypertable(
    'alerts', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- =============================================================
-- VIEWS — pre-built JSON-ready queries for the dashboard API
-- =============================================================

-- View: forecast ribbon (matches dashboard data contract exactly)
CREATE OR REPLACE VIEW v_forecast_ribbon AS
SELECT
    time                            AS timestamp,
    plant_id,
    ROUND(p10_mw::NUMERIC, 2)       AS "P10",
    ROUND(p50_mw::NUMERIC, 2)       AS "P50",
    ROUND(p90_mw::NUMERIC, 2)       AS "P90",
    horizon_hours
FROM forecasts
ORDER BY plant_id, time;

-- View: QCA dashboard feed (matches dashboard data contract exactly)
CREATE OR REPLACE VIEW v_qca_dashboard AS
SELECT
    time                                AS timestamp,
    plant_id,
    block_number                        AS block,
    ROUND(declared_mw::NUMERIC, 2)      AS declared_mw,
    ROUND(actual_mw::NUMERIC, 2)        AS actual_mw,
    ROUND(deviation_mw::NUMERIC, 2)     AS deviation,
    ROUND(dsm_charge_rs::NUMERIC, 2)    AS dsm_charge,
    is_within_band
FROM qca_compliance
ORDER BY plant_id, time, block_number;

-- View: daily DSM penalty roll-up (for the Rs 191 Cr projection card)
CREATE OR REPLACE VIEW v_dsm_daily_penalty AS
SELECT
    DATE(time AT TIME ZONE 'Asia/Kolkata')  AS date_ist,
    plant_id,
    SUM(dsm_charge_rs)                      AS total_dsm_rs,
    COUNT(*)                                AS blocks_reported,
    SUM(CASE WHEN NOT is_within_band THEN 1 ELSE 0 END) AS non_compliant_blocks
FROM qca_compliance
GROUP BY 1, 2
ORDER BY 1 DESC, 2;

-- =============================================================
-- SEED DATA — one reference plant row so the dashboard
-- doesn't render empty on first boot
-- =============================================================

INSERT INTO forecasts (time, plant_id, model_run, horizon_hours, p10_mw, p50_mw, p90_mw, ghi_wm2, temp_c)
VALUES
    (NOW(), 'PAVAGADA_SOLAR', NOW(), 1, 180.5, 210.3, 235.7, 620.0, 32.5),
    (NOW() + INTERVAL '15 min', 'PAVAGADA_SOLAR', NOW(), 1, 175.0, 205.0, 230.0, 610.0, 33.0)
ON CONFLICT DO NOTHING;

-- Confirm schema creation
DO $$
BEGIN
    RAISE NOTICE '✅ OptiCast AI schema initialised successfully.';
    RAISE NOTICE '   Tables: forecasts, actuals, qca_compliance, alerts';
    RAISE NOTICE '   Views:  v_forecast_ribbon, v_qca_dashboard, v_dsm_daily_penalty';
END $$;
