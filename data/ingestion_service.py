"""
OptiCast AI — ingestion_service.py
===================================
Three services in one runner:

  Service A — NWP Ingestion
    Fetches GHI + temperature from ERA5 (Copernicus CDS API).
    Falls back to a physics-based synthetic generator if no CDS key is set.

  Service B — SCADA Simulator
    Generates realistic 15-min actual generation data for Pavagada Solar Park
    using a clear-sky model + cloud variability noise + sensor jitter.

  Service C — DSM Engine
    Computes all 96 KERC blocks for the day:
    deviation, UI rate (frequency-linked), DSM charge/credit.
    Writes results to qca_compliance table.

Usage:
  python ingestion_service.py            # run all three services
  python ingestion_service.py --service era5
  python ingestion_service.py --service scada
  python ingestion_service.py --service dsm
  python ingestion_service.py --dry-run  # print output, skip DB writes

Environment variables (or .env file):
  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
  CDS_KEY   (leave blank to use synthetic ERA5 mock)
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# Optional dependencies — graceful fallback if not installed
# ---------------------------------------------------------------------------
try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env is optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("opticast.ingestion")

# ---------------------------------------------------------------------------
# Constants — Pavagada Solar Park, Karnataka
# ---------------------------------------------------------------------------
PLANT_ID        = "PAVAGADA_SOLAR"
PLANT_LAT       = 14.10       # °N
PLANT_LON       = 77.28       # °E
PLANT_CAPACITY  = 2050.0      # MW (installed)
PANEL_EFF       = 0.185       # module efficiency (18.5 %)
AREA_M2         = PLANT_CAPACITY * 1e6 / PANEL_EFF  # effective capture area

# KERC DSM parameters (Karnataka Electricity Regulatory Commission)
BLOCKS_PER_DAY  = 96          # 15-min blocks
BLOCK_MINUTES   = 15
FREQ_NOMINAL    = 50.0        # Hz
# UI charge rate table (simplified — Rs/kWh vs frequency band)
# Source: KERC DSM Regulations 2014, Schedule I
UI_RATE_TABLE = [
    (49.02, 49.20, 7.82),
    (49.20, 49.50, 5.80),
    (49.50, 49.80, 3.54),
    (49.80, 50.00, 1.40),
    (50.00, 50.20, 0.00),   # within normal band — no UI charge
    (50.20, 50.50, -1.40),  # over-injection credit
    (50.50, 50.80, -3.54),
]
DEVIATION_BAND_PCT = 15.0    # ±15 % — free band (no penalty)


# ===========================================================================
# DATABASE HELPER
# ===========================================================================

def get_db_connection() -> psycopg2.extensions.connection:
    """Return a psycopg2 connection using env vars."""
    dsn = {
        "host":     os.getenv("DB_HOST", "localhost"),
        "port":     int(os.getenv("DB_PORT", "5432")),
        "dbname":   os.getenv("DB_NAME", "opticast_db"),
        "user":     os.getenv("DB_USER", "opticast"),
        "password": os.getenv("DB_PASSWORD", "opticast_secret"),
    }
    log.info("Connecting to TimescaleDB at %(host)s:%(port)s/%(dbname)s …", dsn)
    conn = psycopg2.connect(**dsn)
    conn.autocommit = False
    return conn


def wait_for_db(max_retries: int = 12, delay: float = 5.0) -> psycopg2.extensions.connection:
    """Retry connection until TimescaleDB is ready (useful in Docker Compose)."""
    for attempt in range(1, max_retries + 1):
        try:
            conn = get_db_connection()
            log.info("✅ Database connection established.")
            return conn
        except psycopg2.OperationalError as exc:
            log.warning("DB not ready (attempt %d/%d): %s", attempt, max_retries, exc)
            if attempt == max_retries:
                raise
            time.sleep(delay)


# ===========================================================================
# SERVICE A — NWP / ERA5 INGESTION
# ===========================================================================

class ERA5Ingestion:
    """
    Fetches GHI and 2 m temperature from Copernicus ERA5 via CDS API.
    Falls back to a deterministic synthetic generator when CDS_KEY is absent
    or cdsapi is not installed — so the pipeline works in demo mode.
    """

    def __init__(self, lat: float = PLANT_LAT, lon: float = PLANT_LON):
        self.lat = lat
        self.lon = lon
        self.cds_key = os.getenv("CDS_KEY", "").strip()
        self.use_real_api = bool(self.cds_key) and HAS_CDSAPI
        if self.use_real_api:
            log.info("ERA5 mode: LIVE (Copernicus CDS API)")
        else:
            log.info("ERA5 mode: SYNTHETIC (mock physics model — no CDS key)")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(self, date: datetime) -> list[dict]:
        """
        Return list of hourly ERA5 records for `date` (UTC).
        Each record: {time, ghi_wm2, temp_c, wind_speed_ms}
        """
        if self.use_real_api:
            return self._fetch_cds(date)
        return self._synthetic(date)

    # ------------------------------------------------------------------
    # Real CDS fetch
    # ------------------------------------------------------------------

    def _fetch_cds(self, date: datetime) -> list[dict]:
        """Download ERA5 single-level hourly data for the target date."""
        import cdsapi  # noqa: PLC0415
        client = cdsapi.Client(
            url=os.getenv("CDS_URL", "https://cds.climate.copernicus.eu/api/v2"),
            key=self.cds_key,
            quiet=True,
        )
        date_str = date.strftime("%Y-%m-%d")
        out_path = f"/app/data/era5_{date_str}.nc"
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "surface_solar_radiation_downwards",
                    "2m_temperature",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                ],
                "year":  date.strftime("%Y"),
                "month": date.strftime("%m"),
                "day":   date.strftime("%d"),
                "time":  [f"{h:02d}:00" for h in range(24)],
                "area":  [self.lat + 0.5, self.lon - 0.5,
                          self.lat - 0.5, self.lon + 0.5],
                "format": "netcdf",
            },
            out_path,
        )
        return self._parse_netcdf(out_path, date)

    def _parse_netcdf(self, path: str, date: datetime) -> list[dict]:
        """Parse downloaded NetCDF into list of dicts."""
        try:
            import netCDF4 as nc  # type: ignore
            import numpy as np
        except ImportError:
            log.warning("netCDF4/numpy not installed — falling back to synthetic data.")
            return self._synthetic(date)

        ds = nc.Dataset(path)
        times = nc.num2date(ds["time"][:], ds["time"].units)
        # ERA5 SSRD is in J/m² accumulated — convert to W/m² (mean over hour)
        ssrd  = ds["ssrd"][:, 0, 0] / 3600.0
        t2m   = ds["t2m"][:, 0, 0] - 273.15          # K → °C
        u10   = ds["u10"][:, 0, 0]
        v10   = ds["v10"][:, 0, 0]
        ws    = np.sqrt(u10**2 + v10**2)
        records = []
        for i, t in enumerate(times):
            records.append({
                "time":          datetime(t.year, t.month, t.day, t.hour,
                                          tzinfo=timezone.utc),
                "ghi_wm2":       max(0.0, float(ssrd[i])),
                "temp_c":        float(t2m[i]),
                "wind_speed_ms": float(ws[i]),
            })
        ds.close()
        return records

    # ------------------------------------------------------------------
    # Synthetic / mock ERA5 (physics-based clear-sky + noise)
    # ------------------------------------------------------------------

    def _synthetic(self, date: datetime) -> list[dict]:
        """
        Generates plausible hourly GHI for Pavagada using a simple
        clear-sky model (Bird model approximation) with day-of-year
        seasonality and random cloud perturbation.
        """
        records = []
        doy = date.timetuple().tm_yday          # day of year 1–365
        decl = 23.45 * math.sin(math.radians(360 / 365 * (doy - 81)))  # solar declination °

        for hour in range(24):
            t = date.replace(hour=hour, minute=0, second=0, microsecond=0,
                             tzinfo=timezone.utc)
            # Solar hour angle (15° per hour, noon = 12 IST = 06:30 UTC)
            solar_noon_utc = 12.0 - (PLANT_LON / 15.0)
            ha = 15.0 * (hour - solar_noon_utc)
            # Solar elevation angle
            sin_elev = (
                math.sin(math.radians(PLANT_LAT)) * math.sin(math.radians(decl))
                + math.cos(math.radians(PLANT_LAT)) * math.cos(math.radians(decl))
                * math.cos(math.radians(ha))
            )
            elev = math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))

            if elev <= 0:
                ghi = 0.0
            else:
                # Clear-sky GHI (Bird simplified) — W/m²
                cos_z = math.cos(math.radians(90 - elev))
                am    = 1 / (cos_z + 0.50572 * (96.07995 - (90 - elev)) ** -1.6364)
                ghi_clear = 1361 * cos_z * 0.7 ** (am ** 0.678)
                # Cloud factor with day-to-day reproducibility (seeded by doy+hour)
                rng = random.Random(doy * 100 + hour)
                cloud_factor = rng.uniform(0.55, 1.0)
                ghi = max(0.0, ghi_clear * cloud_factor)

            temp_c = 28 + 8 * math.sin(math.radians((hour - 6) * 15)) + random.gauss(0, 0.5)
            wind_ms = max(0.0, random.gauss(4.5, 1.2))

            records.append({
                "time":          t,
                "ghi_wm2":       round(ghi, 2),
                "temp_c":        round(temp_c, 2),
                "wind_speed_ms": round(wind_ms, 2),
            })
        return records

    # ------------------------------------------------------------------
    # Write to DB
    # ------------------------------------------------------------------

    def write_to_db(self, conn: psycopg2.extensions.connection,
                    records: list[dict]) -> int:
        """Upsert ERA5 records into forecasts table as p50 horizon-0 entries."""
        sql = """
            INSERT INTO forecasts
                (time, plant_id, model_run, horizon_hours,
                 p10_mw, p50_mw, p90_mw, ghi_wm2, temp_c, wind_speed_ms)
            VALUES
                (%(time)s, %(plant_id)s, %(model_run)s, 0,
                 %(p10)s, %(p50)s, %(p90)s, %(ghi)s, %(temp)s, %(wind)s)
            ON CONFLICT DO NOTHING
        """
        model_run = datetime.now(tz=timezone.utc)
        rows = []
        for r in records:
            # Rough capacity-factor conversion: GHI → MW (plant-level)
            cf  = min(1.0, r["ghi_wm2"] / 1000.0 * PANEL_EFF)
            p50 = round(cf * PLANT_CAPACITY, 1)
            rows.append({
                "time":      r["time"],
                "plant_id":  PLANT_ID,
                "model_run": model_run,
                "p10":       round(p50 * 0.88, 1),
                "p50":       p50,
                "p90":       round(p50 * 1.10, 1),
                "ghi":       r["ghi_wm2"],
                "temp":      r["temp_c"],
                "wind":      r["wind_speed_ms"],
            })
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, rows, page_size=200)
        conn.commit()
        log.info("ERA5: wrote %d NWP records → forecasts table.", len(rows))
        return len(rows)


# ===========================================================================
# SERVICE B — SCADA SIMULATOR
# ===========================================================================

class SCADASimulator:
    """
    Generates synthetic actual-generation time-series at 15-min resolution
    for the Pavagada Solar Park.

    Model layers (applied in order):
      1. Clear-sky output (deterministic, hour-of-day + season)
      2. Cloud variability   (AR(1) process — realistic ramp events)
      3. Temperature de-rating (panels lose ~0.4 %/°C above 25 °C)
      4. Inverter/transformer losses (~4 %)
      5. Sensor quantisation jitter (±0.5 MW)
    """

    def __init__(self):
        self._cloud_state = 0.0    # AR(1) state variable

    def generate_day(self, date: datetime, era5_records: list[dict]) -> list[dict]:
        """
        Generate 96 × 15-min actual generation records for `date`.
        `era5_records` provides hourly NWP context (temperature, GHI).
        """
        records = []
        # Build hourly lookup
        nwp = {r["time"].hour: r for r in era5_records}

        for block in range(BLOCKS_PER_DAY):
            minutes_offset = block * BLOCK_MINUTES
            t = date.replace(hour=0, minute=0, second=0, microsecond=0,
                             tzinfo=timezone.utc) + timedelta(minutes=minutes_offset)
            hour = t.hour
            nwp_hour = nwp.get(hour, {"ghi_wm2": 0, "temp_c": 30, "wind_speed_ms": 4})

            ghi  = nwp_hour["ghi_wm2"]
            temp = nwp_hour["temp_c"]

            # --- Layer 1: base output from GHI ---
            cf_base = min(1.0, ghi / 1000.0 * PANEL_EFF)

            # --- Layer 2: AR(1) cloud variability ---
            self._cloud_state = 0.7 * self._cloud_state + random.gauss(0, 0.08)
            cloud_factor = max(0.0, min(1.0, 1.0 + self._cloud_state))

            # --- Layer 3: temperature de-rating ---
            temp_factor = 1.0 - max(0.0, (temp - 25.0)) * 0.004

            # --- Layer 4: system losses ---
            system_loss = 0.96

            actual_mw = (
                PLANT_CAPACITY * cf_base * cloud_factor * temp_factor * system_loss
            )

            # --- Layer 5: sensor jitter ---
            actual_mw += random.gauss(0, 0.5)
            actual_mw  = round(max(0.0, actual_mw), 2)

            records.append({
                "time":        t,
                "plant_id":    PLANT_ID,
                "actual_mw":   actual_mw,
                "data_source": "SCADA_SIM",
                "block":       block + 1,   # 1-indexed for DSM engine
            })

        log.info("SCADA: generated %d blocks for %s.", len(records), date.date())
        return records

    def write_to_db(self, conn: psycopg2.extensions.connection,
                    records: list[dict]) -> int:
        sql = """
            INSERT INTO actuals (time, plant_id, actual_mw, data_source)
            VALUES (%(time)s, %(plant_id)s, %(actual_mw)s, %(data_source)s)
            ON CONFLICT DO NOTHING
        """
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records, page_size=200)
        conn.commit()
        log.info("SCADA: wrote %d actual records → actuals table.", len(records))
        return len(records)


# ===========================================================================
# SERVICE C — DSM ENGINE
# ===========================================================================

def _ui_rate(freq_hz: float) -> float:
    """Look up UI rate (Rs/kWh) for given grid frequency."""
    for lo, hi, rate in UI_RATE_TABLE:
        if lo <= freq_hz < hi:
            return rate
    # Outside table range
    if freq_hz < UI_RATE_TABLE[0][0]:
        return UI_RATE_TABLE[0][2]   # very low freq — max charge
    return UI_RATE_TABLE[-1][2]      # very high freq — max credit


def _simulate_grid_frequency(block: int) -> float:
    """
    Simulate realistic grid frequency variation for a block (1–96).
    Uses a sinusoidal diurnal pattern with random noise.
    Peak load → lower frequency around block 56 (14:00 IST).
    """
    base = 50.0
    diurnal = -0.15 * math.sin(math.pi * block / 96)   # nadir at midday
    noise   = random.gauss(0, 0.05)
    freq    = base + diurnal + noise
    return round(max(49.0, min(51.0, freq)), 3)


class DSMEngine:
    """
    Computes KERC DSM compliance metrics for 96 blocks.

    Deviation = Actual − Declared (positive = over-injection)
    DSM Charge = UI_Rate × |Deviation| × (15/60)    [Rs, per block]

    Reference: KERC (Terms and Conditions for Determination of Tariff)
    Regulations, 2014 — Schedule I (Unscheduled Interchange)
    """

    def compute(self,
                scada_records: list[dict],
                declared_schedule: Optional[list[float]] = None) -> list[dict]:
        """
        `declared_schedule`: list of 96 declared MW values.
        If None, uses P50 forecast-based schedule (90 % of actual as proxy).
        """
        if declared_schedule is None:
            # Proxy: declared = 90 % of actual (typical uncertainty margin)
            declared_schedule = [r["actual_mw"] * 0.90 for r in scada_records]

        results = []
        for i, record in enumerate(scada_records):
            declared  = round(declared_schedule[i], 2)
            actual    = record["actual_mw"]
            deviation = round(actual - declared, 2)
            dev_pct   = (
                round(deviation / declared * 100, 2) if declared > 0 else 0.0
            )
            within    = abs(dev_pct) <= DEVIATION_BAND_PCT

            freq      = _simulate_grid_frequency(record["block"])
            ui_rate   = _ui_rate(freq)

            # Energy imbalance per block (MWh = MW × 15min / 60)
            energy_imbalance_mwh = abs(deviation) * BLOCK_MINUTES / 60.0
            # Convert MWh → kWh for rate application
            dsm_charge = round(ui_rate * energy_imbalance_mwh * 1000, 2)
            # Negative charge = credit (over-frequency + under-injection)
            if deviation < 0 and ui_rate < 0:
                dsm_charge = -dsm_charge

            results.append({
                "time":           record["time"],
                "plant_id":       PLANT_ID,
                "block_number":   record["block"],
                "declared_mw":    declared,
                "actual_mw":      actual,
                "deviation_mw":   deviation,
                "deviation_pct":  dev_pct,
                "dsm_charge_rs":  dsm_charge,
                "frequency_hz":   freq,
                "ui_rate_rs_kwh": ui_rate,
                "is_within_band": within,
            })

        total_penalty = sum(r["dsm_charge_rs"] for r in results)
        non_compliant = sum(1 for r in results if not r["is_within_band"])
        log.info(
            "DSM: %d blocks | non-compliant: %d | total charge: Rs %.0f",
            len(results), non_compliant, total_penalty,
        )
        return results

    def write_to_db(self, conn: psycopg2.extensions.connection,
                    results: list[dict]) -> int:
        sql = """
            INSERT INTO qca_compliance
                (time, plant_id, block_number, declared_mw, actual_mw,
                 deviation_mw, deviation_pct, dsm_charge_rs,
                 frequency_hz, ui_rate_rs_kwh, is_within_band)
            VALUES
                (%(time)s, %(plant_id)s, %(block_number)s, %(declared_mw)s,
                 %(actual_mw)s, %(deviation_mw)s, %(deviation_pct)s,
                 %(dsm_charge_rs)s, %(frequency_hz)s, %(ui_rate_rs_kwh)s,
                 %(is_within_band)s)
            ON CONFLICT (plant_id, time, block_number) DO UPDATE SET
                actual_mw      = EXCLUDED.actual_mw,
                deviation_mw   = EXCLUDED.deviation_mw,
                deviation_pct  = EXCLUDED.deviation_pct,
                dsm_charge_rs  = EXCLUDED.dsm_charge_rs,
                frequency_hz   = EXCLUDED.frequency_hz,
                is_within_band = EXCLUDED.is_within_band
        """
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, results, page_size=200)
        conn.commit()
        log.info("DSM: wrote %d compliance records → qca_compliance table.", len(results))
        return len(results)


# ===========================================================================
# ORCHESTRATOR
# ===========================================================================

def run_pipeline(services: list[str], dry_run: bool = False,
                 target_date: Optional[datetime] = None) -> None:
    """
    Run selected services in dependency order.
    ERA5 → SCADA → DSM (DSM needs SCADA output).
    """
    if target_date is None:
        target_date = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    log.info("=" * 60)
    log.info("OptiCast AI — Ingestion Pipeline")
    log.info("Date: %s | Services: %s | Dry-run: %s",
             target_date.date(), services, dry_run)
    log.info("=" * 60)

    conn = None if dry_run else wait_for_db()

    # ── Service A: ERA5 ────────────────────────────────────────
    era5_records = []
    if "era5" in services or "all" in services:
        log.info("▶ Service A: ERA5 NWP Ingestion")
        era5 = ERA5Ingestion()
        era5_records = era5.fetch(target_date)
        log.info("  Fetched %d hourly NWP records.", len(era5_records))
        if not dry_run and conn:
            era5.write_to_db(conn, era5_records)
        else:
            for r in era5_records[:3]:
                log.info("  [DRY-RUN] %s", r)

    # ── Service B: SCADA ───────────────────────────────────────
    scada_records = []
    if "scada" in services or "all" in services:
        log.info("▶ Service B: SCADA Simulator")
        sim = SCADASimulator()
        scada_records = sim.generate_day(target_date, era5_records)
        if not dry_run and conn:
            sim.write_to_db(conn, scada_records)
        else:
            for r in scada_records[:3]:
                log.info("  [DRY-RUN] Block %02d — %.1f MW", r["block"], r["actual_mw"])

    # ── Service C: DSM ─────────────────────────────────────────
    if "dsm" in services or "all" in services:
        if not scada_records:
            log.warning("DSM engine requires SCADA data. "
                        "Add 'scada' to --service or run after scada.")
        else:
            log.info("▶ Service C: DSM Compliance Engine")
            dsm = DSMEngine()
            dsm_results = dsm.compute(scada_records)
            if not dry_run and conn:
                dsm.write_to_db(conn, dsm_results)
            else:
                total = sum(r["dsm_charge_rs"] for r in dsm_results)
                log.info("  [DRY-RUN] Total DSM charge for day: Rs %.0f", total)

    if conn:
        conn.close()

    log.info("=" * 60)
    log.info("✅ Pipeline complete.")
    log.info("=" * 60)


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OptiCast AI Ingestion Service")
    parser.add_argument(
        "--service",
        nargs="+",
        choices=["era5", "scada", "dsm", "all"],
        default=["all"],
        help="Which service(s) to run (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print output without writing to the database",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date YYYY-MM-DD (default: today UTC)",
    )
    args = parser.parse_args()

    target = None
    if args.date:
        target = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    try:
        run_pipeline(
            services=args.service,
            dry_run=args.dry_run,
            target_date=target,
        )
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        sys.exit(1)
