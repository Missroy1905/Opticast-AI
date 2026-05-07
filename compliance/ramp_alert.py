"""
OptiCast AI — compliance/ramp_alert.py
========================================
Phase 4C: Ramp event detection and SLDC alert system.

Implements PDF Layer 4 spec:
  • Fires when 30-minute forecast gradient exceeds 50 MW
  • Spatial timing offset between plant clusters (cloud front propagation)
  • SLDC notification with timing estimate
  • Alert severity: LOW / MEDIUM / HIGH / CRITICAL
  • Integrates with alerts DB table

PDF spec trigger:
  "Ramp Alert Detector: fires when 30-minute forecast gradient exceeds 50 MW"

Spatial propagation model:
  Cloud fronts typically cross Karnataka at 20–40 km/h.
  Pavagada → Chitradurga: ~90 km → ~2.5–4.5 hours lead time delta
  Alert includes estimated hit time per downstream cluster.

Usage:
  python compliance/ramp_alert.py                          # monitor live
  python compliance/ramp_alert.py --once                   # single check
  python compliance/ramp_alert.py --dry-run                # print, no DB
  python compliance/ramp_alert.py --simulate-cloud-front   # demo ramp event
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("opticast.ramp_alert")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Ramp detection constants ──────────────────────────────────────────────
RAMP_THRESHOLD_MW   = 50.0      # MW per 30-min — PDF spec trigger
RAMP_WINDOW_BLOCKS  = 2         # 2 × 15-min blocks = 30-min window
CRITICAL_RAMP_MW    = 200.0     # MW per 30-min — CRITICAL severity
HIGH_RAMP_MW        = 100.0     # MW per 30-min — HIGH severity

# Karnataka plant spatial registry (lat, lon, cluster)
PLANT_GEO = {
    "PAVAGADA_SOLAR":   {"lat": 14.10, "lon": 77.28, "cluster": "Tumkur"},
    "CHITRADURGA_WIND": {"lat": 14.22, "lon": 76.39, "cluster": "Chitradurga"},
    "KOPPAL_WIND":      {"lat": 15.35, "lon": 76.15, "cluster": "Koppal"},
    "GADAG_WIND":       {"lat": 15.41, "lon": 75.63, "cluster": "Gadag"},
    "RAICHUR_SOLAR":    {"lat": 16.20, "lon": 77.36, "cluster": "Raichur"},
}

# Typical Karnataka monsoon cloud front speed (km/h)
CLOUD_FRONT_SPEED_KMH = 28.0    # representative value (range: 20–40)


# ===========================================================================
# DATA CLASSES
# ===========================================================================

@dataclass
class RampEvent:
    """A detected ramp event at one plant."""
    plant_id:        str
    detected_at:     datetime
    block_start:     int
    block_end:       int
    ramp_mw:         float          # positive = ramp-up, negative = ramp-down
    ramp_type:       str            # 'RAMP_UP' | 'RAMP_DOWN'
    severity:        str            # 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
    p50_at_start:    float
    p50_at_end:      float
    trigger_source:  str            # 'FORECAST' | 'ACTUAL' | 'BOTH'
    message:         str = field(default="")
    propagation:     list = field(default_factory=list)  # downstream timing

    def __post_init__(self):
        direction = "↑" if self.ramp_mw > 0 else "↓"
        self.message = (
            f"{self.plant_id} — {self.ramp_type} {direction} "
            f"{abs(self.ramp_mw):.0f} MW over 30 min | "
            f"Blocks {self.block_start}–{self.block_end} | "
            f"Severity: {self.severity}"
        )

    def to_dict(self) -> dict:
        return {
            "plant_id":       self.plant_id,
            "detected_at":    self.detected_at.isoformat(),
            "block_start":    self.block_start,
            "block_end":      self.block_end,
            "ramp_mw":        round(self.ramp_mw, 1),
            "ramp_type":      self.ramp_type,
            "severity":       self.severity,
            "p50_at_start":   round(self.p50_at_start, 1),
            "p50_at_end":     round(self.p50_at_end, 1),
            "trigger_source": self.trigger_source,
            "message":        self.message,
            "propagation":    self.propagation,
        }


# ===========================================================================
# SPATIAL PROPAGATION MODEL
# ===========================================================================

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two lat/lon points."""
    R   = 6371.0
    phi = np.radians([lat1, lat2])
    d   = np.radians([lat2 - lat1, lon2 - lon1])
    a   = (np.sin(d[0] / 2) ** 2
           + np.cos(phi[0]) * np.cos(phi[1]) * np.sin(d[1] / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_propagation_timing(
    source_plant: str,
    ramp_time:    datetime,
    cloud_speed_kmh: float = CLOUD_FRONT_SPEED_KMH,
) -> list[dict]:
    """
    Estimate when a cloud front originating at `source_plant` will hit
    downstream plants, given propagation speed.

    Returns list of dicts sorted by estimated arrival time.
    """
    src_geo = PLANT_GEO.get(source_plant, {})
    if not src_geo:
        return []

    propagation = []
    for plant_id, geo in PLANT_GEO.items():
        if plant_id == source_plant:
            continue
        dist_km   = _haversine_km(src_geo["lat"], src_geo["lon"],
                                   geo["lat"],     geo["lon"])
        delay_hrs = dist_km / cloud_speed_kmh
        hit_time  = ramp_time + timedelta(hours=delay_hrs)

        propagation.append({
            "plant_id":        plant_id,
            "cluster":         geo["cluster"],
            "distance_km":     round(dist_km, 1),
            "delay_hours":     round(delay_hrs, 2),
            "estimated_hit":   hit_time.strftime("%H:%M IST"),
            "hit_timestamp":   hit_time.isoformat(),
        })

    return sorted(propagation, key=lambda x: x["delay_hours"])


# ===========================================================================
# RAMP DETECTOR
# ===========================================================================

class RampDetector:
    """
    Monitors forecast time-series for ramp events exceeding 50 MW / 30 min.

    Call update() each block with the latest P50 forecast.
    Fires alert when gradient threshold is crossed.
    """

    def __init__(self, plant_id: str, threshold_mw: float = RAMP_THRESHOLD_MW):
        self.plant_id     = plant_id
        self.threshold_mw = threshold_mw
        self._history: list[tuple[int, float, datetime]] = []   # (block, p50, ts)
        self.active_alerts: list[RampEvent] = []

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------

    def update(
        self,
        block_number: int,
        p50_mw:       float,
        timestamp:    datetime,
        source:       str = "FORECAST",
    ) -> Optional[RampEvent]:
        """
        Register one block's P50 forecast.
        Returns a RampEvent if threshold is crossed, else None.
        """
        self._history.append((block_number, p50_mw, timestamp))
        # Keep only the rolling window we need
        if len(self._history) > RAMP_WINDOW_BLOCKS + 1:
            self._history.pop(0)

        if len(self._history) < RAMP_WINDOW_BLOCKS + 1:
            return None

        # Compute 30-min gradient
        old_block, old_p50, old_ts = self._history[-(RAMP_WINDOW_BLOCKS + 1)]
        new_block, new_p50, new_ts = self._history[-1]
        delta_mw = new_p50 - old_p50

        if abs(delta_mw) < self.threshold_mw:
            return None

        # Severity classification
        abs_delta = abs(delta_mw)
        if abs_delta >= CRITICAL_RAMP_MW:
            severity = "CRITICAL"
        elif abs_delta >= HIGH_RAMP_MW:
            severity = "HIGH"
        elif abs_delta >= RAMP_THRESHOLD_MW:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        ramp_type = "RAMP_DOWN" if delta_mw < 0 else "RAMP_UP"

        # Spatial propagation timing
        propagation = compute_propagation_timing(self.plant_id, timestamp)

        event = RampEvent(
            plant_id      = self.plant_id,
            detected_at   = timestamp,
            block_start   = old_block,
            block_end     = new_block,
            ramp_mw       = round(delta_mw, 1),
            ramp_type     = ramp_type,
            severity      = severity,
            p50_at_start  = round(old_p50, 1),
            p50_at_end    = round(new_p50, 1),
            trigger_source= source,
            propagation   = propagation,
        )

        self.active_alerts.append(event)
        self._log_alert(event)
        return event

    # ------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------

    def _log_alert(self, event: RampEvent) -> None:
        icon = {"LOW": "🟡", "MEDIUM": "🟠",
                "HIGH": "🔴", "CRITICAL": "🚨"}.get(event.severity, "⚪")
        log.warning(
            "%s RAMP ALERT: %s | %+.0f MW over 30 min | %s",
            icon, event.plant_id, event.ramp_mw, event.severity,
        )
        if event.propagation:
            log.warning(
                "  Propagation estimate (cloud @ %.0f km/h):",
                CLOUD_FRONT_SPEED_KMH,
            )
            for p in event.propagation[:3]:
                log.warning(
                    "    → %s (%s km)  ETA: %s (+%.1f h)",
                    p["plant_id"], p["distance_km"],
                    p["estimated_hit"], p["delay_hours"],
                )


# ===========================================================================
# FLEET RAMP MONITOR
# ===========================================================================

class FleetRampMonitor:
    """
    Runs one RampDetector per plant and writes alerts to DB + local log.
    """

    ALERT_LOG = OUTPUT_DIR / "ramp_alerts.jsonl"

    def __init__(self, threshold_mw: float = RAMP_THRESHOLD_MW):
        self.detectors: dict[str, RampDetector] = {
            pid: RampDetector(pid, threshold_mw)
            for pid in PLANT_GEO
        }
        self._all_events: list[RampEvent] = []

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------

    def update(
        self,
        plant_id:     str,
        block_number: int,
        p50_mw:       float,
        timestamp:    datetime,
        source:       str = "FORECAST",
    ) -> Optional[RampEvent]:
        if plant_id not in self.detectors:
            self.detectors[plant_id] = RampDetector(plant_id)

        event = self.detectors[plant_id].update(
            block_number, p50_mw, timestamp, source
        )
        if event:
            self._all_events.append(event)
            self._write_alert_log(event)
        return event

    def write_alert_to_db(self, conn, event: RampEvent) -> None:
        """Insert alert into TimescaleDB `alerts` table."""
        sql = """
            INSERT INTO alerts
                (time, plant_id, alert_type, severity, message)
            VALUES (%s, %s, %s, %s, %s)
        """
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    event.detected_at,
                    event.plant_id,
                    event.ramp_type,
                    event.severity,
                    event.message,
                ))
            conn.commit()
        except Exception as exc:
            log.warning("DB alert write failed: %s", exc)

    def _write_alert_log(self, event: RampEvent) -> None:
        with open(self.ALERT_LOG, "a") as f:
            import json  # noqa: PLC0415
            f.write(json.dumps(event.to_dict()) + "\n")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    @property
    def active_alert_count(self) -> int:
        return len(self._all_events)

    def summary(self) -> dict:
        return {
            "total_alerts":    len(self._all_events),
            "critical":        sum(1 for e in self._all_events if e.severity == "CRITICAL"),
            "high":            sum(1 for e in self._all_events if e.severity == "HIGH"),
            "medium":          sum(1 for e in self._all_events if e.severity == "MEDIUM"),
            "low":             sum(1 for e in self._all_events if e.severity == "LOW"),
            "ramp_down_events":sum(1 for e in self._all_events if e.ramp_type == "RAMP_DOWN"),
            "ramp_up_events":  sum(1 for e in self._all_events if e.ramp_type == "RAMP_UP"),
            "plants_affected": list({e.plant_id for e in self._all_events}),
        }


# ===========================================================================
# SIMULATION — cloud front demo
# ===========================================================================

def simulate_cloud_front(
    source_plant:  str  = "PAVAGADA_SOLAR",
    drop_fraction: float = 0.55,    # 55% generation drop (PDF: 400 MW in 8 min)
    duration_blocks: int = 4,
) -> None:
    """
    Simulate a cloud transient at source_plant and show propagation.
    Matches PDF Section 1.3 Failure Mode 3 scenario.
    """
    log.info("=" * 60)
    log.info("CLOUD FRONT SIMULATION — source: %s", source_plant)
    log.info("Drop: %.0f%% over %d blocks (%.0f min)",
             drop_fraction * 100, duration_blocks, duration_blocks * 15)
    log.info("=" * 60)

    monitor   = FleetRampMonitor()
    now       = datetime.now(tz=timezone.utc)
    cap       = PLANT_GEO.get(source_plant, {})
    from qca_generator import PLANTS as _PLANTS  # noqa: PLC0415
    capacity  = _PLANTS.get(source_plant, {}).get("mw", 2050)

    # Pre-cloud baseline
    for block in range(48, 56):
        ts  = now + timedelta(minutes=15 * (block - 48))
        p50 = capacity * 0.38      # ~38% CF baseline
        monitor.update(source_plant, block, p50, ts)

    # Cloud transient
    log.info("\n⚡ Cloud front hits %s at block 56…\n", source_plant)
    for block in range(56, 56 + duration_blocks):
        ts  = now + timedelta(minutes=15 * (block - 48))
        # Progressive drop
        drop_progress = (block - 56) / max(duration_blocks - 1, 1)
        p50 = capacity * 0.38 * (1 - drop_fraction * drop_progress)
        event = monitor.update(source_plant, block, max(0, p50), ts,
                               source="ACTUAL")
        if event:
            log.warning("  Alert fired! Propagation timeline:")
            for p in event.propagation:
                log.warning("    %s: %s (+%.1f h)",
                            p["plant_id"], p["estimated_hit"], p["delay_hours"])

    summary = monitor.summary()
    log.info("\nSimulation summary: %d alert(s) fired — %s",
             summary["total_alerts"],
             {k: v for k, v in summary.items() if k in ("critical", "high", "medium")})


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OptiCast Ramp Alert Detector")
    parser.add_argument("--once",              action="store_true",
                        help="Run single check then exit")
    parser.add_argument("--dry-run",           action="store_true",
                        help="Demo mode — no DB writes")
    parser.add_argument("--simulate-cloud-front", action="store_true",
                        help="Simulate Pavagada cloud transient and show propagation")
    parser.add_argument("--source-plant",      type=str,
                        default="PAVAGADA_SOLAR",
                        help="Source plant for cloud front simulation")
    args = parser.parse_args()

    if args.simulate_cloud_front:
        simulate_cloud_front(source_plant=args.source_plant)
    else:
        # Live monitoring mode — integrates with deviation_tracker
        from deviation_tracker import run_tracker  # noqa: PLC0415
        log.info("Ramp alert monitor active — embedded in deviation_tracker.")
        log.info("Run: python compliance/deviation_tracker.py")
        log.info("Or:  python compliance/ramp_alert.py --simulate-cloud-front")
