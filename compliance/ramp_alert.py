"""
compliance/ramp_alert.py
------------------------
Ramp event detection engine for OptiCast AI.

Fires SLDC alerts when the 30-minute forecast gradient exceeds 50 MW.
Provides spatial timing offset between plant clusters based on
prevailing wind vector and inter-plant distances.

Phase 1: scalar threshold detection (this file)
Phase 2: ST-GNN spatial propagation oracle (see roadmap)
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

# Karnataka plant cluster coordinates (approximate centroids)
PLANT_CLUSTERS = {
    "pavagada":    {"lat": 14.10, "lon": 77.28, "type": "solar"},
    "chitradurga": {"lat": 14.22, "lon": 76.39, "type": "wind"},
    "koppal":      {"lat": 15.35, "lon": 76.15, "type": "wind"},
    "gadag":       {"lat": 15.41, "lon": 75.63, "type": "wind"},
    "raichur":     {"lat": 16.20, "lon": 77.36, "type": "solar"},
}

# Alert threshold: MW change over 30-minute window (2 × 15-min blocks)
RAMP_THRESHOLD_MW     = 50.0
RAMP_WINDOW_BLOCKS    = 2      # 30 minutes = 2 × 15-min blocks
CLOUD_SPEED_KM_PER_HR = 40.0  # Approximate cloud front propagation speed


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def detect_ramp_event(
    forecast_mw: np.ndarray,
    threshold:   float = RAMP_THRESHOLD_MW,
    window:      int   = RAMP_WINDOW_BLOCKS,
) -> List[Dict]:
    """
    Scans a 96-block forecast array for ramp events.

    Returns a list of alert dicts, one per detected event.
    """
    alerts = []
    for i in range(window, len(forecast_mw)):
        delta = forecast_mw[i] - forecast_mw[i - window]
        if abs(delta) >= threshold:
            direction = "ramp_down" if delta < 0 else "ramp_up"
            alerts.append({
                "block_start":    i - window + 1,
                "block_end":      i + 1,
                "time_window":    f"{(i - window) * 15 // 60:02d}:{(i - window) * 15 % 60:02d}"
                                  f" -> {i * 15 // 60:02d}:{i * 15 % 60:02d}",
                "delta_mw":       round(float(delta), 1),
                "direction":      direction,
                "severity":       "CRITICAL" if abs(delta) > 150 else "HIGH",
                "threshold_mw":   threshold,
            })
    return alerts


def propagation_timeline(
    origin_plant: str,
    wind_speed_kmh: float = CLOUD_SPEED_KM_PER_HR,
) -> List[Dict]:
    """
    Estimates when a cloud front originating at `origin_plant` will reach
    other clusters, based on simple distance / speed approximation.

    Phase 2 will replace this with ST-GNN directional propagation.
    """
    if origin_plant not in PLANT_CLUSTERS:
        return []

    origin = PLANT_CLUSTERS[origin_plant]
    timeline = []

    for name, cluster in PLANT_CLUSTERS.items():
        if name == origin_plant:
            continue
        dist_km  = haversine_km(origin["lat"], origin["lon"], cluster["lat"], cluster["lon"])
        delay_hr = dist_km / wind_speed_kmh
        delay_min = round(delay_hr * 60, 0)
        timeline.append({
            "plant":       name,
            "distance_km": round(dist_km, 1),
            "eta_minutes": int(delay_min),
            "eta_label":   f"T+{int(delay_min)} min",
            "asset_type":  cluster["type"],
        })

    timeline.sort(key=lambda x: x["eta_minutes"])
    return timeline


def format_alert_message(
    plant_id: str,
    alert:    Dict,
    timeline: List[Dict],
) -> str:
    """
    Formats an SLDC-ready alert message for the dashboard panel.
    """
    sign   = "-" if alert["delta_mw"] < 0 else "+"
    lines  = [
        f"WARN RAMP ALERT | {plant_id.upper()} | {alert['severity']}",
        f"  {alert['direction'].replace('_', ' ').title()}: "
        f"{sign}{abs(alert['delta_mw']):.0f} MW over {RAMP_WINDOW_BLOCKS * 15} min",
        f"  Window: {alert['time_window']}",
        "",
        "  Propagation forecast (Phase 1 estimate):",
    ]
    for t in timeline[:3]:
        lines.append(f"    {t['plant']:15s} ETA {t['eta_label']}  ({t['distance_km']} km)")

    lines.append("")
    lines.append("  -> Pre-position reactive power support NOW")
    return "\n".join(lines)


def main():
    print("OptiCast Ramp Alert Detector — Demo")
    print("=" * 40)

    np.random.seed(7)
    t = np.linspace(0, 2 * np.pi, 96)
    solar_forecast = np.clip(600 * np.sin(t) + np.random.normal(0, 10, 96), 0, 700)

    # Inject a cloud transient at block 40 (10:00 AM)
    solar_forecast[40:48] *= 0.35

    print(f"\nScanning 96-block forecast for ramps > {RAMP_THRESHOLD_MW} MW / 30 min...")
    alerts = detect_ramp_event(solar_forecast)
    print(f"  Detected {len(alerts)} ramp event(s)")

    for i, alert in enumerate(alerts):
        print(f"\nAlert {i+1}: {alert['direction']} | {alert['delta_mw']:+.0f} MW | "
              f"Block {alert['block_start']}–{alert['block_end']}")
        timeline = propagation_timeline("pavagada")
        msg = format_alert_message("pavagada", alert, timeline)
        print(msg)


if __name__ == "__main__":
    main()
