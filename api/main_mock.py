"""
OptiCast AI — Mock FastAPI Backend
===================================
All endpoints return hardcoded, realistic simulated data.
No ML models are imported or called.

Story baked in:
  • 50 MW ramp drop detected at Pavagada Solar Park at 14:00 IST
  • DSM penalty warning triggered (UI-0.5 deviation band breached)
  • SHAP explainability points to cloud-cover & wind-speed features
  • QCA compliance score dips to 0.71 during the ramp window

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import random
from datetime import datetime, date, timedelta
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OptiCast AI — Mock API",
    description=(
        "Renewable Energy Forecasting & KERC DSM Compliance "
        "— Simulated backend for dashboard development"
    ),
    version="0.1.0-mock",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # relax for dashboard dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

TODAY = date.today().isoformat()
NOW_ISO = datetime.now().isoformat()

HOURS = [f"{h:02d}:00" for h in range(24)]


def _jitter(base: float, pct: float = 0.05) -> float:
    """Return base value ± pct noise."""
    noise = base * pct
    return round(base + random.uniform(-noise, noise), 2)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health_check() -> dict[str, Any]:
    """Simple liveness probe."""
    return {
        "status": "ok",
        "mode": "mock",
        "timestamp": NOW_ISO,
        "service": "OptiCast AI Mock Backend v0.1.0",
    }


# ---------------------------------------------------------------------------
# /forecast/day-ahead
# ---------------------------------------------------------------------------

# Realistic solar generation curve for a 200 MW plant in Karnataka.
# The 14:00 slot shows the 50 MW ramp drop event.
_BASE_SOLAR = [
    0, 0, 0, 0, 0, 0,
    12, 45, 90, 130, 165, 185,
    195, 198, 148,  # ← 14:00 ramp drop (198 → 148 MW, Δ = -50 MW)
    165, 155, 130, 95, 60, 22,
    5, 0,
]

_ACTUAL_SOLAR = [
    0, 0, 0, 0, 0, 0,
    10, 43, 88, 128, 162, 183,
    193, 197, 147,
    163, 153, 128, 93, 58, 20,
    4, 0, 0,
]


@app.get("/forecast/day-ahead", tags=["forecast"])
def day_ahead_forecast(
    plant: str = Query("Pavagada", description="Plant identifier"),
    date_str: str = Query(TODAY, alias="date"),
) -> dict[str, Any]:
    """
    24-hour day-ahead solar generation forecast vs actuals.
    Includes confidence intervals and the 14:00 ramp-drop event marker.
    """
    slots = []
    for i, hour in enumerate(HOURS):
        forecast_mw = _jitter(_BASE_SOLAR[i], 0.03)
        actual_mw = _jitter(_ACTUAL_SOLAR[i], 0.02)
        error_pct = round((actual_mw - forecast_mw) / max(forecast_mw, 1) * 100, 2)
        slots.append(
            {
                "hour": hour,
                "forecast_mw": forecast_mw,
                "actual_mw": actual_mw,
                "lower_ci_mw": round(forecast_mw * 0.92, 2),
                "upper_ci_mw": round(forecast_mw * 1.08, 2),
                "error_pct": error_pct,
                "ramp_event": hour == "14:00",
            }
        )

    return {
        "plant": plant,
        "date": date_str,
        "capacity_mw": 200,
        "model": "TFT-v2 (mock)",
        "rmse_mw": 6.4,
        "mae_mw": 4.1,
        "forecast_slots": slots,
        "summary": {
            "peak_forecast_mw": 198,
            "peak_actual_mw": 197,
            "total_forecast_kwh": round(sum(_BASE_SOLAR) * 1000, 0),
            "total_actual_kwh": round(sum(_ACTUAL_SOLAR) * 1000, 0),
            "ramp_event_detected": True,
            "ramp_event_time": "14:00",
            "ramp_magnitude_mw": -50,
        },
    }


# ---------------------------------------------------------------------------
# /forecast/wind
# ---------------------------------------------------------------------------

_BASE_WIND = [
    42, 44, 46, 50, 53, 55,
    58, 61, 65, 68, 72, 74,
    75, 73, 70, 65, 60, 55,
    50, 48, 46, 44, 43, 42,
]


@app.get("/forecast/wind", tags=["forecast"])
def wind_forecast(
    plant: str = Query("Chitradurga", description="Wind plant identifier"),
    date_str: str = Query(TODAY, alias="date"),
) -> dict[str, Any]:
    """24-hour wind power forecast for Karnataka wind farms."""
    slots = [
        {
            "hour": hour,
            "forecast_mw": _jitter(_BASE_WIND[i], 0.04),
            "wind_speed_ms": round(_jitter(8.5 + i * 0.1, 0.1), 2),
            "capacity_factor": round(_BASE_WIND[i] / 100, 3),
        }
        for i, hour in enumerate(HOURS)
    ]
    return {
        "plant": plant,
        "date": date_str,
        "capacity_mw": 100,
        "model": "TFT-v2-wind (mock)",
        "forecast_slots": slots,
    }


# ---------------------------------------------------------------------------
# /compliance/qca
# ---------------------------------------------------------------------------

# QCA = % of 15-min scheduling blocks where injection ≈ schedule ± 12%
# Score dips during the 14:00 ramp window (blocks 56–59 ≈ hours 14:00-14:45)
_QCA_SCORES_BY_HOUR = [
    0.95, 0.96, 0.97, 0.97, 0.96, 0.95,
    0.94, 0.93, 0.94, 0.95, 0.96, 0.97,
    0.97, 0.96, 0.71,   # ← 14:00 ramp drop causes compliance dip
    0.85, 0.92, 0.94, 0.95, 0.95, 0.96,
    0.96, 0.97, 0.97,
]


@app.get("/compliance/qca", tags=["compliance"])
def qca_compliance(
    plant: str = Query("Pavagada"),
    date_str: str = Query(TODAY, alias="date"),
) -> dict[str, Any]:
    """
    KERC QCA (Qualifying Criteria of Availability) compliance report.
    Score drops to 0.71 at 14:00 due to the ramp event.
    """
    hourly = [
        {
            "hour": hour,
            "qca_score": round(_jitter(_QCA_SCORES_BY_HOUR[i], 0.01), 3),
            "in_band": _QCA_SCORES_BY_HOUR[i] >= 0.80,
            "injection_mw": _jitter(_ACTUAL_SOLAR[i], 0.02),
            "schedule_mw": _jitter(_BASE_SOLAR[i], 0.02),
            "deviation_pct": round(
                (_ACTUAL_SOLAR[i] - _BASE_SOLAR[i]) / max(_BASE_SOLAR[i], 1) * 100, 2
            ),
        }
        for i, hour in enumerate(HOURS)
    ]

    daily_qca = round(sum(_QCA_SCORES_BY_HOUR) / len(_QCA_SCORES_BY_HOUR), 3)
    breach_hours = [h["hour"] for h in hourly if not h["in_band"]]

    return {
        "plant": plant,
        "date": date_str,
        "daily_qca_score": daily_qca,
        "kerc_threshold": 0.80,
        "compliant": daily_qca >= 0.80,
        "breach_hours": breach_hours,
        "hourly": hourly,
        "summary": {
            "total_blocks": 96,
            "non_compliant_blocks": 4,
            "worst_hour": "14:00",
            "worst_qca": 0.71,
            "avg_deviation_pct": 3.2,
        },
    }


# ---------------------------------------------------------------------------
# /compliance/dsm
# ---------------------------------------------------------------------------

@app.get("/compliance/dsm", tags=["compliance"])
def dsm_compliance(
    plant: str = Query("Pavagada"),
    date_str: str = Query(TODAY, alias="date"),
) -> dict[str, Any]:
    """
    KERC DSM (Deviation Settlement Mechanism) charges/credits summary.
    The 14:00 ramp event triggers a penalty in the UI-0.5 deviation band.
    """
    return {
        "plant": plant,
        "date": date_str,
        "grid_frequency_hz": 50.02,
        "deviation_bands": [
            {
                "band": "UI+2",
                "description": "Excess injection > +12%",
                "blocks_triggered": 0,
                "charge_inr": 0,
            },
            {
                "band": "UI+1",
                "description": "Excess injection +6% to +12%",
                "blocks_triggered": 2,
                "charge_inr": 4200,
            },
            {
                "band": "Compliant",
                "description": "Within ±6%",
                "blocks_triggered": 89,
                "charge_inr": 0,
            },
            {
                "band": "UI-1",
                "description": "Under injection -6% to -12%",
                "blocks_triggered": 1,
                "charge_inr": 3100,
            },
            {
                "band": "UI-0.5",
                "description": "Under injection -12% to -25% (RAMP EVENT)",
                "blocks_triggered": 4,
                "charge_inr": 87500,   # ← primary penalty from ramp
                "ramp_triggered": True,
            },
        ],
        "summary": {
            "total_penalty_inr": 94800,
            "total_credit_inr": 0,
            "net_dsm_inr": -94800,
            "ramp_event_contribution_inr": -87500,
            "ramp_event_time": "14:00",
            "currency": "INR",
        },
        "recommendation": (
            "Deploy 10 MWh BESS buffer at Pavagada to absorb ramp transients "
            "and reduce DSM exposure by ~92%."
        ),
    }


# ---------------------------------------------------------------------------
# /alerts/ramps
# ---------------------------------------------------------------------------

@app.get("/alerts/ramps", tags=["alerts"])
def ramp_alerts(
    plant: str = Query("Pavagada"),
    date_str: str = Query(TODAY, alias="date"),
) -> dict[str, Any]:
    """
    Active and historical ramp alerts for the specified plant.
    The 14:00 event is the primary active alert.
    """
    return {
        "plant": plant,
        "date": date_str,
        "active_alerts": [
            {
                "alert_id": "RA-20260504-001",
                "severity": "CRITICAL",
                "type": "ramp_down",
                "detected_at": f"{date_str}T13:47:00+05:30",
                "event_time": f"{date_str}T14:00:00+05:30",
                "duration_minutes": 45,
                "ramp_magnitude_mw": -50,
                "ramp_rate_mw_per_min": -1.11,
                "plant": "Pavagada Solar Park — Zone B",
                "cause": "Rapid cloud-cover ingress from south-west quadrant",
                "dsm_risk_inr": 87500,
                "recommended_action": (
                    "Issue real-time revised schedule to SLDC. "
                    "Activate demand response at 13:55 IST. "
                    "Notify grid operator via WBES portal."
                ),
                "status": "acknowledged",
            }
        ],
        "historical_today": [
            {
                "alert_id": "RA-20260504-000",
                "severity": "LOW",
                "type": "ramp_up",
                "event_time": f"{date_str}T06:15:00+05:30",
                "ramp_magnitude_mw": 12,
                "status": "resolved",
            }
        ],
        "alert_stats_30d": {
            "critical": 3,
            "high": 7,
            "medium": 14,
            "low": 28,
            "total_dsm_exposure_inr": 412000,
        },
    }


# ---------------------------------------------------------------------------
# /explain/shap
# ---------------------------------------------------------------------------

@app.get("/explain/shap", tags=["explainability"])
def shap_explanation(
    plant: str = Query("Pavagada"),
    hour: str = Query("14:00", description="Hour slot to explain (HH:MM)"),
) -> dict[str, Any]:
    """
    SHAP feature-importance explanation for the forecast at the given hour.
    At 14:00 the ramp event shows cloud_cover as dominant negative driver.
    """
    is_ramp_hour = hour == "14:00"

    base_value = 195.0  # model expected value in MW

    if is_ramp_hour:
        feature_contributions = [
            {
                "feature": "cloud_cover_pct",
                "shap_value": -48.3,
                "feature_value": 78.4,
                "unit": "%",
                "direction": "negative",
                "rank": 1,
            },
            {
                "feature": "ghi_w_m2",
                "shap_value": -9.1,
                "feature_value": 310.0,
                "unit": "W/m²",
                "direction": "negative",
                "rank": 2,
            },
            {
                "feature": "wind_speed_ms",
                "shap_value": 3.2,
                "feature_value": 6.4,
                "unit": "m/s",
                "direction": "positive",
                "rank": 3,
            },
            {
                "feature": "temperature_c",
                "shap_value": -1.8,
                "feature_value": 36.2,
                "unit": "°C",
                "direction": "negative",
                "rank": 4,
            },
            {
                "feature": "hour_of_day_sin",
                "shap_value": 0.8,
                "feature_value": 0.866,
                "unit": "encoded",
                "direction": "positive",
                "rank": 5,
            },
        ]
        prediction_mw = 148.0
        narrative = (
            "The 50 MW generation drop at 14:00 is primarily explained by "
            "a sudden spike in cloud_cover (78%) which alone accounts for "
            "−48.3 MW versus the model baseline. Reduced GHI contributed a "
            "further −9.1 MW. Wind speed had a small positive offset."
        )
    else:
        feature_contributions = [
            {
                "feature": "ghi_w_m2",
                "shap_value": 42.1,
                "feature_value": 820.0,
                "unit": "W/m²",
                "direction": "positive",
                "rank": 1,
            },
            {
                "feature": "cloud_cover_pct",
                "shap_value": -5.2,
                "feature_value": 12.0,
                "unit": "%",
                "direction": "negative",
                "rank": 2,
            },
            {
                "feature": "temperature_c",
                "shap_value": -3.1,
                "feature_value": 38.5,
                "unit": "°C",
                "direction": "negative",
                "rank": 3,
            },
            {
                "feature": "wind_speed_ms",
                "shap_value": 1.9,
                "feature_value": 7.2,
                "unit": "m/s",
                "direction": "positive",
                "rank": 4,
            },
            {
                "feature": "hour_of_day_sin",
                "shap_value": 1.1,
                "feature_value": 0.5,
                "unit": "encoded",
                "direction": "positive",
                "rank": 5,
            },
        ]
        prediction_mw = _jitter(_BASE_SOLAR[int(hour.split(":")[0])], 0.02)
        narrative = (
            f"At {hour} the model is driven primarily by high GHI "
            "indicating clear-sky irradiance. Minor losses from elevated "
            "panel temperature."
        )

    return {
        "plant": plant,
        "hour": hour,
        "model": "TFT-v2 (mock)",
        "base_value_mw": base_value,
        "prediction_mw": prediction_mw,
        "feature_contributions": feature_contributions,
        "narrative": narrative,
        "ramp_event_hour": is_ramp_hour,
    }


# ---------------------------------------------------------------------------
# /metrics/financial
# ---------------------------------------------------------------------------

@app.get("/metrics/financial", tags=["metrics"])
def financial_metrics(
    plant: str = Query("Pavagada"),
    date_str: str = Query(TODAY, alias="date"),
) -> dict[str, Any]:
    """
    Financial impact summary: DSM savings vs baseline and potential upside.
    """
    return {
        "plant": plant,
        "date": date_str,
        "currency": "INR",
        "baseline_dsm_penalty_inr": 412000,
        "opticast_dsm_penalty_inr": 94800,
        "savings_today_inr": 317200,
        "savings_pct": 77.0,
        "projected_monthly_savings_inr": 9516000,
        "projected_annual_savings_inr": 114192000,
        "bess_recommendation": {
            "capacity_mwh": 10,
            "estimated_capex_inr": 45000000,
            "payback_years": 0.39,
            "additional_dsm_reduction_pct": 92,
        },
        "tariff_revenue_today_inr": 2640000,
        "net_revenue_today_inr": 2545200,
    }


# ---------------------------------------------------------------------------
# /metrics/grid
# ---------------------------------------------------------------------------

@app.get("/metrics/grid", tags=["metrics"])
def grid_metrics() -> dict[str, Any]:
    """Live (mocked) Karnataka grid snapshot."""
    return {
        "timestamp": NOW_ISO,
        "grid_frequency_hz": 50.02,
        "state": "Karnataka",
        "sldc_zone": "Southern Region",
        "renewable_share_pct": 68.4,
        "total_generation_mw": 12840,
        "renewable_generation_mw": 8783,
        "solar_generation_mw": 5210,
        "wind_generation_mw": 2890,
        "demand_mw": 13100,
        "import_export_mw": -260,   # slight deficit
        "dsm_pool_balance_cr_inr": 142.6,
    }


# ---------------------------------------------------------------------------
# /plants
# ---------------------------------------------------------------------------

@app.get("/plants", tags=["meta"])
def list_plants() -> dict[str, Any]:
    """List all plants tracked by this OptiCast deployment."""
    return {
        "plants": [
            {
                "id": "pavagada",
                "name": "Pavagada Solar Park",
                "type": "solar",
                "capacity_mw": 200,
                "state": "Karnataka",
                "district": "Tumkur",
                "active": True,
                "ramp_alert_active": True,
            },
            {
                "id": "chitradurga",
                "name": "Chitradurga Wind Farm",
                "type": "wind",
                "capacity_mw": 100,
                "state": "Karnataka",
                "district": "Chitradurga",
                "active": True,
                "ramp_alert_active": False,
            },
            {
                "id": "gadag",
                "name": "Gadag Wind Cluster",
                "type": "wind",
                "capacity_mw": 50,
                "state": "Karnataka",
                "district": "Gadag",
                "active": True,
                "ramp_alert_active": False,
            },
        ]
    }
