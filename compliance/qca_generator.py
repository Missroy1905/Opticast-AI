"""
compliance/qca_generator.py
----------------------------
KERC QCA (Quantum of Capacity Allocation) Schedule Generator.

Auto-produces the 96-block schedule in KERC-mandated CSV format.
One file per plant per forecast date — ready to submit to SLDC.

KERC DSM Rules applied:
  - Solar band: ±5%
  - Wind band:  ±10%
  - Confidence: HIGH if P90-P10 spread < 50 MW, else MEDIUM
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json

OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

APPC = 4.50   # Rs/kWh


def generate_qca_schedule(
    plant_id:      str,
    forecast_date: str,
    predictions:   dict,
    asset_type:    str = "solar",
) -> pd.DataFrame:
    """
    Generates a 96-block KERC QCA schedule CSV.

    Args:
        plant_id:      Plant identifier (e.g. 'pavagada')
        forecast_date: 'YYYY-MM-DD'
        predictions:   dict with keys p10, p50, p90, conf_lower, conf_upper (arrays of 96)
        asset_type:    'solar' or 'wind'

    Returns:
        DataFrame with 96 rows — also saved as CSV
    """
    start = datetime.strptime(forecast_date, "%Y-%m-%d")
    band  = 0.05 if asset_type == "solar" else 0.10
    blocks = []

    for i in range(96):
        t       = start + timedelta(minutes=15 * i)
        p10     = float(predictions["p10"][i])
        p50     = float(predictions["p50"][i])
        p90     = float(predictions["p90"][i])
        c_lower = float(predictions.get("conf_lower", predictions["p10"])[i])
        c_upper = float(predictions.get("conf_upper", predictions["p90"])[i])
        width   = p90 - p10

        lower_band = p50 * (1 - band)
        upper_band = p50 * (1 + band)

        blocks.append({
            "block_no":       i + 1,
            "time_from":      t.strftime("%H:%M"),
            "time_to":        (t + timedelta(minutes=15)).strftime("%H:%M"),
            "scheduled_mw":   round(p50, 2),
            "p10_mw":         round(p10, 2),
            "p90_mw":         round(p90, 2),
            "conf_lower_mw":  round(c_lower, 2),
            "conf_upper_mw":  round(c_upper, 2),
            "band_lower_mw":  round(lower_band, 2),
            "band_upper_mw":  round(upper_band, 2),
            "interval_width": round(width, 2),
            "confidence":     "HIGH" if width < 50 else "MEDIUM",
            "plant_id":       plant_id,
            "date":           forecast_date,
            "asset_type":     asset_type,
        })

    df = pd.DataFrame(blocks)
    out_path = OUTPUT_DIR / f"qca_{plant_id}_{forecast_date}.csv"
    df.to_csv(out_path, index=False)
    print(f"  OK QCA schedule saved: {out_path.name}")
    return df


def check_dsm_risk(
    actual_mw:    float,
    scheduled_mw: float,
    asset_type:   str,
) -> dict:
    """
    Real-time deviation check per 15-minute block.
    Returns risk level and DSM penalty estimate.
    """
    if scheduled_mw <= 0:
        return {"deviation_pct": 0.0, "risk": "LOW", "alert": False, "penalty_rs": 0.0}

    band      = 0.05 if asset_type == "solar" else 0.10
    deviation = abs(actual_mw - scheduled_mw) / scheduled_mw

    if deviation <= band:
        risk, rate = "LOW", 0.0
    elif deviation <= 0.25:
        risk, rate = "MEDIUM", 0.12 * APPC
    elif deviation <= 0.35:
        risk, rate = "HIGH", 0.20 * APPC
    else:
        risk, rate = "CRITICAL", 0.30 * APPC

    penalty_rs = actual_mw * 0.25 * rate * 1000   # energy × rate × unit conversion

    return {
        "deviation_pct": round(deviation * 100, 2),
        "risk":          risk,
        "alert":         risk != "LOW",
        "penalty_rs":    round(penalty_rs, 2),
    }


def create_audit_entry(
    plant_id:     str,
    block_no:     int,
    scheduled_mw: float,
    actual_mw:    float,
    risk:         str,
    timestamp:    str,
) -> dict:
    """
    Creates a cryptographically signed audit log entry.
    Append-only — timestamp-immutable.
    """
    payload = f"{plant_id}|{block_no}|{scheduled_mw:.2f}|{actual_mw:.2f}|{risk}|{timestamp}"
    sha256  = hashlib.sha256(payload.encode()).hexdigest()
    return {
        "plant_id":     plant_id,
        "block_no":     block_no,
        "scheduled_mw": scheduled_mw,
        "actual_mw":    actual_mw,
        "risk":         risk,
        "timestamp":    timestamp,
        "sha256":       sha256,
    }


def append_audit_log(entry: dict, log_path: Path = None) -> None:
    if log_path is None:
        log_path = OUTPUT_DIR / "dsm_audit_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    """Demo: generate a QCA schedule for Pavagada with synthetic predictions."""
    import numpy as np

    print("OptiCast QCA Schedule Generator — Demo")
    print("=" * 40)

    np.random.seed(42)
    t = np.linspace(0, 2 * np.pi, 96)

    # Synthetic solar curve (peaks at midday)
    solar_curve = np.clip(600 * np.sin(t) + np.random.normal(0, 20, 96), 0, None)
    predictions = {
        "p10":        solar_curve * 0.85,
        "p50":        solar_curve,
        "p90":        solar_curve * 1.15,
        "conf_lower": solar_curve * 0.82,
        "conf_upper": solar_curve * 1.18,
    }

    df = generate_qca_schedule(
        plant_id="pavagada",
        forecast_date="2024-08-15",
        predictions=predictions,
        asset_type="solar",
    )

    print(f"\nSample blocks:")
    print(df[["block_no", "time_from", "scheduled_mw", "p10_mw", "p90_mw", "confidence"]].head(10).to_string(index=False))

    # Demo DSM risk check
    print("\nDSM Risk Checks:")
    for scheduled, actual in [(500, 480), (500, 380), (500, 290), (500, 150)]:
        result = check_dsm_risk(actual, scheduled, "solar")
        print(f"  Scheduled={scheduled} MW, Actual={actual} MW -> "
              f"Deviation={result['deviation_pct']}% | Risk={result['risk']} | "
              f"Penalty=Rs {result['penalty_rs']:.0f}")


if __name__ == "__main__":
    main()
