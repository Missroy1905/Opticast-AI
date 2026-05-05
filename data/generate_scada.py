"""
data/generate_scada.py
----------------------
Generates synthetic SCADA generation profiles from ERA5 weather data
with physically-grounded solar and wind models.

Injects 15 cloud transient anomaly events for ramp alert validation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

INPUT_DIR  = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent

np.random.seed(42)


# ─── Physics models ───────────────────────────────────────────────────────────

def solar_output(ghi: float, temp: float, capacity_mw: float, year: int = 7) -> float:
    """
    GHI-to-Power conversion with:
      - Temperature derating: -0.4%/°C above 25°C
      - Soiling loss: 3%
      - Panel degradation: 0.7%/year
    """
    efficiency   = 0.195
    temp_coeff   = -0.004        # per °C above 25
    soiling      = 0.97
    degrade      = (1 - 0.007) ** year
    temp_factor  = 1 + temp_coeff * (temp - 25)
    base = (ghi / 1000) * capacity_mw * efficiency * temp_factor * soiling * degrade
    # 2% Gaussian noise to simulate measurement variability
    return float(max(0.0, base * np.random.normal(1.0, 0.02)))


def wind_output(wind_speed: float, capacity_mw: float) -> float:
    """
    Cubic power curve with cut-in / rated / cut-out bands.
    Jensen wake-effect reduction applied at cluster level elsewhere.
    """
    cut_in, rated, cut_out = 3.0, 12.0, 25.0
    if wind_speed < cut_in or wind_speed > cut_out:
        return 0.0
    if wind_speed >= rated:
        return float(capacity_mw)
    return float(capacity_mw * ((wind_speed - cut_in) / (rated - cut_in)) ** 3)


# ─── Anomaly injection ────────────────────────────────────────────────────────

def inject_cloud_transient(
    df: pd.DataFrame,
    idx: int,
    drop_pct: float = 0.60,
    duration: int = 8,
) -> pd.DataFrame:
    """
    Injects a cloud-shadow transient at row `idx`.
    Simulates 400 MW-class drop over `duration` 15-min blocks (~2 hours).
    """
    end_idx = min(idx + duration, len(df) - 1)
    df.loc[idx:end_idx, "generation_mw"] = (
        df.loc[idx:end_idx, "generation_mw"] * (1 - drop_pct)
    )
    df.loc[idx:end_idx, "anomaly"] = True
    return df


# ─── Main generation ──────────────────────────────────────────────────────────

def generate_for_plant(name: str, era5_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(era5_path)

    records = []
    for _, row in df.iterrows():
        asset_type   = row["asset_type"]
        capacity_mw  = row["capacity_mw"]

        if asset_type == "solar":
            ghi  = row.get("shortwave_radiation", 0) or 0
            temp = row.get("temperature_2m", 25) or 25
            gen  = solar_output(ghi, temp, capacity_mw)
        else:
            ws   = row.get("wind_speed_100m", 0) or 0
            gen  = wind_output(ws, capacity_mw)

        records.append({
            "time":          row["time"],
            "plant_id":      name,
            "asset_type":    asset_type,
            "capacity_mw":   capacity_mw,
            "generation_mw": gen,
            "anomaly":       False,
        })

    result = pd.DataFrame(records)

    # Inject 15 anomaly events spread across the 2-year window
    n_rows   = len(result)
    spacing  = n_rows // 16
    for i in range(1, 16):
        idx        = spacing * i
        drop_pct   = np.random.uniform(0.40, 0.70)
        duration   = np.random.randint(4, 12)
        result     = inject_cloud_transient(result, idx, drop_pct, duration)

    print(f"  {name}: {len(result)} rows, {result['anomaly'].sum()} anomaly blocks injected")
    return result


def main():
    print("OptiCast Synthetic SCADA Generator")
    print("=" * 40)

    all_frames = []
    for name in ["pavagada", "chitradurga", "koppal", "gadag", "raichur"]:
        era5_path = INPUT_DIR / f"era5_{name}.parquet"
        if not era5_path.exists():
            print(f"  ⚠ ERA5 file not found for {name} — run fetch_era5.py first")
            continue
        df = generate_for_plant(name, era5_path)
        out_path = OUTPUT_DIR / f"scada_{name}.parquet"
        df.to_parquet(out_path, index=False)
        all_frames.append(df)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = OUTPUT_DIR / "scada_all_plants.parquet"
        combined.to_parquet(combined_path, index=False)
        print(f"\n✓ Combined SCADA: {len(combined)} rows")
        print(f"✓ Total anomaly blocks: {combined['anomaly'].sum()}")
        print(f"✓ Saved to: {combined_path}")


if __name__ == "__main__":
    main()
