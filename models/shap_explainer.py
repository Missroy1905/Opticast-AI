"""
models/shap_explainer.py
------------------------
SHAP DeepExplainer feature attribution for OptiCast AI.

Produces operator-readable explanations per 15-minute forecast block:
  "Predicted 38.2 MW. GHI: +12.4 MW; Temp derating: -2.3 MW; Cloud: -1.5 MW"

Compatible with TFT Variable Selection Network outputs.
"""

import numpy as np
from typing import List, Dict


FEATURE_NAMES = [
    "ghi",
    "direct_normal_irradiance",
    "wind_speed_100m",
    "wind_direction_10m",
    "temperature_2m",
    "cloud_cover",
    "time_of_day",
    "day_of_week",
    "month",
    "capacity_mw",
    "latitude",
    "longitude",
    "commissioning_year",
]

FEATURE_LABELS = {
    "ghi":                     "GHI (solar irradiance)",
    "direct_normal_irradiance": "Direct irradiance",
    "wind_speed_100m":         "Wind speed (100m hub)",
    "wind_direction_10m":      "Wind direction",
    "temperature_2m":          "Temperature derating",
    "cloud_cover":             "Cloud cover",
    "time_of_day":             "Time of day",
    "day_of_week":             "Day of week",
    "month":                   "Month / season",
    "capacity_mw":             "Plant capacity",
    "latitude":                "Location (lat)",
    "longitude":               "Location (lon)",
    "commissioning_year":      "Panel age / degradation",
}


def generate_explanation(
    prediction_mw:  float,
    shap_values:    np.ndarray,
    feature_names:  List[str] = None,
    min_threshold:  float = 0.5,
) -> Dict:
    """
    Converts raw SHAP values into an operator-readable explanation card.

    Args:
        prediction_mw: The model's point forecast in MW
        shap_values:   SHAP values array, shape (n_features,)
        feature_names: Optional override for feature name list
        min_threshold: Minimum |SHAP value| in MW to include in explanation

    Returns:
        dict with predicted_mw, key_drivers list, and plain_text summary
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    drivers = []
    for feat, val in zip(feature_names, shap_values):
        contribution = float(val)
        if abs(contribution) > min_threshold:
            drivers.append({
                "feature":         feat,
                "label":           FEATURE_LABELS.get(feat, feat),
                "contribution_mw": round(contribution, 1),
                "direction":       "positive" if contribution > 0 else "negative",
            })

    drivers.sort(key=lambda x: abs(x["contribution_mw"]), reverse=True)
    top3 = drivers[:3]

    # Plain text for dashboard card and demo script
    plain = f"Predicted {prediction_mw:.0f} MW. "
    parts = []
    for d in top3:
        sign = "+" if d["contribution_mw"] > 0 else ""
        parts.append(f"{d['label']}: {sign}{d['contribution_mw']:.1f} MW")
    plain += "; ".join(parts) + "."

    return {
        "predicted_mw": round(prediction_mw, 1),
        "key_drivers":  drivers,
        "top3":         top3,
        "plain_text":   plain,
        "base_value":   round(float(np.mean(shap_values)), 1),
    }


def mock_shap_explanation(plant_id: str, block_no: int) -> Dict:
    """
    Returns a plausible synthetic SHAP explanation for demo/testing.
    Mimics the explanation shown in the demo script at minute 2:30.
    """
    np.random.seed(hash(f"{plant_id}_{block_no}") % (2**31))

    if "pavagada" in plant_id or "raichur" in plant_id:
        # Solar plant profile
        prediction_mw = float(np.random.uniform(200, 600))
        shap_values = np.array([
             12.4,   # GHI — main driver
              3.1,   # DNI
              0.2,   # wind
              0.0,   # wind direction
             -2.3,   # temperature derating
             -1.5,   # cloud cover
              1.8,   # time of day
              0.4,   # day of week
              0.3,   # month
              0.0,   # capacity
              0.0,   # lat
              0.0,   # lon
             -0.6,   # panel degradation
        ])
    else:
        # Wind plant profile
        prediction_mw = float(np.random.uniform(50, 250))
        shap_values = np.array([
              0.1,   # GHI — minimal for wind
              0.0,   # DNI
             18.7,   # wind speed — dominant driver
             -3.2,   # wind direction
             -0.4,   # temperature
             -0.8,   # cloud cover
              0.2,   # time of day
              1.1,   # day of week
              2.3,   # month / season
              0.0,   # capacity
              0.0,   # lat
              0.0,   # lon
              0.0,   # degradation
        ])

    shap_values += np.random.normal(0, 0.3, size=len(shap_values))
    return generate_explanation(prediction_mw, shap_values)


def main():
    print("OptiCast SHAP Explainer — Demo Output")
    print("=" * 40)

    # Pavagada solar at block 14:30
    result = mock_shap_explanation("pavagada", 58)
    print(f"\nPavagada Solar — Block 58 (14:30)")
    print(f"  Plain text: {result['plain_text']}")
    print(f"  All drivers:")
    for d in result["key_drivers"]:
        bar = "▓" * int(abs(d["contribution_mw"]) / 2)
        sign = "+" if d["contribution_mw"] > 0 else ""
        print(f"    {d['label']:30s} {sign}{d['contribution_mw']:6.1f} MW  {bar}")

    # Chitradurga wind at block 30
    result2 = mock_shap_explanation("chitradurga", 30)
    print(f"\nChitradurga Wind — Block 30 (07:30)")
    print(f"  Plain text: {result2['plain_text']}")


if __name__ == "__main__":
    main()
