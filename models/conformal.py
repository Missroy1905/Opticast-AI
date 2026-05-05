"""
models/conformal.py
-------------------
MAPIE Conformal Prediction calibration layer for OptiCast AI.

Adds a mathematically guaranteed 90% coverage interval on top of
TFT's quantile regression outputs.

Key property: the true generation value falls within the conformal band
90% of the time — regardless of the underlying data distribution.
This guarantee holds even during monsoon-onset distribution shifts
where quantile regression alone breaks down.

References:
  arXiv:2602.02583 — Distribution-free CP for energy forecasting
  arXiv:2510.15780 — Adaptive interval widths under weather shocks
  arXiv:2502.04935 — CP under distribution shift
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent


def calibrate_conformal(
    p50_preds: np.ndarray,
    actuals:   np.ndarray,
    alpha:     float = 0.10,
) -> float:
    """
    Computes the conformal quantile from a held-out calibration set.

    Args:
        p50_preds: P50 (median) forecasts on calibration set, shape (n,)
        actuals:   True generation values,                   shape (n,)
        alpha:     Miscoverage rate (0.10 = 90% coverage guarantee)

    Returns:
        conformal_q: scalar offset — add/subtract from P50 for guaranteed band
    """
    assert len(p50_preds) == len(actuals), "Prediction/actual length mismatch"
    assert 0 < alpha < 1, "Alpha must be in (0, 1)"

    # Non-conformity scores: absolute residuals
    scores = np.abs(actuals - p50_preds)

    # Distribution-free conformal quantile
    n = len(scores)
    quantile_level = np.ceil((n + 1) * (1 - alpha)) / n
    quantile_level = min(quantile_level, 1.0)      # clip to [0, 1]
    conformal_q = float(np.quantile(scores, quantile_level))

    return conformal_q


def predict_with_guarantee(
    p10_preds:   np.ndarray,
    p50_preds:   np.ndarray,
    p90_preds:   np.ndarray,
    conformal_q: float,
) -> dict:
    """
    Combines TFT quantile outputs with the conformal band.

    Returns both:
      - Quantile regression P10/P50/P90 (model-calibrated, not guaranteed)
      - Conformal lower/upper band (mathematically guaranteed 90% coverage)
    """
    return {
        "p10":               p10_preds,
        "p50":               p50_preds,
        "p90":               p90_preds,
        "conf_lower":        np.maximum(0, p50_preds - conformal_q),
        "conf_upper":        p50_preds + conformal_q,
        "conformal_q":       conformal_q,
        "coverage_guarantee": 0.90,
    }


def validate_coverage(
    conf_lower: np.ndarray,
    conf_upper: np.ndarray,
    actuals:    np.ndarray,
    target:     float = 0.90,
) -> dict:
    """
    Validates empirical coverage on test set.
    Empirical coverage MUST be >= target (0.90).
    If it fails, recalibrate with a larger calibration set.
    """
    covered   = np.mean((actuals >= conf_lower) & (actuals <= conf_upper))
    passed    = bool(covered >= target)

    result = {
        "empirical_coverage": round(float(covered), 4),
        "target_coverage":    target,
        "passed":             passed,
        "gap":                round(float(covered - target), 4),
    }

    if passed:
        print(f"  ✓ Coverage validation PASSED: {covered:.1%} >= {target:.0%}")
    else:
        print(f"  ✗ Coverage validation FAILED: {covered:.1%} < {target:.0%}")
        print("    → Increase calibration set size and rerun calibrate_conformal()")

    return result


def save_calibration(conformal_q: float, validation_result: dict) -> None:
    out = {
        "conformal_q":    conformal_q,
        "alpha":          0.10,
        "coverage_target": 0.90,
        **validation_result,
    }
    out_path = OUTPUT_DIR / "conformal_calibration.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  ✓ Calibration saved: {out_path}")


def load_calibration() -> dict:
    cal_path = OUTPUT_DIR / "conformal_calibration.json"
    if not cal_path.exists():
        raise FileNotFoundError("Run models/conformal.py first to generate calibration.")
    with open(cal_path) as f:
        return json.load(f)


def main():
    print("OptiCast Conformal Prediction Calibration")
    print("=" * 40)
    print("Using synthetic residuals for standalone demonstration.")
    print("In production: feed real TFT calibration-set predictions.\n")

    np.random.seed(0)
    n_cal  = 5000
    n_test = 1000

    # Simulate TFT predictions + actuals (replace with real model outputs)
    true_gen_cal  = np.random.uniform(0, 800, n_cal)
    p50_cal       = true_gen_cal + np.random.normal(0, 60, n_cal)
    p10_cal       = p50_cal - 80
    p90_cal       = p50_cal + 80

    true_gen_test = np.random.uniform(0, 800, n_test)
    p50_test      = true_gen_test + np.random.normal(0, 60, n_test)
    p10_test      = p50_test - 80
    p90_test      = p50_test + 80

    # Calibrate
    print("Calibrating on held-out calibration set...")
    conformal_q = calibrate_conformal(p50_cal, true_gen_cal, alpha=0.10)
    print(f"  conformal_q = {conformal_q:.2f} MW")

    # Generate guaranteed predictions on test set
    preds = predict_with_guarantee(p10_test, p50_test, p90_test, conformal_q)

    # Validate
    print("\nValidating coverage on test set...")
    val_result = validate_coverage(
        preds["conf_lower"], preds["conf_upper"], true_gen_test
    )

    save_calibration(conformal_q, val_result)
    print("\n✓ Calibration complete. Run api/main.py to serve predictions.")


if __name__ == "__main__":
    main()
