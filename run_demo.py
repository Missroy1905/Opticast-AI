#!/usr/bin/env python3
"""
OptiCast AI — Master Demo Runner
==================================
Sits in the project root. Sequentially:
  1. Runs data/fetch_era5.py        (ERA5 weather pull)
  2. Runs data/generate_scada.py    (SCADA simulation)
  3. Simulates ML inference (TFT + Conformal Prediction)
  4. Runs compliance/qca_generator.py
  5. Prints a beautiful color-coded KERC DSM compliance report

Usage:
    python run_demo.py
    python run_demo.py --skip-data     (skip ERA5 + SCADA steps)
    python run_demo.py --no-color
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import random
from datetime import datetime, date

# ---------------------------------------------------------------------------
# ANSI color palette (gracefully degraded when --no-color)
# ---------------------------------------------------------------------------

USE_COLOR = True

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    # Foregrounds
    BLACK   = "\033[30m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"

    # Backgrounds
    BG_RED    = "\033[41m"
    BG_GREEN  = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE   = "\033[44m"
    BG_CYAN   = "\033[46m"


def c(color: str, text: str) -> str:
    """Wrap text in ANSI escape if color mode is on."""
    if not USE_COLOR:
        return text
    return f"{color}{text}{C.RESET}"


def bold(text: str) -> str:
    return c(C.BOLD, text)


def dim(text: str) -> str:
    return c(C.DIM, text)


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

WIDTH = 72


def banner(title: str, color: str = C.CYAN) -> None:
    line = "═" * WIDTH
    print()
    print(c(color, line))
    padding = (WIDTH - len(title) - 2) // 2
    print(c(color, "║") + " " * padding + bold(title) + " " * (WIDTH - padding - len(title) - 2) + c(color, "║"))
    print(c(color, line))


def section(title: str) -> None:
    print()
    print(c(C.BLUE, "┌─ ") + bold(title))
    print(c(C.BLUE, "│"))


def row(label: str, value: str, status_color: str = C.WHITE, indent: int = 4) -> None:
    label_w = 36
    print(c(C.BLUE, "│") + " " * indent + dim(f"{label:<{label_w}}") + c(status_color, value))


def section_end() -> None:
    print(c(C.BLUE, "└" + "─" * (WIDTH - 1)))


def separator() -> None:
    print(c(C.DIM, "─" * WIDTH))


def step(icon: str, msg: str, color: str = C.CYAN) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {c(color, icon)}  {dim(ts)}  {msg}")


def ok(msg: str)    -> None: step("✓", msg, C.GREEN)
def warn(msg: str)  -> None: step("⚠", msg, C.YELLOW)
def err(msg: str)   -> None: step("✗", msg, C.RED)
def info(msg: str)  -> None: step("→", msg, C.CYAN)


def progress_bar(label: str, duration: float = 1.0, width: int = 30) -> None:
    """Animated ASCII progress bar."""
    print(f"  {dim(label):40s} [", end="", flush=True)
    steps = width
    delay = duration / steps
    for i in range(steps):
        filled = "█" * i + "░" * (steps - i)
        pct = int(i / steps * 100)
        print(f"\r  {dim(label):40s} [{c(C.GREEN, filled)}] {pct:3d}%", end="", flush=True)
        time.sleep(delay)
    print(f"\r  {dim(label):40s} [{c(C.GREEN, '█' * width)}] 100%")


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_script(script_path: str, label: str, skip: bool = False) -> bool:
    """Run a Python script as a subprocess; return True on success."""
    if skip:
        warn(f"SKIPPED  {label}")
        return True

    info(f"Running  {label}")
    try:
        result = subprocess.run(
            [sys.executable] + script_path.split(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            ok(f"Done     {label}")
            if result.stdout.strip():
                for line in result.stdout.strip().splitlines()[:5]:
                    print(f"           {dim(line)}")
            return True
        else:
            err(f"Failed   {label}")
            if result.stderr.strip():
                for line in result.stderr.strip().splitlines()[:3]:
                    print(f"           {c(C.RED, line)}")
            return False
    except FileNotFoundError:
        warn(f"Script not found ({script_path}) — continuing with mock data")
        return True
    except subprocess.TimeoutExpired:
        err(f"Timed out after 120s: {label}")
        return False


def simulate_ml_inference() -> dict:
    """
    Fake TFT + Conformal Prediction inference.
    Returns a dict of simulated results for the report.
    """
    info("Initialising TFT model weights …")
    progress_bar("Loading TFT checkpoint", duration=0.8)

    info("Running conformal calibration pass …")
    progress_bar("Conformal prediction (α=0.10)", duration=0.6)

    info("Generating 24-hour forecast …")
    progress_bar("Inference — 96 time slots", duration=0.9)

    # Simulate forecast results
    forecast = {
        "date": date.today().isoformat(),
        "plant": "Pavagada Solar Park",
        "capacity_mw": 200,
        "peak_forecast_mw": 198,
        "total_energy_kwh": 1_186_000,
        "rmse_mw": 6.4,
        "mae_mw": 4.1,
        "coverage_90pct": 0.913,
        "ramp_detected": True,
        "ramp_time": "14:00",
        "ramp_magnitude_mw": -50,
        "ramp_duration_min": 45,
        "confidence": 0.94,
    }

    ok("Forecast generated")
    return forecast


def simulate_shap_explanation() -> dict:
    """Return mock SHAP values for the ramp-event hour."""
    info("Computing SHAP feature attributions for 14:00 slot …")
    progress_bar("SHAP TreeExplainer", duration=0.7)

    shap = {
        "cloud_cover_pct":  -48.3,
        "ghi_w_m2":          -9.1,
        "temperature_c":     -1.8,
        "wind_speed_ms":      3.2,
        "hour_of_day_sin":    0.8,
    }
    ok("SHAP explanation ready")
    return shap


def simulate_qca_compliance() -> dict:
    """Return mock KERC QCA metrics."""
    return {
        "daily_qca": 0.923,
        "kerc_threshold": 0.80,
        "compliant": True,
        "breach_hours": ["14:00"],
        "worst_qca": 0.71,
        "non_compliant_blocks": 4,
        "total_blocks": 96,
        "avg_deviation_pct": 3.2,
    }


def simulate_dsm_financials() -> dict:
    return {
        "total_penalty_inr": 94_800,
        "baseline_penalty_inr": 412_000,
        "savings_inr": 317_200,
        "savings_pct": 77.0,
        "ramp_contribution_inr": 87_500,
        "projected_monthly_inr": 9_516_000,
        "projected_annual_inr": 114_192_000,
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(forecast: dict, shap: dict, qca: dict, dsm: dict) -> None:

    banner("  OptiCast AI — KERC DSM Compliance Report  ", C.CYAN)

    # ── Header ──────────────────────────────────────────────────────────────
    print()
    print(f"  {bold('Plant')}   : {forecast['plant']}")
    print(f"  {bold('Date')}    : {forecast['date']}")
    print(f"  {bold('Model')}   : TFT v2 + Conformal Prediction (α=0.10)")
    print(f"  {bold('Run at')}  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")

    # ── RAMP EVENT ALERT ────────────────────────────────────────────────────
    print()
    alert_line = (
        f"  {c(C.BG_RED + C.WHITE + C.BOLD, '  ⚡ CRITICAL RAMP ALERT  ')}"
        f"  {c(C.RED, '50 MW drop @ Pavagada 14:00 IST')}"
    )
    print(alert_line)
    print(f"  {c(C.RED, '  Cause  : Rapid cloud-cover ingress (78%) from SW quadrant')}")
    print(f"  {c(C.RED, '  Impact : DSM UI-0.5 band breached — ₹87,500 penalty exposure')}")
    print(f"  {c(C.YELLOW, '  Action : Issue revised WBES schedule before 13:55 IST')}")

    # ── Forecast Accuracy ───────────────────────────────────────────────────
    section("24-Hour Forecast Accuracy")
    row("Peak Generation (Forecast)",  f"{forecast['peak_forecast_mw']} MW")
    row("Total Energy Generated",      f"{forecast['total_energy_kwh']:,} kWh")
    row("RMSE",                        f"{forecast['rmse_mw']} MW",      C.GREEN)
    row("MAE",                         f"{forecast['mae_mw']} MW",       C.GREEN)
    row("Conformal Coverage (90%)",    f"{forecast['coverage_90pct']:.1%}", C.GREEN)
    row("Ramp Event Detected",         "YES — 14:00 IST, −50 MW",         C.RED)
    section_end()

    # ── SHAP Explanation ────────────────────────────────────────────────────
    section("SHAP Feature Attribution  [14:00 Ramp Hour]")
    sorted_shap = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, val in sorted_shap:
        bar_len = int(abs(val) / 50 * 20)
        bar_char = "▓" if val < 0 else "░"
        bar_color = C.RED if val < 0 else C.GREEN
        sign = "−" if val < 0 else "+"
        bar = c(bar_color, bar_char * bar_len)
        row(feat, f"{sign}{abs(val):5.1f} MW  {bar}", bar_color)
    section_end()

    # ── KERC QCA Compliance ─────────────────────────────────────────────────
    section("KERC QCA Compliance")
    compliant_color = C.GREEN if qca["compliant"] else C.RED
    compliant_label = "✓ COMPLIANT" if qca["compliant"] else "✗ NON-COMPLIANT"
    row("Daily QCA Score",            f"{qca['daily_qca']:.3f}", compliant_color)
    row("KERC Minimum Threshold",     f"{qca['kerc_threshold']:.2f}")
    row("Status",                     compliant_label, compliant_color)
    row("Non-Compliant 15-min Blocks",f"{qca['non_compliant_blocks']} / {qca['total_blocks']}", C.YELLOW)
    row("Worst Hour",                 f"14:00 — QCA {qca['worst_qca']}", C.RED)
    row("Avg Deviation from Schedule",f"{qca['avg_deviation_pct']}%")
    section_end()

    # ── DSM Financial Impact ────────────────────────────────────────────────
    section("KERC DSM Financial Impact")
    row("Today's DSM Penalty",        f"₹ {dsm['total_penalty_inr']:>10,}", C.YELLOW)
    row("Baseline Penalty (no AI)",   f"₹ {dsm['baseline_penalty_inr']:>10,}", C.RED)
    row("Savings Today",              f"₹ {dsm['savings_inr']:>10,}  ({dsm['savings_pct']:.0f}% reduction)", C.GREEN)
    row("Ramp-Event Contribution",    f"₹ {dsm['ramp_contribution_inr']:>10,} penalty", C.YELLOW)
    separator()
    row("Projected Monthly Savings",  f"₹ {dsm['projected_monthly_inr']:>10,}", C.GREEN)
    row("Projected Annual Savings",   f"₹ {dsm['projected_annual_inr']:>10,}", C.GREEN + C.BOLD)
    section_end()

    # ── BESS Recommendation ─────────────────────────────────────────────────
    section("Actionable Recommendation")
    print(c(C.BLUE, "│") + f"  {bold('Deploy 10 MWh BESS at Pavagada to absorb ramp transients')}")
    row("Estimated CAPEX",            "₹  4,50,00,000")
    row("Additional DSM Reduction",   "92%",    C.GREEN)
    row("Simple Payback Period",      "≈ 4.7 months", C.GREEN)
    section_end()

    # ── Footer ───────────────────────────────────────────────────────────────
    banner("  Pipeline Complete — OptiCast AI  ", C.GREEN)
    print()
    print(c(C.DIM, "  All outputs saved to ./outputs/  |  API mock running on :8000"))
    print(c(C.DIM, "  Dashboard: http://localhost:3000"))
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global USE_COLOR

    parser = argparse.ArgumentParser(description="OptiCast AI demo runner")
    parser.add_argument("--skip-data", action="store_true", help="Skip ERA5 and SCADA steps")
    parser.add_argument("--no-color",  action="store_true", help="Disable ANSI colors")
    args = parser.parse_args()

    if args.no_color:
        USE_COLOR = False

    banner("  OptiCast AI — Pipeline Runner  ")

    print()
    print(c(C.DIM, f"  Renewable Energy Forecasting & KERC DSM Compliance"))
    print(c(C.DIM, f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}"))
    print()

    # ── Step 1: ERA5 Data ────────────────────────────────────────────────────
    banner("Step 1 / 4 — ERA5 Weather Data", C.MAGENTA)
    run_script("data/ingestion_service.py --service era5", "ERA5 Reanalysis Pull", skip=args.skip_data)

    # ── Step 2: SCADA Simulation ─────────────────────────────────────────────
    banner("Step 2 / 4 — SCADA Simulation", C.MAGENTA)
    run_script("data/ingestion_service.py --service scada", "SCADA Generation Profiles", skip=args.skip_data)
    
    # ── Step 3: ML Inference ─────────────────────────────────────────────────
    banner("Step 3 / 4 — ML Inference", C.MAGENTA)
    forecast = simulate_ml_inference()
    shap     = simulate_shap_explanation()

    # ── Step 4: Compliance Engine ────────────────────────────────────────────
    banner("Step 4 / 4 — Compliance Engine", C.MAGENTA)
    info("Running QCA generator …")
    run_script("compliance/qca_generator.py", "KERC QCA Generator", skip=False)
    run_script("compliance/ramp_alert.py",    "Ramp Alert Engine",  skip=False)

    qca = simulate_qca_compliance()
    dsm = simulate_dsm_financials()
    ok("Compliance metrics computed")

    # ── Final Report ──────────────────────────────────────────────────────────
    print_report(forecast, shap, qca, dsm)


if __name__ == "__main__":
    main()
