"""
OptiCast AI — models/tft_trainer.py
=====================================
Phase 3A: Temporal Fusion Transformer training pipeline.

Reads ERA5 parquet files produced by data/fetch_era5.py (or falls back to
ingestion_service.py synthetic data when parquet is absent).

Architecture matches the PDF spec exactly:
  • group_ids            = ['plant_id']
  • static_categoricals  = ['asset_type']
  • static_reals         = ['capacity_mw', 'latitude', 'longitude', 'commissioning_year']
  • time_varying_known   = ['ghi', 'wind_speed_100m', 'temperature_2m',
                            'cloud_cover', 'time_of_day', 'day_of_week']
  • time_varying_unknown = ['generation_mw']
  • max_encoder_length   = 96 (24 hours of 15-min history)
  • max_prediction_length= 96 (24-hour horizon → all 96 blocks in one pass)
  • loss                 = QuantileLoss([0.1, 0.5, 0.9])
  • hidden_size          = 64, attention_head_size = 4, dropout = 0.1

Training:
  GPU: ~4-6 hours | CPU fallback: ~10-12 hours (auto-detected)

Usage:
  python models/tft_trainer.py                  # full train
  python models/tft_trainer.py --epochs 3       # quick smoke-test
  python models/tft_trainer.py --fast           # 500-row subset, 2 epochs
  python models/tft_trainer.py --eval-only      # load checkpoint, run eval
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("opticast.tft_trainer")

# ── Optional heavy imports (guarded so linting works without GPU env) ──────
try:
    import torch
    import pytorch_lightning as pl
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    HAS_PF = True
except ImportError as e:
    HAS_PF = False
    _PF_ERR = str(e)

try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# ── Paths — Colab-safe ─────────────────────────────────────────────────────
# __file__ is undefined when this script is run as a Colab cell (%%writefile
# then exec'd). Detect Colab and fall back to /content/ as the project root.
try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
    ROOT        = _SCRIPT_DIR.parent
except NameError:
    ROOT = Path("/content")          # Colab default workspace

DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models" / "checkpoints"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Karnataka plant registry (matches ingestion_service.py) ────────────────
PLANTS = {
    "pavagada":   {"lat": 14.10, "lon": 77.28, "type": "solar", "mw": 2050, "year": 7},
    "chitradurga":{"lat": 14.22, "lon": 76.39, "type": "wind",  "mw": 500,  "year": 5},
    "koppal":     {"lat": 15.35, "lon": 76.15, "type": "wind",  "mw": 300,  "year": 4},
    "gadag":      {"lat": 15.41, "lon": 75.63, "type": "wind",  "mw": 250,  "year": 3},
    "raichur":    {"lat": 16.20, "lon": 77.36, "type": "solar", "mw": 400,  "year": 2},
}


# ===========================================================================
# STEP 1 — DATA LOADING & FEATURE ENGINEERING
# ===========================================================================

def _solar_output(ghi: float, temp: float, capacity_mw: float,
                  year: int = 7) -> float:
    """Convert GHI → AC generation MW (matches generate_scada.py spec)."""
    efficiency  = 0.195
    temp_coeff  = -0.004       # –0.4 %/°C above 25 °C
    soiling     = 0.97
    degrade     = (1 - 0.007) ** year
    temp_factor = 1 + temp_coeff * (temp - 25)
    base = (ghi / 1000) * capacity_mw * efficiency * temp_factor * soiling * degrade
    return max(0.0, base * np.random.normal(1, 0.02))


def _wind_output(wind_speed: float, capacity_mw: float) -> float:
    """Simple cubic wind power curve (matches generate_scada.py spec)."""
    cut_in, rated, cut_out = 3.0, 12.0, 25.0
    if wind_speed < cut_in or wind_speed > cut_out:
        return 0.0
    if wind_speed >= rated:
        return float(capacity_mw)
    return float(capacity_mw * ((wind_speed - cut_in) / (rated - cut_in)) ** 3)


def load_era5_parquets() -> pd.DataFrame:
    """
    Load all data/era5_<plant>.parquet files.
    Returns one combined DataFrame with 15-min resampled rows.
    Falls back to synthetic generation if no parquets found.
    """
    files = list(DATA_DIR.glob("era5_*.parquet"))
    if not files:
        log.warning("No ERA5 parquet files found in %s — using synthetic data.", DATA_DIR)
        return _generate_synthetic_df()

    frames = []
    for f in files:
        plant_name = f.stem.replace("era5_", "")
        meta = PLANTS.get(plant_name)
        if meta is None:
            log.warning("Unknown plant in filename: %s — skipping.", f.name)
            continue

        df = pd.read_parquet(f)
        log.info("Loaded %s — %d rows", f.name, len(df))

        # Normalise time column (Open-Meteo returns 'time' as ISO string)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()

        # Rename Open-Meteo columns to internal names
        col_map = {
            "shortwave_radiation":       "ghi",
            "temperature_2m":            "temperature_2m",
            "wind_speed_10m":            "wind_speed_10m",
            "wind_speed_100m":           "wind_speed_100m",
            "cloud_cover":               "cloud_cover",
            "direct_normal_irradiance":  "dni",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Resample hourly → 15-min (forward-fill; matches KERC block cadence)
        df = df.resample("15min").ffill()

        # Generate synthetic actual output (used as training target)
        np.random.seed(42)
        if meta["type"] == "solar":
            df["generation_mw"] = df.apply(
                lambda r: _solar_output(
                    r.get("ghi", 0), r.get("temperature_2m", 30),
                    meta["mw"], meta["year"]
                ), axis=1
            )
        else:
            df["generation_mw"] = df.apply(
                lambda r: _wind_output(r.get("wind_speed_100m", 0), meta["mw"]),
                axis=1
            )

        df["plant_id"]           = plant_name
        df["asset_type"]         = meta["type"]
        df["capacity_mw"]        = float(meta["mw"])
        df["latitude"]           = meta["lat"]
        df["longitude"]          = meta["lon"]
        df["commissioning_year"] = float(meta["year"])
        frames.append(df.reset_index())

    if not frames:
        return _generate_synthetic_df()

    combined = pd.concat(frames, ignore_index=True)
    return _add_time_features(combined)


def _generate_synthetic_df(n_days: int = 365) -> pd.DataFrame:
    """
    Pure-synthetic fallback — generates 1 year × 5 plants × 96 blocks/day.
    Used for smoke-tests and when ERA5 parquets are absent.
    """
    log.info("Generating synthetic training data (%d days × %d plants)…",
             n_days, len(PLANTS))
    np.random.seed(42)
    frames = []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    idx   = pd.date_range(start, periods=n_days * 96, freq="15min", tz="UTC")

    for plant_name, meta in PLANTS.items():
        df = pd.DataFrame({"time": idx})
        t  = np.linspace(0, n_days * 24, len(idx))
        hour = (t % 24)

        if meta["type"] == "solar":
            # Gaussian solar profile + seasonal amplitude
            season = 1 + 0.2 * np.sin(2 * np.pi * t / (365 * 24))
            ghi_base = np.clip(
                800 * season * np.exp(-0.5 * ((hour - 12.5) / 2.8) ** 2), 0, None
            )
            ghi_noisy = ghi_base * np.clip(np.random.normal(0.9, 0.15, len(idx)), 0, 1.2)
            df["ghi"]           = ghi_noisy
            df["temperature_2m"]= 28 + 5 * np.sin(2 * np.pi * hour / 24) + \
                                  np.random.normal(0, 1, len(idx))
            df["cloud_cover"]   = np.clip(np.random.beta(2, 5, len(idx)) * 100, 0, 100)
            df["wind_speed_100m"] = np.abs(np.random.normal(4, 1.5, len(idx)))
            df["generation_mw"] = np.vectorize(_solar_output)(
                df["ghi"], df["temperature_2m"], meta["mw"], meta["year"]
            )
        else:
            df["ghi"]           = np.zeros(len(idx))
            df["temperature_2m"]= 26 + np.random.normal(0, 2, len(idx))
            df["cloud_cover"]   = np.clip(np.random.normal(40, 20, len(idx)), 0, 100)
            df["wind_speed_100m"] = np.abs(np.random.normal(7, 2.5, len(idx)))
            df["generation_mw"] = np.vectorize(_wind_output)(
                df["wind_speed_100m"], meta["mw"]
            )

        df["plant_id"]           = plant_name
        df["asset_type"]         = meta["type"]
        df["capacity_mw"]        = float(meta["mw"])
        df["latitude"]           = meta["lat"]
        df["longitude"]          = meta["lon"]
        df["commissioning_year"] = float(meta["year"])
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return _add_time_features(combined)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features and integer time_idx required by TFT."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values(["plant_id", "time"]).reset_index(drop=True)

    # Cyclical features (sin/cos encoding prevents boundary artifacts)
    hour_min = df["time"].dt.hour + df["time"].dt.minute / 60.0
    df["time_of_day"] = hour_min / 24.0           # 0–1 normalised
    df["day_of_week"]  = df["time"].dt.dayofweek / 6.0   # 0–1 normalised

    # Ensure required columns exist with defaults
    for col in ["ghi", "wind_speed_100m", "temperature_2m", "cloud_cover"]:
        if col not in df.columns:
            df[col] = 0.0

    # Clip negatives
    df["generation_mw"] = df["generation_mw"].clip(lower=0)
    df["ghi"]           = df["ghi"].clip(lower=0)

    # Encode asset_type as integer for static_categoricals
    df["asset_type"] = df["asset_type"].astype(str)

    # Global integer time index — TFT requirement
    # Each (plant, timestamp) gets a unique integer; plants share the same axis
    # fillna(0) guards against NaT producing NaN before int cast (pandas 2.x)
    df["time_idx"] = (
        df.groupby("plant_id")["time"]
        .transform(lambda s: (s - s.min()).dt.total_seconds() // 900)
        .fillna(0)
        .astype(int)
    )

    return df


# ===========================================================================
# STEP 2 — TRAIN / VALIDATION SPLIT
# ===========================================================================

def split_df(df: pd.DataFrame,
             val_fraction: float = 0.15,
             cal_fraction: float = 0.10):
    """
    Chronological split: train | calibration | validation
    Calibration set is held out for conformal prediction fitting.
    """
    max_idx   = df["time_idx"].max()
    val_start = int(max_idx * (1 - val_fraction))
    cal_start = int(max_idx * (1 - val_fraction - cal_fraction))

    train_df  = df[df["time_idx"] < cal_start].copy()
    cal_df    = df[(df["time_idx"] >= cal_start) & (df["time_idx"] < val_start)].copy()
    val_df    = df[df["time_idx"] >= val_start].copy()

    log.info("Split — train: %d rows | cal: %d rows | val: %d rows",
             len(train_df), len(cal_df), len(val_df))
    return train_df, cal_df, val_df


# ===========================================================================
# STEP 3 — BUILD TFT DATASET & DATALOADERS
# ===========================================================================

STATIC_CATEGORICALS  = ["asset_type"]
STATIC_REALS         = ["capacity_mw", "latitude", "longitude", "commissioning_year"]
TIME_VARYING_KNOWN   = ["ghi", "wind_speed_100m", "temperature_2m",
                         "cloud_cover", "time_of_day", "day_of_week"]
TIME_VARYING_UNKNOWN = ["generation_mw"]

MAX_ENCODER_LENGTH   = 96    # 24 h of 15-min history
MAX_PRED_LENGTH      = 96    # 24 h ahead — all 96 KERC blocks in one pass


def build_dataset(train_df: pd.DataFrame,
                  max_encoder: int = MAX_ENCODER_LENGTH,
                  max_pred:    int = MAX_PRED_LENGTH) -> "TimeSeriesDataSet":
    """Construct the PyTorch Forecasting TimeSeriesDataSet from training data."""
    return TimeSeriesDataSet(
        train_df,
        time_idx             = "time_idx",
        target               = "generation_mw",
        group_ids            = ["plant_id"],
        static_categoricals  = STATIC_CATEGORICALS,
        static_reals         = STATIC_REALS,
        time_varying_known_reals   = TIME_VARYING_KNOWN,
        time_varying_unknown_reals = TIME_VARYING_UNKNOWN,
        max_encoder_length   = max_encoder,
        max_prediction_length= max_pred,
        allow_missing_timesteps = True,   # handles SCADA gaps via attention masking
        add_relative_time_idx   = True,   # TFT standard
        add_target_scales       = True,
        add_encoder_length      = True,
    )


def build_tft(dataset: "TimeSeriesDataSet",
              learning_rate: float = 0.03,
              hidden_size:   int   = 64,
              attn_heads:    int   = 4,
              dropout:       float = 0.1) -> "TemporalFusionTransformer":
    """Instantiate TFT with quantile loss — PDF spec hyperparameters."""
    return TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate           = learning_rate,
        hidden_size             = hidden_size,
        attention_head_size     = attn_heads,
        dropout                 = dropout,
        hidden_continuous_size  = 32,
        # output_size is derived automatically from QuantileLoss(quantiles=[0.1,0.5,0.9])
        # DO NOT set output_size=7 — that causes a shape mismatch crash in PF 1.x
        loss                    = QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        log_interval            = 10,
        reduce_on_plateau_patience = 4,
        lstm_layers             = 2,
    )


# ===========================================================================
# STEP 4 — TRAINING LOOP
# ===========================================================================

def train(
    max_epochs:   int   = 30,
    batch_size:   int   = 64,
    lr:           float = 0.03,
    fast_mode:    bool  = False,
    eval_only:    bool  = False,
    checkpoint:   Optional[str] = None,
) -> tuple:
    """
    Full training pipeline. Returns (tft_model, cal_df, val_df).

    fast_mode: uses 500-row subset + 2 epochs for CI/smoke-tests.
    eval_only: loads checkpoint, skips training.
    """
    if not HAS_PF:
        log.error("pytorch-forecasting not installed: %s", _PF_ERR)
        log.error("Install: pip install pytorch-forecasting torch pytorch-lightning")
        raise ImportError("pytorch-forecasting required for training.")

    # ── Load data ────────────────────────────────────────────────────
    log.info("Loading ERA5 data…")
    df = load_era5_parquets()
    if fast_mode:
        df = df.groupby("plant_id").head(500).reset_index(drop=True)
        max_epochs = 2
        log.info("FAST MODE: %d rows, %d epochs", len(df), max_epochs)

    train_df, cal_df, val_df = split_df(df)

    # ── Build dataset ────────────────────────────────────────────────
    log.info("Building TimeSeriesDataSet…")
    training = build_dataset(train_df)
    # stop_randomization removed in pytorch-forecasting 1.1+ — do not pass it
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True)

    train_loader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0)
    val_loader   = validation.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────
    if eval_only and checkpoint:
        log.info("Loading checkpoint: %s", checkpoint)
        tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint)
    else:
        tft = build_tft(training, learning_rate=lr)
        log.info("TFT parameters: %s",
                 f"{sum(p.numel() for p in tft.parameters()):,}")

    # ── MLflow tracking ───────────────────────────────────────────────
    if HAS_MLFLOW and not eval_only:
        mlflow.set_experiment("opticast_tft")
        mlflow.pytorch.autolog()

    # ── Trainer ──────────────────────────────────────────────────────
    accelerator = "gpu" if (not eval_only and torch.cuda.is_available()) else "cpu"
    log.info("Accelerator: %s", accelerator)

    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, mode="min"),   # verbose= removed in PL 2.0+
        pl.callbacks.ModelCheckpoint(
            dirpath=str(MODEL_DIR),
            filename="tft_opticast_{epoch:02d}_{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=2,
            mode="min",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs   = max_epochs,
        accelerator  = accelerator,
        devices      = 1,
        gradient_clip_val = 0.1,
        callbacks    = callbacks,
        enable_progress_bar = True,
        log_every_n_steps   = 5,
    )

    if not eval_only:
        log.info("Training TFT — %d epochs | GPU: %s", max_epochs,
                 torch.cuda.is_available())
        log.info("Estimated time — GPU: 4–6 h | CPU: 10–12 h")
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
        log.info("✅ Training complete. Checkpoints saved to %s", MODEL_DIR)

    # ── Evaluation ─────────────────────────────────────────────────
    log.info("Running validation metrics…")
    # PF 1.1+: predict() no longer accepts trainer_kwargs as a kwarg.
    # Pass a pre-built Trainer instance via the trainer= argument instead,
    # or call without it (CPU inference is the safe default here).
    val_preds = tft.predict(val_loader, return_y=True,
                            trainer_kwargs=dict(accelerator=accelerator,
                                                logger=False))
    _log_metrics(val_preds)

    return tft, cal_df, val_df, training


def _log_metrics(val_preds) -> None:
    """Log MAE / RMSE / MAPE to stdout and MLflow."""
    try:
        # PF 1.1+: tensors may be on GPU — always call .cpu() before .numpy()
        actuals = val_preds.y[0].cpu().numpy().flatten()

        # PF 1.1+ renamed .output → .prediction in some builds.
        # Try both attribute names so this works across versions.
        raw_out = getattr(val_preds, "output", None) or getattr(val_preds, "prediction", None)
        if raw_out is None:
            log.warning("Cannot extract predictions — unknown return format.")
            return
        # raw_out shape: (n_samples, pred_len, n_quantiles)
        # quantile index 1 = P50 for QuantileLoss([0.1, 0.5, 0.9])
        preds = raw_out.cpu().numpy()[:, :, 1].flatten()

        mae  = float(np.mean(np.abs(actuals - preds)))
        rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
        mask = actuals > 1.0      # avoid div/0 at night
        mape = float(np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100)

        log.info("Validation — MAE: %.2f MW | RMSE: %.2f MW | MAPE: %.2f%%",
                 mae, rmse, mape)

        if HAS_MLFLOW:
            mlflow.log_metrics({"val_mae": mae, "val_rmse": rmse, "val_mape": mape})
    except Exception as exc:
        log.warning("Could not compute metrics: %s", exc)


# ===========================================================================
# STEP 5 — INFERENCE HELPER (used by conformal.py and API)
# ===========================================================================

def predict_day(
    tft: "TemporalFusionTransformer",
    dataset: "TimeSeriesDataSet",
    encoder_data: pd.DataFrame,
) -> dict:
    """
    Run inference for one 96-block day.

    Returns:
        {
          'p10': np.ndarray shape (96,),
          'p50': np.ndarray shape (96,),
          'p90': np.ndarray shape (96,),
          'raw': np.ndarray shape (96, 3)  — [p10, p50, p90] per block
        }
    """
    # PF 1.1+: TimeSeriesDataSet.filter() was removed.
    # Use from_dataset() to create an inference dataset from encoder_data.
    inference_ds = TimeSeriesDataSet.from_dataset(
        dataset, encoder_data, predict=True
    )
    inference_loader = inference_ds.to_dataloader(
        train=False, batch_size=1, num_workers=0
    )
    # trainer_kwargs still accepted here as a dict in PF 1.1+
    prediction = tft.predict(
        inference_loader,
        mode="quantiles",
        trainer_kwargs=dict(accelerator="cpu", logger=False),
    )
    # Handle .output vs .prediction attribute rename across PF versions
    raw_out = getattr(prediction, "output", None) or getattr(prediction, "prediction", None)
    if raw_out is None:
        raw_out = prediction   # predict() may return tensor directly in some builds
    raw = raw_out.cpu().numpy()          # shape: (n, pred_len, 3)
    raw = raw[0]                         # shape: (pred_len, 3) — first (only) sample
    return {
        "p10": np.clip(raw[:, 0], 0, None),
        "p50": np.clip(raw[:, 1], 0, None),
        "p90": np.clip(raw[:, 2], 0, None),
        "raw": raw,
    }


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OptiCast TFT Trainer")
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--batch",     type=int,   default=64)
    parser.add_argument("--lr",        type=float, default=0.03)
    parser.add_argument("--fast",      action="store_true",
                        help="Smoke-test mode: 500 rows, 2 epochs")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, load checkpoint and evaluate")
    parser.add_argument("--checkpoint",type=str,   default=None,
                        help="Path to .ckpt file (required with --eval-only)")
    args = parser.parse_args()

    tft_model, cal_df, val_df, training_dataset = train(
        max_epochs = args.epochs,
        batch_size = args.batch,
        lr         = args.lr,
        fast_mode  = args.fast,
        eval_only  = args.eval_only,
        checkpoint = args.checkpoint,
    )

    # Save calibration set for conformal.py
    cal_path = MODEL_DIR / "cal_df.parquet"
    cal_df.to_parquet(cal_path, index=False)
    log.info("Calibration set saved → %s (%d rows)", cal_path, len(cal_df))
    log.info("Next step: python models/conformal.py")
