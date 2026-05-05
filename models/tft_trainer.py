"""
models/tft_trainer.py
---------------------
Temporal Fusion Transformer training pipeline for OptiCast AI.

Architecture:
  - Single model for all solar and wind assets
  - Static covariates: plant type, capacity, lat/lon, commissioning year
  - Time-varying known: GHI, wind speed, temperature, cloud cover, time features
  - Time-varying unknown: generation_mw (target)
  - Output: P10 / P50 / P90 quantiles for 96 × 15-minute blocks

GPU:  ~4–6 hours  (30 epochs)
CPU:  ~10–12 hours (30 epochs)
Fallback: run lgbm_fallback.py for LightGBM quantile model in ~1–2 hours
"""

import pandas as pd
import numpy as np
import mlflow
import torch
from pathlib import Path

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENCODER_LEN  = 96    # 24 hours of 15-min history
DECODER_LEN  = 96    # 24 hours of 15-min forecast
MAX_EPOCHS   = 30
BATCH_SIZE   = 64
HIDDEN_SIZE  = 64
ATTN_HEADS   = 4
DROPOUT      = 0.1
LEARNING_RATE = 0.03


def load_and_prepare(scada_path: Path, era5_path: Path) -> pd.DataFrame:
    scada = pd.read_parquet(scada_path)
    era5  = pd.read_parquet(era5_path)

    df = scada.merge(
        era5[["time", "plant_id", "shortwave_radiation", "direct_normal_irradiance",
              "cloud_cover", "temperature_2m", "wind_speed_100m", "wind_direction_10m"]],
        on=["time", "plant_id"],
        how="left",
    )

    df = df.rename(columns={
        "shortwave_radiation": "ghi",
        "wind_speed_100m":     "wind_speed_100m",
    })

    # Time features
    df["time"]        = pd.to_datetime(df["time"])
    df["time_of_day"] = df["time"].dt.hour + df["time"].dt.minute / 60
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"]       = df["time"].dt.month

    # Integer time index per plant
    df = df.sort_values(["plant_id", "time"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("plant_id").cumcount()

    # Static metadata
    df["commissioning_year"] = 2017    # Approximate for Karnataka assets
    df["ghi"]                = df["ghi"].fillna(0).clip(lower=0)
    df["cloud_cover"]        = df["cloud_cover"].fillna(0)
    df["temperature_2m"]     = df["temperature_2m"].fillna(25)
    df["wind_speed_100m"]    = df["wind_speed_100m"].fillna(0)
    df["wind_direction_10m"] = df["wind_direction_10m"].fillna(0)

    return df


def build_dataset(df: pd.DataFrame):
    training_cutoff = int(df["time_idx"].max() * 0.75)

    dataset = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="generation_mw",
        group_ids=["plant_id"],
        static_categoricals=["asset_type"],
        static_reals=["capacity_mw", "latitude", "longitude", "commissioning_year"],
        time_varying_known_reals=[
            "ghi", "wind_speed_100m", "wind_direction_10m",
            "temperature_2m", "cloud_cover",
            "time_of_day", "day_of_week", "month",
        ],
        time_varying_unknown_reals=["generation_mw"],
        max_encoder_length=ENCODER_LEN,
        max_prediction_length=DECODER_LEN,
        target_normalizer=GroupNormalizer(groups=["plant_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    val_dataset = TimeSeriesDataSet.from_dataset(
        dataset,
        df[df["time_idx"] > training_cutoff - ENCODER_LEN],
        predict=True,
        stop_randomization=True,
    )

    return dataset, val_dataset


def train(df: pd.DataFrame):
    training, validation = build_dataset(df)

    train_loader = training.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=0
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=BATCH_SIZE * 2, num_workers=0
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTN_HEADS,
        dropout=DROPOUT,
        hidden_continuous_size=16,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters: {tft.size() / 1e3:.1f}k")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ModelCheckpoint(
            dirpath=str(OUTPUT_DIR),
            filename="tft_best",
            monitor="val_loss",
            mode="min",
        ),
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {accelerator.upper()}")
    if accelerator == "cpu":
        print("⚠ CPU training will take 10–12 hours. Consider running lgbm_fallback.py instead.")

    with mlflow.start_run(run_name="tft_training"):
        mlflow.log_params({
            "hidden_size": HIDDEN_SIZE,
            "attention_heads": ATTN_HEADS,
            "encoder_length": ENCODER_LEN,
            "decoder_length": DECODER_LEN,
            "max_epochs": MAX_EPOCHS,
            "learning_rate": LEARNING_RATE,
        })

        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator=accelerator,
            devices=1,
            gradient_clip_val=0.1,
            callbacks=callbacks,
            enable_progress_bar=True,
        )
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_model_path = OUTPUT_DIR / "tft_best.ckpt"
        mlflow.log_artifact(str(best_model_path))
        print(f"\n✓ Best model saved: {best_model_path}")

    return tft, trainer


def main():
    print("OptiCast TFT Trainer")
    print("=" * 40)

    scada_path = DATA_DIR / "scada_all_plants.parquet"
    era5_path  = DATA_DIR / "era5_all_plants.parquet"

    if not scada_path.exists():
        print("⚠ Run data/generate_scada.py first")
        return
    if not era5_path.exists():
        print("⚠ Run data/fetch_era5.py first")
        return

    print("Loading and merging datasets...")
    df = load_and_prepare(scada_path, era5_path)
    print(f"  Rows: {len(df):,} | Plants: {df['plant_id'].nunique()} | "
          f"Date range: {df['time'].min().date()} → {df['time'].max().date()}")

    print("\nStarting TFT training...")
    tft, trainer = train(df)
    print("\n✓ Training complete. Run models/conformal.py to calibrate prediction intervals.")


if __name__ == "__main__":
    main()
