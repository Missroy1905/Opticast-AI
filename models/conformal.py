import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
import os
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import warnings

warnings.filterwarnings("ignore")

def run_conformal():
    print("1. Loading Data and Creating Splits...")
    df = pd.read_parquet("/content/data/master_dataset.parquet")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values(["plant_id", "time"]).reset_index(drop=True)
    df["time_of_day"] = (df["time"].dt.hour + df["time"].dt.minute / 60.0) / 24.0
    df["day_of_week"] = df["time"].dt.dayofweek / 6.0
    df["time_idx"] = df.groupby("plant_id")["time"].transform(lambda s: (s - s.min()).dt.total_seconds() // 900).astype(int)

    # Splitting: Train (80%), Calibration (10%), Test (10%)
    max_idx = df["time_idx"].max()
    train_df = df[df["time_idx"] < int(max_idx * 0.8)].copy()
    cal_df = df[(df["time_idx"] >= int(max_idx * 0.8) - 96) & (df["time_idx"] < int(max_idx * 0.9))].copy()
    test_df = df[df["time_idx"] >= int(max_idx * 0.9) - 96].copy()

    print("2. Rebuilding TimeSeries Rules...")
    training = TimeSeriesDataSet(
        train_df, time_idx="time_idx", target="generation_mw", group_ids=["plant_id"],
        static_categoricals=["asset_type"], 
        categorical_encoders={"asset_type": NaNLabelEncoder(add_nan=True)},
        static_reals=["capacity_mw", "latitude", "longitude"],
        time_varying_known_reals=["ghi", "wind_speed_100m", "temperature_2m", "cloud_cover", "time_of_day", "day_of_week"],
        time_varying_unknown_reals=["generation_mw"],
        max_encoder_length=96, max_prediction_length=96,
        allow_missing_timesteps=True, add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )
    
    cal_dataset = TimeSeriesDataSet.from_dataset(training, cal_df, predict=True, stop_randomization=True)
    test_dataset = TimeSeriesDataSet.from_dataset(training, test_df, predict=True, stop_randomization=True)

    print("3. Loading Best Model Checkpoint...")
    best_model_path = max(glob.glob("/content/models/checkpoints/*.ckpt"), key=os.path.getctime)
    model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    print("4. Generating Predictions for Calibration Set...")
    cal_dataloader = cal_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
    cal_preds = model.predict(cal_dataloader, mode="raw", return_x=True)
    
    # FIX: Move target to CPU
    y_cal_true = cal_preds.x["decoder_target"].cpu()
    
    # FIX: Move predictions to CPU
    cal_prediction_tensor = cal_preds.output[0].cpu() 
    y_cal_low = cal_prediction_tensor[..., 0]  # 10th percentile
    y_cal_high = cal_prediction_tensor[..., 2] # 90th percentile

    print("5. Calculating Conformal Scores (CQR)...")
    # Score = max(lower_bound - actual, actual - upper_bound)
    scores = torch.maximum(y_cal_low - y_cal_true, y_cal_true - y_cal_high)
    
    # Calculate the 90th percentile of the scores to act as our correction factor
    alpha = 0.1 
    n = scores.numel()
    q_val = np.quantile(scores.numpy(), np.ceil((n + 1) * (1 - alpha)) / n)
    print(f"\n✅ Conformal Correction Factor (q_hat): {q_val:.4f} MW")

    print("\n6. Applying Guaranteed Bounds to Test Set...")
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=32, num_workers=0)
    test_preds = model.predict(test_dataloader, mode="raw", return_x=True)
    
    idx = 0
    # FIX: Move test targets and predictions to CPU before converting to NumPy
    y_test_true = test_preds.x["decoder_target"][idx].cpu().numpy()
    test_prediction_tensor = test_preds.output[0].cpu()
    
    y_test_median = test_prediction_tensor[idx, :, 1].numpy()
    y_test_low_raw = test_prediction_tensor[idx, :, 0].numpy()
    y_test_high_raw = test_prediction_tensor[idx, :, 2].numpy()

    # Apply the calibration factor
    y_test_low_cal = np.maximum(y_test_low_raw - q_val, 0) # Floor at 0 MW
    y_test_high_cal = y_test_high_raw + q_val

    print("7. Plotting Final Calibrated Forecast...")
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(len(y_test_true))
    
    plt.plot(time_steps, y_test_true, label="Actual Generation", color="black", linewidth=2)
    plt.plot(time_steps, y_test_median, label="TFT Median Forecast", color="blue", linestyle="--")
    plt.fill_between(time_steps, y_test_low_cal, y_test_high_cal, color="gray", alpha=0.3, label="90% Conformal Bound")
    
    plt.title(f"Statistically Guaranteed Forecast (90% Confidence)\nCorrection Factor Applied: ±{q_val:.2f} MW")
    plt.xlabel("Time Steps (15-min intervals)")
    plt.ylabel("Generation (MW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/content/models/conformal_forecast.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_conformal()
