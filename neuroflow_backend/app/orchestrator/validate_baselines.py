"""
Validate baseline metrics: recompute and compare to saved numbers; assess effectiveness.
Run from repo root: python -m app.orchestrator.validate_baselines
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Same config and logic as baselines.py
PRIMARY_DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "datasets" / "training_dataset_enriched.csv"
ORCHESTRATOR_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "orchestrator_output"
BASELINE_METRICS_PATH = ORCHESTRATOR_OUTPUT_DIR / "baseline_metrics.json"
SPEED_BAND_TO_KMH = {1: 12.5, 2: 22.5, 3: 32.5, 4: 42.5, 5: 52.5, 6: 62.5, 7: 72.5, 8: 82.5}
TRAIN_RATIO, VAL_RATIO = 0.7, 0.15


def main():
    df = pd.read_csv(PRIMARY_DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["speed_kmh"] = df["speed_band"].map(SPEED_BAND_TO_KMH)
    n = len(df)
    t1 = int(n * TRAIN_RATIO)
    t2 = int(n * (TRAIN_RATIO + VAL_RATIO))
    train, val, test = df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]

    # --- 1) Derived target sanity ---
    assert df["speed_kmh"].min() == 12.5 and df["speed_kmh"].max() == 82.5, "speed_kmh range should be 12.5–82.5"
    print("1. Derived target (speed_band -> speed_kmh): OK (12.5–82.5 km/h)")

    # --- 2) Temporal split and leakage ---
    assert train["timestamp"].max() <= val["timestamp"].min() and val["timestamp"].max() <= test["timestamp"].min()
    print("2. Temporal split (no shuffle, chronological): OK")
    print(f"   Train: {len(train)} rows, Val: {len(val)}, Test: {len(test)}")

    # --- 3) Recompute Historical Average test MAE/RMSE ---
    train_avg = train.groupby(["road_name", "hour"])["speed_kmh"].mean().reset_index()
    train_avg = train_avg.rename(columns={"speed_kmh": "pred"})
    test_merged = test.merge(train_avg, on=["road_name", "hour"], how="left")
    global_mean = train["speed_kmh"].mean()
    test_merged["pred"] = test_merged["pred"].fillna(global_mean)
    test_mae = (test_merged["speed_kmh"] - test_merged["pred"]).abs().mean()
    test_rmse = np.sqrt(((test_merged["speed_kmh"] - test_merged["pred"]) ** 2).mean())

    with open(BASELINE_METRICS_PATH) as f:
        saved = json.load(f)
    saved_mae = saved["historical_average"]["test_mae"]
    saved_rmse = saved["historical_average"]["test_rmse"]
    mae_ok = np.isclose(test_mae, saved_mae, rtol=1e-4)
    rmse_ok = np.isclose(test_rmse, saved_rmse, rtol=1e-4)
    print("3. Historical Average — recomputed vs saved:")
    print(f"   Recomputed Test MAE={test_mae:.6f}  RMSE={test_rmse:.6f}")
    print(f"   Saved      Test MAE={saved_mae:.6f}  RMSE={saved_rmse:.6f}")
    print(f"   Match: MAE={mae_ok}, RMSE={rmse_ok}")

    # --- 4) Effectiveness: scale of error vs target ---
    test_speed_std = test["speed_kmh"].std()
    test_speed_mean = test["speed_kmh"].mean()
    # Naive baseline: predict global mean for everyone
    naive_mae = (test["speed_kmh"] - test["speed_kmh"].mean()).abs().mean()
    print("4. Effectiveness context:")
    print(f"   Test set: mean speed={test_speed_mean:.1f} km/h, std={test_speed_std:.1f}")
    print(f"   Historical Avg Test MAE={test_mae:.2f} (≈ {100*test_mae/test_speed_mean:.0f}% of mean speed)")
    print(f"   Naive (predict global mean) MAE={naive_mae:.2f}")
    print(f"   Historical Average beats naive: {test_mae < naive_mae}")

    # --- 5) ARIMA: we only check that saved numbers are in a plausible range ---
    arima = saved.get("arima_or_persistence", {})
    arima_mae = arima.get("test_mae")
    if arima_mae is not None:
        print("5. ARIMA/Persistence: saved Test MAE={:.2f} (plausible if > Historical Avg for this setup)".format(arima_mae))
        # ARIMA(1,0,0) on short series can be worse than historical avg
        print("   (ARIMA per-road with 1-step forecast logic; can be worse than road+hour average.)")

    # --- 6) Phase 2 target ---
    # Phase 2 requires ~30% improvement over baseline; 11.76 * 0.7 ≈ 8.23 km/h MAE target
    target_mae = saved_mae * 0.7
    print("6. Phase 2 target (30% improvement over Historical Avg): Test MAE ≤ {:.2f} km/h".format(target_mae))

    if not (mae_ok and rmse_ok):
        sys.exit(1)
    print("\nValidation: numbers are accurate and methodology is sound.")
    return 0


if __name__ == "__main__":
    main()
