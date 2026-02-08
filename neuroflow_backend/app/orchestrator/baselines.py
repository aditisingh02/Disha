"""
Phase 1 — Baseline models: Historical Average, ARIMA.
Temporal train/val/test split; no shuffling across time. Metrics: MAE, RMSE.
Derived target: speed (km/h) from speed_band via midpoint mapping (1->12.5, 2->22.5, ..., 8->87.5).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.orchestrator.config import PRIMARY_DATASET_PATH, ORCHESTRATOR_OUTPUT_DIR, BASELINE_METRICS_PATH

logger = logging.getLogger("neuroflow.orchestrator.baselines")

# Explicit derivation: speed_band (1-8) -> speed km/h midpoint for regression (no synthetic labels; derived from ordinal)
SPEED_BAND_TO_KMH = {1: 12.5, 2: 22.5, 3: 32.5, 4: 42.5, 5: 52.5, 6: 62.5, 7: 72.5, 8: 82.5}


def _load_and_split(train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Load dataset, derive speed from speed_band, temporal split (no shuffle)."""
    df = pd.read_csv(PRIMARY_DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["speed_kmh"] = df["speed_band"].map(SPEED_BAND_TO_KMH)
    n = len(df)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    train = df.iloc[:t1]
    val = df.iloc[t1:t2]
    test = df.iloc[t2:]
    logger.info("Temporal split: train=%d val=%d test=%d (no shuffle)", len(train), len(val), len(test))
    return train, val, test


def baseline_historical_average(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Historical Average: predict mean of training target (global or per-road per hour).
    Per-road per hour: for each (road_name, hour) use mean speed in train.
    """
    logger.info("Baseline: Historical Average (per road_name, hour)")
    # Per (road_name, hour) average speed in train
    train_avg = train.groupby(["road_name", "hour"])["speed_kmh"].mean().reset_index()
    train_avg = train_avg.rename(columns={"speed_kmh": "pred"})
    val_merged = val.merge(train_avg, on=["road_name", "hour"], how="left")
    test_merged = test.merge(train_avg, on=["road_name", "hour"], how="left")
    # Fill missing (unseen road/hour) with global train mean
    global_mean = train["speed_kmh"].mean()
    val_merged["pred"] = val_merged["pred"].fillna(global_mean)
    test_merged["pred"] = test_merged["pred"].fillna(global_mean)
    val_mae = (val_merged["speed_kmh"] - val_merged["pred"]).abs().mean()
    val_rmse = np.sqrt(((val_merged["speed_kmh"] - val_merged["pred"]) ** 2).mean())
    test_mae = (test_merged["speed_kmh"] - test_merged["pred"]).abs().mean()
    test_rmse = np.sqrt(((test_merged["speed_kmh"] - test_merged["pred"]) ** 2).mean())
    logger.info("Historical Avg — Val MAE=%.4f RMSE=%.4f | Test MAE=%.4f RMSE=%.4f", val_mae, val_rmse, test_mae, test_rmse)
    return {"model": "historical_average", "val_mae": float(val_mae), "val_rmse": float(val_rmse), "test_mae": float(test_mae), "test_rmse": float(test_rmse)}


def baseline_arima(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    ARIMA: per-road univariate time series. Use simple AR(1) or statsmodels ARIMA if available.
    Fallback: last-value (persistence) if ARIMA not installed.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        has_arima = True
    except ImportError:
        has_arima = False
    if not has_arima:
        logger.info("Baseline: ARIMA skipped (statsmodels not installed); using persistence (last value)")
        # Persistence: predict last known value per road
        last_speed = train.groupby("road_name")["speed_kmh"].last().reindex(val["road_name"].unique()).fillna(train["speed_kmh"].mean())
        val_pred = val["road_name"].map(last_speed).fillna(train["speed_kmh"].mean())
        last_speed_test = train.groupby("road_name")["speed_kmh"].last().reindex(test["road_name"].unique()).fillna(train["speed_kmh"].mean())
        test_pred = test["road_name"].map(last_speed_test).fillna(train["speed_kmh"].mean())
        val_mae = (val["speed_kmh"] - val_pred.values).abs().mean()
        val_rmse = np.sqrt(((val["speed_kmh"] - val_pred.values) ** 2).mean())
        test_mae = (test["speed_kmh"] - test_pred.values).abs().mean()
        test_rmse = np.sqrt(((test["speed_kmh"] - test_pred.values) ** 2).mean())
        return {"model": "persistence", "val_mae": float(val_mae), "val_rmse": float(val_rmse), "test_mae": float(test_mae), "test_rmse": float(test_rmse)}
    # ARIMA per road (order 1,0,0) for simplicity
    results = []
    for road in train["road_name"].unique():
        ts = train[train["road_name"] == road].sort_values("timestamp")["speed_kmh"].values
        if len(ts) < 10:
            continue
        try:
            model = ARIMA(ts, order=(1, 0, 0))
            fit = model.fit()
            v = val[val["road_name"] == road]["speed_kmh"].values
            t = test[test["road_name"] == road]["speed_kmh"].values
            if len(v) > 0:
                pred_v = fit.forecast(steps=len(v))
                results.append(("val", road, v, pred_v))
            if len(t) > 0:
                pred_t = fit.forecast(steps=len(t))
                results.append(("test", road, t, pred_t))
        except Exception as e:
            logger.debug("ARIMA fit failed for %s: %s", road, e)
    if not results:
        return baseline_historical_average(train, val, test)
    val_res = [(r[2], r[3]) for r in results if r[0] == "val"]
    test_res = [(r[2], r[3]) for r in results if r[0] == "test"]
    val_mae = np.mean([np.abs(y - p).mean() for y, p in val_res])
    val_rmse = np.mean([np.sqrt(((y - p) ** 2).mean()) for y, p in val_res])
    test_mae = np.mean([np.abs(y - p).mean() for y, p in test_res])
    test_rmse = np.mean([np.sqrt(((y - p) ** 2).mean()) for y, p in test_res])
    return {"model": "arima", "val_mae": float(val_mae), "val_rmse": float(val_rmse), "test_mae": float(test_mae), "test_rmse": float(test_rmse)}


def run_baselines() -> dict:
    """Run Historical Average and ARIMA; save metrics to BASELINE_METRICS_PATH."""
    logger.info("Baseline models started")
    train, val, test = _load_and_split()
    out = {
        "historical_average": baseline_historical_average(train, val, test),
        "arima_or_persistence": baseline_arima(train, val, test),
    }
    ORCHESTRATOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_METRICS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Baseline metrics saved to %s", BASELINE_METRICS_PATH)
    return out


if __name__ == "__main__":
    run_baselines()
