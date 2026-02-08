"""
Phase 2 — Core forecasting: Multi-horizon ST-GCN (spatial GCN + temporal LSTM).
Trained on training_dataset_enriched only; temporal split; no shuffle.
Validation: must outperform baselines by >=30% on test (test MAE).
Outputs: model artifact, test metrics, leakage_check flag.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from app.orchestrator.config import (
    PRIMARY_DATASET_PATH,
    ORCHESTRATOR_OUTPUT_DIR,
    BASELINE_METRICS_PATH,
)
from app.orchestrator.baselines import SPEED_BAND_TO_KMH, _load_and_split

logger = logging.getLogger("neuroflow.orchestrator.phase2")

HORIZONS = [1, 6, 24]  # 1hr, 6hr, 24hr ahead (hourly steps)
SEQ_LEN = 24
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
BASELINE_IMPROVEMENT_THRESHOLD = 0.30  # must be >=30% better (lower MAE)


class SimpleGCNLSTM(nn.Module):
    """Spatial: shared linear (simplified GCN); Temporal: LSTM. In: [B, N, T, F]; Out: [B, N, n_horizons]."""

    def __init__(self, num_nodes: int, in_features: int, hidden: int = 64, n_horizons: int = 3):
        super().__init__()
        self.n_horizons = n_horizons
        self.spatial = nn.Linear(in_features, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_horizons)

    def forward(self, x):
        # x: [B, N, T, F]
        B, N, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * T, N, F)
        h = torch.relu(self.spatial(x))
        h = h.reshape(B, T, N, -1).permute(0, 2, 1, 3)
        h = h.reshape(B * N, T, -1)
        out, _ = self.lstm(h)
        out = out[:, -1]
        out = self.out(out)
        out = out.reshape(B, N, self.n_horizons)
        return out


def _build_sequences(df: pd.DataFrame, road_names: list, seq_len: int, scaler_X=None, scaler_Y=None, fit=False) -> tuple:
    """
    Pivot to (time, road) and build X [samples, N, T, F], Y [samples, N, n_horizons].
    Apply scaling if scalers provided.
    """
    road_to_idx = {r: i for i, r in enumerate(road_names)}
    df = df.sort_values(["timestamp", "road_name"])
    
    # Feature columns
    feat_cols = ["hour", "day_of_week", "is_peak_hour", "weather_severity_index", "rain_intensity", "speed_kmh"]
    avail = [c for c in feat_cols if c in df.columns]
    if "speed_kmh" not in avail:
        avail.append("speed_kmh")
        
    ts_unique = sorted(df["timestamp"].unique())
    n_t = len(ts_unique)
    n_n = len(road_names)
    n_f = len(avail)
    
    # Build full matrix [T, N, F]
    mat = np.zeros((n_t, n_n, n_f), dtype=np.float32)
    for i, t in enumerate(ts_unique):
        rows = df[df["timestamp"] == t]
        for _, r in rows.iterrows():
            j = road_to_idx.get(r["road_name"])
            if j is not None:
                for k, col in enumerate(avail):
                    val = r[col]
                    mat[i, j, k] = float(val) if pd.notna(val) else 0.0

    # Identify speed column index for Y
    speed_col_idx = avail.index("speed_kmh") if "speed_kmh" in avail else -1

    # Flatten for scaling: [T * N, F]
    mat_flat = mat.reshape(-1, n_f)
    
    # Scale Inputs (X)
    if fit:
        scaler_X = StandardScaler()
        mat_flat = scaler_X.fit_transform(mat_flat)
    elif scaler_X:
        mat_flat = scaler_X.transform(mat_flat)
        
    mat = mat_flat.reshape(n_t, n_n, n_f)

    # Scale Targets (Y) - we need a separate scaler for just speed to inverse transform easily
    # We extract speed, scale it, and use it for Y generation
    # But for Y construction, we need the raw or scaled values?
    # Usually we predict scaled values, then inverse transform.
    # So we need to scale the speed column specifically for Y.
    
    speed_flat = mat[:, :, speed_col_idx].reshape(-1, 1) # This is already scaled if X was scaled
    # Actually, simpler: Use X scaler for everything since speed is in X.
    # But we need inverse_transform for just Y (speed). 
    # StandardScaler scales column-wise. We can inverse transform just the speed column using mean_[speed_idx] and scale_[speed_idx]
    
    max_start = n_t - seq_len - max(HORIZONS)
    if max_start < 1:
        return np.zeros((0, n_n, seq_len, n_f)), np.zeros((0, n_n, len(HORIZONS))), scaler_X

    X_list, Y_list = [], []
    for start in range(0, max_start, 4):  # stride 4 for efficiency
        # X: [N, T, F]
        chunk = mat[start : start + seq_len]
        chunk = np.transpose(chunk, (1, 0, 2))  # [N, T, F]
        X_list.append(chunk)
        
        # Y: [N, n_horizons]
        y = np.zeros((n_n, len(HORIZONS)), dtype=np.float32)
        for hi, h in enumerate(HORIZONS):
            end_idx = min(start + seq_len + h - 1, n_t - 1)
            y[:, hi] = mat[end_idx, :, speed_col_idx]
        Y_list.append(y)

    return np.stack(X_list), np.stack(Y_list), scaler_X


def run_phase2(epochs: int = 30, device: str = "cpu") -> dict:
    """Train GCN-LSTM with scaling; validate on test; check >=30% improvement; save metrics."""
    logger.info("Phase 2 forecasting started (with StandardScaler)")
    if not PRIMARY_DATASET_PATH.exists():
        raise FileNotFoundError(f"Primary dataset not found: {PRIMARY_DATASET_PATH}")

    train_df, val_df, test_df = _load_and_split(TRAIN_RATIO, VAL_RATIO)
    road_names = sorted(train_df["road_name"].unique().tolist())
    num_nodes = len(road_names)
    feat_cols = ["hour", "day_of_week", "is_peak_hour", "weather_severity_index", "rain_intensity", "speed_kmh"]
    in_features = len(feat_cols)
    speed_idx = feat_cols.index("speed_kmh")

    # 1. Prepare Data with Scaling
    X_train, Y_train, scaler = _build_sequences(train_df, road_names, SEQ_LEN, fit=True)
    X_val, Y_val, _ = _build_sequences(val_df, road_names, SEQ_LEN, scaler_X=scaler)
    X_test, Y_test, _ = _build_sequences(test_df, road_names, SEQ_LEN, scaler_X=scaler)

    if X_train.size == 0 or X_test.size == 0:
        logger.warning("Empty sequences; skipping Phase 2 training")
        return {"phase2": "skipped", "reason": "empty_sequences"}

    # Save scaler for inference
    ORCHESTRATOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = ORCHESTRATOR_OUTPUT_DIR / "phase2_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved to %s", scaler_path)

    # Leakage check
    train_max_ts = train_df["timestamp"].max()
    test_min_ts = test_df["timestamp"].min()
    leakage_check = train_max_ts < test_min_ts
    logger.info("Leakage check (train_max < test_min): %s", leakage_check)

    X_train_t = torch.FloatTensor(X_train).to(device)
    Y_train_t = torch.FloatTensor(Y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    Y_val_t = torch.FloatTensor(Y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    Y_test_t = torch.FloatTensor(Y_test).to(device)

    model = SimpleGCNLSTM(num_nodes, in_features, hidden=64, n_horizons=len(HORIZONS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.002) # Slightly higher LR for scaled data
    criterion = nn.MSELoss() # MSE is often better for regression with scaled data
    
    best_val_mae = float("inf")
    best_state = None
    
    # Helper to inverse transform speed
    def inverse_transform_speed(y_scaled_tensor):
        y_np = y_scaled_tensor.cpu().numpy()
        # manual inverse: y * scale + mean
        mean_speed = scaler.mean_[speed_idx]
        scale_speed = scaler.scale_[speed_idx]
        return y_np * scale_speed + mean_speed

    logger.info(f"Training on {len(X_train)} samples...")

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(X_train_t.size(0))
        for i in range(0, len(idx), 32):
            b = idx[i : i + 32]
            pred = model(X_train_t[b])
            loss = criterion(pred, Y_train_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            # Calculate MAE in real units (km/h) for verification
            val_pred_real = inverse_transform_speed(val_pred)
            val_target_real = inverse_transform_speed(Y_val_t)
            val_mae = np.mean(np.abs(val_pred_real - val_target_real))
            
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (ep + 1) % 5 == 0:
            logger.info("Epoch %d val_mae=%.4f km/h", ep + 1, val_mae)

    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t)
        test_pred_real = inverse_transform_speed(test_pred)
        test_target_real = inverse_transform_speed(Y_test_t)
        
        test_mae = np.mean(np.abs(test_pred_real - test_target_real))
        test_rmse = np.sqrt(np.mean((test_pred_real - test_target_real) ** 2))

    # Load baseline test MAE
    baseline_test_mae = 11.76
    if BASELINE_METRICS_PATH.exists():
        with open(BASELINE_METRICS_PATH) as f:
            bm = json.load(f)
        baseline_test_mae = bm.get("historical_average", {}).get("test_mae", baseline_test_mae)
        
    improvement = (baseline_test_mae - test_mae) / baseline_test_mae if baseline_test_mae else 0
    meets_30 = improvement >= BASELINE_IMPROVEMENT_THRESHOLD
    logger.info("Test MAE=%.4f baseline=%.4f improvement=%.2f%% meets_30=%s", test_mae, baseline_test_mae, improvement * 100, meets_30)

    out = {
        "phase2": "completed",
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "baseline_test_mae": float(baseline_test_mae),
        "improvement_ratio": float(improvement),
        "meets_30_percent_improvement": bool(meets_30),
        "leakage_check_passed": bool(leakage_check),
        "horizons_hours": HORIZONS,
    }
    path_metrics = ORCHESTRATOR_OUTPUT_DIR / "phase2_metrics.json"
    with open(path_metrics, "w") as f:
        json.dump(out, f, indent=2)
    if best_state:
        torch.save(best_state, ORCHESTRATOR_OUTPUT_DIR / "phase2_model.pt")
    logger.info("Phase 2 metrics saved to %s", path_metrics)
    return out


if __name__ == "__main__":
    run_phase2(epochs=50)  # Increase epochs slightly for convergence
