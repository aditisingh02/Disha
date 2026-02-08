"""
NeuroFlow BharatFlow — Residual GCN Training (q90 only)
Trains with quantile pinball loss (0.9), regime-weighted. Does not optimize global MAE.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

_current_dir = Path(__file__).resolve().parent
_backend_dir = _current_dir.parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.engine.models import ResidualGCN
from app.engine.baseline_forecaster import BaselineForecaster
from app.engine.spatial_features import enrich_spatial_features, build_default_adjacency
from app.utils.scaler import ManualScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neuroflow.train_residual_gcn")

FEATURE_COLS = [
    "speed", "volume", "occupancy",
    "rain_intensity", "weather_severity_index",
    "event_attendance", "holiday_intensity_score",
    "is_peak_hour", "is_weekday",
]
HORIZONS = 48
TIME_STEPS = 24
QUANTILE = 0.9


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, q: float = 0.9) -> torch.Tensor:
    """Quantile pinball loss."""
    e = target - pred
    return (torch.where(e >= 0, q * e, (q - 1) * e)).mean()


class ResidualGCNDataset(Dataset):
    """Input: [N, T, 13] enriched; target: [N, 48] residual (actual - baseline_q50)."""

    def __init__(self, data: np.ndarray, baseline_q50: np.ndarray, residuals: np.ndarray, regime_weights: np.ndarray, num_nodes: int):
        # data: (T_total, N, 9); baseline_q50 and residuals: (num_samples, N, 48); regime_weights: (num_samples,)
        self.data = data
        self.baseline_q50 = baseline_q50
        self.residuals = residuals
        self.regime_weights = regime_weights
        self.num_nodes = num_nodes

    def __len__(self):
        return len(self.residuals)

    def __getitem__(self, idx):
        # Each sample: one (start_time) giving X [N, T, 9], baseline [N, 48], residual [N, 48], weight
        t = idx
        T_total, N, _ = self.data.shape
        if t + TIME_STEPS + HORIZONS > T_total:
            t = max(0, T_total - TIME_STEPS - HORIZONS - 1)
        X = self.data[t : t + TIME_STEPS]
        X = X.transpose(1, 0, 2)
        base = self.baseline_q50[idx]
        res = self.residuals[idx]
        w = self.regime_weights[idx]
        return torch.FloatTensor(X), torch.FloatTensor(base), torch.FloatTensor(res), torch.FloatTensor([w])


def run_baseline_on_sequence(baseline: BaselineForecaster, data_slice: np.ndarray, city: str) -> np.ndarray:
    """data_slice: (T, N, 9). Return (N, 48) q50 using last timestep features."""
    T, N, F = data_slice.shape
    last = data_slice[-1]
    readings = [dict(zip(FEATURE_COLS, last[i].tolist())) for i in range(N)]
    for i in range(N):
        readings[i]["segment_id"] = f"seg_{i}"
    return baseline.predict_q50(readings, city)


def train_residual_gcn_for_city(
    city: str,
    data_dir: Path,
    weights_dir: Path,
    epochs: int = 20,
    batch_size: int = 8,
):
    weights_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{city}_train_v2.csv"
    if not csv_path.exists():
        logger.error(f"Data not found: {csv_path}")
        return
    baseline_path = weights_dir / f"baseline_{city}_v1.pkl"
    if not baseline_path.exists():
        logger.error(f"Baseline not found: {baseline_path}. Run train_baseline.py first.")
        return

    df = pd.read_csv(csv_path)
    df["is_peak_hour"] = df["is_peak_hour"].astype(float)
    df["is_weekday"] = df["is_weekday"].astype(float)
    nodes = sorted(df["segment_id"].unique())
    num_nodes = len(nodes)
    node_map = {n: i for i, n in enumerate(nodes)}
    df["node_idx"] = df["segment_id"].map(node_map)
    df = df.sort_values(["segment_id", "timestamp"])
    df["time_idx"] = df.groupby("segment_id").cumcount()
    max_t = int(df.groupby("segment_id")["time_idx"].max().min())
    if max_t < TIME_STEPS + HORIZONS + 1:
        logger.error("Not enough timesteps")
        return

    data_matrix = np.zeros((max_t, num_nodes, len(FEATURE_COLS)), dtype=np.float32)
    for i, col in enumerate(FEATURE_COLS):
        piv = df.pivot(index="time_idx", columns="node_idx", values=col).fillna(0)
        data_matrix[: len(piv), :, i] = piv.values
    scaler = ManualScaler()
    flat = data_matrix.reshape(-1, len(FEATURE_COLS))
    data_matrix = scaler.fit_transform(flat).reshape(max_t, num_nodes, len(FEATURE_COLS))

    baseline = BaselineForecaster(weights_dir)
    baseline.initialize(city)
    n_samples = max_t - TIME_STEPS - HORIZONS
    baseline_q50_arr = np.zeros((n_samples, num_nodes, HORIZONS), dtype=np.float32)
    target_speed = np.zeros((n_samples, num_nodes, HORIZONS), dtype=np.float32)
    regime_weights = np.ones(n_samples, dtype=np.float32)
    for t in range(n_samples):
        sl = data_matrix[t : t + TIME_STEPS]
        baseline_q50_arr[t] = run_baseline_on_sequence(baseline, sl, city)
        for h in range(HORIZONS):
            target_speed[t, :, h] = data_matrix[t + TIME_STEPS + h, :, 0]
        base_mean = baseline_q50_arr[t].mean()
        is_peak = data_matrix[t + TIME_STEPS - 1, 0, FEATURE_COLS.index("is_peak_hour")] > 0.5
        if is_peak or base_mean < 25:
            regime_weights[t] = 2.0
    residuals = target_speed - baseline_q50_arr

    adj = build_default_adjacency(num_nodes)
    split = int(n_samples * 0.85)
    train_idx = np.arange(split)
    val_idx = np.arange(split, n_samples)
    train_X_list = []
    for t in train_idx:
        X = data_matrix[t : t + TIME_STEPS].transpose(1, 0, 2)
        base_t = baseline_q50_arr[t]
        enriched = enrich_spatial_features(X[:, -1, :], base_t, np.zeros(num_nodes), adj)
        extra = np.broadcast_to(enriched[:, 9:13][:, None, :], (num_nodes, TIME_STEPS, 4))
        X_enriched = np.concatenate([X, extra], axis=-1)
        train_X_list.append(X_enriched)
    train_X = np.array(train_X_list)
    train_res = residuals[train_idx]
    train_base = baseline_q50_arr[train_idx]
    train_w = regime_weights[train_idx]

    class SimpleResidualDataset(Dataset):
        def __init__(self, X, base, res, w):
            self.X, self.base, self.res, self.w = X, base, res, w
        def __len__(self): return len(self.res)
        def __getitem__(self, i):
            return torch.FloatTensor(self.X[i]), torch.FloatTensor(self.base[i]), torch.FloatTensor(self.res[i]), torch.FloatTensor([self.w[i]])

    train_ds = SimpleResidualDataset(train_X, train_base, train_res, train_w)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_X_list = []
    for t in val_idx:
        X = data_matrix[t : t + TIME_STEPS].transpose(1, 0, 2)
        base_t = baseline_q50_arr[t]
        enriched = enrich_spatial_features(X[:, -1, :], base_t, np.zeros(num_nodes), adj)
        extra = np.broadcast_to(enriched[:, 9:13][:, None, :], (num_nodes, TIME_STEPS, 4))
        X_enriched = np.concatenate([X, extra], axis=-1)
        val_X_list.append(X_enriched)
    val_X = np.array(val_X_list)
    val_ds = SimpleResidualDataset(val_X, baseline_q50_arr[val_idx], residuals[val_idx], regime_weights[val_idx])
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualGCN(num_nodes=num_nodes, in_features=13, hidden_dim=64, output_horizons=HORIZONS, temporal_steps=TIME_STEPS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, _base, res, w in train_loader:
            X = X.to(device)
            res = res.to(device)
            w = w.to(device).view(-1, 1, 1)
            optimizer.zero_grad()
            pred = model(X)
            loss = (pinball_loss(pred, res, QUANTILE) * w).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, _base, res, w in val_loader:
                X, res = X.to(device), res.to(device)
                pred = model(X)
                val_loss += pinball_loss(pred, res, QUANTILE).item()
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1} train_loss={train_loss/len(train_loader):.4f} val_pinball={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), weights_dir / f"residual_gcn_{city}_v1.pth")
    logger.info(f"Saved ResidualGCN to {weights_dir / f'residual_gcn_{city}_v1.pth'}")
    joblib.dump(scaler, weights_dir / f"scaler_residual_{city}_v1.pkl")


def main():
    data_dir = _backend_dir / "data" / "datasets_v2"
    if not data_dir.exists():
        data_dir = _backend_dir / "data" / "datasets"
    weights_dir = _backend_dir / "ml_models" / "weights"
    for city in ["bengaluru", "mumbai", "delhi"]:
        if (data_dir / f"{city}_train_v2.csv").exists() and (weights_dir / f"baseline_{city}_v1.pkl").exists():
            try:
                train_residual_gcn_for_city(city, data_dir, weights_dir, epochs=10)
            except Exception as e:
                logger.error(f"train_residual_gcn {city}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
