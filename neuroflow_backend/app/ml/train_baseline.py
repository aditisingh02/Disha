"""
NeuroFlow BharatFlow — LightGBM Baseline Training (q50)
Trains per-horizon LightGBM models on synthetic data with chronological split.
Optimizes MAE. Persists baseline_mae_reference for safeguard.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure backend on path
_current_dir = Path(__file__).resolve().parent
_backend_dir = _current_dir.parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neuroflow.train_baseline")

FEATURE_COLS = [
    "speed",
    "volume",
    "occupancy",
    "rain_intensity",
    "weather_severity_index",
    "event_attendance",
    "holiday_intensity_score",
    "is_peak_hour",
    "is_weekday",
]
HORIZONS = 48


def load_and_build_xy(csv_path: Path):
    """Load CSV, build X (features per segment per time) and Y (48 future speeds). Chronological."""
    df = pd.read_csv(csv_path)
    df["is_peak_hour"] = df["is_peak_hour"].astype(float)
    df["is_weekday"] = df["is_weekday"].astype(float)
    nodes = sorted(df["segment_id"].unique())
    # Per-segment time index (chronological)
    df = df.sort_values(["segment_id", "timestamp"])
    df["time_idx"] = df.groupby("segment_id").cumcount()
    max_t = int(df.groupby("segment_id")["time_idx"].max().min())
    if max_t < HORIZONS + 1:
        raise ValueError("Not enough timesteps for 48-horizon targets")
    rows = []
    for seg_id in nodes:
        seg_df = df[df["segment_id"] == seg_id].sort_values("time_idx").reset_index(drop=True)
        for t in range(max_t - HORIZONS):
            if t + 1 + HORIZONS > len(seg_df):
                break
            x = seg_df.loc[seg_df.index[t], FEATURE_COLS].values.astype(np.float32)
            y = seg_df.loc[seg_df.index[t + 1 : t + 1 + HORIZONS], "speed"].values.astype(np.float32)
            if len(y) != HORIZONS:
                continue
            rows.append((x, y))
    if not rows:
        raise ValueError("No valid (X, Y) rows")
    X = np.array([r[0] for r in rows], dtype=np.float32)
    Y = np.array([r[1] for r in rows], dtype=np.float32)
    return X, Y, max_t


def train_baseline_for_city(
    city: str,
    data_dir: Path,
    weights_dir: Path,
    train_fraction: float = 0.85,
):
    """Train 48 LightGBM models; save weights and MAE reference."""
    import joblib
    import lightgbm as lgb

    csv_path = data_dir / f"{city}_train_v2.csv"
    if not csv_path.exists():
        logger.error(f"Data not found: {csv_path}")
        return
    logger.info(f"Loading {csv_path}...")
    X, Y, valid_t = load_and_build_xy(csv_path)
    n = len(X)
    split = int(n * train_fraction)
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    logger.info(f"Train {X_train.shape[0]}, Val {X_val.shape[0]}")

    models = []
    for h in range(HORIZONS):
        m = lgb.LGBMRegressor(
            objective="regression_l1",
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=100,
            verbosity=-1,
            random_state=42,
        )
        m.fit(X_train, Y_train[:, h])
        models.append(m)

    # Validation MAE (overall)
    val_pred = np.zeros_like(Y_val)
    for h in range(HORIZONS):
        val_pred[:, h] = models[h].predict(X_val)
    val_mae = np.abs(val_pred - Y_val).mean()
    logger.info(f"Val MAE: {val_mae:.4f}")

    weights_dir.mkdir(parents=True, exist_ok=True)
    out_path = weights_dir / f"baseline_{city}_v1.pkl"
    joblib.dump({"models": models}, out_path)
    logger.info(f"Saved baseline to {out_path}")

    # Corridor 75th percentile: speed below which we consider high congestion (75th %ile congestion = 25th %ile speed)
    q50_75p_speed = float(np.percentile(Y_train, 25))
    from app.engine.regime import save_corridor_75p
    save_corridor_75p(weights_dir, city, q50_75p_speed)

    # Safeguard: save MAE reference
    from app.ml.check_baseline_mae import save_reference, check_baseline_mae
    save_reference(weights_dir, city, float(val_mae), {"horizons": HORIZONS})
    if not check_baseline_mae(city, float(val_mae), weights_dir):
        logger.error("Baseline MAE regression exceeded 2%; check failed.")
        raise SystemExit(1)


def main():
    data_dir = _backend_dir / "data" / "datasets_v2"
    if not data_dir.exists():
        data_dir = _current_dir.parent.parent / "data" / "datasets"
    weights_dir = _backend_dir / "ml_models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    for city in ["bengaluru", "mumbai", "delhi"]:
        if (data_dir / f"{city}_train_v2.csv").exists():
            try:
                train_baseline_for_city(city, data_dir, weights_dir)
            except Exception as e:
                logger.error(f"Train baseline {city}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
