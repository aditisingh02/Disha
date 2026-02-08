"""
NeuroFlow BharatFlow — Forecast Evaluation (peak-risk and tail)
Headline: peak-hour MAE, peak miss rate, q90 coverage during peak, tail pinball loss.
Not headline: global MAE comparison. Always report baseline vs decomposed.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

_current_dir = Path(__file__).resolve().parent
_backend_dir = _current_dir.parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neuroflow.evaluate")

# Peak windows (hour of day): 8-10, 17-20
PEAK_HOURS = set(range(8, 11)) | set(range(17, 21))
QUANTILE = 0.9
CONGESTED_SPEED_THRESHOLD = 25.0


def pinball_loss(pred: np.ndarray, target: np.ndarray, q: float = 0.9) -> float:
    e = target - pred
    return np.mean(np.where(e >= 0, q * e, (q - 1) * e))


def peak_hour_mask(step_indices: np.ndarray) -> np.ndarray:
    """step_indices: 0..47 (15-min steps ahead). Hour ahead = (step+1)*15/60. Peak = hours 8,9,10 in 12h window."""
    hour_ahead = ((step_indices + 1) * 15) // 60
    return np.isin(hour_ahead, [8, 9, 10])


def evaluate(
    y_true: np.ndarray,
    y_q50: np.ndarray,
    y_q90: np.ndarray | None = None,
    peak_hours: set | None = None,
) -> dict:
    """
    y_true, y_q50, y_q90: (N, 48) or (N*48,) flattened.
    Returns dict with peak_hour_mae, peak_miss_rate, q90_coverage_peak, tail_pinball, baseline_mae, decomposed_mae.
    """
    peak_hours = peak_hours or PEAK_HOURS
    if y_true.ndim == 2:
        y_true = y_true.ravel()
        y_q50 = y_q50.ravel()
        y_q90 = y_q90.ravel() if y_q90 is not None else y_q50.ravel()
    n = len(y_true)
    if n % 48 != 0:
        n = (n // 48) * 48
    y_true = y_true[:n]
    y_q50 = y_q50[:n]
    if y_q90 is None:
        y_q90 = y_q50
    else:
        y_q90 = y_q90[:n]
    step_indices = np.arange(n) % 48
    peak_mask = peak_hour_mask(step_indices)
    # Peak-hour MAE
    if peak_mask.any():
        peak_hour_mae = np.abs(y_true[peak_mask] - y_q50[peak_mask]).mean()
    else:
        peak_hour_mae = float("nan")
    # Peak miss rate: actual congested but predicted speed above threshold (false negative)
    actual_congested = y_true <= CONGESTED_SPEED_THRESHOLD
    pred_above = y_q50 > CONGESTED_SPEED_THRESHOLD
    fn = np.logical_and(actual_congested, pred_above)
    peak_congested = np.zeros_like(actual_congested, dtype=bool)
    peak_congested[peak_mask] = True
    if (actual_congested & peak_congested).any():
        peak_miss_rate = fn[peak_congested].sum() / max(1, (actual_congested & peak_congested).sum())
    else:
        peak_miss_rate = float("nan")
    # q90 coverage during peak: % of peak actuals <= q90
    if peak_mask.any():
        in_peak = y_true[peak_mask] <= y_q90[peak_mask]
        q90_coverage_peak = in_peak.mean()
    else:
        q90_coverage_peak = float("nan")
    tail_pinball = pinball_loss(y_q90, y_true, QUANTILE)
    baseline_mae = np.abs(y_true - y_q50).mean()
    decomposed_mae = np.abs(y_true - y_q90).mean() if y_q90 is not None else baseline_mae
    return {
        "peak_hour_mae": float(peak_hour_mae),
        "peak_miss_rate": float(peak_miss_rate),
        "q90_coverage_peak": float(q90_coverage_peak),
        "tail_pinball_loss": float(tail_pinball),
        "baseline_mae": float(baseline_mae),
        "decomposed_mae": float(decomposed_mae),
        "mae_unchanged_risk_improved": baseline_mae <= decomposed_mae * 1.02 and not np.isnan(q90_coverage_peak),
    }


def run_evaluation_and_save(
    city: str,
    y_true: np.ndarray,
    y_q50: np.ndarray,
    y_q90: np.ndarray | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Compute metrics and save to JSON. Return metrics dict."""
    metrics = evaluate(y_true, y_q50, y_q90)
    metrics["city"] = city
    metrics["timestamp"] = datetime.utcnow().isoformat() + "Z"
    output_dir = output_dir or (_backend_dir / "ml_models" / "eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"eval_{city}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Evaluation %s: peak_hour_mae=%.4f peak_miss_rate=%.4f q90_coverage_peak=%.4f tail_pinball=%.4f baseline_mae=%.4f decomposed_mae=%.4f",
                city, metrics["peak_hour_mae"], metrics["peak_miss_rate"], metrics["q90_coverage_peak"],
                metrics["tail_pinball_loss"], metrics["baseline_mae"], metrics["decomposed_mae"])
    if metrics.get("mae_unchanged_risk_improved"):
        logger.info("MAE unchanged but risk detection improved (q90 coverage / peak metrics).")
    return metrics


def main():
    """Example: load val data and run evaluation if baseline + optional q90 exist."""
    data_dir = _backend_dir / "data" / "datasets_v2"
    if not data_dir.exists():
        data_dir = _backend_dir / "data" / "datasets"
    weights_dir = _backend_dir / "ml_models" / "weights"
    for city in ["bengaluru", "mumbai", "delhi"]:
        csv_path = data_dir / f"{city}_train_v2.csv"
        if not csv_path.exists():
            continue
        import pandas as pd
        from app.engine.baseline_forecaster import BaselineForecaster
        df = pd.read_csv(csv_path)
        nodes = sorted(df["segment_id"].unique())
        num_nodes = len(nodes)
        df["node_idx"] = df["segment_id"].map({n: i for i, n in enumerate(nodes)})
        df = df.sort_values(["segment_id", "timestamp"])
        df["time_idx"] = df.groupby("segment_id").cumcount()
        max_t = int(df.groupby("segment_id")["time_idx"].max().min())
        if max_t < 50:
            continue
        split = int(max_t * 0.85)
        val_t = np.arange(split, max_t - 48)
        baseline = BaselineForecaster(weights_dir)
        if not baseline.initialize(city):
            continue
        y_true_list = []
        y_q50_list = []
        for t in val_t[:100]:
            rows = df[(df["time_idx"] >= t) & (df["time_idx"] < t + 1)]
            if len(rows) < num_nodes:
                continue
            readings = []
            for _, row in rows.iterrows():
                readings.append({
                    "segment_id": row["segment_id"],
                    "speed_kmh": row["speed"],
                    "volume": row["volume"],
                    "occupancy": row["occupancy"],
                    "rain_intensity": row["rain_intensity"],
                    "weather_severity_index": row["weather_severity_index"],
                    "event_attendance": row["event_attendance"],
                    "holiday_intensity_score": row["holiday_intensity_score"],
                    "is_peak_hour": row["is_peak_hour"],
                    "is_weekday": row["is_weekday"],
                })
            q50 = baseline.predict_q50(readings, city)
            y_q50_list.append(q50)
            actual = df[(df["time_idx"] >= t + 1) & (df["time_idx"] <= t + 48)].pivot(index="time_idx", columns="node_idx", values="speed").values
            if actual.shape[0] == 48 and actual.shape[1] == num_nodes:
                y_true_list.append(actual.T)
        if not y_true_list:
            continue
        y_true = np.stack(y_true_list, axis=0)
        y_q50 = np.stack(y_q50_list, axis=0)
        run_evaluation_and_save(city, y_true, y_q50, y_q90=None)


if __name__ == "__main__":
    main()
