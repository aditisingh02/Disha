"""
Multi-horizon forecasting (1h, 3h, 6h, 12h, 24h) for PS: "hourly or daily forecasts".
Uses historical (road_name, hour) means from training_dataset_enriched; applies event context.
"""

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.orchestrator.baselines import _load_and_split, SPEED_BAND_TO_KMH
from app.orchestrator.congestion import CongestionClassifier

HORIZONS_HOURS = [1, 3, 6, 12, 24]
DEFAULT_FREE_FLOW_KMH = 80.0


def _event_factor(event_context: str) -> float:
    if event_context == "accident":
        return 0.70
    if event_context == "weather":
        return 0.85
    if event_context == "public_event":
        return 0.80
    return 1.0


def predict_multi_horizon(
    origin: str,
    destination: str,
    departure_time: str,
    event_context: str = "none",
    distance_km: float | None = None,
    free_flow_kmh: float = DEFAULT_FREE_FLOW_KMH,
) -> dict[str, Any]:
    """
    Returns multi-horizon forecasts and congestion classification per horizon.
    Keys: multi_horizon_forecasts (dict by "1h", "3h", ...), congestion_classification (for 1h),
    avg_speed_kmh (1h), model_version.
    """
    train, _, _ = _load_and_split(0.7, 0.15)
    # Per (road_name, hour) mean speed
    road_hour_mean = train.groupby(["road_name", "hour"])["speed_kmh"].mean().reset_index()
    road_hour_mean = road_hour_mean.rename(columns={"speed_kmh": "mean_speed"})
    global_mean = float(train["speed_kmh"].mean())
    factor = _event_factor(event_context)

    try:
        dep_dt = datetime.fromisoformat(departure_time.replace("Z", "+00:00"))
    except Exception:
        dep_dt = datetime.now(timezone.utc)
    current_hour = dep_dt.hour

    classifier = CongestionClassifier(default_free_flow_kmh=free_flow_kmh)
    multi_horizon_forecasts: dict[str, Any] = {}

    for h in HORIZONS_HOURS:
        future_hour = (current_hour + h) % 24
        o_val = road_hour_mean[(road_hour_mean["road_name"] == origin) & (road_hour_mean["hour"] == future_hour)]
        d_val = road_hour_mean[(road_hour_mean["road_name"] == destination) & (road_hour_mean["hour"] == future_hour)]
        o_speed = float(o_val["mean_speed"].iloc[0]) if len(o_val) else global_mean
        d_speed = float(d_val["mean_speed"].iloc[0]) if len(d_val) else global_mean
        speed = (o_speed + d_speed) / 2.0 * factor
        speed = round(speed, 2)
        low_80 = round(speed * 0.85, 2)
        up_80 = round(speed * 1.15, 2)
        cong = classifier.classify_congestion(speed, free_flow_kmh, distance_km)
        multi_horizon_forecasts[f"{h}h"] = {
            "speed_kmh": speed,
            "level": cong["level"],
            "score": cong["score"],
            "delay_vs_freeflow_min": cong.get("delay_vs_freeflow_min"),
            "ci_80_lower": low_80,
            "ci_80_upper": up_80,
        }

    avg_speed_1h = multi_horizon_forecasts["1h"]["speed_kmh"]
    congestion_classification = classifier.classify_congestion(
        avg_speed_1h, free_flow_kmh, distance_km
    )

    return {
        "multi_horizon_forecasts": multi_horizon_forecasts,
        "congestion_classification": congestion_classification,
        "avg_speed_kmh": avg_speed_1h,
        "model_version": "multi_horizon_historical_v1",
        "departure_time": departure_time,
        "horizons_hours": HORIZONS_HOURS,
    }
