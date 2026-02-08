"""
NeuroFlow BharatFlow — Regime Gating for Residual GCN
Activates residual model only during abnormal/peak congestion regimes.
"""

import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger("neuroflow.regime")

# Default thresholds when no config is loaded
DEFAULT_WEATHER_SEVERITY_THRESHOLD = 3.0
DEFAULT_HOLIDAY_FLAG_THRESHOLD = 0.5


def load_corridor_75p(weights_dir: Path, city: str) -> float | None:
    """Load corridor 75th percentile (of congestion / low speed) for city. Returns None if missing."""
    path = weights_dir / f"corridor_75p_{city}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return float(data.get("q50_75p_speed", data.get("speed_75p", 0.0)))
    except Exception as e:
        logger.warning(f"Could not load corridor 75p for {city}: {e}")
        return None


def save_corridor_75p(weights_dir: Path, city: str, q50_75p_speed: float) -> None:
    """Persist corridor 75th percentile (speed below which we consider high congestion)."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    path = weights_dir / f"corridor_75p_{city}.json"
    with open(path, "w") as f:
        json.dump({"q50_75p_speed": q50_75p_speed, "city": city}, f, indent=2)
    logger.info(f"Saved corridor 75p for {city}: q50_75p_speed={q50_75p_speed:.2f}")


def regime_gate_active(
    baseline_q50_speeds: List[float] | None,
    readings: List[dict],
    city: str,
    weights_dir: Path | None = None,
    weather_threshold: float = DEFAULT_WEATHER_SEVERITY_THRESHOLD,
    holiday_threshold: float = DEFAULT_HOLIDAY_FLAG_THRESHOLD,
) -> bool:
    """
    Return True if residual model should be activated.
    Activates when:
      - any baseline_q50 (median speed) <= corridor_specific_75th_percentile (i.e. high congestion), OR
      - weather_severity_index > threshold, OR
      - holiday_flag == true (holiday_intensity_score > holiday_threshold).
    """
    if not readings:
        return False
    # Holiday / weather from first reading (same for whole batch typically)
    r0 = readings[0]
    weather_severity = float(r0.get("weather_severity_index", 0.0))
    holiday_score = float(r0.get("holiday_intensity_score", 0.0))
    if weather_severity > weather_threshold:
        logger.debug(f"Regime gate active: weather_severity={weather_severity} > {weather_threshold}")
        return True
    if holiday_score > holiday_threshold:
        logger.debug(f"Regime gate active: holiday_intensity_score={holiday_score} > {holiday_threshold}")
        return True
    # Corridor 75p: baseline q50 below 75th percentile speed means high congestion
    if baseline_q50_speeds:
        weights_dir = weights_dir or Path(__file__).resolve().parent.parent / "ml_models" / "weights"
        q50_75p = load_corridor_75p(weights_dir, city)
        if q50_75p is not None:
            # 75th percentile stored as speed threshold: when median speed is below this, corridor is congested
            if any(s <= q50_75p for s in baseline_q50_speeds):
                logger.debug(f"Regime gate active: baseline_q50 below 75p threshold {q50_75p}")
                return True
    return False
