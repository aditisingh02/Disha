"""
NeuroFlow BharatFlow — LightGBM Baseline Forecaster (q50)
Predicts median (q50) congestion/speed for 48 horizons using temporal and exogenous features.
No spatial learning. Optimized for MAE/RMSE.
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger("neuroflow.baseline")

# Feature columns matching training (order matters)
BASELINE_FEATURE_COLS = [
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


class BaselineForecaster:
    """
    LightGBM-based baseline that owns q50 (median) prediction per segment.
    Outputs 48-step speed forecast per segment from current readings.
    """

    def __init__(self, weights_dir: str | Path | None = None) -> None:
        self.weights_dir = Path(weights_dir) if weights_dir else Path(__file__).resolve().parent.parent.parent / "ml_models" / "weights"
        self._models: Dict[str, list] = {}  # city -> list of 48 LGBM models
        self._initialized: Dict[str, bool] = {}

    def _reading_to_features(self, r: dict) -> np.ndarray:
        """Extract 9-d feature vector from a single reading."""
        return np.array([
            float(r.get("speed_kmh", r.get("speed", 30.0))),
            float(r.get("volume", 500)),
            float(r.get("occupancy", 0.1)),
            float(r.get("rain_intensity", 0.0)),
            float(r.get("weather_severity_index", 0.0)),
            float(r.get("event_attendance", 0)),
            float(r.get("holiday_intensity_score", 0.0)),
            float(r.get("is_peak_hour", 0.0)),
            float(r.get("is_weekday", 1.0)),
        ], dtype=np.float32)

    def initialize(self, city: str) -> bool:
        """Load 48 LightGBM models for the city. Returns True if loaded."""
        if self._initialized.get(city):
            return True
        path = self.weights_dir / f"baseline_{city}_v1.pkl"
        if not path.exists():
            logger.warning(f"Baseline weights not found: {path}")
            return False
        try:
            import joblib
            payload = joblib.load(path)
            # payload: list of 48 models, or dict with "models" key
            if isinstance(payload, dict):
                models = payload.get("models", payload.get("lgb_models"))
            else:
                models = payload
            if not models or len(models) != HORIZONS:
                logger.warning(f"Baseline {city}: expected {HORIZONS} models, got {len(models) if models else 0}")
                return False
            self._models[city] = models
            self._initialized[city] = True
            logger.info(f"Loaded LightGBM baseline for {city} ({HORIZONS} horizons)")
            return True
        except Exception as e:
            logger.warning(f"Could not load baseline for {city}: {e}")
            return False

    def predict_q50(self, current_readings: List[dict], city: str = "bengaluru") -> np.ndarray:
        """
        Predict q50 (median) speed for 48 horizons for each segment.
        current_readings: list of dicts with segment_id and feature keys.
        Returns: shape (N, 48) float array, one row per reading in same order.
        """
        import pandas as pd
        import warnings
        
        if not current_readings:
            return np.zeros((0, HORIZONS), dtype=np.float32)
        if not self.initialize(city):
            # Fallback: repeat current speed for all horizons
            out = np.zeros((len(current_readings), HORIZONS), dtype=np.float32)
            for i, r in enumerate(current_readings):
                s = float(r.get("speed_kmh", r.get("speed", 30.0)))
                out[i, :] = s
            return out
        models = self._models[city]
        X = np.array([self._reading_to_features(r) for r in current_readings], dtype=np.float32)
        
        # Convert to DataFrame with feature names to suppress LightGBM warning
        X_df = pd.DataFrame(X, columns=BASELINE_FEATURE_COLS)
        
        # Predict each horizon with warnings suppressed
        preds = np.zeros((X.shape[0], HORIZONS), dtype=np.float32)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            for h in range(HORIZONS):
                preds[:, h] = models[h].predict(X_df)
        return preds

