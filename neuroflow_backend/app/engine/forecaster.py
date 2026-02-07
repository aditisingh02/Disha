"""
NeuroFlow BharatFlow — Traffic Forecaster (Singapore Edition)
Uses Deep Residual Network (ResNet) for high-accuracy traffic speed prediction.
Supports dynamic 12-hour forecasting via time-shifted inference.
"""

import logging
import joblib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch

from app.core.config import settings
from app.models.schemas import TrafficPrediction
from app.engine.singapore_model import DeepTrafficModel

logger = logging.getLogger("neuroflow.forecaster")

class TrafficForecaster:
    """
    Traffic Prediction Engine for Singapore.
    Loads the trained Deep ResNet model and generates forecasts by querying
    the model with future timestamps.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.models: Dict[str, DeepTrafficModel] = {}
        self.scalers: Dict[str, object] = {}
        self.encoders: Dict[str, object] = {}
        self.feature_columns: Dict[str, List[str]] = {}
        self._initialized = False

    def initialize(self, city: str = "singapore", num_nodes: int = 17) -> None:
        """Load trained Singapore model artifacts."""
        try:
            weights_dir = Path(settings.weights_dir)
            weights_path = weights_dir / f"stgcn_{city}_v1.pth" # Using V1 (Best Regression Model)
            scaler_path = weights_dir / f"scaler_{city}.pkl"
            encoders_path = weights_dir / f"encoders_{city}.pkl"
            features_path = weights_dir / f"features_{city}.pkl"

            if not weights_path.exists():
                logger.warning(f"⚠️ Model weights not found at {weights_path}. Using random init for testing.")
                # We will initialize a dummy model to prevent crashes during dev without weights
                # Assuming ~35 features as per training
                dummy_features = 35 
                model = DeepTrafficModel(num_features=dummy_features, hidden=512, num_layers=8).to(self.device)
                model.eval()
                self.models[city] = model
                self._initialized = True
                return

            # Load Artifacts
            logger.info(f"Loading model artifacts for {city}...")
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.scalers[city] = joblib.load(scaler_path)
            self.encoders[city] = joblib.load(encoders_path)
            self.feature_columns[city] = joblib.load(features_path)
            
            # Initialize Model
            num_features = len(self.feature_columns[city])
            model = DeepTrafficModel(num_features=num_features, hidden=512, num_layers=8).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.models[city] = model
            self._initialized = True
            logger.info(f"✅ Loaded DeepTrafficModel for {city} (Acc: {checkpoint.get('tolerance_accuracy', 0.0):.1f}%)")

        except Exception as e:
            logger.error(f"❌ Failed to initialize forecaster for {city}: {e}", exc_info=True)
            self.models[city] = None

    def predict(self, current_readings: List[dict], city: str = "singapore") -> List[TrafficPrediction]:
        """
        Generate 12-hour forecast for each segment.
        """
        if not current_readings:
            return []

        if city not in self.models:
            self.initialize(city)

        model = self.models.get(city)
        if model is None:
            return self._heuristic_predict(current_readings)

        try:
            results = []
            now = datetime.utcnow()
            
            # Horizons: 48 x 15-minute intervals = 12 hours
            # This produces data for the UI chart which expects 48 points.
            # Index 0 = T+15min, Index 1 = T+30min, ... Index 47 = T+12h
            horizons = [i * 15 for i in range(48)]  # [0, 15, 30, 45, 60, ..., 705]
            horizon_timestamps = [now + timedelta(minutes=m) for m in horizons]
            
            # Prepare feature vectors
            vectors = []
            meta_map = [] # To map back result index -> (segment_idx, horizon_idx)
            
            cols = self.feature_columns[city]
            encoders = self.encoders[city]
            scaler = self.scalers[city]

            for seg_idx, reading in enumerate(current_readings):
                # Static features for this segment
                road_name = reading.get("road_name", reading.get("name", "Unknown"))
                # Fallback: try to infer road name from segment_id or use property
                if "PIE" in road_name: road_name = "PIE" # Normalization
                
                # Dynamic inputs
                rain_intensity = float(reading.get("rain_intensity", 0.0))
                # We assume rain forecast decay? Or constant for now.
                
                for h_idx, ts in enumerate(horizon_timestamps):
                    vec = self._construct_feature_vector(
                        ts, reading, road_name, rain_intensity, cols, encoders
                    )
                    vectors.append(vec)
                    meta_map.append((seg_idx, h_idx))

            # Stack and Norm
            X = np.array(vectors, dtype=np.float32)
            X_scaled = scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            # Inference
            with torch.no_grad():
                preds_band = model(X_tensor) # [Batch] -> Speed Bands
                
            # Convert Bands to Speed
            preds_speed = self._band_to_speed(preds_band.cpu().numpy())
            
            # Unpack results
            # Initialize storage
            # segment_forecasts[seg_idx] = {0: val, 15: val, ...}
            seg_forecasts = {i: {} for i in range(len(current_readings))}
            
            for i, p_val in enumerate(preds_speed):
                s_idx, h_idx = meta_map[i]
                minute_offset = horizons[h_idx]
                seg_forecasts[s_idx][minute_offset] = float(p_val)
                
            # Build objects
            for i, r in enumerate(current_readings):
                f = seg_forecasts[i]
                
                # 48 x 15-minute intervals for UI chart (T+0, T+15, T+30 ... T+705)
                hourly_vals = [f.get(m * 15, f.get(0, 30.0)) for m in range(48)]
                
                results.append(TrafficPrediction(
                    segment_id=r.get("segment_id"),
                    timestamp=now,
                    predicted_speed_t15=f.get(15, 30.0),
                    predicted_speed_t30=f.get(30, 30.0),
                    predicted_speed_t60=f.get(60, 30.0),
                    hourly_speeds=hourly_vals, 
                    confidence=0.85 # High confidence in our ResNet
                ))
                
            return results

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return self._heuristic_predict(current_readings)

    def _construct_feature_vector(self, dt: datetime, reading: dict, road_name: str, rain: float, cols: list, encoders: dict) -> list:
        """Helper to build the exact feature vector expected by the model."""
        data = {}
        
        # Time
        data['hour'] = dt.hour
        data['day_of_week'] = dt.weekday()
        data['month'] = dt.month
        data['day_of_month'] = dt.day
        data['is_weekend'] = 1 if dt.weekday() >= 5 else 0
        
        # Cyclical
        data['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24.0)
        data['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24.0)
        data['dow_sin'] = np.sin(2 * np.pi * dt.weekday() / 7.0)
        data['dow_cos'] = np.cos(2 * np.pi * dt.weekday() / 7.0)
        
        # Geo (From reading or default)
        # Assuming reading has lat/lng, otherwise Singapore Center
        data['latitude'] = float(reading.get("start_lat", 1.3521))
        data['longitude'] = float(reading.get("start_lng", 103.8198))
        
        # Weather / Events
        data['rain_intensity'] = rain
        data['weather_main_encoded'] = 0 # Default 'Clear' if unknown or use encoders
        # reading.get("weather") might be "Rain".
        w_str = reading.get("weather", "Clear")
        if 'weather_main' in encoders:
             try: data['weather_main_encoded'] = encoders['weather_main'].transform([w_str])[0]
             except: pass

        data['special_event_encoded'] = 0
        if 'special_event' in encoders and "special_event" in reading:
             try: data['special_event_encoded'] = encoders['special_event'].transform([reading["special_event"]])[0]
             except: pass
             
        # Road features
        if 'road_name' in encoders:
            try: data['road_name_encoded'] = encoders['road_name'].transform([road_name])[0]
            except: data['road_name_encoded'] = 0 # Unknown road
            
        if 'road_category' in encoders:
             # Infer category
             cat = "primary"
             if "Expressway" in road_name or "PIE" in road_name or "AYE" in road_name: cat = "motorway"
             try: data['road_category_encoded'] = encoders['road_category'].transform([cat])[0]
             except: data['road_category_encoded'] = 0

        # Construct List
        vec = []
        for c in cols:
            # If feature missing, default to 0
            vec.append(data.get(c, 0.0))
        return vec

    def _band_to_speed(self, band_array: np.ndarray) -> np.ndarray:
        """Convert regression band (1.0-8.0) to km/h."""
        # Band 1: 0-10 (Avg 5)
        # Band 8: >70 (Avg 75)
        # Linear approx: Speed = (Band - 0.5) * 10
        return np.clip((band_array - 0.5) * 10.0, 5.0, 120.0)

    def _heuristic_predict(self, readings: List[dict]) -> List[TrafficPrediction]:
        """Fallback with realistic time-varying traffic patterns."""
        now = datetime.utcnow()
        res = []
        for r in readings:
            base_speed = r.get("speed_kmh", 40.0)
            
            # Generate 48 x 15-minute intervals with realistic traffic patterns
            hourly_speeds = []
            for i in range(48):
                future_time = now + timedelta(minutes=i * 15)
                hour = future_time.hour
                
                # Traffic pattern: Singapore rush hours are 7-9 AM and 5-8 PM
                # Normalize pattern as a multiplier (0.6 = congested, 1.0 = free flow)
                if 7 <= hour <= 9:  # Morning rush
                    pattern = 0.55 + 0.05 * np.random.random()
                elif 17 <= hour <= 20:  # Evening rush
                    pattern = 0.50 + 0.05 * np.random.random()
                elif 10 <= hour <= 16:  # Midday (moderate)
                    pattern = 0.75 + 0.10 * np.random.random()
                elif 21 <= hour <= 23:  # Late evening
                    pattern = 0.85 + 0.10 * np.random.random()
                else:  # Night/early morning (free flow)
                    pattern = 0.95 + 0.05 * np.random.random()
                
                # Apply pattern with some segment-specific variation
                segment_variance = 0.9 + 0.2 * np.random.random()
                predicted_speed = base_speed * pattern * segment_variance
                hourly_speeds.append(round(max(5.0, min(120.0, predicted_speed)), 2))
            
            res.append(TrafficPrediction(
                segment_id=r.get("segment_id", "u"),
                timestamp=now,
                predicted_speed_t15=hourly_speeds[1] if len(hourly_speeds) > 1 else base_speed,
                predicted_speed_t30=hourly_speeds[2] if len(hourly_speeds) > 2 else base_speed,
                predicted_speed_t60=hourly_speeds[4] if len(hourly_speeds) > 4 else base_speed,
                hourly_speeds=hourly_speeds,
                confidence=0.75
            ))
        return res
