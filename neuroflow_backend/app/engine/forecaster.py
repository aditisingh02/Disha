"""
NeuroFlow BharatFlow — Indo-Traffic ST-GCN Forecaster
Spatio-Temporal Graph Convolutional Network for traffic speed prediction.
Uses simplified ST-MLP architecture for robustness and lightweight inference.

Architecture:
    Input: [N, T, F] — N nodes, T timesteps, F features (Speed)
    Spatial: Shared MLP across nodes
    Temporal: 1D Conv layers along time axis
    Output: Predicted speed for T+15, T+30, T+60 minutes
"""

import logging
from typing import Optional, Dict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.config import settings
from app.models.schemas import TrafficPrediction

logger = logging.getLogger("neuroflow.forecaster")


from app.engine.models import IndoTrafficSTGCN, PhysicsInformedLoss


class TrafficForecaster:
    """
    High-level interface for traffic prediction.
    Wraps the ST-GCN model with data preprocessing and postprocessing.
    Supports multi-city models.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.models: Dict[str, IndoTrafficSTGCN] = {}
        self.scalers: Dict[str, object] = {}
        self._node_count: Dict[str, int] = {}
        self._initialized = False

    def initialize(self, city: str = "bengaluru", num_nodes: int = 250) -> None:
        """Initialize the model for a specific city."""
        import joblib
        
        # V2 Model Config
        # Input features: 9
        # Output horizons: 48 (12 hours)
        # Temporal steps: 24 (6 hours)
        
        model = IndoTrafficSTGCN(
            num_nodes=num_nodes,
            in_features=9, 
            hidden_dim=64, 
            output_horizons=48,
            temporal_steps=24
        ).to(self.device)

        # Load weights
        print(f"Forecaster settings ID: {id(settings)}")
        weights_path = f"{settings.weights_dir}/stgcn_{city}_v2.pth"
        scaler_path = f"{settings.weights_dir}/scaler_{city}_v2.pkl"
        
        import os
        logger.info(f"Looking for weights at: {os.path.abspath(weights_path)}")
        if not os.path.exists(weights_path):
            logger.error(f"Weights file not found at: {os.path.abspath(weights_path)}")
        
        try:
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            logger.info(f"✅ Loaded ST-GCN V2 weights for {city}")
            
            # Load Scaler
            self.scalers[city] = joblib.load(scaler_path)
            logger.info(f"✅ Loaded Scaler for {city}")
            
        except FileNotFoundError:
            logger.warning(f"⚠️ No weights found for {city}. Using random initialization.")
            self.scalers[city] = None
        except Exception as e:
            logger.warning(f"❌ Could not load weights for {city}: {e}")
            self.scalers[city] = None

        model.eval()
        self.models[city] = model
        self._node_count[city] = num_nodes
        self._initialized = True
        logger.info(f"TrafficForecaster initialized for {city}")

    def predict(self, current_readings: list[dict], city: str = "bengaluru") -> list[TrafficPrediction]:
        """
        Generate predictions from current traffic readings.
        """
        if not current_readings:
            return []
            
        if city not in self.models:
            # Lazy init
            logger.info(f"Lazy initializing model for {city}...")
            num_nodes = len(current_readings) 
            self.initialize(city, num_nodes)

        model = self.models.get(city)
        scaler = self.scalers.get(city)
        
        if not model or not scaler:
            return self._heuristic_predict(current_readings)

        # PREPROCESS: Convert readings to tensor [1, N, T, F]
        # We need to sort readings to match node order (lexicographic segment_id)
        sorted_readings = sorted(current_readings, key=lambda r: r.get("segment_id", ""))
        
        # Extract features: [speed, volume, occupancy, rain, severity, attendance, holiday, peak, weekday]
        feats = []
        for r in sorted_readings:
            # Default to average/safe values if missing in payload
            f = [
                float(r.get("speed_kmh", 30.0)),
                float(r.get("volume", 500)),
                float(r.get("occupancy", 0.1)),
                float(r.get("rain_intensity", 0.0)),
                float(r.get("weather_severity_index", 0.0)),
                float(r.get("event_attendance", 0)),
                float(r.get("holiday_intensity_score", 0.0)),
                float(r.get("is_peak_hour", 0.0)),
                float(r.get("is_weekday", 1.0))
            ]
            feats.append(f)
            
        # [N, F]
        feats_array = np.array(feats, dtype=np.float32)
        
        # Normalize using loaded scaler
        try:
            feats_normalized = scaler.transform(feats_array)
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            return self._heuristic_predict(current_readings)
            
        x_node = torch.FloatTensor(feats_normalized).to(self.device).unsqueeze(0) # [1, N, F]
        
        # Replicate for history [1, N, T=24, F]
        # (Assuming cold start / steady state)
        # We assume steady state = current state for the whole lookback window
        x_input = x_node.unsqueeze(2).repeat(1, 1, 24, 1) # [1, N, 24, F]
        # x_input = x_input.transpose(1, 2) # Removed incorrect transpose
        x_input = x_input.contiguous()
        # Model expects: [Batch, N, T, F] or [Batch, T, N, F]?
        # IndoTrafficSTGCN forward: B, N, T, F = x.shape
        # So [1, N, 24, F] is correct.
        
        # My previous line: x_input = x_node.unsqueeze(2).repeat(1, 1, 24, 1) -> [1, N, 24, F]
        # This is correct.
        
        with torch.no_grad():
            # Output: [1, N, 48] -> 12 hours
            preds = model(x_input)
            
        # Post-process
        pred_numpy = preds.cpu().numpy()[0] # [N, 48]
        
        results = []
        now = datetime.utcnow()
        for i, r in enumerate(sorted_readings):
            if i >= len(pred_numpy): break
            
            p = pred_numpy[i] # [48] - Normalized Speed
            
            # Inverse Transform
            # Speed is idx 0
            if scaler:
                speed_mean = scaler.mean[0]
                speed_std = scaler.std[0]
                p = p * speed_std + speed_mean
            
            # Extract key horizons: 1 hour (4), 6 hours (24), 12 hours (47)
            # Indices: T+15m=0, T+30m=1, ... T+60m=3
            
            # Confidence based on variance
            confidence = float(max(0.0, 1.0 - (p.std() / (p.mean() + 1e-5)))) 
            
            results.append(TrafficPrediction(
                segment_id=r.get("segment_id"),
                timestamp=now,
                predicted_speed_t15=float(p[0]),
                predicted_speed_t30=float(p[1]),
                predicted_speed_t60=float(p[3]), # 1 hour
                hourly_speeds=[float(x) for x in p], # Full 12 hours
                confidence=round(confidence, 2)
            ))
            
        return results

    def _heuristic_predict(self, readings: list[dict]) -> list[TrafficPrediction]:
        """Fallback heuristic prediction."""
        now = datetime.utcnow()
        predictions = []
        for r in readings:
            speed = r.get("speed_kmh", 30.0)
            predictions.append(TrafficPrediction(
                segment_id=r.get("segment_id", "unknown"),
                timestamp=now,
                predicted_speed_t15=speed,
                predicted_speed_t30=speed,
                predicted_speed_t60=speed,
                hourly_speeds=[speed] * 48,
                confidence=0.5,
            ))
        return predictions
