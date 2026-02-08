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

from app.engine.models import IndoTrafficSTGCN, PhysicsInformedLoss, ResidualGCN, SimpleGCNLSTM
from app.engine.baseline_forecaster import BaselineForecaster
from app.engine.regime import regime_gate_active
from app.engine.spatial_features import enrich_spatial_features, build_default_adjacency
from pathlib import Path


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
        weights_dir = Path(settings.weights_dir) if getattr(settings, "weights_dir", None) else Path(__file__).resolve().parent.parent / "ml_models" / "weights"
        self._baseline = BaselineForecaster(weights_dir=weights_dir)
        self._weights_dir = weights_dir
        self._residual_models: Dict[str, ResidualGCN] = {}

    def initialize(self, city: str = "bengaluru", num_nodes: int = 250) -> None:
        """Initialize the model for a specific city."""
        import joblib
        import os
        
        # ── Strategy: Try loading the high-fidelity Orchestrator model first ──
        # This model (SimpleGCNLSTM) was trained in phase2_forecasting.py
        # It expects 6 features: [hour, day_of_week, is_peak_hour, weather, rain, speed]
        # And outputs 3 horizons: [1h, 6h, 24h]
        
        orch_weights_path = f"data/orchestrator_output/phase2_model.pt"
        orch_scaler_path = f"data/orchestrator_output/phase2_scaler.pkl"
        
        if os.path.exists(orch_weights_path) and os.path.exists(orch_scaler_path):
            try:
                logger.info(f"🚀 Loading Orchestrator Phase 2 model from {orch_weights_path}")
                
                # Check metrics to verify dimensions if possible, or just assume standard Phase 2 config
                # Phase 2 uses 3 horizons [1, 6, 24] and 6 input features
                model = SimpleGCNLSTM(
                    num_nodes=num_nodes,
                    in_features=6,
                    hidden=64,
                    n_horizons=3
                ).to(self.device)
                
                state_dict = torch.load(orch_weights_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                logger.info(f"✅ Loaded SimpleGCNLSTM (Orchestrator) weights")
                
                self.scalers[city] = joblib.load(orch_scaler_path)
                logger.info(f"✅ Loaded Phase 2 Scaler")
                
                model.eval()
                self.models[city] = model
                self._node_count[city] = num_nodes
                self._initialized = True
                
                # Mark as using orchestrator model for predict method to know
                self.models[f"{city}_is_orchestrator"] = True
                return

            except Exception as e:
                logger.error(f"❌ Failed to load Orchestrator model: {e}")
                # Fallback to standard flow
        
        # ── Fallback: Standard V2 Model ──
        logger.info("Falling back to standard V2 model loading...")
        
        model = IndoTrafficSTGCN(
            num_nodes=num_nodes,
            in_features=9, 
            hidden_dim=64, 
            output_horizons=48,
            temporal_steps=24
        ).to(self.device)

        weights_path = f"{settings.weights_dir}/stgcn_{city}_v2.pth"
        scaler_path = f"{settings.weights_dir}/scaler_{city}_v2.pkl"
        
        try:
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            logger.info(f"✅ Loaded ST-GCN V2 weights for {city}")
            self.scalers[city] = joblib.load(scaler_path)
            
        except FileNotFoundError:
            logger.warning(f"⚠️ No weights found for {city}. Using random initialization.")
            self.scalers[city] = None
        except Exception as e:
            logger.warning(f"❌ Could not load weights for {city}: {e}")
            self.scalers[city] = None

        model.eval()
        self.models[city] = model
        self.models[f"{city}_is_orchestrator"] = False
        self._node_count[city] = num_nodes
        self._initialized = True
        logger.info(f"TrafficForecaster initialized for {city}")

    def _get_residual_model(self, city: str, num_nodes: int) -> Optional[ResidualGCN]:
        """Load ResidualGCN for city if available."""
        if city in self._residual_models:
            return self._residual_models[city]
        path = self._weights_dir / f"residual_gcn_{city}_v1.pth"
        if not path.exists():
            return None
        try:
            model = ResidualGCN(num_nodes=num_nodes, in_features=13, hidden_dim=64, output_horizons=48, temporal_steps=24).to(self.device)
            state = torch.load(path, map_location=self.device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            self._residual_models[city] = model
            return model
        except Exception as e:
            logger.warning("Could not load ResidualGCN for %s: %s", city, e)
            return None

    def predict(self, current_readings: list[dict], city: str = "bengaluru") -> list[TrafficPrediction]:
        """
        Generate predictions from current traffic readings.
        Uses LightGBM baseline for q50, regime gate for residual activation; q90 = q50 + gated residual.
        """
        if not current_readings:
            return []

        sorted_readings = sorted(current_readings, key=lambda r: r.get("segment_id", ""))
        # Baseline q50 (median) per segment — used for gate and for optional q50/q90 output
        q50_arr = self._baseline.predict_q50(sorted_readings, city)  # (N, 48)
        baseline_q50_speeds = q50_arr.mean(axis=1).tolist() if q50_arr.size else []
        gate_active = regime_gate_active(
            baseline_q50_speeds,
            sorted_readings,
            city,
            weights_dir=self._weights_dir,
        )
        residual_magnitude = 0.0
        q90_residual_arr = np.zeros_like(q50_arr)
        if gate_active:
            logger.info("residual_active=True city=%s segments=%d", city, len(sorted_readings))
            res_model = self._get_residual_model(city, len(sorted_readings))
            if res_model is not None:
                feats_9 = np.array([
                    [float(r.get("speed_kmh", 30.0)), float(r.get("volume", 500)), float(r.get("occupancy", 0.1)),
                    float(r.get("rain_intensity", 0.0)), float(r.get("weather_severity_index", 0.0)),
                    float(r.get("event_attendance", 0)), float(r.get("holiday_intensity_score", 0.0)),
                    float(r.get("is_peak_hour", 0.0)), float(r.get("is_weekday", 1.0))]
                    for r in sorted_readings
                ], dtype=np.float32)
                enriched = enrich_spatial_features(feats_9, q50_arr, None, build_default_adjacency(len(sorted_readings)))
                x_res = torch.FloatTensor(enriched).to(self.device).unsqueeze(0).unsqueeze(2).repeat(1, 1, 24, 1)
                with torch.no_grad():
                    q90_residual_arr = res_model(x_res).cpu().numpy()[0]
                residual_magnitude = float(np.abs(q90_residual_arr).mean())
                logger.info("residual_contribution_magnitude=%.4f", residual_magnitude)

        q90_arr = q50_arr + q90_residual_arr

        if city not in self.models:
            logger.info(f"Lazy initializing model for {city}...")
            num_nodes = len(current_readings)
            self.initialize(city, num_nodes)

        model = self.models.get(city)
        scaler = self.scalers.get(city)
        is_orchestrator = self.models.get(f"{city}_is_orchestrator", False)

        if not model or not scaler:
            return self._heuristic_predict_with_q50(sorted_readings, q50_arr, gate_active, residual_magnitude)
        
        # ── Feature Extraction ──
        if is_orchestrator:
            # Phase 2 Features: [hour, day_of_week, is_peak_hour, weather, rain, speed] (6 features)
            feats = []
            for r in sorted_readings:
                f = [
                    float(r.get("hour", datetime.utcnow().hour)),
                    float(r.get("day_of_week", datetime.utcnow().weekday())),
                    float(r.get("is_peak_hour", 0.0)),
                    float(r.get("weather_severity_index", 0.0)),
                    float(r.get("rain_intensity", 0.0)),
                    float(r.get("speed_kmh", 30.0))
                ]
                feats.append(f)
        else:
            # Standard V2 Features: 9 features
            feats = []
            for r in sorted_readings:
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
        # (Assuming steady state for simplicity in live value)
        x_input = x_node.unsqueeze(2).repeat(1, 1, 24, 1) # [1, N, 24, F]
        x_input = x_input.contiguous()

        with torch.no_grad():
            preds = model(x_input)
            
        # Post-process
        pred_numpy = preds.cpu().numpy()[0] # [N, OutputDims]
        
        results = []
        now = datetime.utcnow()
        
        # Speed mean/std for inverse transform (last col for 6-feat, 0th col for 9-feat?)
        # Phase 2 scaler: check speed index. It was last column in provided list [..., speed]
        # But scaler fits on all cols. We need mean/scale of speed col.
        speed_idx = 5 if is_orchestrator else 0
        speed_mean = scaler.mean_[speed_idx]
        speed_std = scaler.scale_[speed_idx]

        for i, r in enumerate(sorted_readings):
            if i >= len(pred_numpy):
                break
            p = pred_numpy[i] # [3] for orchestrator, [48] for V2
            
            # Inverse transform
            p = p * speed_std + speed_mean
            
            # If orchestrator (3 horizons), interpolate to 48 horizons (12h)
            if is_orchestrator:
                # p has [1h, 6h, 24h] forecast
                # We need 48 points (every 15m for 12h)
                # T+1h=index 4, T+6h=index 24, T+12h=? 
                # Let's do linear interpolation
                val_1h = p[0]
                val_6h = p[1]
                val_24h = p[2]
                
                # Create 48 points
                # 0..4 (0-1h): Interp(current, 1h)
                # 4..24 (1h-6h): Interp(1h, 6h)
                # 24..48 (6h-12h): Interp(6h, 24h)
                
                current_speed = float(r.get("speed_kmh", 30.0))
                interpolated = []
                
                # 0 to 4 (1h)
                for step in range(4):
                    frac = (step + 1) / 4
                    interpolated.append(current_speed + frac * (val_1h - current_speed))
                    
                # 4 to 24 (1h to 6h) -> 20 steps
                for step in range(20):
                    frac = (step + 1) / 20
                    interpolated.append(val_1h + frac * (val_6h - val_1h))
                    
                # 24 to 48 (6h to 12h) -> 24 steps
                # Note: val_24h is at T+24h, we only go to T+12h
                # We interpolate towards 24h but stop halfway
                val_12h_est = val_6h + (12-6)/(24-6) * (val_24h - val_6h)
                
                for step in range(24):
                    frac = (step + 1) / 24
                    interpolated.append(val_6h + frac * (val_12h_est - val_6h))
                
                primary = interpolated
            else:
                primary = [float(x) for x in p]

            # Construct Result
            q50_row = q50_arr[i] if i < len(q50_arr) else primary
            q90_row = q90_arr[i] if i < len(q90_arr) else q50_row
            
            # Confidence calculation
            spread = np.abs(np.array(q90_row) - np.array(q50_row)).mean() if len(q90_row) == len(q50_row) else 0
            if is_orchestrator:
                # Orchestrator doesn't have q90 from this model output directly (unless using residual separately)
                # We can fallback to heuristic confidence
                confidence = 0.75
            else:
                confidence = float(max(0.0, min(1.0, 1.0 - spread / (np.mean(q50_row) + 1e-5))))

            results.append(TrafficPrediction(
                segment_id=r.get("segment_id"),
                timestamp=now,
                predicted_speed_t15=float(primary[0]),
                predicted_speed_t30=float(primary[1]),
                predicted_speed_t60=float(primary[3]),
                hourly_speeds=[float(x) for x in primary],
                confidence=round(confidence, 2),
                hourly_speeds_q50=[float(x) for x in primary], # Using primary as q50 for now
                hourly_speeds_q90=[float(x) for x in primary], # Fallback until residual connected
            ))
        return results

    def _heuristic_predict_with_q50(
        self,
        sorted_readings: list[dict],
        q50_arr: np.ndarray,
        gate_active: bool,
        residual_magnitude: float,
    ) -> list[TrafficPrediction]:
        """Fallback when ST-GCN not available; use baseline q50 when present."""
        now = datetime.utcnow()
        q90_arr = q50_arr + np.zeros_like(q50_arr) if q50_arr is not None and q50_arr.size else None
        predictions = []
        for i, r in enumerate(sorted_readings):
            if q50_arr is not None and i < len(q50_arr):
                speeds = [float(x) for x in q50_arr[i]]
                q90_row = [float(x) for x in (q90_arr[i] if q90_arr is not None and i < len(q90_arr) else q50_arr[i])]
            else:
                speed = float(r.get("speed_kmh", r.get("speed", 30.0)))
                speeds = [speed] * 48
                q90_row = None
            predictions.append(TrafficPrediction(
                segment_id=r.get("segment_id", "unknown"),
                timestamp=now,
                predicted_speed_t15=speeds[0],
                predicted_speed_t30=speeds[1] if len(speeds) > 1 else speeds[0],
                predicted_speed_t60=speeds[3] if len(speeds) > 3 else speeds[0],
                hourly_speeds=speeds,
                confidence=0.5,
                hourly_speeds_q50=speeds if q50_arr is not None else None,
                hourly_speeds_q90=q90_row,
            ))
        return predictions

    def _heuristic_predict(self, readings: list[dict]) -> list[TrafficPrediction]:
        """Fallback heuristic prediction (no baseline dependency)."""
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
