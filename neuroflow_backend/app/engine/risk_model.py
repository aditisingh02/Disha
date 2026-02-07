"""
NeuroFlow BharatFlow — Gaussian Collision Probability Field (Risk Engine)
Computes dynamic risk scores for every road segment based on traffic characteristics.

Concept: Treat vehicles as charged particles emitting a 'Danger Field'.
    Risk(x,y) = Σ( Mass_i * Velocity_i² * exp( -distance² / 2σ² ) )

India Weighting:
    - Buses: Mass = 5.0 (heavy, slow-stopping)
    - Cars: Mass = 1.0 (baseline)
    - 2-wheelers: Mass = 0.3 but velocity_variance * 2.0 (unpredictable)
    - Auto-rickshaws: Mass = 0.5, high lateral variance
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from app.models.schemas import (
    RiskScore,
    RiskHeatmapResponse,
    GeoJSONPoint,
    GeoJSONFeature,
    GeoJSONFeatureCollection,
)
from app.core.config import settings

logger = logging.getLogger("neuroflow.risk_model")

# ── Indian Vehicle Fleet Characteristics ──
VEHICLE_PROFILES = {
    "bus": {"mass": 5.0, "velocity_variance": 0.3, "proportion": 0.08},
    "car": {"mass": 1.0, "velocity_variance": 0.5, "proportion": 0.25},
    "2_wheeler": {"mass": 0.3, "velocity_variance": 2.0, "proportion": 0.50},  # Dominant in India
    "auto_rickshaw": {"mass": 0.5, "velocity_variance": 1.5, "proportion": 0.12},
    "truck": {"mass": 4.0, "velocity_variance": 0.4, "proportion": 0.05},
}


class GaussianRiskEngine:
    """
    Computes collision probability risk fields for road segments.
    Generates both per-segment scores and hex-binned heatmap data.
    """

    def __init__(self, sigma: float = 0.002, grid_resolution: int = 50) -> None:
        """
        Args:
            sigma: Gaussian kernel spread (in degrees, ~200m at Bengaluru latitude)
            grid_resolution: Number of cells per axis for hex grid
        """
        self.sigma = sigma
        self.grid_resolution = grid_resolution

    def compute_segment_risks(
        self,
        traffic_readings: list[dict],
    ) -> list[RiskScore]:
        """
        Compute risk score for each road segment based on current traffic.

        Args:
            traffic_readings: List of dicts with keys:
                segment_id, speed_kmh, speed_std, volume, location: {coordinates: [lng, lat]}

        Returns:
            List of RiskScore objects.
        """
        risk_scores = []
        now = datetime.utcnow()

        for reading in traffic_readings:
            speed = reading.get("speed_kmh", 30.0)
            speed_std = reading.get("speed_std", 5.0)
            volume = reading.get("volume", 100)
            location = reading.get("location")

            if location is None:
                continue

            coords = location.get("coordinates", [77.5946, 12.9716])

            # ── Compute risk components per vehicle type ──
            total_risk = 0.0
            bus_risk = 0.0
            tw_variance_risk = 0.0
            speed_risk = 0.0

            for vtype, profile in VEHICLE_PROFILES.items():
                # Effective vehicles of this type
                v_count = volume * profile["proportion"]

                # Kinetic energy proxy: mass * velocity²
                kinetic = profile["mass"] * (speed ** 2)

                # Velocity variance contribution (higher for 2-wheelers in India)
                variance_contribution = speed_std * profile["velocity_variance"]

                # Combined risk for this vehicle type
                type_risk = v_count * (kinetic * 1e-4 + variance_contribution * 0.1)

                total_risk += type_risk

                if vtype == "bus":
                    bus_risk = type_risk
                elif vtype == "2_wheeler":
                    tw_variance_risk = type_risk

            # ── Speed factor: very slow = jam risk, very fast = accident risk ──
            if speed < 10:
                speed_risk = 0.8  # Severe congestion risk
            elif speed < 20:
                speed_risk = 0.5
            elif speed > 60:
                speed_risk = 0.6  # High-speed risk
            elif speed > 80:
                speed_risk = 0.9
            else:
                speed_risk = 0.2  # Normal flow

            total_risk += speed_risk * volume * 0.05

            # Normalize to 0-1 range (empirical scaling)
            normalized_risk = min(1.0, total_risk / 5000.0)

            risk_scores.append(
                RiskScore(
                    segment_id=reading.get("segment_id"),
                    location=GeoJSONPoint(coordinates=coords),
                    risk_value=round(normalized_risk, 4),
                    risk_components={
                        "bus_risk": round(bus_risk / max(total_risk, 1), 3),
                        "two_wheeler_variance": round(tw_variance_risk / max(total_risk, 1), 3),
                        "speed_factor": round(speed_risk, 3),
                        "volume_density": round(min(1.0, volume / 2000), 3),
                        "congestion_chaos": round(min(1.0, speed_std / max(speed, 1)), 3),
                    },
                    timestamp=now,
                )
            )

        return risk_scores

    def compute_heatmap(
        self,
        risk_scores: list[RiskScore],
        bounds: Optional[dict] = None,
    ) -> RiskHeatmapResponse:
        """
        Generate hex-binned heatmap data for Deck.gl HexagonLayer.

        Args:
            risk_scores: Per-segment risk scores
            bounds: Optional {north, south, east, west} bounding box

        Returns:
            RiskHeatmapResponse with hex_data and GeoJSON
        """
        if not bounds:
            bounds = {
                "north": 13.02, "south": 12.92,
                "east": 77.65, "west": 77.55,
            }

        hex_data = []
        features = []
        now = datetime.utcnow()

        for score in risk_scores:
            coords = score.location.coordinates
            hex_data.append({
                "position": coords,  # [lng, lat]
                "risk": score.risk_value,
                "elevation": score.risk_value * 1000,  # Scale for 3D visualization
                "segment_id": score.segment_id,
                "components": score.risk_components,
            })

            features.append(GeoJSONFeature(
                geometry={"type": "Point", "coordinates": coords},
                properties={
                    "risk": score.risk_value,
                    "segment_id": score.segment_id,
                    **score.risk_components,
                },
            ))

        return RiskHeatmapResponse(
            timestamp=now,
            center=[
                (bounds["west"] + bounds["east"]) / 2,
                (bounds["south"] + bounds["north"]) / 2,
            ],
            hex_data=hex_data,
            geojson=GeoJSONFeatureCollection(features=features),
        )

    def compute_risk_grid(
        self,
        risk_scores: list[RiskScore],
        bounds: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Compute a 2D risk field grid using Gaussian kernel smoothing.
        Useful for interpolating risk between discrete sensor points.

        Returns:
            2D numpy array of shape (grid_resolution, grid_resolution)
        """
        if not bounds:
            bounds = {
                "north": 13.02, "south": 12.92,
                "east": 77.65, "west": 77.55,
            }

        grid = np.zeros((self.grid_resolution, self.grid_resolution), dtype=np.float64)

        lat_range = bounds["north"] - bounds["south"]
        lng_range = bounds["east"] - bounds["west"]

        for score in risk_scores:
            lng, lat = score.location.coordinates

            # Map to grid coordinates
            col = int((lng - bounds["west"]) / lng_range * (self.grid_resolution - 1))
            row = int((lat - bounds["south"]) / lat_range * (self.grid_resolution - 1))

            if 0 <= row < self.grid_resolution and 0 <= col < self.grid_resolution:
                grid[row, col] += score.risk_value

        # Apply Gaussian smoothing (simulates the danger field diffusion)
        sigma_pixels = self.sigma / (lat_range / self.grid_resolution)
        smoothed = gaussian_filter(grid, sigma=max(1, sigma_pixels))

        # Normalize to 0-1
        max_val = smoothed.max()
        if max_val > 0:
            smoothed /= max_val

        return smoothed
