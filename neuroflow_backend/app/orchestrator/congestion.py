"""
Congestion level classification from predicted speed (HCM-style Level of Service).
PS requirement: "Predict traffic congestion levels" — output level + score + delay.
"""

from typing import Literal
import numpy as np

CongestionLevel = Literal["FREE_FLOW", "LIGHT", "MODERATE", "HEAVY", "GRIDLOCK"]


class CongestionClassifier:
    """
    Convert speed predictions to congestion levels (Highway Capacity Manual style).
    """

    def __init__(self, default_free_flow_kmh: float = 80.0):
        self.default_free_flow_kmh = default_free_flow_kmh

    def classify_congestion(
        self,
        predicted_speed_kmh: float,
        free_flow_speed_kmh: float | None = None,
        distance_km: float | None = None,
    ) -> dict:
        """
        Returns level (FREE_FLOW/LIGHT/MODERATE/HEAVY/GRIDLOCK), score in [0,1],
        delay_vs_freeflow in minutes (if distance_km given), and speed_ratio.
        """
        free_flow = free_flow_speed_kmh if free_flow_speed_kmh is not None else self.default_free_flow_kmh
        speed_ratio = predicted_speed_kmh / max(free_flow, 1.0)
        speed_ratio = min(speed_ratio, 1.0)

        if speed_ratio > 0.85:
            level: CongestionLevel = "FREE_FLOW"
            score = 0.0 + (1 - speed_ratio) * 3.0  # ~0.0–0.45
        elif speed_ratio > 0.67:
            level = "LIGHT"
            score = 0.15 + (0.85 - speed_ratio) * 2.0  # ~0.15–0.51
        elif speed_ratio > 0.50:
            level = "MODERATE"
            score = 0.33 + (0.67 - speed_ratio) * 2.0  # ~0.33–0.67
        elif speed_ratio > 0.40:
            level = "HEAVY"
            score = 0.50 + (0.50 - speed_ratio) * 3.0  # ~0.50–0.80
        else:
            level = "GRIDLOCK"
            score = 0.60 + (0.40 - speed_ratio) * 1.0  # ~0.60–1.0

        score = float(np.clip(score, 0.0, 1.0))

        delay_minutes: float | None = None
        if distance_km is not None and predicted_speed_kmh > 0:
            time_at_speed_min = (distance_km / predicted_speed_kmh) * 60
            time_free_flow_min = (distance_km / max(free_flow, 1.0)) * 60
            delay_minutes = round(time_at_speed_min - time_free_flow_min, 2)

        color = {
            "FREE_FLOW": "green",
            "LIGHT": "lightgreen",
            "MODERATE": "yellow",
            "HEAVY": "orange",
            "GRIDLOCK": "red",
        }.get(level, "gray")

        return {
            "level": level,
            "score": round(score, 4),
            "delay_vs_freeflow_min": delay_minutes,
            "speed_ratio": round(speed_ratio, 4),
            "color": color,
        }
