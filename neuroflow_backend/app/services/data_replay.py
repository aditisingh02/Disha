"""
NeuroFlow BharatFlow — IUDX Data Replay Generator
Streams historical or synthetic traffic data as if it were live sensor events.
Used when real IUDX API access is pending — simulates real-time traffic feed.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncGenerator

import numpy as np

from app.core.config import settings
from app.models.schemas import TrafficReading, GeoJSONPoint

logger = logging.getLogger("neuroflow.data_replay")


class DataReplayService:
    """
    Async generator that replays traffic data as a live stream.
    Falls back to synthetic generation if no IUDX JSON files are available.
    """

    def __init__(self, speed_multiplier: float = 1.0) -> None:
        self.speed_multiplier = speed_multiplier or settings.data_replay_speed
        self.data_dir = Path(settings.data_dir)
        self._running = True

    def stop(self) -> None:
        self._running = False

    async def stream(self, segment_ids: list[str] | None = None) -> AsyncGenerator[list[TrafficReading], None]:
        """
        Main entry point: yields batches of TrafficReading objects.
        First tries to replay stored IUDX JSON files;
        Falls back to synthetic generation using historical patterns.
        """
        iudx_files = sorted(self.data_dir.glob("iudx_*.json"))

        if iudx_files:
            logger.info(f"Replaying {len(iudx_files)} IUDX JSON snapshots...")
            async for batch in self._replay_iudx_files(iudx_files):
                if not self._running:
                    break
                yield batch
        else:
            logger.info("No IUDX data found. Generating synthetic live traffic stream...")
            async for batch in self._generate_synthetic_stream(segment_ids):
                if not self._running:
                    break
                yield batch

    async def _replay_iudx_files(self, files: list[Path]) -> AsyncGenerator[list[TrafficReading], None]:
        """Replay stored IUDX JSON snapshots at configured speed."""
        for file_path in files:
            if not self._running:
                break

            try:
                data = await asyncio.to_thread(self._read_json, file_path)
                readings = []

                for item in data if isinstance(data, list) else [data]:
                    reading = TrafficReading(
                        segment_id=item.get("id", item.get("segment_id", "unknown")),
                        timestamp=datetime.utcnow(),
                        speed_kmh=item.get("currentSpeed", item.get("speed_kmh", 30.0)),
                        volume=item.get("vehicleCount", item.get("volume", 50)),
                        occupancy=item.get("occupancy", 0.3),
                        speed_std=item.get("speedVariance", item.get("speed_std", 5.0)),
                        location=GeoJSONPoint(
                            coordinates=[
                                item.get("longitude", item.get("lng", 77.5946)),
                                item.get("latitude", item.get("lat", 12.9716)),
                            ]
                        ) if "latitude" in item or "lat" in item else None,
                    )
                    readings.append(reading)

                yield readings

                # Wait based on speed multiplier (1s real-time / multiplier)
                await asyncio.sleep(1.0 / self.speed_multiplier)

            except Exception as e:
                logger.error(f"Error replaying {file_path}: {e}")
                continue

    async def _generate_synthetic_stream(
        self, segment_ids: list[str] | None = None
    ) -> AsyncGenerator[list[TrafficReading], None]:
        """
        Generates synthetic real-time traffic data for Bengaluru.
        Models Indian traffic patterns with heterogeneous flow characteristics.
        """
        if not segment_ids:
            segment_ids = [f"seg_{i}" for i in range(200)]

        # Base characteristics per segment
        segment_profiles = {}
        for seg_id in segment_ids:
            segment_profiles[seg_id] = {
                "base_speed": np.random.uniform(15, 60),
                "lat": 12.915 + np.random.uniform(0, 0.065),
                "lng": 77.608 + np.random.uniform(0, 0.037),
                "road_capacity": np.random.randint(800, 3000),
            }

        tick = 0
        while self._running:
            now = datetime.utcnow()
            hour = now.hour
            minute = now.minute

            readings = []
            for seg_id in segment_ids:
                profile = segment_profiles[seg_id]
                base = profile["base_speed"]

                # ── Indian traffic congestion model ──
                # Morning rush: 8–10 AM
                # Evening rush: 5–8 PM (heavier in India due to staggered office hours)
                congestion = self._compute_congestion(hour, minute, now.weekday())

                # Add noise: Indian traffic is highly stochastic
                noise = np.random.normal(0, base * 0.15)
                current_speed = max(2.0, base * (1 - congestion) + noise)

                # Speed standard deviation — proxy for mixed traffic chaos
                # Higher during peak hours (2-wheelers weaving through)
                speed_std = abs(congestion * base * np.random.uniform(0.2, 0.6))

                # Volume proportional to congestion
                volume = int(profile["road_capacity"] * (0.3 + 0.7 * congestion) * np.random.uniform(0.8, 1.2))

                reading = TrafficReading(
                    segment_id=seg_id,
                    timestamp=now,
                    speed_kmh=round(current_speed, 2),
                    volume=volume,
                    occupancy=round(max(0.0, min(1.0, congestion + np.random.uniform(-0.1, 0.1))), 3),
                    speed_std=round(speed_std, 2),
                    location=GeoJSONPoint(
                        coordinates=[profile["lng"], profile["lat"]]
                    ),
                )
                readings.append(reading)

            yield readings

            tick += 1
            await asyncio.sleep(settings.simulation_tick_seconds / self.speed_multiplier)

    @staticmethod
    def _compute_congestion(hour: int, minute: int, weekday: int) -> float:
        """
        Compute congestion factor (0-1) based on Indian urban traffic patterns.
        Bengaluru-specific patterns considered.
        """
        t = hour + minute / 60.0
        is_weekday = weekday < 5

        if not is_weekday:
            # Weekend: lighter, with a shopping-hour bump
            if 10 <= t <= 14:
                return np.random.uniform(0.2, 0.45)
            elif 17 <= t <= 20:
                return np.random.uniform(0.15, 0.35)
            else:
                return np.random.uniform(0.05, 0.15)

        # Weekday Indian metro pattern
        if 7.5 <= t <= 10.5:  # Morning rush
            peak_factor = 1 - abs(t - 9.0) / 1.5  # Peak at 9 AM
            return np.clip(peak_factor * np.random.uniform(0.6, 0.9), 0.3, 0.9)
        elif 12.5 <= t <= 14:  # Lunch movement
            return np.random.uniform(0.2, 0.4)
        elif 16.5 <= t <= 20.5:  # Evening rush (Bengaluru's notorious IT corridor rush)
            peak_factor = 1 - abs(t - 18.5) / 2.0  # Peak at 6:30 PM
            return np.clip(peak_factor * np.random.uniform(0.65, 0.95), 0.35, 0.95)
        elif 22 <= t or t <= 5:  # Late night / early morning
            return np.random.uniform(0.02, 0.08)
        else:
            return np.random.uniform(0.1, 0.25)

    @staticmethod
    def _read_json(path: Path) -> dict | list:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
