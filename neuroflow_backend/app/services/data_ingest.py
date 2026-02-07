"""
NeuroFlow BharatFlow — Data Ingestion Service
Reads Uber Movement, xMap CSVs and populates MongoDB with historical ground truth.
"""

import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from app.core.database import get_database
from app.core.config import settings

logger = logging.getLogger("neuroflow.data_ingest")


class DataIngestService:
    """Handles bulk ingestion of historical traffic datasets into MongoDB."""

    def __init__(self) -> None:
        self.data_dir = Path(settings.data_dir)

    async def ingest_uber_data(self, csv_path: str | None = None) -> int:
        """
        Ingest Uber Movement Bangalore travel time data.
        Expected CSV columns: sourceid, dstid, month, mean_travel_time, std_travel_time
        Maps zone-to-zone travel times as historical speed baselines.
        """
        path = Path(csv_path) if csv_path else self.data_dir / "uber_bangalore.csv"
        if not path.exists():
            logger.warning(f"Uber data file not found: {path}. Generating synthetic data instead.")
            return await self._generate_synthetic_historical()

        logger.info(f"Ingesting Uber Movement data from {path}...")
        df = pd.read_csv(path)

        documents = []
        for _, row in df.iterrows():
            documents.append({
                "segment_id": f"uber_{row.get('sourceid', 0)}_{row.get('dstid', 0)}",
                "hour_of_day": row.get("hod", np.random.randint(0, 24)),
                "day_of_week": row.get("dow", np.random.randint(0, 7)),
                "mean_speed": row.get("mean_travel_time", 30.0),
                "std_speed": row.get("standard_deviation_travel_time", 5.0),
                "source": "uber_movement",
                "ingested_at": datetime.utcnow(),
            })

        db = get_database()
        if documents:
            result = await db.historical_speeds.insert_many(documents)
            count = len(result.inserted_ids)
            logger.info(f"Ingested {count} Uber Movement records.")
            return count
        return 0

    async def ingest_xmap_data(self, csv_path: str | None = None) -> int:
        """
        Ingest xMap traffic speed data.
        Expected: segment_id, timestamp, avg_speed, std_speed, volume
        High std_speed = chaotic mixed Indian traffic.
        """
        path = Path(csv_path) if csv_path else self.data_dir / "xmap_sample.csv"
        if not path.exists():
            logger.warning(f"xMap data file not found: {path}. Generating synthetic data instead.")
            return await self._generate_synthetic_historical()

        logger.info(f"Ingesting xMap data from {path}...")
        df = pd.read_csv(path)

        documents = []
        for _, row in df.iterrows():
            ts = pd.to_datetime(row.get("timestamp", datetime.utcnow()))
            documents.append({
                "segment_id": str(row.get("segment_id", "unknown")),
                "hour_of_day": ts.hour,
                "day_of_week": ts.dayofweek,
                "mean_speed": float(row.get("avg_speed", 25.0)),
                "std_speed": float(row.get("std_speed", 8.0)),
                "volume": int(row.get("volume", 100)),
                "source": "xmap",
                "ingested_at": datetime.utcnow(),
            })

        db = get_database()
        if documents:
            result = await db.historical_speeds.insert_many(documents)
            count = len(result.inserted_ids)
            logger.info(f"Ingested {count} xMap records.")
            return count
        return 0

    async def _generate_synthetic_historical(self) -> int:
        """
        Generate synthetic historical speed data for Bengaluru corridors.
        Models Indian traffic patterns: morning rush 8-10, evening rush 5-8.
        """
        logger.info("Generating synthetic historical data for Bengaluru...")
        db = get_database()

        # Create ~500 synthetic segments with realistic Indian traffic patterns
        segments = [f"seg_{i}" for i in range(500)]
        documents = []

        for seg in segments:
            base_speed = np.random.uniform(15, 60)  # Base free-flow speed

            for hour in range(24):
                for dow in range(7):
                    # Indian traffic patterns
                    is_weekday = dow < 5
                    if is_weekday:
                        if 8 <= hour <= 10:  # Morning rush
                            congestion = np.random.uniform(0.4, 0.7)
                        elif 17 <= hour <= 20:  # Evening rush (later in India)
                            congestion = np.random.uniform(0.5, 0.8)
                        elif 13 <= hour <= 14:  # Lunch hour slight dip
                            congestion = np.random.uniform(0.2, 0.4)
                        else:
                            congestion = np.random.uniform(0.05, 0.2)
                    else:
                        if 10 <= hour <= 13:  # Weekend shopping hours
                            congestion = np.random.uniform(0.2, 0.5)
                        else:
                            congestion = np.random.uniform(0.05, 0.15)

                    speed = base_speed * (1 - congestion)
                    std = base_speed * congestion * np.random.uniform(0.3, 0.8)  # Higher chaos during congestion

                    documents.append({
                        "segment_id": seg,
                        "hour_of_day": hour,
                        "day_of_week": dow,
                        "mean_speed": round(float(speed), 2),
                        "std_speed": round(float(std), 2),
                        "source": "synthetic",
                        "ingested_at": datetime.utcnow(),
                    })

        # Batch insert
        batch_size = 5000
        total = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            result = await db.historical_speeds.insert_many(batch)
            total += len(result.inserted_ids)

        logger.info(f"Generated {total} synthetic historical records.")
        return total
