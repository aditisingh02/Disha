"""
NeuroFlow BharatFlow — MongoDB Connection Layer
Uses Motor (async MongoDB driver) for non-blocking database operations.
Provides 2dsphere geospatial indexes for road segment & traffic queries.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional
import logging

from app.core.config import settings

logger = logging.getLogger("neuroflow.database")

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


async def connect_to_mongodb() -> None:
    """Initialize the MongoDB connection and create indexes."""
    global _client, _db

    logger.info("Connecting to MongoDB Atlas...")
    _client = AsyncIOMotorClient(settings.mongodb_uri)
    _db = _client[settings.mongodb_db_name]

    # Verify connection
    await _client.admin.command("ping")
    logger.info(f"Connected to MongoDB database: {settings.mongodb_db_name}")

    # ── Create Indexes ──
    await _create_indexes()


async def _create_indexes() -> None:
    """Create geospatial and performance indexes on collections."""
    db = get_database()

    # Road segments — geospatial index on geometry
    await db.road_segments.create_index([("geometry", "2dsphere")])
    await db.road_segments.create_index("osm_id", unique=True, sparse=True)

    # Traffic readings — compound index for time-series queries
    await db.traffic_readings.create_index(
        [("segment_id", 1), ("timestamp", -1)]
    )
    await db.traffic_readings.create_index([("location", "2dsphere")])
    await db.traffic_readings.create_index("timestamp", expireAfterSeconds=86400 * 7)  # TTL: 7 days

    # Historical speeds — for ML training lookups
    await db.historical_speeds.create_index(
        [("segment_id", 1), ("hour_of_day", 1), ("day_of_week", 1)]
    )

    # Risk scores — latest per segment
    await db.risk_scores.create_index(
        [("segment_id", 1), ("timestamp", -1)]
    )
    await db.risk_scores.create_index([("location", "2dsphere")])

    # Emission factors — static lookup
    await db.emission_factors.create_index("vehicle_type", unique=True)

    # Predictions cache
    await db.predictions.create_index(
        [("timestamp", -1)], expireAfterSeconds=3600  # TTL: 1 hour
    )

    logger.info("MongoDB indexes created/verified.")


async def close_mongodb_connection() -> None:
    """Gracefully close the MongoDB connection."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed.")


def get_database() -> AsyncIOMotorDatabase:
    """Return the active database instance. Raises if not connected."""
    if _db is None:
        raise RuntimeError(
            "MongoDB is not connected. Call connect_to_mongodb() first."
        )
    return _db


def get_client() -> AsyncIOMotorClient:
    """Return the active client instance."""
    if _client is None:
        raise RuntimeError(
            "MongoDB client is not initialized. Call connect_to_mongodb() first."
        )
    return _client
