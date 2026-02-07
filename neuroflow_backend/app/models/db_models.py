"""
NeuroFlow BharatFlow — MongoDB Document Models
Defines the document structures stored in MongoDB collections.
These are not ORM models (MongoDB is schema-less), but serve as
reference documentation and validation templates.
"""

from datetime import datetime
from typing import Optional


# These are plain dict templates for reference.
# MongoDB doesn't require schemas, but these document the expected structure.

ROAD_SEGMENT_TEMPLATE = {
    "osm_id": "123456",
    "name": "100 Feet Road",
    "geometry": {
        "type": "LineString",
        "coordinates": [[77.5946, 12.9716], [77.6002, 12.9754]],
    },
    "properties": {
        "highway_type": "primary",
        "maxspeed_kmh": 50,
        "lanes": 4,
        "oneway": True,
        "length_km": 0.85,
        "roughness": 0.3,
    },
    "created_at": datetime.utcnow(),
}

TRAFFIC_READING_TEMPLATE = {
    "segment_id": "seg_1234",
    "timestamp": datetime.utcnow(),
    "speed_kmh": 28.5,
    "volume": 450,
    "occupancy": 0.65,
    "speed_std": 12.3,
    "location": {
        "type": "Point",
        "coordinates": [77.5946, 12.9716],
    },
}

HISTORICAL_SPEED_TEMPLATE = {
    "segment_id": "seg_1234",
    "hour_of_day": 9,
    "day_of_week": 1,
    "mean_speed": 22.4,
    "std_speed": 11.7,
    "source": "uber_movement",
    "ingested_at": datetime.utcnow(),
}

RISK_SCORE_TEMPLATE = {
    "segment_id": "seg_1234",
    "timestamp": datetime.utcnow(),
    "risk_value": 0.73,
    "risk_components": {
        "bus_risk": 0.35,
        "two_wheeler_variance": 0.28,
        "speed_factor": 0.10,
        "volume_density": 0.82,
        "congestion_chaos": 0.45,
    },
    "location": {
        "type": "Point",
        "coordinates": [77.5946, 12.9716],
    },
}

EMISSION_FACTOR_TEMPLATE = {
    "vehicle_type": "car_petrol",
    "factor_kgco2_per_km": 0.140,
    "label": "Car (Petrol BS-VI)",
}
