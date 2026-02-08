"""
NeuroFlow BharatFlow — Application Configuration
Uses pydantic-settings to load from .env file and environment variables.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    # ── MongoDB ──
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017/neuroflow",
        description="MongoDB Atlas connection string",
    )
    mongodb_db_name: str = Field(default="neuroflow", description="Database name")

    # ── Geoapify ──
    geoapify_api_key: str = Field(default="", description="Geoapify API key for maps, routing, and geocoding")
    
    # ── Google Maps ──
    google_maps_api_key: str = Field(default="", description="Google Maps API key for routing and geocoding")

    # ── PyTorch ──
    torch_device: Literal["cuda", "cpu", "mps"] = Field(
        default="cuda", description="Device for PyTorch inference"
    )

    # ── Singapore Pilot Zone ──
    singapore_center_lat: float = Field(default=1.3521)
    singapore_center_lng: float = Field(default=103.8198)

    # ── Singapore Zone ──
    singapore_center_lat: float = Field(default=1.3521)
    singapore_center_lng: float = Field(default=103.8198)

    # ── API Server ──
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    debug: bool = Field(default=True)

    # ── Simulation ──
    simulation_tick_seconds: float = Field(
        default=5.0, description="Seconds between simulation ticks"
    )
    data_replay_speed: float = Field(
        default=1.0, description="Multiplier for data replay speed (1.0 = real-time)"
    )

    # ── Paths ──
    data_dir: str = Field(default="data/datasets")
    weights_dir: str = Field(default="ml_models/weights")
    cache_dir: str = Field(default="data/cache")

    # ── ARAI Emission Factors (kg CO2 per km) ──
    emission_2_wheeler: float = 0.035
    emission_3_wheeler_lpg: float = 0.065
    emission_car_petrol: float = 0.140
    emission_bus_diesel: float = 0.750

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton instance
settings = Settings()
