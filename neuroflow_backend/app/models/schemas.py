"""
NeuroFlow BharatFlow — Pydantic V2 Schemas
All request/response models for the API layer.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════

class RoutingMode(str, Enum):
    FASTEST = "fastest"
    ECO = "eco"
    NASH = "nash"


class VehicleType(str, Enum):
    TWO_WHEELER = "2_wheeler"
    THREE_WHEELER = "3_wheeler_lpg"
    CAR_PETROL = "car_petrol"
    BUS_DIESEL = "bus_diesel"


# ═══════════════════════════════════════════════════════════════
# GeoJSON Primitives
# ═══════════════════════════════════════════════════════════════

class GeoJSONPoint(BaseModel):
    type: str = "Point"
    coordinates: list[float] = Field(..., min_length=2, max_length=3, description="[lng, lat] or [lng, lat, alt]")


class GeoJSONLineString(BaseModel):
    type: str = "LineString"
    coordinates: list[list[float]] = Field(..., description="Array of [lng, lat] pairs")


class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: dict
    properties: dict = Field(default_factory=dict)


class GeoJSONFeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: list[GeoJSONFeature] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Traffic Data Models
# ═══════════════════════════════════════════════════════════════

class TrafficReading(BaseModel):
    """A single traffic sensor reading for a road segment."""
    segment_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    speed_kmh: float = Field(..., ge=0, le=200, description="Current speed in km/h")
    volume: int = Field(default=0, ge=0, description="Vehicle count in the interval")
    occupancy: float = Field(default=0.0, ge=0, le=1.0, description="Lane occupancy ratio")
    speed_std: float = Field(default=0.0, ge=0, description="Speed standard deviation — high = chaotic mixed traffic")
    location: Optional[GeoJSONPoint] = None


class TrafficPrediction(BaseModel):
    """ST-GCN prediction output for a road segment."""
    segment_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    predicted_speed_t15: float = Field(..., description="Predicted speed at T+15 min")
    predicted_speed_t30: float = Field(..., description="Predicted speed at T+30 min")
    predicted_speed_t60: float = Field(..., description="Predicted speed at T+60 min")
    hourly_speeds: list[float] = Field(..., description="Predicted speeds for next 12 hours (48 steps)")
    confidence: float = Field(default=0.0, ge=0, le=1.0)


class TrafficPredictionResponse(BaseModel):
    """API response wrapping multiple predictions."""
    timestamp: datetime
    predictions: list[TrafficPrediction]
    model_version: str = "stgcn_india_v1"


# ═══════════════════════════════════════════════════════════════
# Risk Models
# ═══════════════════════════════════════════════════════════════

class RiskScore(BaseModel):
    """Gaussian collision probability score for a road segment / hex cell."""
    segment_id: Optional[str] = None
    hex_id: Optional[str] = None
    location: GeoJSONPoint
    risk_value: float = Field(..., ge=0, description="Normalized risk 0-1")
    risk_components: dict = Field(
        default_factory=dict,
        description="Breakdown: { 'bus_risk', 'two_wheeler_variance', 'speed_factor' }",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RiskHeatmapResponse(BaseModel):
    """Deck.gl HexagonLayer compatible response."""
    timestamp: datetime
    center: list[float] = Field(description="[lng, lat] center of the heatmap")
    hex_data: list[dict] = Field(description="Array of { position: [lng, lat], risk: float, elevation: float }")
    geojson: Optional[GeoJSONFeatureCollection] = None


# ═══════════════════════════════════════════════════════════════
# Routing Models
# ═══════════════════════════════════════════════════════════════

class RouteRequest(BaseModel):
    origin: list[float] = Field(..., min_length=2, max_length=2, description="[lat, lng]")
    destination: list[float] = Field(..., min_length=2, max_length=2, description="[lat, lng]")
    mode: RoutingMode = RoutingMode.NASH
    vehicle_type: VehicleType = VehicleType.CAR_PETROL
    alpha: float = Field(default=0.5, ge=0, le=1, description="Weight for travel time")
    beta: float = Field(default=0.3, ge=0, le=1, description="Weight for emissions")
    gamma: float = Field(default=0.2, ge=0, le=1, description="Weight for road roughness/risk")


class SingleRoute(BaseModel):
    """A single computed route."""
    path_index: int = 0
    travel_time_seconds: float
    distance_km: float
    emission_kgco2: float
    distribution_weight: float = Field(
        default=1.0, description="Nash distribution: what fraction of drivers should take this path"
    )
    geometry: GeoJSONLineString
    segments: list[str] = Field(default_factory=list, description="Ordered list of segment IDs")
    is_eco_optimal: bool = False


class RouteResponse(BaseModel):
    """API response with one or more routes."""
    mode: RoutingMode
    routes: list[SingleRoute]
    braess_warning: bool = Field(
        default=False,
        description="True if Braess Paradox conditions detected — fastest route would cause system-wide slowdown",
    )
    system_emission_saved_kg: float = Field(
        default=0.0, description="kg CO2 saved vs. everyone taking fastest route"
    )


# ═══════════════════════════════════════════════════════════════
# Analytics Models
# ═══════════════════════════════════════════════════════════════

class CorridorStats(BaseModel):
    """Aggregated stats for the Silk Board – Indiranagar corridor."""
    corridor_name: str = "Silk Board → Indiranagar"
    avg_speed_kmh: float
    avg_travel_time_min: float
    congestion_index: float = Field(description="0-1 where 1 = fully jammed")
    total_vehicles_estimated: int
    dominant_vehicle_type: VehicleType
    timestamp: datetime


class EmissionComparison(BaseModel):
    """Compare eco-route vs fastest route emissions."""
    fastest_route_emission_kg: float
    eco_route_emission_kg: float
    savings_kg: float
    savings_percent: float
    equivalent_trees_per_year: float = Field(description="CO2 savings expressed as tree equivalent")


class BraessParadoxData(BaseModel):
    """Data for visualizing the Braess Paradox."""
    user_equilibrium_total_time: float = Field(description="Total system travel time under selfish routing")
    system_optimum_total_time: float = Field(description="Total system travel time under optimal routing")
    improvement_percent: float
    paradox_edges: list[dict] = Field(description="Edges where removing them would improve flow")


# ═══════════════════════════════════════════════════════════════
# WebSocket Models
# ═══════════════════════════════════════════════════════════════

class WSTrafficUpdate(BaseModel):
    """Real-time update pushed via WebSocket."""
    event: str = "traffic_update"
    timestamp: datetime
    readings: list[TrafficReading]
    predictions: Optional[list[TrafficPrediction]] = None
    risk_summary: Optional[dict] = None
