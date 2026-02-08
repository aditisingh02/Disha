"""
NeuroFlow BharatFlow — LTA DataMall API Endpoints
Exposes Singapore's real-time traffic data to the frontend.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from app.services.lta_client import lta_client

logger = logging.getLogger("neuroflow.lta.api")

router = APIRouter(prefix="/lta", tags=["LTA DataMall"])


# ═══════════════════════════════════════════════════════════════
# Response Models
# ═══════════════════════════════════════════════════════════════

class CameraResponse(BaseModel):
    id: str
    latitude: float
    longitude: float
    image_url: str
    description: str
    fetched_at: str


class SpeedBandResponse(BaseModel):
    link_id: str
    road_name: str
    road_category: int
    speed_band: int
    min_speed: int
    max_speed: int
    start: list[float]
    end: list[float]


class IncidentResponse(BaseModel):
    type: str
    latitude: float
    longitude: float
    message: str
    fetched_at: str


class TravelTimeResponse(BaseModel):
    expressway: str
    direction: int
    start_point: str
    end_point: str
    far_end_point: str
    est_time_mins: int


class LTAStatusResponse(BaseModel):
    status: str
    api_key_configured: bool
    cameras_count: Optional[int] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════

@router.get("/status", response_model=LTAStatusResponse)
async def lta_status():
    """Check LTA API connection status."""
    result = await lta_client.test_connection()
    return result


@router.get("/cameras", response_model=list[CameraResponse])
async def get_cameras():
    """
    Get all traffic cameras with current image URLs.
    
    Returns ~70 cameras across Singapore's expressways.
    Image URLs expire after 5 minutes - re-fetch if needed.
    """
    cameras = await lta_client.get_traffic_cameras()
    return cameras


@router.get("/cameras/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: str):
    """Get a specific camera by ID."""
    camera = await lta_client.get_camera_by_id(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    return camera


@router.get("/cameras/near/{lat}/{lng}", response_model=list)
async def get_cameras_near(
    lat: float,
    lng: float,
    radius_km: float = Query(default=2.0, ge=0.1, le=20.0)
):
    """
    Find cameras within a radius of a location.
    Useful for finding cameras near reported incidents.
    
    - **lat/lng**: Location coordinates
    - **radius_km**: Search radius in kilometers (default 2km, max 20km)
    """
    cameras = await lta_client.get_cameras_near(lat, lng, radius_km)
    return cameras


@router.get("/speedbands", response_model=list[SpeedBandResponse])
async def get_speed_bands():
    """
    Get current traffic speed bands for all roads.
    
    Speed bands indicate current traffic conditions:
    - Band 1: 0-9 km/h (Stationary/Very Slow)
    - Band 2: 10-19 km/h (Slow)
    - Band 3: 20-29 km/h (Moderate Slow)
    - Band 4: 30-39 km/h (Moderate)
    - Band 5: 40-49 km/h (Moderate Fast)
    - Band 6: 50-59 km/h (Fast)
    - Band 7: 60-69 km/h (Very Fast)
    - Band 8: 70+ km/h (Free Flow)
    
    Updates every 5 minutes.
    """
    bands = await lta_client.get_speed_bands()
    return bands


@router.get("/speedbands/geojson")
async def get_speed_bands_geojson():
    """
    Get speed bands as GeoJSON LineString collection.
    Suitable for direct rendering on Mapbox/Maplibre.
    
    Roads are colored by speed band:
    - Red (1-2): Very slow
    - Orange (3-4): Slow
    - Yellow (5-6): Moderate
    - Green (7-8): Fast/Free flow
    """
    bands = await lta_client.get_speed_bands()
    
    # Speed band to color mapping
    band_colors = {
        1: "#dc2626",  # Red
        2: "#ea580c",  # Red-Orange
        3: "#f97316",  # Orange
        4: "#facc15",  # Yellow
        5: "#a3e635",  # Lime
        6: "#4ade80",  # Light Green
        7: "#22c55e",  # Green
        8: "#16a34a",  # Dark Green
    }
    
    features = []
    for band in bands:
        speed_band = band.get("speed_band", 5)
        features.append({
            "type": "Feature",
            "properties": {
                "link_id": band["link_id"],
                "road_name": band["road_name"],
                "road_category": band["road_category"],
                "speed_band": speed_band,
                "min_speed": band["min_speed"],
                "max_speed": band["max_speed"],
                "color": band_colors.get(speed_band, "#22c55e"),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    band["start"],  # [lng, lat]
                    band["end"],    # [lng, lat]
                ]
            }
        })
    
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "count": len(features),
            "band_legend": {
                "1-2": "Very slow (0-19 km/h)",
                "3-4": "Slow (20-39 km/h)",
                "5-6": "Moderate (40-59 km/h)",
                "7-8": "Fast (60+ km/h)",
            }
        }
    }


@router.get("/incidents", response_model=list[IncidentResponse])
async def get_incidents():
    """
    Get current traffic incidents.
    
    Incident types include:
    - Accident
    - Roadwork
    - Vehicle breakdown
    - Weather
    - Obstacle
    - Road Block
    - Heavy Traffic
    - Miscellaneous
    - Diversion
    - Unattended Vehicle
    
    Updates every 2 minutes.
    """
    incidents = await lta_client.get_incidents()
    return incidents


@router.get("/incidents/geojson")
async def get_incidents_geojson():
    """Get incidents as GeoJSON Point collection for map display."""
    incidents = await lta_client.get_incidents()
    
    # Incident type to icon/color mapping
    type_styles = {
        "Accident": {"color": "#dc2626", "icon": "accident"},
        "Roadwork": {"color": "#f97316", "icon": "roadwork"},
        "Vehicle breakdown": {"color": "#facc15", "icon": "breakdown"},
        "Heavy Traffic": {"color": "#ea580c", "icon": "traffic"},
        "Weather": {"color": "#3b82f6", "icon": "weather"},
        "Road Block": {"color": "#ef4444", "icon": "block"},
    }
    
    features = []
    for inc in incidents:
        inc_type = inc.get("type", "Miscellaneous")
        style = type_styles.get(inc_type, {"color": "#6b7280", "icon": "misc"})
        
        features.append({
            "type": "Feature",
            "properties": {
                "type": inc_type,
                "message": inc["message"],
                "color": style["color"],
                "icon": style["icon"],
            },
            "geometry": {
                "type": "Point",
                "coordinates": [inc["longitude"], inc["latitude"]]
            }
        })
    
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "count": len(features),
        }
    }


@router.get("/travel-times", response_model=list[TravelTimeResponse])
async def get_travel_times():
    """
    Get estimated travel times for expressway segments.
    
    Shows real-time estimated travel times in minutes
    for each segment of Singapore's expressways.
    """
    times = await lta_client.get_travel_times()
    return times
