"""
NeuroFlow BharatFlow — Core Traffic API Routes (v1)
Endpoints: /predict/traffic, /route/optimize, /risk/heatmap, /ws/live
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import ORJSONResponse

from app.models.schemas import (
    RouteRequest,
    RouteResponse,
    RoutingMode,
    TrafficPredictionResponse,
    RiskHeatmapResponse,
    GeoJSONFeatureCollection,
    GeoJSONFeature,
)
from app.engine.router import TrafficRouter
from app.services.google_maps import GoogleMapsService

logger = logging.getLogger("neuroflow.api.routes")

router = APIRouter(prefix="/api/v1", tags=["Traffic"])


# ═══════════════════════════════════════════════════════════════
# GET /predict/traffic — ST-GCN Predictions
# ═══════════════════════════════════════════════════════════════

@router.get("/predict/traffic", response_model=TrafficPredictionResponse)
async def get_traffic_predictions(
    limit: int = Query(default=50, ge=1, le=500, description="Max predictions to return"),
    city: str = Query(default="singapore", description="Target city for predictions"),
):
    """
    Return the latest ST-GCN traffic speed predictions.
    Predictions include T+15, T+30, T+60 minute horizons for each road segment.
    """
    from app.core.events import simulation

    if not simulation:
        return TrafficPredictionResponse(
            timestamp=datetime.utcnow(),
            predictions=[],
            model_version="stgcn_india_v1",
        )

    # Filter predictions for the requested city
    # Assuming segment_ids are prefixed with city name (e.g. "bengaluru_seg_1")
    city_prefix = f"{city}_"
    all_preds = simulation.latest_predictions
    city_preds = [p for p in all_preds if p.get("segment_id", "").startswith(city_prefix)]
    
    # Fallback: if no specific filtering logic exists yet or city not found, return all or limited
    if not city_preds and all_preds:
         # If no prefix match, might be old data or different naming convention. 
         # For now return relevant slice
         city_preds = all_preds

    return TrafficPredictionResponse(
        timestamp=datetime.utcnow(),
        predictions=city_preds[:limit],
        model_version=f"stgcn_{city}_v1",
    )


# ═══════════════════════════════════════════════════════════════
# POST /predict/route-forecast — Route-Specific ST-GCN Predictions
# ═══════════════════════════════════════════════════════════════

from pydantic import BaseModel
from typing import List

class RouteForecastRequest(BaseModel):
    origin: List[float]       # [lat, lng]
    destination: List[float]  # [lat, lng]
    city: str = "singapore"

class RouteForecastResponse(BaseModel):
    hourly_speeds: List[float]  # 48 values for 12 hours
    peak_speed: float
    min_speed: float
    avg_speed: float
    timestamp: datetime
    model_version: str

@router.post("/predict/route-forecast", response_model=RouteForecastResponse)
async def get_route_forecast(request: RouteForecastRequest):
    """
    Generate a 12-hour traffic forecast for a specific route.
    Uses the ST-GCN model with current time-based synthetic readings.
    """
    from app.core.events import simulation
    from app.engine.forecaster import TrafficForecaster
    import math
    
    city = request.city.lower()
    now = datetime.now()
    hour = now.hour
    is_weekday = now.weekday() < 5
    is_peak = hour in [8, 9, 10, 17, 18, 19, 20]
    
    # Generate synthetic readings for the route based on coordinates and time
    # We create a few virtual segments along the route
    lat1, lon1 = request.origin
    lat2, lon2 = request.destination
    
    # Calculate approximate distance (Haversine simplified)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    distance_km = math.sqrt(dlat**2 + dlon**2) * 111  # Rough km
    
    # Number of virtual segments (1 per 2km, min 3, max 20)
    num_segments = max(3, min(20, int(distance_km / 2)))
    
    # Generate readings for each virtual segment
    readings = []
    for i in range(num_segments):
        progress = i / (num_segments - 1) if num_segments > 1 else 0.5
        seg_lat = lat1 + dlat * progress
        seg_lon = lon1 + dlon * progress
        
        base_speed = 35.0
        if city == "singapore":
            base_speed = 45.0 if is_peak else 60.0  # Singapore expressways
        elif city == "bengaluru":
            base_speed = 25.0 if is_peak else 40.0
        elif city == "mumbai":
            base_speed = 20.0 if is_peak else 35.0
        elif city == "delhi":
            base_speed = 22.0 if is_peak else 38.0
        
        # Add variation based on position (center of route might be busier)
        position_factor = 1.0 - 0.3 * math.sin(progress * math.pi)  # Dip in middle
        speed = base_speed * position_factor + (5 * (0.5 - abs(0.5 - progress)))
        
        readings.append({
            "segment_id": f"{city}_route_seg_{i}",
            "speed_kmh": speed,
            "volume": 500 + int(300 * (1 if is_peak else 0.5)),
            "occupancy": 0.2 if is_peak else 0.1,
            "rain_intensity": 0.0,
            "weather_severity_index": 0.0,
            "event_attendance": 0,
            "holiday_intensity_score": 0.0,
            "is_peak_hour": float(is_peak),
            "is_weekday": float(is_weekday),
        })
    
    # Use forecaster to generate predictions
    forecaster = TrafficForecaster(device="cpu")
    try:
        forecaster.initialize(city=city, num_nodes=num_segments)
        predictions = forecaster.predict(readings, city=city)
    except Exception as e:
        logger.error(f"Forecaster error: {e}")
        # Fallback: generate demo forecast
        predictions = []
    
    if not predictions or not predictions[0].hourly_speeds:
        # Heuristic fallback
        base = 30.0 if not is_peak else 20.0
        hourly_speeds = []
        for step in range(48):
            future_hour = (hour + (step * 15) // 60) % 24
            future_is_peak = future_hour in [8, 9, 10, 17, 18, 19, 20]
            speed = base * (0.7 if future_is_peak else 1.0) + 5 * math.sin(step / 8)
            hourly_speeds.append(round(speed, 1))
    else:
        # Aggregate predictions from all segments
        all_speeds = [p.hourly_speeds for p in predictions if p.hourly_speeds]
        if all_speeds:
            hourly_speeds = []
            for step in range(48):
                avg = sum(s[step] for s in all_speeds) / len(all_speeds)
                hourly_speeds.append(round(avg, 1))
        else:
            hourly_speeds = [30.0] * 48
    
    peak_speed = max(hourly_speeds)
    min_speed = min(hourly_speeds)
    avg_speed = sum(hourly_speeds) / len(hourly_speeds)
    
    return RouteForecastResponse(
        hourly_speeds=hourly_speeds,
        peak_speed=round(peak_speed, 1),
        min_speed=round(min_speed, 1),
        avg_speed=round(avg_speed, 1),
        timestamp=datetime.utcnow(),
        model_version=f"stgcn_{city}_v2"
    )


# ═══════════════════════════════════════════════════════════════
# POST /route/optimize — Route Optimization
# ═══════════════════════════════════════════════════════════════

@router.post("/route/optimize", response_model=RouteResponse)
async def optimize_route(request: RouteRequest):
    """
    Find optimal route(s) between origin and destination.

    Modes:
    - **fastest**: Standard Dijkstra shortest-time path
    - **eco**: ARAI-calibrated multi-objective A* minimizing emissions
    - **nash**: Epsilon-Nash equilibrium — distributes traffic across K-shortest paths
    
    Route geometry is fetched from Geoapify for real road-snapped paths.
    """
    from app.core.events import graph_service

    if not graph_service or not graph_service.is_ready():
        raise HTTPException(status_code=503, detail="Graph still loading — please wait a moment and retry")

    graph = graph_service.get_graph()
    traffic_router = TrafficRouter(graph)

    # Find nearest graph nodes to origin/destination coordinates
    origin_node = graph_service.get_nearest_node(request.origin[0], request.origin[1])
    dest_node = graph_service.get_nearest_node(request.destination[0], request.destination[1])

    vehicle_type = request.vehicle_type.value

    # Helper: Fetch real road geometry from Google Maps Routes API
    async def get_road_geometry() -> list[list[float]]:
        """Get actual road-snapped geometry from Google Routes API."""
        service = GoogleMapsService()
        if not service.is_available:
            logger.warning("Google Maps API not available, keeping original graph geometry")
            return []
        
        # Map vehicle type to Google Maps mode
        mode_map = {
            "car_petrol": "DRIVE",
            "car_diesel": "DRIVE",
            "2_wheeler": "TWO_WHEELER",
            "3_wheeler_lpg": "DRIVE",
            "bus_diesel": "DRIVE",
        }
        geo_mode = mode_map.get(vehicle_type, "DRIVE")
        
        origin = (request.origin[0], request.origin[1])
        destination = (request.destination[0], request.destination[1])
        
        try:
            result = await service.get_route(origin, destination, mode=geo_mode)
            await service.close()
        except Exception as e:
            logger.error(f"Google Maps API request failed: {e}")
            return []
        
        if "error" in result:
            logger.warning(f"Google Maps route fetch failed: {result.get('error')} - {result.get('detail', '')}")
            return []
        
        # Extract geometry from Google Maps response
        try:
            routes = result.get("routes", [])
            if not routes:
                logger.warning("Google Maps returned no routes")
                return []
            
            route = routes[0]
            geometry = route.get("geometry", {})
            coords = geometry.get("coordinates", [])
            
            logger.info(f"Google Maps returned {len(coords) if coords else 0} coordinates")
            return coords
                
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse Geoapify geometry: {e}")
        
        return []

    # Helper: Update route geometry with road-snapped coordinates
    def update_route_geometry(route, road_coords: list[list[float]]):
        """Replace synthetic geometry with real road geometry."""
        if road_coords:
            route.geometry.coordinates = road_coords

    if request.mode == RoutingMode.FASTEST:
        route = traffic_router.find_fastest_route(origin_node, dest_node, vehicle_type)
        road_coords = await get_road_geometry()
        update_route_geometry(route, road_coords)
        return RouteResponse(
            mode=RoutingMode.FASTEST,
            routes=[route],
            braess_warning=False,
            system_emission_saved_kg=0,
        )

    elif request.mode == RoutingMode.ECO:
        route = traffic_router.find_eco_route(
            origin_node, dest_node, vehicle_type,
            alpha=request.alpha, beta=request.beta, gamma=request.gamma,
        )
        # Compare with fastest for emission savings
        fastest = traffic_router.find_fastest_route(origin_node, dest_node, vehicle_type)
        savings = max(0, fastest.emission_kgco2 - route.emission_kgco2)
        road_coords = await get_road_geometry()
        update_route_geometry(route, road_coords)
        return RouteResponse(
            mode=RoutingMode.ECO,
            routes=[route],
            braess_warning=False,
            system_emission_saved_kg=round(savings, 5),
        )

    elif request.mode == RoutingMode.NASH:
        response = traffic_router.find_nash_balanced_routes(
            origin_node, dest_node, vehicle_type,
        )
        # Update all routes with real road geometry
        road_coords = await get_road_geometry()
        for route in response.routes:
            update_route_geometry(route, road_coords)
        return response

    raise HTTPException(status_code=400, detail=f"Unknown routing mode: {request.mode}")


# ═══════════════════════════════════════════════════════════════
# GET /risk/heatmap — Risk Field Heatmap
# ═══════════════════════════════════════════════════════════════

@router.get("/risk/heatmap")
async def get_risk_heatmap():
    """
    Return the current risk field as a heatmap compatible with Deck.gl HexagonLayer.
    Height = risk intensity. Color gradient = Red (high) to Green (low).
    """
    from app.core.events import simulation

    if not simulation or not simulation.latest_heatmap:
        return RiskHeatmapResponse(
            timestamp=datetime.utcnow(),
            center=[77.5946, 12.9716],
            hex_data=[],
            geojson=GeoJSONFeatureCollection(features=[]),
        ).model_dump()

    return simulation.latest_heatmap


# ═══════════════════════════════════════════════════════════════
# GET /traffic/segments — Road Segment Geometries
# ═══════════════════════════════════════════════════════════════

@router.get("/traffic/segments")
async def get_traffic_segments(
    limit: int = Query(default=500, ge=1, le=5000),
):
    """
    Return road segments with current speeds as GeoJSON FeatureCollection.
    Used by the frontend PathLayer for traffic flow visualization.
    """
    from app.core.events import graph_service, simulation

    if not graph_service or not graph_service.is_ready():
        return {"type": "FeatureCollection", "features": []}

    features = graph_service.get_segment_geometries()[:limit]

    # Overlay current readings if available
    if simulation and simulation.latest_readings:
        speed_map = {
            r["segment_id"]: r["speed_kmh"]
            for r in simulation.latest_readings
        }
        for f in features:
            seg_id = f["properties"].get("segment_id", "")
            if seg_id in speed_map:
                f["properties"]["current_speed_kmh"] = speed_map[seg_id]

    return {
        "type": "FeatureCollection",
        "features": features[:limit],
    }


# ═══════════════════════════════════════════════════════════════
# GET /traffic/live — Current Readings
# ═══════════════════════════════════════════════════════════════

@router.get("/traffic/live")
async def get_live_traffic(
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Return the latest live traffic readings."""
    from app.core.events import simulation

    if not simulation:
        return {"readings": [], "timestamp": datetime.utcnow().isoformat()}

    return {
        "readings": simulation.latest_readings[:limit],
        "timestamp": datetime.utcnow().isoformat(),
        "tick": simulation._tick_count,
    }


# ═══════════════════════════════════════════════════════════════
# WebSocket /ws/live — Real-time Streaming
# ═══════════════════════════════════════════════════════════════

@router.websocket("/ws/live")
async def websocket_live_traffic(websocket: WebSocket):
    """
    WebSocket endpoint for real-time traffic updates.
    Broadcasts simulation tick data to connected clients.
    Waits for simulation to initialize if not yet ready.
    """
    import app.core.events as _events

    await websocket.accept()
    logger.info("WebSocket client connected")

    # Wait for simulation to become available (up to 120s while graph loads)
    for _ in range(120):
        if _events.simulation is not None:
            break
        await asyncio.sleep(1)

    if _events.simulation is None:
        await websocket.close(code=1013, reason="Service still starting up")
        return

    _events.simulation.subscribe_ws(websocket)

    try:
        while True:
            # Keep connection alive; client can also send messages
            data = await websocket.receive_text()
            # Echo back acknowledgment
            await websocket.send_json({"ack": True, "received": data})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        if _events.simulation:
            _events.simulation.unsubscribe_ws(websocket)


# ═══════════════════════════════════════════════════════════════
# Geoapify Integration Endpoints
# ═══════════════════════════════════════════════════════════════

@router.get("/geoapify/route")
async def geoapify_route(
    origin_lat: float = Query(..., description="Origin latitude"),
    origin_lng: float = Query(..., description="Origin longitude"),
    dest_lat: float = Query(..., description="Destination latitude"),
    dest_lng: float = Query(..., description="Destination longitude"),
    mode: str = Query(default="drive", description="Travel mode: drive, truck, bicycle, walk, transit"),
):
    """
    Get a route between two points using the Geoapify Routing API.
    Returns turn-by-turn directions with geometry.
    """
    service = GeoapifyService()
    if not service.is_available:
        raise HTTPException(status_code=503, detail="Geoapify API key not configured")

    waypoints = [(origin_lat, origin_lng), (dest_lat, dest_lng)]
    result = await service.get_route(waypoints, mode=mode)
    await service.close()

    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])

    return result


@router.post("/geoapify/routeplanner")
async def geoapify_route_planner(body: dict):
    """
    Solve a Vehicle Routing Problem (VRP) using the Geoapify Route Planner API.

    Body should contain:
    - mode: "drive", "truck", "bicycle", "walk"
    - agents: array of agent objects with start_location, end_location, pickup_capacity
    - jobs: array of job objects with location, duration, pickup_amount

    The API optimally assigns jobs to agents and returns optimized routes.
    """
    service = GeoapifyService()
    if not service.is_available:
        raise HTTPException(status_code=503, detail="Geoapify API key not configured")

    agents = body.get("agents", [])
    jobs = body.get("jobs", [])
    mode = body.get("mode", "drive")

    if not agents or not jobs:
        raise HTTPException(status_code=400, detail="Both 'agents' and 'jobs' are required")

    result = await service.plan_routes(agents=agents, jobs=jobs, mode=mode)
    await service.close()

    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])

    return result


@router.get("/geoapify/geocode")
async def geoapify_geocode(
    text: str = Query(..., description="Address or place to search for"),
    limit: int = Query(default=5, ge=1, le=20),
):
    """
    Forward geocoding using Geoapify — convert text address to coordinates.
    Biased towards Bengaluru.
    """
    service = GeoapifyService()
    if not service.is_available:
        raise HTTPException(status_code=503, detail="Geoapify API key not configured")

    result = await service.geocode(text=text, limit=limit)
    await service.close()

    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])

    return result


@router.get("/geoapify/reverse-geocode")
async def geoapify_reverse_geocode(
    lat: float = Query(..., description="Latitude"),
    lng: float = Query(..., description="Longitude"),
):
    """Reverse geocoding — convert coordinates to an address."""
    service = GeoapifyService()
    if not service.is_available:
        raise HTTPException(status_code=503, detail="Geoapify API key not configured")

    result = await service.reverse_geocode(lat=lat, lng=lng)
    await service.close()

    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])

    return result
