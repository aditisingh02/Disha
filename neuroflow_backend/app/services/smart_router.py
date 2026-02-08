"""
NeuroFlow BharatFlow — Smart Route Distribution Service (Epsilon-Nash Equilibrium)

Solves the Braess Paradox by distributing users across multiple near-optimal routes
using epsilon-Nash equilibrium principles.

Key Concepts:
1. Epsilon-Nash Equilibrium: A route assignment where no user can improve their 
   travel time by more than epsilon (15%) by switching routes.
2. Hash-based assignment ensures deterministic, fair distribution.
3. Uses Geoapify Routing API for fetching alternative routes.
"""

import logging
import hashlib
import math
from typing import Optional
from dataclasses import dataclass

import httpx

from app.core.config import settings

logger = logging.getLogger("neuroflow.smart_router")

GEOAPIFY_ROUTING_URL = "https://api.geoapify.com/v1/routing"

# Epsilon: Maximum relative travel time difference for Nash equilibrium
# Routes within 15% of optimal are considered "equivalent" choices
EPSILON = 0.15


@dataclass
class AssignedRoute:
    """A route assigned to a specific user via epsilon-Nash equilibrium."""
    route_index: int
    total_routes: int
    polyline: str  # Encoded polyline for the route
    distance_meters: int
    distance_text: str
    duration_seconds: int
    duration_text: str
    start_address: str
    end_address: str
    summary: str  # Brief description (main road names)
    steps: list[dict]  # Turn-by-turn directions
    bounds: dict  # Viewport bounds
    waypoints: list[tuple[float, float]]  # Route waypoints for rendering
    
    def to_dict(self) -> dict:
        return {
            "route_index": self.route_index,
            "total_routes": self.total_routes,
            "polyline": self.polyline,
            "distance_meters": self.distance_meters,
            "distance_text": self.distance_text,
            "duration_seconds": self.duration_seconds,
            "duration_text": self.duration_text,
            "start_address": self.start_address,
            "end_address": self.end_address,
            "summary": self.summary,
            "steps": self.steps,
            "bounds": self.bounds,
            "waypoints": self.waypoints,
        }


def format_duration(seconds: int) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds} sec"
    elif seconds < 3600:
        mins = seconds // 60
        return f"{mins} min"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        if mins > 0:
            return f"{hours} hr {mins} min"
        return f"{hours} hr"


def format_distance(meters: int) -> str:
    """Format meters into human-readable distance."""
    if meters < 1000:
        return f"{meters} m"
    else:
        km = meters / 1000
        return f"{km:.1f} km"


def encode_polyline(coordinates: list[list[float]]) -> str:
    """
    Encode a list of [lng, lat] coordinates into Google polyline format.
    This is needed since Geoapify returns raw coordinates.
    """
    def encode_value(value: int) -> str:
        """Encode a single value using Google's polyline algorithm."""
        value = ~(value << 1) if value < 0 else (value << 1)
        chunks = []
        while value >= 0x20:
            chunks.append(chr((0x20 | (value & 0x1f)) + 63))
            value >>= 5
        chunks.append(chr(value + 63))
        return ''.join(chunks)
    
    encoded = []
    prev_lat = 0
    prev_lng = 0
    
    for coord in coordinates:
        lng, lat = coord[0], coord[1]
        lat_int = round(lat * 1e5)
        lng_int = round(lng * 1e5)
        
        d_lat = lat_int - prev_lat
        d_lng = lng_int - prev_lng
        
        prev_lat = lat_int
        prev_lng = lng_int
        
        encoded.append(encode_value(d_lat))
        encoded.append(encode_value(d_lng))
    
    return ''.join(encoded)


class SmartRouteDistributor:
    """
    Distributes users across multiple near-optimal routes using epsilon-Nash equilibrium.
    """
    
    def __init__(self):
        self._api_key = settings.geoapify_api_key
        self._client = httpx.AsyncClient(timeout=30.0)
        
        if not self._api_key:
            logger.warning("GEOAPIFY_API_KEY not set — Smart routing disabled")
    
    @property
    def is_available(self) -> bool:
        return bool(self._api_key)
    
    async def close(self):
        await self._client.aclose()
    
    def _hash_user_id(self, user_id: str) -> int:
        """Generate a consistent hash from user ID for deterministic assignment."""
        return int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    
    async def get_route_alternatives(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
    ) -> list[dict]:
        """
        Fetch route from Geoapify Routing API.
        Generates alternatives using different routing preferences.
        """
        if not self._api_key:
            logger.error("Geoapify API key not configured")
            return []
        
        routes = []
        waypoints = f"{origin_lat},{origin_lng}|{dest_lat},{dest_lng}"
        
        # Route variations using API parameters
        route_configs = [
            {"name": "Fastest Route", "params": {}},
            {"name": "No Tolls", "params": {"avoid": "tolls"}},
            {"name": "Local Roads", "params": {"avoid": "highways"}},
        ]
        
        for config in route_configs:
            try:
                params = {
                    "waypoints": waypoints,
                    "mode": "drive",
                    "details": "instruction_details",
                    "apiKey": self._api_key,
                    **config["params"]
                }
                
                response = await self._client.get(GEOAPIFY_ROUTING_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                if data.get("features") and len(data["features"]) > 0:
                    feature = data["features"][0]
                    props = feature.get("properties", {})
                    
                    # Deduplicate
                    is_duplicate = False
                    for existing in routes:
                        if abs(existing["duration_s"] - props.get("time", 0)) < 10 and \
                           abs(existing["distance_m"] - props.get("distance", 0)) < 50:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        routes.append({
                            "name": config["name"],
                            "distance_m": props.get("distance", 0),
                            "duration_s": props.get("time", 0),
                            "geometry": feature.get("geometry", {}),
                            "legs": props.get("legs", []),
                            "waypoints": props.get("waypoints", []),
                        })
                        logger.info(f"📍 Route '{config['name']}': {format_distance(props.get('distance', 0))}, {format_duration(props.get('time', 0))}")
                    
            except Exception as e:
                logger.warning(f"Failed to get route '{config['name']}': {e}")
                continue
        
        if not routes:
            logger.error("No routes could be fetched from Geoapify")
            return []
        
        # Sort by duration
        routes.sort(key=lambda r: r["duration_s"])
        
        # Apply epsilon-Nash filter
        filtered_routes = []
        if routes:
            fastest_duration = routes[0]["duration_s"]
            max_acceptable_duration = fastest_duration * (1 + EPSILON)
            filtered_routes = [r for r in routes if r["duration_s"] <= max_acceptable_duration]
            logger.info(f"🎯 Epsilon-Nash filter: {len(routes)} routes → {len(filtered_routes)} routes (within {EPSILON*100}%)")
        
        return filtered_routes
    
    def _extract_bounds(self, geometry: dict) -> dict:
        coords = geometry.get("coordinates", [[]])
        if isinstance(coords[0], list) and isinstance(coords[0][0], (int, float)):
            all_coords = coords
        elif isinstance(coords[0], list) and isinstance(coords[0][0], list):
            all_coords = [c for segment in coords for c in segment]
        else:
            return {}
        
        if not all_coords: return {}
        lngs = [c[0] for c in all_coords]
        lats = [c[1] for c in all_coords]
        return {
            "northeast": {"lat": max(lats), "lng": max(lngs)},
            "southwest": {"lat": min(lats), "lng": min(lngs)},
        }
    
    def _extract_waypoints(self, geometry: dict) -> list[tuple[float, float]]:
        coords = geometry.get("coordinates", [])
        if isinstance(coords[0], list) and isinstance(coords[0][0], (int, float)):
            return [(c[1], c[0]) for c in coords]
        elif isinstance(coords[0], list) and isinstance(coords[0][0], list):
            flat = [c for segment in coords for c in segment]
            return [(c[1], c[0]) for c in flat]
        return []
    
    def _parse_route(self, route: dict, index: int, total: int) -> AssignedRoute:
        geometry = route.get("geometry", {})
        coords = geometry.get("coordinates", [])
        if coords and isinstance(coords[0], list):
            if isinstance(coords[0][0], list):
                flat_coords = [c for segment in coords for c in segment]
            else:
                flat_coords = coords
        else:
            flat_coords = []
        
        steps = []
        for leg in route.get("legs", []):
            for step in leg.get("steps", []):
                steps.append({
                    "instruction": step.get("instruction", {}).get("text", ""),
                    "distance": format_distance(step.get("distance", 0)),
                    "duration": format_duration(step.get("time", 0)),
                })
        
        polyline = encode_polyline(flat_coords) if flat_coords else ""
        
        return AssignedRoute(
            route_index=index,
            total_routes=total,
            polyline=polyline,
            distance_meters=route.get("distance_m", 0),
            distance_text=format_distance(route.get("distance_m", 0)),
            duration_seconds=route.get("duration_s", 0),
            duration_text=format_duration(route.get("duration_s", 0)),
            start_address="",
            end_address="",
            summary=route.get("name", "Route"),
            steps=steps,
            bounds=self._extract_bounds(geometry),
            waypoints=self._extract_waypoints(geometry),
        )
    
    async def get_assigned_route(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
        user_id: str,
    ) -> tuple[Optional[AssignedRoute], list[AssignedRoute]]:
        """
        Get assigned route and all alternatives.
        """
        routes = await self.get_route_alternatives(origin_lat, origin_lng, dest_lat, dest_lng)
        
        if not routes:
            return None, []
        
        parsed_routes = [self._parse_route(r, i, len(routes)) for i, r in enumerate(routes)]
        
        user_hash = self._hash_user_id(user_id)
        route_index = user_hash % len(routes)
        
        assigned = parsed_routes[route_index]
        assigned.route_index = route_index # Ensure index matches position in list
        
        return assigned, parsed_routes


smart_router = SmartRouteDistributor()
