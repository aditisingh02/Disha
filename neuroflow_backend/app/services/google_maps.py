"""
Google Maps API Service for routing and geocoding.
Uses Google Routes API for real road-snapped routes.
"""

import logging
from typing import Optional
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

GOOGLE_ROUTES_API = "https://routes.googleapis.com/directions/v2:computeRoutes"
GOOGLE_GEOCODING_API = "https://maps.googleapis.com/maps/api/geocode/json"


class GoogleMapsService:
    """Service for Google Maps API interactions."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(settings, 'google_maps_api_key', '')
        self._client = httpx.AsyncClient(timeout=30.0)

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    async def close(self):
        await self._client.aclose()

    async def get_route(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        mode: str = "DRIVE",
    ) -> dict:
        """
        Get route from Google Routes API.
        
        Args:
            origin: (lat, lng) tuple
            destination: (lat, lng) tuple
            mode: DRIVE, BICYCLE, WALK, TWO_WHEELER, TRANSIT
            
        Returns:
            Route data with geometry coordinates
        """
        if not self.is_available:
            logger.warning("Google Maps API key not configured")
            return {"error": "API key not configured"}

        # Map our modes to Google's travel modes
        travel_mode_map = {
            "drive": "DRIVE",
            "car": "DRIVE",
            "truck": "DRIVE",
            "bicycle": "BICYCLE",
            "walk": "WALK",
            "two_wheeler": "TWO_WHEELER",
        }
        travel_mode = travel_mode_map.get(mode.lower(), "DRIVE")

        request_body = {
            "origin": {
                "location": {
                    "latLng": {
                        "latitude": origin[0],
                        "longitude": origin[1]
                    }
                }
            },
            "destination": {
                "location": {
                    "latLng": {
                        "latitude": destination[0],
                        "longitude": destination[1]
                    }
                }
            },
            "travelMode": travel_mode,
            "routingPreference": "TRAFFIC_AWARE",
            "computeAlternativeRoutes": False,
            "polylineQuality": "HIGH_QUALITY",
            "polylineEncoding": "GEO_JSON_LINESTRING",
        }

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.polyline,routes.legs.polyline",
        }

        try:
            resp = await self._client.post(
                GOOGLE_ROUTES_API,
                json=request_body,
                headers=headers,
            )
            
            if resp.status_code != 200:
                logger.error(f"Google Routes API error: {resp.status_code} - {resp.text}")
                return {"error": f"API error: {resp.status_code}", "detail": resp.text}

            data = resp.json()
            
            if "routes" not in data or not data["routes"]:
                logger.warning("No routes returned from Google Routes API")
                return {"error": "No route found"}

            route = data["routes"][0]
            
            # Extract polyline coordinates
            # Google returns GeoJSON LineString with [lng, lat] format
            polyline_data = route.get("polyline", {})
            geo_json = polyline_data.get("geoJsonLinestring", {})
            coordinates = geo_json.get("coordinates", [])
            
            return {
                "routes": [{
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates,  # Already in [lng, lat] format
                    },
                    "distance_meters": route.get("distanceMeters", 0),
                    "duration_seconds": int(route.get("duration", "0s").replace("s", "")),
                }]
            }

        except httpx.RequestError as e:
            logger.error(f"Google Routes API request failed: {e}")
            return {"error": "Request failed", "detail": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error in Google Routes API: {e}")
            return {"error": "Unexpected error", "detail": str(e)}

    async def geocode(self, address: str) -> dict:
        """Geocode an address to coordinates."""
        if not self.is_available:
            return {"error": "API key not configured"}

        params = {
            "address": address,
            "key": self.api_key,
        }

        try:
            resp = await self._client.get(GOOGLE_GEOCODING_API, params=params)
            return resp.json()
        except Exception as e:
            logger.error(f"Geocoding failed: {e}")
            return {"error": str(e)}


# Singleton instance
google_maps_service = GoogleMapsService()
