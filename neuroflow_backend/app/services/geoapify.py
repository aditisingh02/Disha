"""
NeuroFlow BharatFlow — Geoapify API Integration Service
Provides routing, route planning (VRP), and geocoding via Geoapify APIs.

Geoapify API Key: Configured via GEOAPIFY_API_KEY environment variable.

APIs used:
  - Route Planner (VRP): POST /v1/routeplanner
  - Routing:             GET  /v1/routing
  - Geocoding:           GET  /v1/geocode/search
  - Reverse Geocoding:   GET  /v1/geocode/reverse
"""

import logging
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger("neuroflow.geoapify")

GEOAPIFY_BASE = "https://api.geoapify.com/v1"


class GeoapifyService:
    """Async client for Geoapify APIs — routing, route planning, and geocoding."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or settings.geoapify_api_key
        if not self._api_key:
            logger.warning("GEOAPIFY_API_KEY not set — Geoapify features disabled")
        self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    # ═══════════════════════════════════════════════════════════
    # Routing API — A-to-B directions
    # ═══════════════════════════════════════════════════════════

    async def get_route(
        self,
        waypoints: list[tuple[float, float]],
        mode: str = "drive",
        units: str = "metric",
    ) -> dict:
        """
        Get turn-by-turn route between waypoints.

        Args:
            waypoints: List of (lat, lng) tuples
            mode: "drive", "truck", "bicycle", "walk", "transit"
            units: "metric" or "imperial"

        Returns:
            Geoapify Routing API response (GeoJSON-like)
        """
        if not self._api_key:
            logger.error("Geoapify API key not configured")
            return {"error": "API key not configured"}

        # Geoapify expects "lat,lon|lat,lon" format
        wp_str = "|".join(f"{lat},{lng}" for lat, lng in waypoints)

        params = {
            "waypoints": wp_str,
            "mode": mode,
            "units": units,
            "apiKey": self._api_key,
        }

        try:
            resp = await self._client.get(f"{GEOAPIFY_BASE}/routing", params=params)
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"Geoapify route: {len(waypoints)} waypoints, mode={mode}")
            return data
        except httpx.HTTPStatusError as e:
            logger.error(f"Geoapify routing error {e.response.status_code}: {e.response.text}")
            return {"error": f"HTTP {e.response.status_code}", "detail": e.response.text}
        except Exception as e:
            logger.error(f"Geoapify routing failed: {e}")
            return {"error": str(e)}

    # ═══════════════════════════════════════════════════════════
    # Route Planner API — Vehicle Routing Problem (VRP)
    # ═══════════════════════════════════════════════════════════

    async def plan_routes(
        self,
        agents: list[dict],
        jobs: list[dict],
        mode: str = "drive",
    ) -> dict:
        """
        Solve a Vehicle Routing Problem using the Geoapify Route Planner API.

        This is the VRP solver — assigns jobs to agents optimally.

        Args:
            agents: List of agent objects with start_location, end_location, pickup_capacity
                    e.g. [{"start_location": [lng, lat], "end_location": [lng, lat], "pickup_capacity": 4}]
            jobs:   List of job objects with location, duration, pickup_amount
                    e.g. [{"location": [lng, lat], "duration": 300, "pickup_amount": 1}]
            mode:   "drive", "truck", "bicycle", "walk"

        Returns:
            Geoapify Route Planner response with optimized routes per agent
        """
        if not self._api_key:
            logger.error("Geoapify API key not configured")
            return {"error": "API key not configured"}

        payload = {
            "mode": mode,
            "agents": agents,
            "jobs": jobs,
        }

        try:
            resp = await self._client.post(
                f"{GEOAPIFY_BASE}/routeplanner",
                params={"apiKey": self._api_key},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            logger.info(
                f"Geoapify VRP solved: {len(agents)} agents, {len(jobs)} jobs, mode={mode}"
            )
            return data
        except httpx.HTTPStatusError as e:
            logger.error(f"Geoapify route planner error {e.response.status_code}: {e.response.text}")
            return {"error": f"HTTP {e.response.status_code}", "detail": e.response.text}
        except Exception as e:
            logger.error(f"Geoapify route planner failed: {e}")
            return {"error": str(e)}

    # ═══════════════════════════════════════════════════════════
    # Geocoding — text → coordinates
    # ═══════════════════════════════════════════════════════════

    async def geocode(
        self,
        text: str,
        bias_lat: float = 12.9716,
        bias_lng: float = 77.5946,
        limit: int = 5,
    ) -> dict:
        """
        Forward geocoding: convert address text to coordinates.
        Biased towards Bengaluru by default.
        """
        if not self._api_key:
            return {"error": "API key not configured"}

        params = {
            "text": text,
            "bias": f"proximity:{bias_lng},{bias_lat}",
            "limit": limit,
            "format": "json",
            "apiKey": self._api_key,
        }

        try:
            resp = await self._client.get(f"{GEOAPIFY_BASE}/geocode/search", params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Geoapify geocoding failed: {e}")
            return {"error": str(e)}

    async def reverse_geocode(self, lat: float, lng: float) -> dict:
        """Reverse geocoding: convert coordinates to address."""
        if not self._api_key:
            return {"error": "API key not configured"}

        params = {
            "lat": lat,
            "lon": lng,
            "format": "json",
            "apiKey": self._api_key,
        }

        try:
            resp = await self._client.get(f"{GEOAPIFY_BASE}/geocode/reverse", params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Geoapify reverse geocoding failed: {e}")
            return {"error": str(e)}
