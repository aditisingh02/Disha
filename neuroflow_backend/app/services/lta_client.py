"""
NeuroFlow BharatFlow — LTA DataMall Client
API wrapper for Singapore's Land Transport Authority DataMall.
Provides access to real-time traffic data including cameras, speed bands, and incidents.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import httpx

from app.core.config import settings

logger = logging.getLogger("neuroflow.lta")

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

LTA_BASE_URL = "https://datamall2.mytransport.sg/ltaodataservice"

# Cache durations in seconds
CACHE_CAMERAS = 120  # 2 minutes (camera list rarely changes)
CACHE_IMAGES = 60   # 1 minutes (image URLs expire in 1 mins)
CACHE_SPEED_BANDS = 60  # 1 minute (updates every 5 mins)
CACHE_INCIDENTS = 120  # 2 minutes (updates every 2 mins)
CACHE_TRAVEL_TIMES = 300  # 5 minutes


# ═══════════════════════════════════════════════════════════════
# Cache Entry
# ═══════════════════════════════════════════════════════════════

@dataclass
class CacheEntry:
    """A cached API response with expiry time."""
    data: Any
    expires_at: float
    
    @property
    def is_valid(self) -> bool:
        return time.time() < self.expires_at


# ═══════════════════════════════════════════════════════════════
# Camera Metadata (with lat/long for map display)
# ═══════════════════════════════════════════════════════════════

# Camera ID to location description mapping (from LTA ANNEX G)
CAMERA_LOCATIONS: Dict[str, str] = {
    "1111": "TPE(PIE) - Exit 2 to Loyang Ave",
    "1112": "TPE(PIE) - Tampines Viaduct",
    "1113": "Tanah Merah Coast Road towards Changi",
    "1701": "CTE (AYE) - Moulmein Flyover",
    "1702": "CTE (AYE) - Braddell Flyover",
    "1703": "CTE (SLE) - St George's Road",
    "1704": "CTE (AYE) - Chin Swee Road Entrance",
    "1705": "CTE (AYE) - Ang Mo Kio Ave 5 Flyover",
    "1706": "CTE (AYE) - Yio Chu Kang Flyover",
    "1707": "CTE (AYE) - Bukit Merah Flyover",
    "1709": "CTE (AYE) - Exit 6 to Bukit Timah Road",
    "1711": "CTE (AYE) - Ang Mo Kio Flyover",
    "2701": "Woodlands Causeway (Towards Johor)",
    "2702": "Woodlands Checkpoint",
    "2703": "BKE (PIE) - Chantek Flyover",
    "2704": "BKE (Checkpoint) - Woodlands Flyover",
    "2705": "BKE (PIE) - Dairy Farm Flyover",
    "2706": "Mandai Rd Entrance (Towards Checkpoint)",
    "2707": "Exit 5 to KJE (towards PIE)",
    "2708": "Exit 5 to KJE (Towards Checkpoint)",
    "3702": "ECP (Changi) - Entrance from PIE",
    "3704": "ECP (Changi) - Entrance from KPE",
    "3705": "ECP (AYE) - Exit 2A to Changi Coast Road",
    "3793": "ECP (Changi) - Laguna Flyover",
    "3795": "ECP (City) - Marine Parade Flyover",
    "3796": "ECP (Changi) - Tanjong Katong Flyover",
    "3797": "ECP (City) - Tanjung Rhu",
    "3798": "ECP (Changi) - Benjamin Sheares Bridge",
    "4701": "AYE (City) - Alexander Road Exit",
    "4702": "AYE (Jurong) - Keppel Viaduct",
    "4703": "Tuas Second Link",
    "4704": "AYE (CTE) - Lower Delta Road Flyover",
    "4705": "AYE (MCE) - Yuan Ching Rd Entrance",
    "4706": "AYE (Jurong) - NUS",
    "4707": "AYE (MCE) - Jln Ahmad Ibrahim Entrance",
    "4708": "AYE (CTE) - ITE College West Dover",
    "4709": "Clementi Ave 6 Entrance",
    "4710": "AYE (Tuas) - Pandan Garden",
    "4712": "AYE (Tuas) - Tuas Ave 8 Exit",
    "4713": "Tuas Checkpoint",
    "4714": "AYE (Tuas) - Near West Coast Walk",
    "4716": "AYE (Tuas) - Benoi Rd Entrance",
    "4798": "Sentosa Tower 1",
    "4799": "Sentosa Tower 2",
    "5794": "PIE (Jurong) - Bedok North",
    "5795": "PIE (Jurong) - Eunos Flyover",
    "5797": "PIE (Jurong) - Paya Lebar Flyover",
    "5798": "PIE (Jurong) - Kallang Sims Drive",
    "5799": "PIE (Changi) - Woodsville Flyover",
    "6701": "PIE (Changi) - Kim Keat",
    "6703": "PIE (Changi) - Toa Payoh Lorong 1",
    "6704": "PIE (Jurong) - Mt Pleasant Flyover",
    "6705": "PIE (Changi) - Adam Flyover",
    "6706": "PIE (Changi) - BKE",
    "6708": "Nanyang Flyover (Towards Changi)",
    "6710": "Jln Anak Bukit Entrance (Towards Changi)",
    "6711": "ECP Entrance (Towards Jurong)",
    "6712": "Exit 27 to Clementi Ave 6",
    "6713": "Simei Ave Entrance (Towards Jurong)",
    "6714": "Exit 35 to KJE (Towards Changi)",
    "6715": "Hong Kah Flyover (Towards Jurong)",
    "6716": "AYE Flyover",
    "7791": "TPE (PIE) - Upper Changi Flyover",
    "7793": "TPE (PIE) - Tampines Ave 10 Entrance",
    "7794": "TPE (SLE) - KPE Exit",
    "7795": "TPE (PIE) - Tampines Flyover Entrance",
    "7796": "TPE (SLE) - Rivervale Drive",
    "7797": "TPE (PIE) - Seletar Flyover",
    "7798": "TPE (SLE) - SLE Flyover",
    "8701": "KJE (PIE) - Choa Chu Kang West Flyover",
    "8702": "KJE (BKE) - Exit To BKE",
    "8704": "KJE (BKE) - Choa Chu Kang Dr Entrance",
    "8706": "KJE (BKE) - Tengah Flyover",
    "9701": "SLE (TPE) - Lentor Flyover",
    "9702": "SLE (TPE) - Thomson Flyover",
    "9703": "SLE (Woodlands) - Woodlands South Flyover",
    "9704": "SLE (TPE) - Ulu Sembawang Flyover",
    "9705": "SLE (TPE) - Woodland Ave 2",
    "9706": "SLE (Woodlands) - Mandai Lake Flyover",
}


# ═══════════════════════════════════════════════════════════════
# LTA Client
# ═══════════════════════════════════════════════════════════════

class LTAClient:
    """
    Async client for LTA DataMall API with built-in caching.
    
    Usage:
        client = LTAClient()
        cameras = await client.get_traffic_cameras()
        speeds = await client.get_speed_bands()
    """
    
    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()
    
    @property
    def api_key(self) -> str:
        return settings.lta_account_key
    
    @property
    def headers(self) -> Dict[str, str]:
        return {
            "AccountKey": self.api_key,
            "accept": "application/json",
        }
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=LTA_BASE_URL,
                headers=self.headers,
                timeout=30.0,
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    def _get_cache(self, key: str) -> Optional[Any]:
        """Get cached data if still valid."""
        entry = self._cache.get(key)
        if entry and entry.is_valid:
            logger.debug(f"Cache HIT for {key}")
            return entry.data
        logger.debug(f"Cache MISS for {key}")
        return None
    
    def _set_cache(self, key: str, data: Any, ttl: float):
        """Cache data with TTL in seconds."""
        self._cache[key] = CacheEntry(
            data=data,
            expires_at=time.time() + ttl,
        )
    
    async def _fetch(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Fetch data from LTA API with pagination support."""
        client = await self._get_client()
        all_records = []
        skip = 0
        
        while True:
            url = endpoint
            request_params = params or {}
            if skip > 0:
                request_params["$skip"] = skip
            
            try:
                response = await client.get(url, params=request_params)
                response.raise_for_status()
                data = response.json()
                
                # Handle paginated responses (500 records per call)
                if "value" in data:
                    records = data["value"]
                    all_records.extend(records)
                    
                    # If less than 500 records, we've got all data
                    if len(records) < 500:
                        break
                    skip += 500
                else:
                    # Non-paginated response
                    return data
                    
            except httpx.HTTPError as e:
                logger.error(f"LTA API error for {endpoint}: {e}")
                raise
        
        return {"value": all_records}
    
    # ─────────────────────────────────────────────────────────
    # Traffic Cameras
    # ─────────────────────────────────────────────────────────
    
    async def get_traffic_cameras(self) -> List[Dict]:
        """
        Get all traffic camera locations and image URLs.
        Returns list of cameras with lat/long/image_url.
        
        Note: Image URLs expire after 5 minutes!
        """
        cache_key = "traffic_cameras"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        async with self._lock:
            # Double-check after acquiring lock
            cached = self._get_cache(cache_key)
            if cached:
                return cached
            
            data = await self._fetch("/Traffic-Imagesv2")
            cameras = []
            
            for cam in data.get("value", []):
                camera_id = str(cam.get("CameraID", ""))
                cameras.append({
                    "id": camera_id,
                    "latitude": float(cam.get("Latitude", 0)),
                    "longitude": float(cam.get("Longitude", 0)),
                    "image_url": cam.get("ImageLink", ""),
                    "description": CAMERA_LOCATIONS.get(camera_id, f"Camera {camera_id}"),
                    "fetched_at": datetime.utcnow().isoformat(),
                })
            
            self._set_cache(cache_key, cameras, CACHE_IMAGES)
            logger.info(f"Fetched {len(cameras)} traffic cameras from LTA")
            return cameras
    
    async def get_camera_by_id(self, camera_id: str) -> Optional[Dict]:
        """Get a specific camera by ID."""
        cameras = await self.get_traffic_cameras()
        for cam in cameras:
            if cam["id"] == camera_id:
                return cam
        return None
    
    async def get_cameras_near(self, lat: float, lng: float, radius_km: float = 1.0) -> List[Dict]:
        """Get cameras within radius of a location."""
        from math import radians, sin, cos, sqrt, atan2
        
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371  # Earth radius in km
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
            return 2 * R * atan2(sqrt(a), sqrt(1-a))
        
        cameras = await self.get_traffic_cameras()
        nearby = []
        
        for cam in cameras:
            dist = haversine(lat, lng, cam["latitude"], cam["longitude"])
            if dist <= radius_km:
                nearby.append({**cam, "distance_km": round(dist, 2)})
        
        return sorted(nearby, key=lambda x: x["distance_km"])
    
    # ─────────────────────────────────────────────────────────
    # Traffic Speed Bands
    # ─────────────────────────────────────────────────────────
    
    async def get_speed_bands(self) -> List[Dict]:
        """
        Get current traffic speed bands for all roads.
        Speed bands: 1 (0-9 km/h) to 8 (70+ km/h).
        """
        cache_key = "speed_bands"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        async with self._lock:
            cached = self._get_cache(cache_key)
            if cached:
                return cached
            
            data = await self._fetch("/v4/TrafficSpeedBands")
            bands = []
            
            for band in data.get("value", []):
                bands.append({
                    "link_id": band.get("LinkID"),
                    "road_name": band.get("RoadName", ""),
                    "road_category": band.get("RoadCategory"),
                    "speed_band": band.get("SpeedBand"),
                    "min_speed": band.get("MinimumSpeed"),
                    "max_speed": band.get("MaximumSpeed"),
                    "start": [float(band.get("StartLon", 0)), float(band.get("StartLat", 0))],
                    "end": [float(band.get("EndLon", 0)), float(band.get("EndLat", 0))],
                })
            
            self._set_cache(cache_key, bands, CACHE_SPEED_BANDS)
            logger.info(f"Fetched {len(bands)} speed band segments from LTA")
            return bands
    
    # ─────────────────────────────────────────────────────────
    # Traffic Incidents
    # ─────────────────────────────────────────────────────────
    
    async def get_incidents(self) -> List[Dict]:
        """
        Get current traffic incidents (accidents, breakdowns, etc).
        """
        cache_key = "incidents"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        async with self._lock:
            cached = self._get_cache(cache_key)
            if cached:
                return cached
            
            data = await self._fetch("/TrafficIncidents")
            incidents = []
            
            for inc in data.get("value", []):
                incidents.append({
                    "type": inc.get("Type", ""),
                    "latitude": float(inc.get("Latitude", 0)),
                    "longitude": float(inc.get("Longitude", 0)),
                    "message": inc.get("Message", ""),
                    "fetched_at": datetime.utcnow().isoformat(),
                })
            
            self._set_cache(cache_key, incidents, CACHE_INCIDENTS)
            logger.info(f"Fetched {len(incidents)} traffic incidents from LTA")
            return incidents
    
    # ─────────────────────────────────────────────────────────
    # Estimated Travel Times
    # ─────────────────────────────────────────────────────────
    
    async def get_travel_times(self) -> List[Dict]:
        """Get estimated travel times for expressway segments."""
        cache_key = "travel_times"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        async with self._lock:
            cached = self._get_cache(cache_key)
            if cached:
                return cached
            
            data = await self._fetch("/EstTravelTimes")
            times = []
            
            for tt in data.get("value", []):
                times.append({
                    "expressway": tt.get("Name", ""),
                    "direction": tt.get("Direction"),
                    "start_point": tt.get("StartPoint", ""),
                    "end_point": tt.get("EndPoint", ""),
                    "far_end_point": tt.get("FarEndPoint", ""),
                    "est_time_mins": tt.get("EstTime"),
                })
            
            self._set_cache(cache_key, times, CACHE_TRAVEL_TIMES)
            logger.info(f"Fetched {len(times)} travel time segments from LTA")
            return times
    
    # ─────────────────────────────────────────────────────────
    # Health Check
    # ─────────────────────────────────────────────────────────
    
    async def test_connection(self) -> Dict:
        """Test API connection and return status."""
        try:
            cameras = await self.get_traffic_cameras()
            return {
                "status": "ok",
                "api_key_configured": bool(self.api_key),
                "cameras_count": len(cameras),
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "api_key_configured": bool(self.api_key),
            }


# ═══════════════════════════════════════════════════════════════
# Singleton Instance
# ═══════════════════════════════════════════════════════════════

lta_client = LTAClient()
