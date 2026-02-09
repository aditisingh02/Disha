"""
NeuroFlow BharatFlow — Vehicle Detection Service
Uses YOLOv8 to detect and count vehicles in traffic camera images.
Provides congestion scoring based on vehicle density.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from io import BytesIO

import httpx
from PIL import Image

logger = logging.getLogger("neuroflow.vehicle_detector")

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# Vehicle classes from COCO dataset that YOLO detects
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle", 
    5: "bus",
    7: "truck",
}

# Congestion thresholds (vehicles per image)
CONGESTION_THRESHOLDS = {
    "low": 5,       # 0-5 vehicles
    "medium": 15,   # 6-15 vehicles
    "high": 25,     # 16-25 vehicles
    "severe": 999,  # 26+ vehicles
}

# Cache duration in seconds
CACHE_DURATION = 30


# ═══════════════════════════════════════════════════════════════
# Cache Entry
# ═══════════════════════════════════════════════════════════════

@dataclass
class CacheEntry:
    """Cached analysis result with expiry."""
    data: Dict[str, Any]
    expires_at: float
    
    @property
    def is_valid(self) -> bool:
        return time.time() < self.expires_at


# ═══════════════════════════════════════════════════════════════
# Analysis Result
# ═══════════════════════════════════════════════════════════════

@dataclass
class VehicleAnalysis:
    """Result of vehicle detection on a camera image."""
    camera_id: str
    vehicle_counts: Dict[str, int]
    total_vehicles: int
    congestion_score: int  # 0-100
    congestion_level: str  # low, medium, high, severe
    detections: List[Dict[str, Any]]  # Individual detection details
    analyzed_at: str
    processing_time_ms: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "vehicle_counts": self.vehicle_counts,
            "total_vehicles": self.total_vehicles,
            "congestion_score": self.congestion_score,
            "congestion_level": self.congestion_level,
            "detections": self.detections,
            "analyzed_at": self.analyzed_at,
            "processing_time_ms": self.processing_time_ms,
        }


# ═══════════════════════════════════════════════════════════════
# Vehicle Detector
# ═══════════════════════════════════════════════════════════════

class VehicleDetector:
    """
    YOLOv8-based vehicle detector for traffic camera analysis.
    
    Usage:
        detector = VehicleDetector()
        result = await detector.analyze_camera(camera_id, image_url)
    """
    
    def __init__(self):
        self._model = None
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for downloading images."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client
    
    def _load_model(self):
        """Lazy-load YOLO model on first use."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                # Use YOLOv8 nano model - smallest and fastest
                self._model = YOLO("yolov8n.pt")
                logger.info("YOLOv8 nano model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                raise
        return self._model
    
    def _get_cache(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis if still valid."""
        entry = self._cache.get(camera_id)
        if entry and entry.is_valid:
            logger.debug(f"Cache HIT for camera {camera_id}")
            return entry.data
        return None
    
    def _set_cache(self, camera_id: str, data: Dict[str, Any]):
        """Cache analysis result."""
        self._cache[camera_id] = CacheEntry(
            data=data,
            expires_at=time.time() + CACHE_DURATION,
        )
    
    async def _download_image(self, image_url: str) -> Image.Image:
        """Download image from URL and return as PIL Image."""
        client = await self._get_http_client()
        
        try:
            response = await client.get(image_url)
            response.raise_for_status()
            image_bytes = response.content
            return Image.open(BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Failed to download image from {image_url}: {e}")
            raise
    
    def _calculate_congestion(self, total_vehicles: int) -> tuple[int, str]:
        """
        Calculate congestion score (0-100) and level based on vehicle count.
        """
        # Score: 0-100 based on vehicle count
        # Assuming 30+ vehicles = 100% congestion
        score = min(100, int((total_vehicles / 30) * 100))
        
        # Level based on thresholds
        if total_vehicles <= CONGESTION_THRESHOLDS["low"]:
            level = "low"
        elif total_vehicles <= CONGESTION_THRESHOLDS["medium"]:
            level = "medium"
        elif total_vehicles <= CONGESTION_THRESHOLDS["high"]:
            level = "high"
        else:
            level = "severe"
        
        return score, level
    
    def _run_detection(self, image: Image.Image) -> tuple[Dict[str, int], List[Dict]]:
        """
        Run YOLO detection on image.
        Returns (vehicle_counts, detections).
        """
        model = self._load_model()
        
        # Run inference
        results = model(image, verbose=False)
        
        # Count vehicles by type
        vehicle_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Only count vehicle classes
                if class_id in VEHICLE_CLASSES:
                    vehicle_type = VEHICLE_CLASSES[class_id]
                    vehicle_counts[vehicle_type] += 1
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detections.append({
                        "type": vehicle_type,
                        "confidence": round(confidence, 3),
                        "bbox": {
                            "x1": round(x1, 1),
                            "y1": round(y1, 1),
                            "x2": round(x2, 1),
                            "y2": round(y2, 1),
                        }
                    })
        
        return vehicle_counts, detections
    
    async def analyze_camera(
        self, 
        camera_id: str, 
        image_url: str,
        use_cache: bool = True
    ) -> VehicleAnalysis:
        """
        Analyze a traffic camera image for vehicles.
        
        Args:
            camera_id: Camera identifier
            image_url: URL to camera image
            use_cache: Whether to use cached results
            
        Returns:
            VehicleAnalysis with counts and congestion score
        """
        # Check cache first
        if use_cache:
            cached = self._get_cache(camera_id)
            if cached:
                return VehicleAnalysis(**cached)
        
        start_time = time.time()
        
        async with self._lock:
            # Double-check cache after acquiring lock
            if use_cache:
                cached = self._get_cache(camera_id)
                if cached:
                    return VehicleAnalysis(**cached)
            
            try:
                # Download image
                image = await self._download_image(image_url)
                
                # Run detection in thread pool (YOLO is CPU-bound)
                loop = asyncio.get_event_loop()
                vehicle_counts, detections = await loop.run_in_executor(
                    None, self._run_detection, image
                )
                
                # Calculate totals and congestion
                total_vehicles = sum(vehicle_counts.values())
                congestion_score, congestion_level = self._calculate_congestion(total_vehicles)
                
                processing_time = int((time.time() - start_time) * 1000)
                
                result = VehicleAnalysis(
                    camera_id=camera_id,
                    vehicle_counts=vehicle_counts,
                    total_vehicles=total_vehicles,
                    congestion_score=congestion_score,
                    congestion_level=congestion_level,
                    detections=detections,
                    analyzed_at=datetime.utcnow().isoformat(),
                    processing_time_ms=processing_time,
                )
                
                # Cache result
                self._set_cache(camera_id, result.to_dict())
                
                logger.info(
                    f"Camera {camera_id}: {total_vehicles} vehicles detected, "
                    f"congestion={congestion_level} ({congestion_score}%), "
                    f"time={processing_time}ms"
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to analyze camera {camera_id}: {e}")
                raise
    
    async def analyze_multiple(
        self,
        cameras: List[Dict[str, str]]
    ) -> List[VehicleAnalysis]:
        """
        Analyze multiple cameras.
        
        Args:
            cameras: List of dicts with 'id' and 'image_url' keys
            
        Returns:
            List of VehicleAnalysis results
        """
        tasks = [
            self.analyze_camera(cam["id"], cam["image_url"])
            for cam in cameras
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        valid_results = [r for r in results if isinstance(r, VehicleAnalysis)]
        return valid_results
    
    async def close(self):
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
    
    def clear_cache(self):
        """Clear all cached results."""
        self._cache.clear()


# ═══════════════════════════════════════════════════════════════
# Singleton Instance
# ═══════════════════════════════════════════════════════════════

vehicle_detector = VehicleDetector()
