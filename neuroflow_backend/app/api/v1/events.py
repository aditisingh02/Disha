from fastapi import APIRouter, Query, HTTPException
from typing import List
from app.schemas.events import EventBase
from app.services.predicthq_service import PredictHQService
from app.core.config import settings

router = APIRouter(tags=["Events"])
service = PredictHQService()

@router.get("/events/upcoming", response_model=List[EventBase])
async def get_upcoming_events(
    lat: float = Query(default=settings.singapore_center_lat, description="Latitude"),
    lon: float = Query(default=settings.singapore_center_lng, description="Longitude"),
    radius: str = Query(default="10km", description="Search radius (e.g., 5km, 10mi)")
):
    """
    Get upcoming events for the next 12 hours from PredictHQ.
    """
    try:
        events = await service.fetch_upcoming_events(lat, lon, radius)
        return events
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
