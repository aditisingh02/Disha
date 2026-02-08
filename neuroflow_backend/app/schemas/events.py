from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class EventBase(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    category: str
    start: datetime
    end: datetime
    country: str
    location: List[float]  # [longitude, latitude]
    rank: int
    phq_rank: Optional[int] = None
    attendance: Optional[int] = None

class EventResponse(BaseModel):
    events: List[EventBase]
