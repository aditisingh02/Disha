import httpx
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from app.core.config import settings

logger = logging.getLogger("neuroflow.predicthq")

class PredictHQService:
    BASE_URL = "https://api.predicthq.com/v1/events/"

    async def fetch_upcoming_events(self, location_lat: float, location_lon: float, radius: str = "10km") -> List[Dict[str, Any]]:
        """
        Fetch upcoming events for the next 12 hours.
        """
        headers = {
            "Authorization": f"Bearer {settings.predicthq_access_token}",
            "Accept": "application/json"
        }

        now = datetime.utcnow()
        end_time = now + timedelta(hours=12)

        params = {
            "active.gte": now.isoformat(),
            "active.lte": end_time.isoformat(),
            "location_around.origin": f"{location_lat},{location_lon}",
            "location_around.scale": radius,
            "sort": "rank",
            "limit": 50,
        }

        try:
            async with httpx.AsyncClient() as client:
                logger.info(f"Fetching PredictHQ events with params: {params}")
                
                response = await client.get(self.BASE_URL, headers=headers, params=params)
                
                if response.status_code != 200:
                    logger.error(f"PredictHQ Error {response.status_code}: {response.text}")
                    response.raise_for_status()

                data = response.json()
                results = data.get("results", [])
                logger.info(f"PredictHQ Response: Found {len(results)} events. Raw count: {data.get('count')}")
                
                events = []
                for item in results:
                    # Safely handle missing phq_attendance
                    attendance = item.get("phq_attendance")
                    
                    events.append({
                        "id": item.get("id"),
                        "title": item.get("title"),
                        "description": item.get("description"),
                        "category": item.get("category"),
                        "start": item.get("start"),
                        "end": item.get("end"),
                        "country": item.get("country"),
                        "location": item.get("location"), # [lon, lat]
                        "rank": item.get("rank"),
                        "phq_rank": item.get("phq_rank"),
                        "attendance": attendance
                    })
                
                return events

        except httpx.HTTPStatusError as e:
            logger.error(f"PredictHQ API Error: {e.response.text}")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch PredictHQ events: {e}")
            return []
