"""
NeuroFlow BharatFlow — Simulation Loop
Async infinite loop that forms the heartbeat of the system:
  1. Fetch traffic data (replay or live IUDX)
  2. Run ST-GCN predictions
  3. Compute risk field
  4. Update graph edge weights
  5. Broadcast updates to connected WebSocket clients
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from app.services.data_replay import DataReplayService
from app.services.graph_builder import GraphBuilderService
from app.engine.forecaster import TrafficForecaster
from app.engine.risk_model import GaussianRiskEngine
from app.core.database import get_database

logger = logging.getLogger("neuroflow.simulation")


class SimulationLoop:
    """
    Core simulation engine. Runs as an asyncio background task.
    Orchestrates data flow from sensors → prediction → risk → broadcast.
    """

    def __init__(
        self,
        graph_service: GraphBuilderService,
        forecaster: TrafficForecaster,
        tick_seconds: float = 5.0,
    ) -> None:
        self.graph_service = graph_service
        self.forecaster = forecaster
        self.risk_engine = GaussianRiskEngine()
        self.tick_seconds = tick_seconds
        self._data_replay = DataReplayService()
        self._running = True
        self._tick_count = 0

        # Latest state (shared with API endpoints)
        self.latest_readings: list[dict] = []
        self.latest_predictions: list[dict] = []
        self.latest_risk_scores: list[dict] = []
        self.latest_heatmap: Optional[dict] = None

        # WebSocket subscribers
        self._ws_clients: set = set()

    def subscribe_ws(self, ws) -> None:
        """Register a WebSocket client for live updates."""
        self._ws_clients.add(ws)

    def unsubscribe_ws(self, ws) -> None:
        """Remove a WebSocket client."""
        self._ws_clients.discard(ws)

    async def run(self) -> None:
        """Main simulation loop — runs indefinitely until cancelled."""
        logger.info(f"Simulation loop started (tick={self.tick_seconds}s)")

        # Get segment IDs from graph for data replay
        graph = self.graph_service.get_graph()
        segment_ids = [
            f"singapore_{u}-{v}-{k}"
            for u, v, k in list(graph.edges(keys=True))[:200]  # Limit for performance
        ]

        async for readings_batch in self._data_replay.stream(segment_ids[:200]):
            if not self._running:
                break

            try:
                await self._process_tick(readings_batch)
                self._tick_count += 1

                if self._tick_count % 10 == 0:
                    logger.info(
                        f"Simulation tick #{self._tick_count}: "
                        f"{len(readings_batch)} readings processed"
                    )

            except asyncio.CancelledError:
                logger.info("Simulation loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Simulation tick error: {e}", exc_info=True)
                await asyncio.sleep(self.tick_seconds)

        self._data_replay.stop()
        logger.info("Simulation loop stopped.")

    async def _process_tick(self, readings: list) -> None:
        """Process one simulation tick."""
        now = datetime.utcnow()

        # 1. Convert readings to dicts
        readings_dicts = [r.model_dump() for r in readings]
        self.latest_readings = readings_dicts

        # 2. Run ST-GCN predictions
        # Group readings by city to call predict for each city model
        city_groups = {}
        for r in readings_dicts:
            # Infer city from segment_id (e.g. "bengaluru_seg_1")
            seg_id = r.get("segment_id", "")
            city = seg_id.split("_")[0] if "_" in seg_id else "singapore"
            if city not in city_groups:
                city_groups[city] = []
            city_groups[city].append(r)
            
        all_predictions = []
        for city, city_readings in city_groups.items():
            try:
                preds = self.forecaster.predict(city_readings, city=city) 
                all_predictions.extend(preds)
            except Exception as e:
                logger.error(f"Prediction failed for {city}: {e}")
                
        self.latest_predictions = [p.model_dump() for p in all_predictions]

        # 3. Compute risk field
        risk_readings = []
        for r in readings_dicts:
            loc = r.get("location")
            if loc:
                risk_readings.append({
                    "segment_id": r["segment_id"],
                    "speed_kmh": r["speed_kmh"],
                    "speed_std": r.get("speed_std", 5.0),
                    "volume": r.get("volume", 100),
                    "location": loc if isinstance(loc, dict) else loc,
                })

        risk_scores = self.risk_engine.compute_segment_risks(risk_readings)
        self.latest_risk_scores = [rs.model_dump() for rs in risk_scores]

        # 4. Compute heatmap
        heatmap = self.risk_engine.compute_heatmap(risk_scores)
        self.latest_heatmap = heatmap.model_dump()

        # 5. Update graph edge weights with current speeds
        graph = self.graph_service.get_graph()
        for r in readings_dicts:
            seg_id = r["segment_id"]
            parts = seg_id.split("-")
            if len(parts) >= 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    self.graph_service.update_edge_speed(u, v, r["speed_kmh"])
                except (ValueError, KeyError):
                    pass

        # 6. Store readings & risk scores to MongoDB (non-blocking)
        asyncio.create_task(self._persist_to_db(readings_dicts, self.latest_risk_scores))

        # 7. Broadcast to WebSocket clients
        await self._broadcast_ws(now)

    async def _persist_to_db(self, readings: list[dict], risk_scores: list[dict]) -> None:
        """Store the latest tick data to MongoDB."""
        try:
            db = get_database()

            if readings:
                # Convert datetime objects for MongoDB
                for r in readings:
                    if "location" in r and r["location"]:
                        loc = r["location"]
                        if isinstance(loc, dict) and "coordinates" in loc:
                            r["location"] = {
                                "type": "Point",
                                "coordinates": loc["coordinates"],
                            }
                await db.traffic_readings.insert_many(readings, ordered=False)

            if risk_scores:
                for rs in risk_scores:
                    if "location" in rs and rs["location"]:
                        loc = rs["location"]
                        if isinstance(loc, dict) and "coordinates" in loc:
                            rs["location"] = {
                                "type": "Point",
                                "coordinates": loc["coordinates"],
                            }
                await db.risk_scores.insert_many(risk_scores, ordered=False)

        except Exception as e:
            logger.debug(f"DB persist error (non-critical): {e}")

    async def _broadcast_ws(self, timestamp: datetime) -> None:
        """Push update to all connected WebSocket clients."""
        if not self._ws_clients:
            return

        import orjson
        payload = orjson.dumps({
            "event": "traffic_update",
            "timestamp": timestamp.isoformat(),
            "tick": self._tick_count,
            "summary": {
                "total_readings": len(self.latest_readings),
                "avg_speed": round(
                    sum(r.get("speed_kmh", 0) for r in self.latest_readings) / max(len(self.latest_readings), 1), 2
                ),
                "avg_risk": round(
                    sum(r.get("risk_value", 0) for r in self.latest_risk_scores) / max(len(self.latest_risk_scores), 1), 4
                ),
                "total_predictions": len(self.latest_predictions),
            },
            "readings_sample": self.latest_readings[:20],
            "predictions": self.latest_predictions[:50],
            "risk_sample": self.latest_risk_scores[:20],
        }).decode()

        dead_clients = set()
        for ws in self._ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead_clients.add(ws)

        self._ws_clients -= dead_clients

    def stop(self) -> None:
        """Signal the simulation loop to stop."""
        self._running = False
        self._data_replay.stop()
