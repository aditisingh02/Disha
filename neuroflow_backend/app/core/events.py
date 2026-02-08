"""
NeuroFlow BharatFlow — FastAPI Lifespan Events
Manages startup (DB connect, graph load, model load, sim start) and shutdown.
Uses the modern FastAPI lifespan context manager pattern.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.core.config import settings
from app.core.database import connect_to_mongodb, close_mongodb_connection, is_connected
from app.services.graph_builder import GraphBuilderService
from app.engine.forecaster import TrafficForecaster
from app.core.simulation import SimulationLoop

logger = logging.getLogger("neuroflow.events")

# ── Global references for DI ──
graph_service: GraphBuilderService | None = None
forecaster: TrafficForecaster | None = None
simulation: SimulationLoop | None = None
_simulation_task: asyncio.Task | None = None
_startup_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.
    MongoDB connects synchronously (fast). Heavy initialization (graph, model,
    simulation) runs in a background task so the server starts accepting
    requests immediately.
    """
    global _startup_task

    logger.info("=" * 60)
    logger.info("  NeuroFlow BharatFlow — Starting Up")
    logger.info("=" * 60)

    # 1. Connect to MongoDB (optional; app runs without it if unavailable)
    await connect_to_mongodb()

    # 2. Seed emission factors only if DB is connected
    if is_connected():
        await _seed_emission_factors()

    # 3. Start heavy initialization in the background so server is immediately responsive
    _startup_task = asyncio.create_task(_initialize_services(), name="startup-init")

    logger.info("Server accepting requests. Graph & simulation loading in background...")

    yield  # ── App is running ──

    # ── Shutdown ──
    logger.info("Shutting down NeuroFlow BharatFlow...")

    if _startup_task and not _startup_task.done():
        _startup_task.cancel()
        try:
            await _startup_task
        except asyncio.CancelledError:
            pass

    if _simulation_task and not _simulation_task.done():
        _simulation_task.cancel()
        try:
            await _simulation_task
        except asyncio.CancelledError:
            pass

    await close_mongodb_connection()
    logger.info("Shutdown complete.")


async def _initialize_services() -> None:
    """Background task: Load graph, init forecaster, start simulation loop."""
    global graph_service, forecaster, simulation, _simulation_task

    try:
        # 3. Build / Load road graph for Bengaluru
        graph_service = GraphBuilderService()
        await graph_service.initialize()

        # 4. Initialize traffic forecaster (loads ST-GCN weights if available)
        forecaster = TrafficForecaster(device=settings.torch_device)
        forecaster.initialize(graph_service.get_graph())

        # 5. Start the simulation loop as a background task
        simulation = SimulationLoop(
            graph_service=graph_service,
            forecaster=forecaster,
            tick_seconds=settings.simulation_tick_seconds,
        )
        _simulation_task = asyncio.create_task(simulation.run(), name="simulation-loop")

        logger.info("All systems online. Serving requests.")

    except Exception as e:
        logger.error(f"Background initialization failed: {e}", exc_info=True)


async def _seed_emission_factors() -> None:
    """Insert ARAI emission constants into MongoDB if not already present."""
    from app.core.database import get_database

    db = get_database()
    existing = await db.emission_factors.count_documents({})
    if existing > 0:
        return

    factors = [
        {"vehicle_type": "2_wheeler", "factor_kgco2_per_km": settings.emission_2_wheeler, "label": "Two-Wheeler (Petrol)"},
        {"vehicle_type": "3_wheeler_lpg", "factor_kgco2_per_km": settings.emission_3_wheeler_lpg, "label": "Auto-Rickshaw (LPG)"},
        {"vehicle_type": "car_petrol", "factor_kgco2_per_km": settings.emission_car_petrol, "label": "Car (Petrol BS-VI)"},
        {"vehicle_type": "bus_diesel", "factor_kgco2_per_km": settings.emission_bus_diesel, "label": "Bus (Diesel BS-IV)"},
    ]
    await db.emission_factors.insert_many(factors)
    logger.info(f"Seeded {len(factors)} ARAI emission factors.")
