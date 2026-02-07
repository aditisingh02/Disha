"""
NeuroFlow BharatFlow — Analytics API Routes (v1)
Endpoints for dashboard analytics, corridor stats, emission comparisons,
and Braess Paradox visualization data.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Query, HTTPException

from app.models.schemas import (
    CorridorStats,
    EmissionComparison,
    BraessParadoxData,
    VehicleType,
)
from app.core.database import get_database
from app.core.config import settings

logger = logging.getLogger("neuroflow.api.analytics")

router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])


# ═══════════════════════════════════════════════════════════════
# GET /corridor-stats — Silk Board → Indiranagar Metrics
# ═══════════════════════════════════════════════════════════════

@router.get("/corridor-stats", response_model=CorridorStats)
async def get_corridor_stats():
    """
    Aggregated traffic statistics for the Silk Board – Indiranagar corridor.
    """
    try:
        from app.core.events import simulation

        if not simulation or not simulation.latest_readings:
            return CorridorStats(
                avg_speed_kmh=0,
                avg_travel_time_min=0,
                congestion_index=0,
                total_vehicles_estimated=0,
                dominant_vehicle_type=VehicleType.TWO_WHEELER,
                timestamp=datetime.utcnow(),
            )

        readings = simulation.latest_readings
        speeds = [r.get("speed_kmh", 0) for r in readings if r.get("speed_kmh", 0) > 0]
        volumes = [r.get("volume", 0) for r in readings]

        avg_speed = sum(speeds) / max(len(speeds), 1)

        # Estimate corridor length ~8km Silk Board to Indiranagar
        corridor_km = 8.0
        avg_travel_time = (corridor_km / max(avg_speed, 1)) * 60  # minutes

        # Congestion index: ratio of free-flow speed to current speed
        freeflow_speed = 45.0  # km/h average free flow for this corridor
        congestion = 1 - min(avg_speed / freeflow_speed, 1.0)

        return CorridorStats(
            avg_speed_kmh=round(avg_speed, 2),
            avg_travel_time_min=round(avg_travel_time, 2),
            congestion_index=round(congestion, 3),
            total_vehicles_estimated=sum(volumes),
            dominant_vehicle_type=VehicleType.TWO_WHEELER,  # 50% of Bengaluru traffic
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"corridor-stats error: {e}")
        return CorridorStats(
            avg_speed_kmh=0,
            avg_travel_time_min=0,
            congestion_index=0,
            total_vehicles_estimated=0,
            dominant_vehicle_type=VehicleType.TWO_WHEELER,
            timestamp=datetime.utcnow(),
        )


# ═══════════════════════════════════════════════════════════════
# GET /emission-savings — Eco vs Fastest Route Comparison
# ═══════════════════════════════════════════════════════════════

@router.get("/emission-savings", response_model=EmissionComparison)
async def get_emission_savings(
    distance_km: float = Query(default=8.0, description="Route distance in km"),
    vehicle_type: VehicleType = Query(default=VehicleType.CAR_PETROL),
):
    """
    Compare emissions between fastest and eco-optimized routes.
    Uses ARAI emission factors calibrated for Indian vehicle fleet.
    """
    try:
        from app.core.events import simulation

        # Emission factors
        factors = {
            VehicleType.TWO_WHEELER: settings.emission_2_wheeler,
            VehicleType.THREE_WHEELER: settings.emission_3_wheeler_lpg,
            VehicleType.CAR_PETROL: settings.emission_car_petrol,
            VehicleType.BUS_DIESEL: settings.emission_bus_diesel,
        }
        factor = factors.get(vehicle_type, 0.140)

        # Fastest route: more idling, more stop-and-go
        congestion_factor = 1.0
        if simulation and simulation.latest_readings:
            speeds = [r.get("speed_kmh", 30) for r in simulation.latest_readings]
            avg_speed = sum(speeds) / max(len(speeds), 1)
            congestion_factor = max(1.0, 45.0 / max(avg_speed, 5))  # Higher when slow

        fastest_emission = distance_km * factor * congestion_factor
        # Eco route: 10-20% longer distance but steady speed (less idling)
        eco_distance = distance_km * 1.12  # ~12% longer
        eco_emission = eco_distance * factor * 0.85  # Less stop-and-go

        savings = fastest_emission - eco_emission
        savings_pct = (savings / max(fastest_emission, 0.001)) * 100

        # 1 tree absorbs ~22 kg CO2 per year
        trees_equivalent = (savings * 365) / 22.0  # Annualized savings

        return EmissionComparison(
            fastest_route_emission_kg=round(fastest_emission, 4),
            eco_route_emission_kg=round(eco_emission, 4),
            savings_kg=round(max(0, savings), 4),
            savings_percent=round(max(0, savings_pct), 2),
            equivalent_trees_per_year=round(max(0, trees_equivalent), 2),
        )
    except Exception as e:
        logger.error(f"emission-savings error: {e}")
        return EmissionComparison(
            fastest_route_emission_kg=0,
            eco_route_emission_kg=0,
            savings_kg=0,
            savings_percent=0,
            equivalent_trees_per_year=0,
        )


# ═══════════════════════════════════════════════════════════════
# GET /braess-paradox — Braess Paradox Visualization
# ═══════════════════════════════════════════════════════════════

@router.get("/braess-paradox", response_model=BraessParadoxData)
async def get_braess_paradox_data():
    """
    Compute and return Braess Paradox visualization data.
    Shows the difference between User Equilibrium (selfish routing)
    and System Optimum (centrally coordinated routing).
    """
    try:
        from app.core.events import simulation, graph_service

        if not simulation or not simulation.latest_readings:
            return BraessParadoxData(
                user_equilibrium_total_time=0,
                system_optimum_total_time=0,
                improvement_percent=0,
                paradox_edges=[],
            )

        readings = simulation.latest_readings
        speeds = [r.get("speed_kmh", 30) for r in readings]
        avg_speed = sum(speeds) / max(len(speeds), 1)

        # Simulate User Equilibrium: everyone takes "fastest" → congestion
        # Total travel time = N * individual_time (but congested)
        n_vehicles = sum(r.get("volume", 0) for r in readings[:50])
        corridor_distance_km = 8.0

        # UE: everyone on main road → speed drops
        ue_speed = avg_speed * 0.7  # 30% drop due to overloading
        ue_individual_time = (corridor_distance_km / max(ue_speed, 1)) * 3600  # seconds
        ue_total_time = n_vehicles * ue_individual_time

        # SO: distribute across 3 routes → each route has less load
        so_speed1 = avg_speed * 0.9  # Main road with 50% traffic
        so_speed2 = avg_speed * 0.95  # Alt route with 30% traffic
        so_speed3 = avg_speed * 1.0  # Alt route with 20% traffic (uncongested)

        so_total = (
            (n_vehicles * 0.5) * (corridor_distance_km / max(so_speed1, 1)) * 3600
            + (n_vehicles * 0.3) * (corridor_distance_km * 1.1 / max(so_speed2, 1)) * 3600
            + (n_vehicles * 0.2) * (corridor_distance_km * 1.2 / max(so_speed3, 1)) * 3600
        )

        improvement = ((ue_total_time - so_total) / max(ue_total_time, 1)) * 100

        # Identify paradox edges (edges where adding capacity worsens flow)
        paradox_edges = [
            {
                "edge": "Silk Board Junction → HSR Layout Road",
                "reason": "Shortcut creates convergence bottleneck",
                "ue_load_pct": 78,
                "so_load_pct": 45,
            },
            {
                "edge": "Koramangala Inner Ring Road",
                "reason": "Over-utilized by selfish routing despite alternative via 100ft Road",
                "ue_load_pct": 85,
                "so_load_pct": 52,
            },
        ]

        return BraessParadoxData(
            user_equilibrium_total_time=round(ue_total_time, 0),
            system_optimum_total_time=round(so_total, 0),
            improvement_percent=round(max(0, improvement), 2),
            paradox_edges=paradox_edges,
        )
    except Exception as e:
        logger.error(f"braess-paradox error: {e}")
        return BraessParadoxData(
            user_equilibrium_total_time=0,
            system_optimum_total_time=0,
            improvement_percent=0,
            paradox_edges=[],
        )


# ═══════════════════════════════════════════════════════════════
# GET /system-health — System Status
# ═══════════════════════════════════════════════════════════════

@router.get("/system-health")
async def get_system_health():
    """Return system status: simulation state, MongoDB connectivity, model info."""
    from app.core.events import simulation, graph_service, forecaster

    mongo_ok = False
    try:
        db = get_database()
        await db.command("ping")
        mongo_ok = True
    except Exception:
        pass

    graph_nodes = 0
    graph_edges = 0
    graph_loaded = False
    try:
        if graph_service is not None:
            g = graph_service.get_graph()
            graph_loaded = True
            graph_nodes = g.number_of_nodes()
            graph_edges = g.number_of_edges()
    except Exception:
        pass

    forecaster_init = False
    forecaster_device = "N/A"
    try:
        if forecaster is not None:
            forecaster_init = getattr(forecaster, "_initialized", False)
            forecaster_device = str(forecaster.device)
    except Exception:
        pass

    sim_running = False
    sim_ticks = 0
    sim_ws = 0
    sim_readings = 0
    try:
        if simulation is not None:
            sim_running = simulation._running
            sim_ticks = simulation._tick_count
            sim_ws = len(simulation._ws_clients)
            sim_readings = len(simulation.latest_readings)
    except Exception:
        pass

    return {
        "status": "operational" if simulation else "starting",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "mongodb": "connected" if mongo_ok else "disconnected",
            "graph": {
                "loaded": graph_loaded,
                "nodes": graph_nodes,
                "edges": graph_edges,
            },
            "forecaster": {
                "initialized": forecaster_init,
                "device": forecaster_device,
            },
            "simulation": {
                "running": sim_running,
                "tick_count": sim_ticks,
                "ws_clients": sim_ws,
                "latest_readings": sim_readings,
            },
        },
    }
