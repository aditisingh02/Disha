"""
NeuroFlow-Orchestrator API — Phase 1+ outputs for frontend sync.
Serves dataset profile, baseline metrics, road network summary (read-only from orchestrator_output).
Also route forecast (multi-horizon + congestion) so frontend shows same output as terminal CLI.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.orchestrator.config import (
    PROFILE_OUTPUT_PATH,
    BASELINE_METRICS_PATH,
    ROAD_GRAPH_PATH,
    ORCHESTRATOR_OUTPUT_DIR,
)

logger = logging.getLogger("neuroflow.api.orchestrator")

router = APIRouter(prefix="/api/v1/orchestrator", tags=["Orchestrator"])


@router.get("/dataset-profile")
async def get_dataset_profile():
    """Phase 1: schema, missing_value_report, temporal_coverage."""
    if not PROFILE_OUTPUT_PATH.exists():
        raise HTTPException(status_code=404, detail="Run dataset profiling first (app.orchestrator.dataset_profiling)")
    with open(PROFILE_OUTPUT_PATH) as f:
        return json.load(f)


@router.get("/baseline-metrics")
async def get_baseline_metrics():
    """Phase 1: Historical Average and ARIMA/persistence MAE, RMSE on temporal split."""
    if not BASELINE_METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Run baselines first (app.orchestrator.baselines)")
    with open(BASELINE_METRICS_PATH) as f:
        return json.load(f)


@router.get("/road-network-summary")
async def get_road_network_summary():
    """Phase 1: nodes and edges count from constructed graph."""
    if not ROAD_GRAPH_PATH.exists():
        raise HTTPException(status_code=404, detail="Build road network first (app.orchestrator.road_network)")
    import pickle
    with open(ROAD_GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "directed": True}


@router.get("/phase2-metrics")
async def get_phase2_metrics():
    """Phase 2: ST-GCN test MAE, improvement vs baseline, leakage check."""
    path = ORCHESTRATOR_OUTPUT_DIR / "phase2_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run Phase 2 training first (app.orchestrator.phase2_forecasting)")
    with open(path) as f:
        return json.load(f)


@router.get("/phase4-evaluation")
async def get_phase4_evaluation():
    """Phase 4: Ablation, robustness, pitch metrics."""
    path = ORCHESTRATOR_OUTPUT_DIR / "phase4_evaluation.json"
    if not path.exists():
        try:
            from app.orchestrator.phase4_evaluation import run_evaluation
            return run_evaluation()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    with open(path) as f:
        return json.load(f)


@router.get("/phase3-innovation")
async def get_phase3_innovation():
    """Phase 3: Dynamic Risk Fields, GreenWave Eco-Routing, Event Impact (from primary dataset)."""
    path = ORCHESTRATOR_OUTPUT_DIR / "phase3_innovation.json"
    if not path.exists():
        try:
            from app.orchestrator.phase3_innovation import run_phase3_and_save
            return run_phase3_and_save()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    with open(path) as f:
        return json.load(f)


# ─── Route forecast (same as terminal CLI output) for frontend/Google Maps ───

class OrchestratorRouteForecastRequest(BaseModel):
    """Accept either road names or lat/lon (nearest graph node used)."""
    origin_road: Optional[str] = None
    destination_road: Optional[str] = None
    origin_lat: Optional[float] = None
    origin_lon: Optional[float] = None
    destination_lat: Optional[float] = None
    destination_lon: Optional[float] = None
    departure_time: Optional[str] = None
    event_context: str = "none"


@router.post("/route-forecast")
async def get_orchestrator_route_forecast(request: OrchestratorRouteForecastRequest):
    """
    Return the same payload as terminal CLI: multi_horizon_forecasts (1h–24h),
    congestion_classification (level, score, delay), risk, routes (path, fastest/eco).
    Frontend can call this with origin_lat/lon + destination_lat/lon (from Google Maps)
    to show the same output as the terminal on the map UI.
    """
    from app.orchestrator.route_forecast_api import get_orchestrator_route_forecast as get_forecast
    return get_forecast(
        origin_road=request.origin_road,
        destination_road=request.destination_road,
        origin_lat=request.origin_lat,
        origin_lon=request.origin_lon,
        destination_lat=request.destination_lat,
        destination_lon=request.destination_lon,
        departure_time=request.departure_time,
        event_context=request.event_context,
    )
