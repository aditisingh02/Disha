"""
NeuroFlow BharatFlow — FastAPI Application Entry Point
The main server that ties together all components.

Run with:
    cd neuroflow_backend
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.core.config import settings
from app.core.events import lifespan
from app.api.v1.routes import router as traffic_router
from app.api.v1.analytics import router as analytics_router

# ── Logging ──
logging.basicConfig(
    level=logging.INFO if settings.debug else logging.WARNING,
    format="%(asctime)s │ %(name)-28s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger("neuroflow")

# ═══════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="NeuroFlow BharatFlow",
    description=(
        "Cyber-Physical Traffic Orchestration System for Indian Urban Dynamics. "
        "Solves the Braess Paradox through Nash Equilibrium routing, "
        "ARAI-calibrated eco-routing, and physics-informed traffic prediction."
    ),
    version="3.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow frontend dev server) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routers ──
app.include_router(traffic_router)
app.include_router(analytics_router)


# ═══════════════════════════════════════════════════════════════
# Root & Health Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "NeuroFlow BharatFlow",
        "version": "3.0.0-production",
        "description": "Cyber-Physical Traffic Orchestration for Indian Metros",
        "pilot_zone": "Bengaluru (Silk Board → Indiranagar Corridor)",
        "docs": "/docs",
        "endpoints": {
            "traffic_predictions": "/api/v1/predict/traffic",
            "route_optimize": "/api/v1/route/optimize",
            "risk_heatmap": "/api/v1/risk/heatmap",
            "live_traffic": "/api/v1/traffic/live",
            "road_segments": "/api/v1/traffic/segments",
            "corridor_stats": "/api/v1/analytics/corridor-stats",
            "emission_savings": "/api/v1/analytics/emission-savings",
            "braess_paradox": "/api/v1/analytics/braess-paradox",
            "system_health": "/api/v1/analytics/system-health",
            "websocket": "ws://localhost:8000/api/v1/ws/live",
        },
    }


@app.get("/health", tags=["Root"])
async def health():
    return {"status": "ok", "service": "neuroflow-bharatflow"}
