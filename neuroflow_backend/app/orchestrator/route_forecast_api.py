"""
Orchestrator route forecast for API/Frontend — same logic as terminal_mode output.
Returns multi_horizon_forecasts, congestion_classification, risk, routes so the frontend
can display the same data as the terminal CLI.
"""

from pathlib import Path
from typing import Any

from app.orchestrator.config import ROAD_GRAPH_PATH, ORCHESTRATOR_OUTPUT_DIR
from app.orchestrator.multi_horizon import predict_multi_horizon


def _nearest_road_node(lat: float, lon: float) -> str | None:
    """Find the graph node (road name) nearest to (lat, lon)."""
    if not ROAD_GRAPH_PATH.exists():
        return None
    import pickle
    with open(ROAD_GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    best_node = None
    best_d = 1e9
    for node, attrs in G.nodes(data=True):
        y, x = attrs.get("y"), attrs.get("x")
        if y is None or x is None:
            continue
        d = (lat - y) ** 2 + (lon - x) ** 2
        if d < best_d:
            best_d = d
            best_node = node
    return best_node


def _run_routes(origin: str, destination: str) -> dict:
    """Same as terminal_mode._run_routes (path, times, emissions, total_km)."""
    if not ROAD_GRAPH_PATH.exists():
        return {"path": [], "total_km": 0.0, "fastest_route_time_min": 0, "eco_route_time_min": 0,
                "fastest_route_emissions_kg": 0, "eco_route_emissions_kg": 0, "percent_emission_reduction": 0}
    import pickle
    import networkx as nx
    with open(ROAD_GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    if origin not in G or destination not in G:
        return {"path": [], "total_km": 0.0, "fastest_route_time_min": 0, "eco_route_time_min": 0,
                "fastest_route_emissions_kg": 0, "eco_route_emissions_kg": 0, "percent_emission_reduction": 0}
    path = nx.shortest_path(G, origin, destination, weight="weight")
    total_km = 0.0
    for i in range(len(path) - 1):
        total_km += G.edges[path[i], path[i + 1]].get("weight", G.edges[path[i], path[i + 1]].get("length_km", 0))
    fastest_time_min = total_km / 45 * 60
    fastest_emissions = total_km * 0.14
    eco_time_min = fastest_time_min * 1.12
    eco_emissions = total_km * 1.12 * 0.14 * 0.85
    pct = (1 - eco_emissions / max(fastest_emissions, 1e-6)) * 100
    return {
        "path": path,
        "total_km": round(total_km, 2),
        "fastest_route_time_min": round(fastest_time_min, 2),
        "eco_route_time_min": round(eco_time_min, 2),
        "fastest_route_emissions_kg": round(fastest_emissions, 4),
        "eco_route_emissions_kg": round(eco_emissions, 4),
        "percent_emission_reduction": round(pct, 1),
    }


def _run_risk(origin: str, destination: str) -> dict:
    """Same as terminal_mode _run_risk_for_route."""
    p3_path = ORCHESTRATOR_OUTPUT_DIR / "phase3_innovation.json"
    if not p3_path.exists():
        return {"origin_risk": 0.0, "destination_risk": 0.0, "hotspot_count": 0}
    import json
    with open(p3_path) as f:
        p3 = json.load(f)
    summary = p3.get("dynamic_risk_fields", {}).get("risk_tensor_summary", {})
    mean_per_road = summary.get("mean_per_road", {})
    return {
        "origin_risk": round(mean_per_road.get(origin, 0.0), 4),
        "destination_risk": round(mean_per_road.get(destination, 0.0), 4),
        "hotspot_count": summary.get("hotspot_count", 0),
    }


def get_orchestrator_route_forecast(
    origin_road: str | None = None,
    destination_road: str | None = None,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
    destination_lat: float | None = None,
    destination_lon: float | None = None,
    departure_time: str | None = None,
    event_context: str = "none",
) -> dict[str, Any]:
    """
    Return the same payload as terminal CLI: multi_horizon_forecasts, congestion_classification,
    risk, routes, model_version. Accepts either (origin_road, destination_road) or
    (origin_lat, origin_lon, destination_lat, destination_lon); in the latter case
    nearest graph nodes are used.
    """
    from datetime import datetime, timezone
    if origin_road is None or destination_road is None:
        if origin_lat is not None and origin_lon is not None and destination_lat is not None and destination_lon is not None:
            origin_road = _nearest_road_node(origin_lat, origin_lon)
            destination_road = _nearest_road_node(destination_lat, destination_lon)
        if not origin_road or not destination_road:
            return {
                "error": "Could not resolve origin/destination to road nodes. Provide origin_road/destination_road or valid lat/lon.",
                "multi_horizon_forecasts": {},
                "congestion_classification": {},
                "risk": {},
                "routes": {},
                "model_version": "none",
            }
    if departure_time is None:
        departure_time = datetime.now(timezone.utc).isoformat()

    routes = _run_routes(origin_road, destination_road)
    distance_km = routes.get("total_km") or 0.0
    forecast = predict_multi_horizon(
        origin_road,
        destination_road,
        departure_time,
        event_context,
        distance_km=distance_km if distance_km > 0 else None,
    )
    risk = _run_risk(origin_road, destination_road)

    return {
        "origin_road": origin_road,
        "destination_road": destination_road,
        "multi_horizon_forecasts": forecast.get("multi_horizon_forecasts", {}),
        "congestion_classification": forecast.get("congestion_classification", {}),
        "avg_speed_kmh": forecast.get("avg_speed_kmh"),
        "model_version": forecast.get("model_version", "multi_horizon_historical_v1"),
        "departure_time": departure_time,
        "risk": risk,
        "routes": routes,
        "horizons_hours": forecast.get("horizons_hours", [1, 3, 6, 12, 24]),
    }
