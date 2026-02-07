"""
NeuroFlow BharatFlow — Nash Equilibrium + GreenWave Eco-Router
Two routing paradigms to solve India's urban traffic:

1. GreenWave Eco-Router (ARAI-Calibrated)
   Multi-Objective A*: Cost = α·TravelTime + β·Emission + γ·Roughness
   Finds routes minimizing carbon output, even if 5% slower.

2. Epsilon-Nash System Balancer
   Solves the Braess Paradox by distributing drivers across K-shortest paths.
   If time difference < ε (10%), distribute: Path A (50%), Path B (30%), Path C (20%).
"""

import logging
import heapq
from typing import Optional
from datetime import datetime

import networkx as nx
import numpy as np

from app.core.config import settings
from app.models.schemas import (
    SingleRoute,
    RouteResponse,
    RoutingMode,
    GeoJSONLineString,
)

logger = logging.getLogger("neuroflow.router")

# ── ARAI Emission Factors (kg CO2 per km) ──
EMISSION_FACTORS = {
    "2_wheeler": settings.emission_2_wheeler,
    "3_wheeler_lpg": settings.emission_3_wheeler_lpg,
    "car_petrol": settings.emission_car_petrol,
    "bus_diesel": settings.emission_bus_diesel,
}


class TrafficRouter:
    """
    Combined eco-routing and Nash equilibrium routing engine.
    Operates on a weighted NetworkX graph with enriched edge attributes.
    """

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self._graph = graph
        # Create a simple DiGraph view for path algorithms (NetworkX requires this)
        self._simple_graph = self._to_weighted_digraph()

    def _to_weighted_digraph(self) -> nx.DiGraph:
        """Convert MultiDiGraph to simple DiGraph, keeping minimum-weight edges."""
        G = nx.DiGraph()
        for u, v, key, data in self._graph.edges(keys=True, data=True):
            if G.has_edge(u, v):
                if data.get("current_time_s", float("inf")) < G[u][v].get("current_time_s", float("inf")):
                    G[u][v].update(data)
            else:
                G.add_edge(u, v, **data)

        # Copy node attributes (lat/lng)
        for n, data in self._graph.nodes(data=True):
            G.nodes[n].update(data)

        return G

    def refresh_graph(self, graph: nx.MultiDiGraph) -> None:
        """Update the underlying graph (called when speeds change)."""
        self._graph = graph
        self._simple_graph = self._to_weighted_digraph()

    # ═══════════════════════════════════════════════════════════
    # ROUTING MODE 1: Fastest (Standard Dijkstra)
    # ═══════════════════════════════════════════════════════════

    def find_fastest_route(
        self,
        origin_node: int,
        dest_node: int,
        vehicle_type: str = "car_petrol",
    ) -> SingleRoute:
        """Standard shortest-time path using Dijkstra."""
        try:
            path = nx.shortest_path(
                self._simple_graph, origin_node, dest_node, weight="current_time_s"
            )
        except nx.NetworkXNoPath:
            logger.warning(f"No path found from {origin_node} to {dest_node}")
            return self._empty_route(0)

        return self._path_to_route(path, 0, vehicle_type, distribution=1.0)

    # ═══════════════════════════════════════════════════════════
    # ROUTING MODE 2: GreenWave Eco-Router (Multi-Objective A*)
    # ═══════════════════════════════════════════════════════════

    def find_eco_route(
        self,
        origin_node: int,
        dest_node: int,
        vehicle_type: str = "car_petrol",
        alpha: float = 0.5,  # Travel time weight
        beta: float = 0.3,   # Emission weight
        gamma: float = 0.2,  # Road roughness weight
    ) -> SingleRoute:
        """
        Multi-Objective A* search with ARAI emission calibration.
        Cost(edge) = α·TravelTime + β·Emission + γ·Roughness
        """
        emission_factor = EMISSION_FACTORS.get(vehicle_type, 0.140)

        def eco_weight(u, v, data):
            time_cost = data.get("current_time_s", data.get("freeflow_time_s", 100))
            emission_cost = data.get("length_km", 0.1) * emission_factor * 1000  # Scale up
            roughness_cost = data.get("roughness", 0.5) * 100
            return alpha * time_cost + beta * emission_cost + gamma * roughness_cost

        def eco_heuristic(n, target):
            """Haversine distance heuristic for A*."""
            n_data = self._simple_graph.nodes.get(n, {})
            t_data = self._simple_graph.nodes.get(target, {})
            lat1, lon1 = n_data.get("y", 0), n_data.get("x", 0)
            lat2, lon2 = t_data.get("y", 0), t_data.get("x", 0)
            # Approximate distance in meters
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
            dist_m = 6371000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            # Convert to cost units (assume 30 km/h average)
            return alpha * (dist_m / 8.33)

        try:
            path = nx.astar_path(
                self._simple_graph,
                origin_node,
                dest_node,
                heuristic=eco_heuristic,
                weight=eco_weight,
            )
        except nx.NetworkXNoPath:
            logger.warning(f"No eco path found from {origin_node} to {dest_node}")
            return self._empty_route(0)

        route = self._path_to_route(path, 0, vehicle_type, distribution=1.0)
        route.is_eco_optimal = True
        return route

    # ═══════════════════════════════════════════════════════════
    # ROUTING MODE 3: Epsilon-Nash System Balancer
    # ═══════════════════════════════════════════════════════════

    def find_nash_balanced_routes(
        self,
        origin_node: int,
        dest_node: int,
        vehicle_type: str = "car_petrol",
        epsilon: float = 0.10,  # 10% tolerance
        k: int = 3,  # Top-K paths
    ) -> RouteResponse:
        """
        Epsilon-Nash Equilibrium routing.

        1. Find Top-K shortest paths (Yen's Algorithm via NetworkX)
        2. If time difference between paths < epsilon, distribute users
        3. Return 'sub-optimal but stable' paths to prevent Braess Paradox

        The key insight: Google Maps sends everyone to the 'fastest' road → jam.
        We distribute traffic across near-equivalent alternatives.
        """
        try:
            # Yen's K-shortest paths
            paths = list(
                self._k_shortest_paths(origin_node, dest_node, k=k)
            )
        except Exception as e:
            logger.warning(f"K-shortest paths failed: {e}. Falling back to single path.")
            fastest = self.find_fastest_route(origin_node, dest_node, vehicle_type)
            return RouteResponse(
                mode=RoutingMode.NASH,
                routes=[fastest],
                braess_warning=False,
                system_emission_saved_kg=0.0,
            )

        if not paths:
            return RouteResponse(
                mode=RoutingMode.NASH,
                routes=[self._empty_route(0)],
                braess_warning=False,
                system_emission_saved_kg=0.0,
            )

        # Compute route objects with travel times
        routes = []
        for idx, path in enumerate(paths):
            route = self._path_to_route(path, idx, vehicle_type)
            routes.append(route)

        # Sort by travel time
        routes.sort(key=lambda r: r.travel_time_seconds)

        fastest_time = routes[0].travel_time_seconds

        # ── Nash Distribution Logic ──
        # Check if times are within epsilon of each other
        within_epsilon = [
            r for r in routes
            if r.travel_time_seconds <= fastest_time * (1 + epsilon)
        ]

        braess_detected = len(within_epsilon) > 1

        if braess_detected:
            # Distribute: 50% / 30% / 20% (or proportional)
            distribution_weights = self._compute_nash_distribution(within_epsilon)
            for route, weight in zip(within_epsilon, distribution_weights):
                route.distribution_weight = weight

            # Routes outside epsilon get 0 weight
            for r in routes:
                if r not in within_epsilon:
                    r.distribution_weight = 0.0
        else:
            routes[0].distribution_weight = 1.0
            for r in routes[1:]:
                r.distribution_weight = 0.0

        # ── Compute emission savings ──
        # Compare: everyone takes fastest vs Nash-balanced distribution
        fastest_emission = routes[0].emission_kgco2
        nash_emission = sum(r.emission_kgco2 * r.distribution_weight for r in routes)
        savings = max(0, fastest_emission - nash_emission)

        return RouteResponse(
            mode=RoutingMode.NASH,
            routes=routes,
            braess_warning=braess_detected,
            system_emission_saved_kg=round(savings, 4),
        )

    def _k_shortest_paths(self, source: int, target: int, k: int = 3):
        """Yen's K-shortest simple paths using NetworkX."""
        count = 0
        for path in nx.shortest_simple_paths(self._simple_graph, source, target, weight="current_time_s"):
            yield path
            count += 1
            if count >= k:
                break

    @staticmethod
    def _compute_nash_distribution(routes: list[SingleRoute]) -> list[float]:
        """
        Compute Nash equilibrium distribution weights.
        Inversely proportional to travel time (better routes get more traffic).
        """
        times = [r.travel_time_seconds for r in routes]
        inv_times = [1.0 / max(t, 1) for t in times]
        total = sum(inv_times)
        weights = [round(w / total, 3) for w in inv_times]

        # Ensure sums to 1.0
        diff = 1.0 - sum(weights)
        weights[0] += diff

        return weights

    # ═══════════════════════════════════════════════════════════
    # Helper: Convert path to RouteResponse
    # ═══════════════════════════════════════════════════════════

    def _path_to_route(
        self,
        path: list[int],
        index: int,
        vehicle_type: str = "car_petrol",
        distribution: float = 1.0,
    ) -> SingleRoute:
        """Convert a list of node IDs to a SingleRoute with geometry and metrics."""
        emission_factor = EMISSION_FACTORS.get(vehicle_type, 0.140)

        total_time = 0.0
        total_distance_km = 0.0
        total_emission = 0.0
        coordinates = []
        segment_ids = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = self._simple_graph.get_edge_data(u, v) or {}

            total_time += edge_data.get("current_time_s", edge_data.get("freeflow_time_s", 60))
            dist_km = edge_data.get("length_km", 0.1)
            total_distance_km += dist_km
            total_emission += dist_km * emission_factor

            # Add idling emission for slow segments
            speed = edge_data.get("current_speed_kmh", edge_data.get("maxspeed_kmh", 30))
            if speed < 10:
                idling_time_min = (edge_data.get("current_time_s", 60)) / 60
                total_emission += idling_time_min * 0.01  # ARAI idling factor

            segment_ids.append(f"{u}-{v}")

            # Geometry
            if "geometry" in edge_data:
                coords = list(edge_data["geometry"].coords)
                coordinates.extend(coords)
            else:
                u_data = self._simple_graph.nodes[u]
                v_data = self._simple_graph.nodes[v]
                coordinates.append([u_data.get("x", 0), u_data.get("y", 0)])
                if i == len(path) - 2:
                    coordinates.append([v_data.get("x", 0), v_data.get("y", 0)])

        # Deduplicate consecutive identical coordinates
        if coordinates:
            deduped = [coordinates[0]]
            for c in coordinates[1:]:
                if c != deduped[-1]:
                    deduped.append(c)
            coordinates = deduped

        return SingleRoute(
            path_index=index,
            travel_time_seconds=round(total_time, 2),
            distance_km=round(total_distance_km, 3),
            emission_kgco2=round(total_emission, 5),
            distribution_weight=distribution,
            geometry=GeoJSONLineString(coordinates=coordinates if coordinates else [[0, 0], [0, 0]]),
            segments=segment_ids,
            is_eco_optimal=False,
        )

    @staticmethod
    def _empty_route(index: int) -> SingleRoute:
        """Return a placeholder empty route."""
        return SingleRoute(
            path_index=index,
            travel_time_seconds=0,
            distance_km=0,
            emission_kgco2=0,
            distribution_weight=0,
            geometry=GeoJSONLineString(coordinates=[[77.5946, 12.9716], [77.6200, 12.9784]]),
            segments=[],
        )
