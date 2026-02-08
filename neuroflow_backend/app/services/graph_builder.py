"""
NeuroFlow BharatFlow — Synthetic Road Graph Builder
Generates a realistic NetworkX MultiDiGraph for the Bengaluru
Silk Board → Indiranagar corridor.  Fully self-contained — no OSMnx required.
"""

import asyncio
import logging
import math
import os
import pickle
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from app.core.config import settings

logger = logging.getLogger("neuroflow.graph_builder")

# ── Default free-flow speeds for Indian road classes (km/h) ──
INDIA_DEFAULT_SPEEDS = {
    "motorway": 80, "motorway_link": 60,
    "trunk": 60, "trunk_link": 40,
    "primary": 50, "primary_link": 35,
    "secondary": 40, "secondary_link": 30,
    "tertiary": 30, "tertiary_link": 25,
    "residential": 25, "living_street": 15,
    "unclassified": 30, "service": 15,
}

# ── Named roads in the corridor ──
_ROAD_NAMES = [
    "PIE (Pan Island Expressway)", "AYE (Ayer Rajah Expressway)", 
    "CTE (Central Expressway)", "TPE (Tampines Expressway)",
    "ECP (East Coast Parkway)", "KPE (Kallang-Paya Lebar Expressway)",
    "SLE (Seletar Expressway)", "MCE (Marina Coastal Expressway)",
    "Orchard Road", "Bukit Timah Road", "Upper Thomson Road",
    "Serangoon Road", "Jalan Bukit Merah", "River Valley Road",
    "Victoria Street", "North Bridge Road", "Nicoll Highway",
    "Dunearn Road", "Adam Road", "Lornie Road", "Braddell Road",
    "Bartley Road", "Paya Lebar Road", "Sims Avenue", "Geylang Road",
]


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in metres between two lat/lng pairs."""
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class GraphBuilderService:
    """Manages the Bengaluru road network graph (synthetic)."""

    def __init__(self) -> None:
        self._graph: Optional[nx.MultiDiGraph] = None
        self._cache_path = Path(settings.cache_dir) / "singapore_synth_graph.pkl"

    # ── Public API ─────────────────────────────────────────

    async def initialize(self) -> None:
        """Load from cache or build a new synthetic graph."""
        if self._cache_path.exists():
            logger.info(f"Loading cached graph from {self._cache_path}")
            await asyncio.to_thread(self._load_from_cache)
        else:
            logger.info("Building synthetic Bengaluru corridor road graph…")
            await asyncio.to_thread(self._build_synthetic_graph)
            await asyncio.to_thread(self._save_to_cache)

        self._enrich_edges()
        logger.info(
            f"Graph ready: {self._graph.number_of_nodes()} nodes, "
            f"{self._graph.number_of_edges()} edges"
        )

    def is_ready(self) -> bool:
        return self._graph is not None

    def get_graph(self) -> nx.MultiDiGraph:
        if self._graph is None:
            raise RuntimeError("Graph not initialized. Call initialize() first.")
        return self._graph

    def get_nearest_node(self, lat: float, lng: float) -> int:
        """Nearest graph node to given lat/lng (squared-Euclidean approx)."""
        if self._graph is None:
            raise RuntimeError("Graph not initialized.")
        best, best_d = 0, float("inf")
        for n, d in self._graph.nodes(data=True):
            dist = (d["y"] - lat) ** 2 + (d["x"] - lng) ** 2
            if dist < best_d:
                best, best_d = n, dist
        return best

    def get_edge_data(self, u: int, v: int) -> dict:
        edges = self._graph.get_edge_data(u, v)
        if edges:
            return edges[next(iter(edges))]
        return {}

    def update_edge_speed(self, u: int, v: int, speed_kmh: float) -> None:
        """Live-update edge weight from simulation tick."""
        if u not in self._graph or v not in self._graph[u]:
            return
        for key in self._graph[u][v]:
            data = self._graph[u][v][key]
            data["current_speed_kmh"] = speed_kmh
            data["current_time_s"] = data["length_m"] / max(speed_kmh / 3.6, 0.1)

    def get_segment_geometries(self) -> list[dict]:
        """Road segments as GeoJSON features for the frontend."""
        features: list[dict] = []
        for u, v, key, data in self._graph.edges(keys=True, data=True):
            ud = self._graph.nodes[u]
            vd = self._graph.nodes[v]
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[ud["x"], ud["y"]], [vd["x"], vd["y"]]],
                },
                "properties": {
                    "segment_id": f"singapore_{u}-{v}-{key}",
                    "maxspeed_kmh": data.get("maxspeed_kmh", 30),
                    "current_speed_kmh": data.get(
                        "current_speed_kmh", data.get("maxspeed_kmh", 30)
                    ),
                    "length_km": data.get("length_km", 0),
                    "road_type": data.get("highway", "unclassified"),
                    "roughness": data.get("roughness", 0.5),
                    "name": data.get("name", "Unknown Road"),
                },
            })
        return features

    # ── Synthetic graph construction ───────────────────────

    def _build_synthetic_graph(self) -> None:
        """
        Create a realistic road network for the
        Silk Board → Indiranagar corridor (~7 km × 4 km).

        Grid: 10 rows (N-S) × 7 cols (E-W) = 70 intersections
        plus diagonal cross-streets and ring-road jumps → ~400+ directed edges.
        """
        G = nx.MultiDiGraph()
        rng = np.random.default_rng(42)

        rows, cols = 10, 7
        lat_lo, lat_hi = 1.26, 1.45
        lng_lo, lng_hi = 103.75, 103.98

        lats = np.linspace(lat_lo, lat_hi, rows)
        lngs = np.linspace(lng_lo, lng_hi, cols)

        # ── nodes ──
        nm: dict[tuple[int, int], int] = {}
        nid = 0
        for r in range(rows):
            for c in range(cols):
                lat = float(lats[r] + rng.uniform(-0.0003, 0.0003))
                lng = float(lngs[c] + rng.uniform(-0.0003, 0.0003))
                G.add_node(nid, y=lat, x=lng, street_count=int(rng.integers(2, 5)))
                nm[(r, c)] = nid
                nid += 1

        _ri = [0]

        def _name() -> str:
            n = _ROAD_NAMES[_ri[0] % len(_ROAD_NAMES)]
            _ri[0] += 1
            return n

        def _bidir(u: int, v: int, hw: str, name: str = "") -> None:
            ud, vd = G.nodes[u], G.nodes[v]
            length = _haversine_m(ud["y"], ud["x"], vd["y"], vd["x"])
            attrs = dict(
                highway=hw,
                name=name or _name(),
                length=round(length, 1),
                oneway=False,
                lanes="2" if hw in ("trunk", "primary") else "1",
                maxspeed=str(INDIA_DEFAULT_SPEEDS.get(hw, 30)),
            )
            G.add_edge(u, v, 0, **attrs)
            G.add_edge(v, u, 0, **attrs)

        # ── E-W roads (horizontal) ──
        ew = ["secondary", "primary", "trunk", "secondary", "tertiary",
              "primary", "secondary", "trunk", "tertiary", "primary"]
        for r in range(rows):
            rn = _name()
            for c in range(cols - 1):
                _bidir(nm[(r, c)], nm[(r, c + 1)], ew[r], rn)

        # ── N-S roads (vertical) ──
        ns = ["primary", "secondary", "trunk", "secondary", "trunk", "secondary", "primary"]
        for c in range(cols):
            rn = _name()
            for r in range(rows - 1):
                _bidir(nm[(r, c)], nm[(r + 1, c)], ns[c], rn)

        # ── Diagonal cross-streets ──
        for r in range(rows - 1):
            for c in range(cols - 1):
                if (r + c) % 3 == 0:
                    _bidir(nm[(r, c)], nm[(r + 1, c + 1)], "tertiary")
                if (r + c) % 4 == 1 and c > 0:
                    _bidir(nm[(r, c)], nm[(r + 1, c - 1)], "residential")

        # ── Ring-road long jumps ──
        mid = cols // 2
        for r in range(0, rows - 2, 2):
            _bidir(nm[(r, mid)], nm[(r + 2, mid)], "trunk", "Inner Ring Road")

        # ── Extra skip-links for route diversity ──
        _bidir(nm[(0, 0)], nm[(2, 2)], "secondary", "BTM Shortcut")
        _bidir(nm[(3, 4)], nm[(5, 6)], "secondary", "Koramangala Bypass")
        _bidir(nm[(6, 1)], nm[(8, 3)], "secondary", "Domlur Link Road")
        _bidir(nm[(7, 5)], nm[(9, 6)], "tertiary", "HAL Link")

        self._graph = G
        os.makedirs(self._cache_path.parent, exist_ok=True)

    # ── Persistence ────────────────────────────────────────

    def _load_from_cache(self) -> None:
        with open(self._cache_path, "rb") as f:
            self._graph = pickle.load(f)

    def _save_to_cache(self) -> None:
        os.makedirs(self._cache_path.parent, exist_ok=True)
        with open(self._cache_path, "wb") as f:
            pickle.dump(self._graph, f, protocol=5)
        logger.info(f"Graph cached to {self._cache_path}")

    # ── Edge enrichment ────────────────────────────────────

    def _enrich_edges(self) -> None:
        """Impute speeds, compute travel time, emission cost, roughness."""
        for u, v, key, data in self._graph.edges(keys=True, data=True):
            # ── Impute maxspeed ──
            if "maxspeed" not in data or data["maxspeed"] is None:
                hw = data.get("highway", "unclassified")
                if isinstance(hw, list):
                    hw = hw[0]
                speed = INDIA_DEFAULT_SPEEDS.get(hw, 30)
            else:
                raw = data["maxspeed"]
                if isinstance(raw, list):
                    raw = raw[0]
                try:
                    speed = float(str(raw).replace(" km/h", "").replace("mph", "").strip())
                except (ValueError, TypeError):
                    speed = 30
            data["maxspeed_kmh"] = speed

            length_m = data.get("length", 100.0)
            data["length_m"] = length_m
            data["length_km"] = length_m / 1000.0

            speed_ms = speed / 3.6
            data["freeflow_time_s"] = length_m / max(speed_ms, 0.1)
            data["emission_cost_kg"] = data["length_km"] * settings.emission_car_petrol
            data["current_time_s"] = data["freeflow_time_s"]

            roughness_map = {
                "motorway": 0.1, "trunk": 0.2, "primary": 0.3,
                "secondary": 0.5, "tertiary": 0.7, "residential": 0.8,
                "unclassified": 0.9, "service": 0.9,
            }
            ht = data.get("highway", "unclassified")
            if isinstance(ht, list):
                ht = ht[0]
            data["roughness"] = roughness_map.get(ht, 0.5)
