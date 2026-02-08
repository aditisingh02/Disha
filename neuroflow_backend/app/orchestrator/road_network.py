"""
Phase 1 — Road network construction from dataset geometry.
Source: dataset geometry columns (latitude, longitude, road_name). Graph: directed, weighted.
Does not mutate raw data; writes graph to ORCHESTRATOR_OUTPUT_DIR.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from app.orchestrator.config import PRIMARY_DATASET_PATH, ORCHESTRATOR_OUTPUT_DIR, ROAD_GRAPH_PATH

logger = logging.getLogger("neuroflow.orchestrator.road_network")


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    a = np.sin(np.radians(lat2 - lat1) / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def build_road_network(distance_km_threshold: float = 5.0) -> nx.DiGraph:
    """
    Build directed weighted graph from dataset. One node per road_name; centroid from (latitude, longitude).
    Edges: between nodes whose centroid distance < distance_km_threshold. Weight = distance_km.
    """
    logger.info("Road network construction started — source=%s", str(PRIMARY_DATASET_PATH))
    if not PRIMARY_DATASET_PATH.exists():
        raise FileNotFoundError(f"Primary dataset not found: {PRIMARY_DATASET_PATH}")

    df = pd.read_csv(PRIMARY_DATASET_PATH)
    logger.info("Preprocessing step: load_csv for road network rows=%d", len(df))

    # One node per road_name: centroid (mean lat, mean lon)
    centroids = df.groupby("road_name").agg({"latitude": "mean", "longitude": "mean"}).reset_index()
    road_names = centroids["road_name"].tolist()
    n = len(road_names)
    name_to_idx = {name: i for i, name in enumerate(road_names)}

    G = nx.DiGraph()
    for i, row in centroids.iterrows():
        node_id = row["road_name"]
        G.add_node(
            node_id,
            y=float(row["latitude"]),
            x=float(row["longitude"]),
            road_category=int(df[df["road_name"] == node_id]["road_category"].iloc[0]) if "road_category" in df.columns else 0,
        )
    logger.info("Added %d nodes (one per road_name)", G.number_of_nodes())

    # Edges: pairs within distance threshold (bidirectional for undirected connectivity, we use DiGraph with both directions)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lat1, lon1 = centroids.iloc[i]["latitude"], centroids.iloc[i]["longitude"]
            lat2, lon2 = centroids.iloc[j]["latitude"], centroids.iloc[j]["longitude"]
            d = _haversine_km(lat1, lon1, lat2, lon2)
            if d <= distance_km_threshold:
                u, v = road_names[i], road_names[j]
                G.add_edge(u, v, weight=d, length_km=d)
                G.add_edge(v, u, weight=d, length_km=d)
    logger.info("Added edges (distance_km <= %.1f): %d edges", distance_km_threshold, G.number_of_edges())

    ORCHESTRATOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ROAD_GRAPH_PATH, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Road network saved to %s", ROAD_GRAPH_PATH)
    return G


def load_road_network() -> nx.DiGraph:
    if not ROAD_GRAPH_PATH.exists():
        return build_road_network()
    with open(ROAD_GRAPH_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    build_road_network()
