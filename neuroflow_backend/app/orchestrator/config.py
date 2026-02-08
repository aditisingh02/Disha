"""
Orchestrator config: primary dataset path only. No overwrite of raw data.
"""

from pathlib import Path

# SINGLE_SOURCE_OF_TRUTH — do not mutate this file
PRIMARY_DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "datasets" / "training_dataset_enriched.csv"
# Derived outputs (logs, profiles, graphs, models) — never write to PRIMARY_DATASET_PATH
ORCHESTRATOR_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "orchestrator_output"
PROFILE_OUTPUT_PATH = ORCHESTRATOR_OUTPUT_DIR / "dataset_profile.json"
ROAD_GRAPH_PATH = ORCHESTRATOR_OUTPUT_DIR / "road_network_graph.pkl"
BASELINE_METRICS_PATH = ORCHESTRATOR_OUTPUT_DIR / "baseline_metrics.json"
