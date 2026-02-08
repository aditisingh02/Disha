"""
Phase 1 — Dataset profiling. Log every step; output schema, missing_value_report, temporal_coverage.
Does not mutate raw data.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from app.orchestrator.config import PRIMARY_DATASET_PATH, ORCHESTRATOR_OUTPUT_DIR, PROFILE_OUTPUT_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
)
logger = logging.getLogger("neuroflow.orchestrator.profiling")


def run_profiling() -> dict:
    """
    Profile training_dataset_enriched.csv. Outputs schema, missing_value_report, temporal_coverage.
    Writes to PROFILE_OUTPUT_PATH and returns the same dict.
    """
    logger.info("Dataset profiling started — primary_dataset=%s", str(PRIMARY_DATASET_PATH))
    if not PRIMARY_DATASET_PATH.exists():
        raise FileNotFoundError(f"Primary dataset not found: {PRIMARY_DATASET_PATH}")

    # Step 1: Load (read-only)
    df = pd.read_csv(PRIMARY_DATASET_PATH)
    logger.info("Preprocessing step: load_csv rows=%d columns=%d", len(df), len(df.columns))

    # Schema
    schema = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        schema.append({"name": col, "dtype": dtype})
    logger.info("Schema: %d columns", len(schema))

    # Missing value report
    missing = df.isnull().sum()
    missing_report = {col: int(missing[col]) for col in df.columns if missing[col] > 0}
    if not missing_report:
        missing_report = {"_message": "No missing values in any column"}
    else:
        logger.info("Missing value report: %s", missing_report)

    # Temporal coverage
    if "timestamp" not in df.columns:
        temporal_coverage = {"error": "No timestamp column"}
        logger.warning("No timestamp column for temporal coverage")
    else:
        df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
        valid_ts = df["_ts"].dropna()
        if len(valid_ts) == 0:
            temporal_coverage = {"error": "No valid timestamps"}
        else:
            t_min = valid_ts.min()
            t_max = valid_ts.max()
            temporal_coverage = {
                "min": t_min.isoformat(),
                "max": t_max.isoformat(),
                "span_days": (t_max - t_min).days,
                "unique_timestamps": int(valid_ts.nunique()),
            }
        logger.info("Temporal coverage: %s", temporal_coverage)

    # Segment/geometry summary (for road network)
    geometry_summary = {}
    if "latitude" in df.columns and "longitude" in df.columns:
        geometry_summary["unique_lat_lon_pairs"] = int(df.groupby(["latitude", "longitude"]).ngroups)
    if "road_name" in df.columns:
        geometry_summary["unique_road_names"] = int(df["road_name"].nunique())
    if "road_category" in df.columns:
        geometry_summary["road_categories"] = df["road_category"].astype(str).unique().tolist()
    logger.info("Geometry summary: %s", geometry_summary)

    # Target variable (speed_band) summary
    target_summary = {}
    if "speed_band" in df.columns:
        target_summary["speed_band_value_counts"] = df["speed_band"].value_counts().sort_index().to_dict()
        target_summary["speed_band_min_max"] = [int(df["speed_band"].min()), int(df["speed_band"].max())]
    logger.info("Target (speed_band) summary: %s", target_summary)

    out = {
        "schema": schema,
        "missing_value_report": missing_report,
        "temporal_coverage": temporal_coverage,
        "geometry_summary": geometry_summary,
        "target_summary": target_summary,
        "n_rows": len(df),
        "profiled_at": datetime.now(timezone.utc).isoformat(),
    }

    ORCHESTRATOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Profile written to %s", PROFILE_OUTPUT_PATH)
    return out


if __name__ == "__main__":
    run_profiling()
