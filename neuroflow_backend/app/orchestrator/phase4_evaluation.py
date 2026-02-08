"""
Phase 4 — Integration evaluation: ablation/robustness checks, metric-backed summary.
Uses only orchestrator outputs (no external data).
"""

import json
import logging
from pathlib import Path

from app.orchestrator.config import ORCHESTRATOR_OUTPUT_DIR, BASELINE_METRICS_PATH

logger = logging.getLogger("neuroflow.orchestrator.phase4")

PHASE4_OUTPUT = ORCHESTRATOR_OUTPUT_DIR / "phase4_evaluation.json"


def run_evaluation() -> dict:
    """Ablation: baseline vs Phase 2. Robustness: leakage check, temporal split reported."""
    out = {"ablation": {}, "robustness": {}, "pitch_metrics": {}}
    if BASELINE_METRICS_PATH.exists():
        with open(BASELINE_METRICS_PATH) as f:
            b = json.load(f)
        out["ablation"]["baseline_historical_avg_test_mae"] = b.get("historical_average", {}).get("test_mae")
        out["ablation"]["baseline_persistence_test_mae"] = b.get("arima_or_persistence", {}).get("test_mae")
    p2 = ORCHESTRATOR_OUTPUT_DIR / "phase2_metrics.json"
    if p2.exists():
        with open(p2) as f:
            p = json.load(f)
        out["ablation"]["phase2_stgcn_test_mae"] = p.get("test_mae")
        out["robustness"]["leakage_check_passed"] = p.get("leakage_check_passed")
        out["robustness"]["temporal_split"] = "train 70% / val 15% / test 15% (chronological)"
        out["pitch_metrics"]["forecast_baseline_mae"] = out["ablation"].get("baseline_historical_avg_test_mae")
        out["pitch_metrics"]["forecast_stgcn_mae"] = p.get("test_mae")
    p3 = ORCHESTRATOR_OUTPUT_DIR / "phase3_innovation.json"
    if p3.exists():
        with open(p3) as f:
            p = json.load(f)
        out["pitch_metrics"]["risk_hotspots_count"] = p.get("dynamic_risk_fields", {}).get("risk_tensor_summary", {}).get("hotspot_count")
        out["pitch_metrics"]["event_impact_speed_kmh"] = p.get("event_impact_encoder", {}).get("attribution_scores", {}).get("event_impact_on_speed_kmh")
    ORCHESTRATOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PHASE4_OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Phase 4 evaluation saved to %s", PHASE4_OUTPUT)
    return out


if __name__ == "__main__":
    run_evaluation()
