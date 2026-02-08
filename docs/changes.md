# Forecast Improvement — Change Log

All architectural decisions and forecast pipeline changes are logged here.  
**Rules:** Log every role change explicitly; log why median accuracy was preserved; log peak-risk improvements separately; timestamp all entries.

---

## 2025-02-08 — Forecast improvement execution (plan bootstrap)

- **Change:** Introduced `docs/changes.md` and baseline MAE safeguard script.
- **Median accuracy:** N/A (baseline not yet implemented).
- **Peak-risk:** N/A.
- **Notes:** Placeholder for iteration logging per plan. `app/ml/check_baseline_mae.py` added to abort if baseline MAE degrades >2% vs stored reference.

---

## 2025-02-08 — Full forecast improvement implementation

- **Change:** LightGBM baseline (q50), regime-gated Residual GCN (q90), pinball loss, spatial features, evaluation, API/UI.
- **Median accuracy:** Baseline owns q50; trained with MAE; safeguard prevents >2% regression via `check_baseline_mae.py`.
- **Peak-risk:** Regime gating activates residual only during peak/weather/holiday; evaluation promotes peak-hour MAE, peak miss rate, q90 coverage, tail pinball; UI shows q90 escalation warnings and uncertainty band.
- **Notes:** `app/engine/baseline_forecaster.py`, `app/engine/regime.py`, `app/engine/spatial_features.py`, `ResidualGCN` in models; `train_baseline.py`, `train_residual_gcn.py`, `evaluate.py`; schemas extended with `hourly_speeds_q50`, `hourly_speeds_q90`; ForecastModal/StatsPanel use q50/q90 and escalation banner; simulation WS includes `predictions` with q50/q90.

---

## 2025-02-08 — NeuroFlow-Orchestrator (training_dataset_enriched)

- **Change:** Orchestrator pipeline using **only** `training_dataset_enriched.csv` (single source of truth). No raw data mutation; temporal train/val/test split; leakage checks.
- **Phase 1:** Dataset profiling (schema, missing report, temporal coverage), road network construction (directed weighted graph from lat/lon + road_name), baseline models (Historical Average, ARIMA/persistence) with MAE/RMSE on test.
- **Phase 2:** Multi-horizon ST-GCN (GCN+LSTM) training; validation reports test MAE and ≥30% improvement check; leakage_check_passed enforced.
- **Phase 3:** Dynamic Risk Fields (risk tensor summary, hotspots), GreenWave Eco-Routing (emission proxy), Event Impact Encoder (attribution from has_major_event).
- **Frontend:** Phase 1–3 reflected in `Phase1Foundation` panel (dataset profile, baselines, road network, Phase 2 metrics, Phase 3 innovation). API: `/api/v1/orchestrator/*`.
