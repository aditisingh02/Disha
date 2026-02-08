# How the ML and Training Work in BharatFlow

This doc explains **where training happens**, **what data it uses**, and **how predictions are produced** at inference (API / dashboard).

---

## 1. Training that exists

### A. Orchestrator Phase 2 (GCN–LSTM)

- **Script:** `python -m app.orchestrator.phase2_forecasting` (or run as part of the orchestrator pipeline).
- **Code:** `app/orchestrator/phase2_forecasting.py`
- **What it does:** Trains a **SimpleGCNLSTM** (spatial linear + LSTM) on `training_dataset_enriched.csv` with a **temporal split** (70% train / 15% val / 15% test, no shuffle). Target: speed at 1h, 6h, 24h ahead.
- **Outputs:** `data/orchestrator_output/phase2_model.pt`, `phase2_metrics.json` (test MAE, improvement vs baseline, leakage check).
- **Inference:** The Phase 2 model is **not** currently loaded by any API. The orchestrator route-forecast and multi-horizon logic use **historical (road_name, hour) means** from the same CSV instead (see §3).

### B. Standalone ML training scripts

All under `app/ml/`, writing to `ml_models/weights/` (or equivalent):

| Script | Model | Data | Output |
|--------|--------|------|--------|
| `train.py` | ST-GCN (PyTorch) | `data/datasets/{city}_train.csv` (synthetic) | `stgcn_{city}_v1.pth` |
| `train_v2.py` | Advanced ST-GCN (48 horizons) | `{city}_train_v2.csv` | `stgcn_{city}_v2.pth` |
| `train_baseline.py` | 48× LightGBM (one per horizon) | `datasets_v2/{city}_train_v2.csv` | `baseline_{city}_v1.pkl` + regime 75th %ile |
| `train_residual_gcn.py` | Residual GCN | V2 dataset | `residual_gcn_{city}_v1.pth` |
| `train_fallback.py` | Fallback trainer | Synthetic | Fallback weights |

- **Data:** V1 uses `data/datasets/` (e.g. from `data_generator`); V2 uses `data/datasets_v2/` (see `synthetic_data_generator_v2.py` or similar). The **orchestrator** uses only `data/datasets/training_dataset_enriched.csv` (Phase 1 baselines + Phase 2).
- **You run training explicitly**, e.g.:
  - `python -m app.ml.train_v2` (for ST-GCN V2),
  - `python -m app.ml.train_baseline` (for LightGBM baseline),
  - `python -m app.orchestrator.phase2_forecasting` (for Phase 2 GCN–LSTM).

---

## 2. Where inference gets its “ML”

### GET `/predict/traffic` and simulation loop

- **Source:** Simulation loop (`app/core/simulation.py`) calls `forecaster.predict(readings, city=...)`.
- **Forecaster:** `app/engine/forecaster.py` (TrafficForecaster).
  - Loads **ST-GCN V2** from `ml_models/weights/stgcn_{city}_v2.pth` (produced by `train_v2.py`).
  - If that file is missing, the model is **random-initialized** (no real learned weights).
  - Can optionally use **Residual GCN** from `residual_gcn_{city}_v1.pth` (from `train_residual_gcn.py`) and **LightGBM baseline** from `baseline_{city}_v1.pkl` (from `train_baseline.py`) for regime-gated / fallback behaviour.

So **real ML** for the main prediction API and dashboard stream is:

- **ST-GCN V2** (trained by `train_v2.py`) → used if weights exist.
- **LightGBM baseline** (trained by `train_baseline.py`) → used for baseline/regime logic if present.
- **Residual GCN** (trained by `train_residual_gcn.py`) → used if present.

### POST `/predict/route-forecast`

- Uses the same **TrafficForecaster** (ST-GCN V2 + baseline/residual when available) to produce route-level hourly speeds (e.g. 48 steps) and optional q50/q90.

### POST `/orchestrator/route-forecast` (terminal-equivalent, dashboard)

- **Does not use the Phase 2 model or ST-GCN.**
- **Logic:** `app/orchestrator/multi_horizon.py` → `predict_multi_horizon()`.
- **How it works:** Loads `training_dataset_enriched.csv`, computes **historical (road_name, hour) mean speeds** from the training split, applies an **event factor** (e.g. accident 0.7, weather 0.85), and returns 1h–24h horizons with congestion classification (from `congestion.py`). So this path is **rule-based / historical averages**, not a trained neural model.

---

## 3. Summary: “Is there training? How is the ML happening?”

- **Yes, there is training:**
  - **Orchestrator:** Phase 2 trains a GCN–LSTM and saves `phase2_model.pt`; Phase 1 baselines (Historical Average, ARIMA) are computed on the same CSV.
  - **Standalone:** `train_v2.py`, `train_baseline.py`, `train_residual_gcn.py`, etc., train on synthetic/V2 datasets and write weights to `ml_models/weights/`.

- **How the ML is used at inference:**
  - **ST-GCN / route-forecast (main API):** If you have run `train_v2.py` (and optionally `train_baseline.py`, `train_residual_gcn.py`) and the weights exist, the **engine forecaster** loads them and predictions are real **trained ML**.
  - **Orchestrator route-forecast (terminal + dashboard):** Today this uses **historical (road_name, hour) means** from the CSV plus event factors and congestion rules — **no** Phase 2 model and **no** ST-GCN in this path. So the “ML” in that flow is the **data-driven historical baseline**, not the trained GCN–LSTM.

- **Gap:** The **Phase 2 model** (`phase2_model.pt`) is trained but **never loaded for any API**. To use it you’d add an inference path (e.g. in `multi_horizon.py` or a new orchestrator forecaster) that loads `phase2_model.pt` and replaces (or complements) the historical-mean logic.

---

## 4. Quick reference: run training before “real” ML

1. **Generate / have data:**  
   - Orchestrator: `training_dataset_enriched.csv` in `data/datasets/`.  
   - V2 ST-GCN / baseline: e.g. `data/datasets_v2/bengaluru_train_v2.csv` (see `app/ml/synthetic_data_generator_v2.py` or data pipeline).

2. **Run orchestrator (baselines + Phase 2 GCN–LSTM):**  
   - Phase 1: baselines (Historical Avg, ARIMA) + dataset profile, road network.  
   - Phase 2: `python -m app.orchestrator.phase2_forecasting` → produces `phase2_model.pt` and metrics (not yet used at inference).

3. **Run standalone ML (so API uses trained models):**  
   - `python -m app.ml.train_baseline` → `baseline_bengaluru_v1.pkl` (used by forecaster if present).  
   - `python -m app.ml.train_v2` → `stgcn_bengaluru_v2.pth` (main ST-GCN used by `/predict/traffic` and route-forecast).  
   - Optionally: `python -m app.ml.train_residual_gcn` → residual GCN for regime-aware prediction.

4. **Result:**  
   - **With weights:** `/predict/traffic` and `/predict/route-forecast` use **trained** ST-GCN + baseline (and optionally residual GCN).  
   - **Orchestrator route-forecast** (terminal + dashboard) still uses **historical means** until you wire Phase 2 (or another model) into that path.
