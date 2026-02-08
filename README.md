# Mumbai Urban Traffic Intelligence Platform

## Project Directive

**City Focus:** Mumbai, India
**Phase:** Phase 1 – Predictive Decision Support (Mumbai)

### Problem Framing
Develop a city-scale traffic forecasting and decision-support system for Mumbai that predicts corridor-level congestion using GPS probe data, provides situational awareness through crowd-sourced incidents, and supports human-in-the-loop traffic management decisions.

**Realism Constraints:**
- No assumption of public loop-sensor availability in India.
- Forecasting based on travel-time and speed indices.
- Explicit use of simulation and historical replay.
- No real-time signal automation claims.

### Core Philosophy: Three-Layer Model
1.  **Predictive:** Forecast congestion and travel times (1–24 hours).
2.  **Perceptive:** Surface real-time disruptions explaining deviations.
3.  **Prescriptive:** Support controller decisions (no direct automated control in Phase 1).

### Data Strategy
*   **Primary Predictive Data:** Uber Movement – Mumbai (aggregated travel time/speed).
*   **Benchmark Data:** METR-LA (for ML validation only).
*   **Perceptive Data (Future):** Waze (crowd-sourced incidents).
*   **Structural Context:** OpenStreetMap (road hierarchy, corridors, bottlenecks).

### ML Pipeline
*   **Objectives:** Predict Travel Time Index (TTI) and congestion risk.
*   **Models:** XGBoost (Primary), LightGBM (Secondary). LSTM optional.
*   **Validation:** Strict chronological split, rolling window validation.

### Technical Stack
*   **Backend:** FastAPI
*   **Frontend:** React (Dashboard with Mapbox/Leaflet integration for Mumbai overlays)
*   **Analysis:** Python, Pandas, Jupyter

### Dashboard Design
*   **Goal:** Operational traffic control console (not a startup analytics product).
*   **Layout:** Top bar (system state), Map panel (corridor intelligence), Forecast panel (risk/confidence), Controller context mechanism.

### Engineering Guardrails
*   ML forecasting independent of perception inputs.
*   Perception affects confidence, not predictions.
*   UI decoupled from ML logic.
*   All architectural decisions logged in `docs/changes.md`.

## Directory Structure
- `neuroflow_backend/`: FastAPI application & Orchestrator.
- `neuroflow_frontend/`: React dashboard application.
- `docs/`: Project documentation and logs.

---

## 🚀 Setup & Execution Guide

### Prerequisites
- **Python 3.10+**
- **Node.js 18+**

### 1. Backend Setup
```bash
cd neuroflow_backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Frontend Setup
```bash
cd neuroflow_frontend
npm install
```

### 3. Running the System

#### Option A: Orchestrator Terminal Mode ( verification & training)
Run the full 4-Phase pipeline (Profiling -> Forecasting -> Innovation -> Evaluation) in your terminal. This trains the models and verifies system integrity.
```bash
cd neuroflow_backend
# Run full timeline
python -m app.orchestrator.terminal_mode --timeline

# Run interactive CLI demo
python -m app.orchestrator.terminal_mode --cli
```

#### Option B: Full Stack (Interactive Dashboard)
Run the backend and frontend in separate terminals.

**Terminal 1 (Backend):**
```bash
cd neuroflow_backend
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 (Frontend):**
```bash
cd neuroflow_frontend
npm run dev
```

Open your browser at [http://localhost:5173](http://localhost:5173).

### Key Features
- **Phase 2 Forecasting:** ST-GCN model predicting traffic 12 hours ahead with uncertainty bands.
- **Eco-Routing:** Calibration for Indian driving cycles to minimize emissions.
- **Risk Heatmaps:** Dynamic identification of high-risk congestion zones.

