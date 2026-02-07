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
- `backend/`: FastAPI application.
- `frontend/`: React dashboard application.
- `data/`: Raw and processed data storage.
- `ml/`: Machine learning notebooks, models, and scripts.
- `simulation/`: Simulation and historical replay logic.
- `docs/`: Project documentation and logs.

