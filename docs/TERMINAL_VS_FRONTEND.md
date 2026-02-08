# Terminal vs Frontend Data Flow

The same orchestrator output that you see in the **terminal CLI** (when running `python -m app.orchestrator.terminal_mode`) is now available on the **frontend** (Google Maps flow) and **reported on the dashboard** (Sidebar + Forecast modal).

## Google Maps API compliance

- **Directions API:** Used with `travelMode: DRIVING`, `drivingOptions.departureTime` (current time), and `trafficModel: BEST_GUESS` so `duration_in_traffic` is returned where supported. Route errors (e.g. `ZERO_RESULTS`) clear the route and computing state so the dashboard and Sidebar stay in sync.
- **Traffic Layer:** Enabled on the map for live traffic display.
- **Map:** Loaded via `@react-google-maps/api` (LoadScript + GoogleMap); origin/destination markers and traffic-colored polylines are drawn from the directions result.

## API that mirrors terminal output

- **Endpoint:** `POST /api/v1/orchestrator/route-forecast`
- **Body:** `origin_lat`, `origin_lon`, `destination_lat`, `destination_lon` (and optional `departure_time`, `event_context`), or `origin_road` / `destination_road` by name.
- **Response:** Same shape as terminal:
  - `multi_horizon_forecasts` — 1h, 3h, 6h, 12h, 24h (speed_kmh, level, score, delay_vs_freeflow_min, CI)
  - `congestion_classification` — level (FREE_FLOW → GRIDLOCK), score, delay_vs_freeflow_min, speed_ratio
  - `risk` — origin_risk, destination_risk, hotspot_count
  - `routes` — path, total_km, fastest_route_time_min, eco_route_time_min, emissions, percent_emission_reduction
  - `model_version`

## How the frontend uses it (live dynamic dashboard)

1. **MapView:** When the user sets origin and destination and a route is computed (Google Directions), the frontend calls `getOrchestratorRouteForecast({ origin_lat, origin_lon, destination_lat, destination_lon })` and stores the result in `trafficStore.orchestratorRouteForecast`. Route errors clear `googleRoute` and computing state so the UI never gets stuck.
2. **Sidebar (dashboard):** When a route exists and `orchestratorRouteForecast` is present, the Route Analytics card shows an **Orchestrator Forecast** line: congestion level, eco emission reduction %, and a **View full forecast** link that opens the modal.
3. **ForecastModal:** When the user opens the “12-Hour Traffic Forecast” modal and `orchestratorRouteForecast` is present, the modal shows an **“Orchestrator Route Forecast (same as terminal)”** section with:
   - Multi-horizon table (horizon, speed, level, score, delay, CI 80%)
   - Congestion classification (level, score, delay, speed ratio)
   - Risk (origin/destination risk, hotspot count)
   - Routes (path, total km, fastest/eco times, emission reduction)
4. **Live status:** Header and bottom bar show **ONLINE/Live** when either WebSocket is connected or REST data was updated in the last 35 seconds.
5. **Clear route:** When the user clicks “Clear” on the route, both `routeForecast` and `orchestratorRouteForecast` are cleared so the modal does not show stale data.

## Parity

- Terminal: `INPUT_RECEIVED` → `MODEL_OUTPUT` (multi_horizon + congestion) → `RISK_ANALYSIS` → `ROUTING_COMPARISON` → `UNCERTAINTY_ESTIMATES` → `FINAL_OUTPUTS`.
- Frontend: The same multi-horizon, congestion, risk, and routes data is shown in the forecast modal when a route is active; the chart above still uses the existing route forecast (hourly_speeds, q50/q90) from `POST /predict/route-forecast`.
