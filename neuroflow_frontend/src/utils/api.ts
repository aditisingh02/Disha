/** API client for NeuroFlow backend */
import { API_BASE } from './constants';
import type {
  RouteRequest,
  RouteResponse,
  TrafficPrediction,
  HeatmapPoint,
  CorridorStats,
  EmissionComparison,
  BraessParadoxData,
  SystemHealth,
  GeoFeatureCollection,
} from '@/types';

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

// ── Traffic ────────────────────────────────────────────
/** Response from GET /predict/traffic (has predictions array) */
export function getTrafficPredictionsResponse(): Promise<{
  predictions: TrafficPrediction[];
  timestamp: string;
  model_version: string;
}> {
  return fetchJSON('/predict/traffic');
}

export function getPredictions(): Promise<TrafficPrediction[]> {
  return getTrafficPredictionsResponse().then((r) => r.predictions ?? []);
}

export function getRiskHeatmap(): Promise<HeatmapPoint[]> {
  return fetchJSON('/risk/heatmap');
}

export function getTrafficSegments(): Promise<GeoFeatureCollection> {
  return fetchJSON('/traffic/segments');
}

/** Full live payload: readings array + timestamp + tick (for dynamic store updates) */
export function getLiveReadings(): Promise<{
  readings: Array<Record<string, unknown>>;
  timestamp: string;
  tick?: number;
}> {
  return fetchJSON('/traffic/live');
}

export function getLiveTraffic(): Promise<{
  readings_count: number;
  predictions_count: number;
  risk_scores_count: number;
  heatmap_points: number;
}> {
  return getLiveReadings().then((r) => ({
    readings_count: r.readings?.length ?? 0,
    predictions_count: 0,
    risk_scores_count: 0,
    heatmap_points: 0,
  }));
}

// ── Routing ────────────────────────────────────────────
export function optimizeRoute(req: RouteRequest): Promise<RouteResponse> {
  return fetchJSON('/route/optimize', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

// ── Orchestrator (Phase 1+ foundation) ─────────────────
export function getOrchestratorDatasetProfile(): Promise<{
  schema: { name: string; dtype: string }[];
  missing_value_report: Record<string, unknown>;
  temporal_coverage: { min?: string; max?: string; span_days?: number; unique_timestamps?: number };
  geometry_summary?: Record<string, unknown>;
  target_summary?: Record<string, unknown>;
  n_rows: number;
}> {
  return fetchJSON('/orchestrator/dataset-profile');
}
export function getOrchestratorBaselineMetrics(): Promise<Record<string, { model: string; val_mae: number; val_rmse: number; test_mae: number; test_rmse: number }>> {
  return fetchJSON('/orchestrator/baseline-metrics');
}
export function getOrchestratorRoadNetworkSummary(): Promise<{ nodes: number; edges: number; directed: boolean }> {
  return fetchJSON('/orchestrator/road-network-summary');
}
export function getOrchestratorPhase2Metrics(): Promise<{
  phase2: string;
  test_mae: number;
  test_rmse: number;
  baseline_test_mae: number;
  improvement_ratio: number;
  meets_30_percent_improvement: boolean;
  leakage_check_passed: boolean;
  horizons_hours: number[];
}> {
  return fetchJSON('/orchestrator/phase2-metrics');
}
export function getOrchestratorPhase3Innovation(): Promise<{
  dynamic_risk_fields: Record<string, unknown>;
  greenwave_eco_routing: Record<string, unknown>;
  event_impact_encoder: Record<string, unknown>;
}> {
  return fetchJSON('/orchestrator/phase3-innovation');
}

/** Same output as terminal CLI: multi_horizon (1h–24h), congestion level/score/delay, risk, routes. Use with map origin/destination. */
export interface OrchestratorRouteForecast {
  origin_road?: string;
  destination_road?: string;
  multi_horizon_forecasts: Record<string, { speed_kmh: number; level: string; score: number; delay_vs_freeflow_min?: number; ci_80_lower?: number; ci_80_upper?: number }>;
  congestion_classification: { level: string; score: number; delay_vs_freeflow_min?: number; speed_ratio?: number };
  avg_speed_kmh?: number;
  model_version: string;
  departure_time?: string;
  risk: { origin_risk: number; destination_risk: number; hotspot_count: number };
  routes: { path: string[]; total_km: number; fastest_route_time_min: number; eco_route_time_min: number; fastest_route_emissions_kg: number; eco_route_emissions_kg: number; percent_emission_reduction: number };
  horizons_hours?: number[];
  error?: string;
}

export function getOrchestratorRouteForecast(params: {
  origin_lat: number;
  origin_lon: number;
  destination_lat: number;
  destination_lon: number;
  departure_time?: string;
  event_context?: string;
}): Promise<OrchestratorRouteForecast> {
  return fetchJSON('/orchestrator/route-forecast', {
    method: 'POST',
    body: JSON.stringify({
      origin_lat: params.origin_lat,
      origin_lon: params.origin_lon,
      destination_lat: params.destination_lat,
      destination_lon: params.destination_lon,
      departure_time: params.departure_time,
      event_context: params.event_context ?? 'none',
    }),
  });
}

// ── Analytics ──────────────────────────────────────────
export function getCorridorStats(): Promise<CorridorStats> {
  return fetchJSON('/analytics/corridor-stats');
}

export function getEmissionSavings(): Promise<EmissionComparison> {
  return fetchJSON('/analytics/emission-savings');
}

export function getBraessParadox(): Promise<BraessParadoxData> {
  return fetchJSON('/analytics/braess-paradox');
}

export function getSystemHealth(): Promise<SystemHealth> {
  return fetchJSON('/analytics/system-health');
}

// ── Route Forecast ─────────────────────────────────────────
export interface RouteForecast {
  hourly_speeds: number[];
  peak_speed: number;
  min_speed: number;
  avg_speed: number;
  timestamp: string;
  model_version: string;
  hourly_speeds_q50?: number[];
  hourly_speeds_q90?: number[];
  regime_active?: boolean;
  residual_contribution_summary?: { mean_abs_residual?: number };
}

export function getRouteForecast(
  origin: [number, number],
  destination: [number, number],
  city: string = 'bengaluru'
): Promise<RouteForecast> {
  return fetchJSON('/predict/route-forecast', {
    method: 'POST',
    body: JSON.stringify({
      origin: [origin[0], origin[1]],
      destination: [destination[0], destination[1]],
      city,
    }),
  });
}

// ── Geoapify (via backend proxy) ───────────────────────
export interface GeoapifyRouteParams {
  origin_lat: number;
  origin_lon: number;
  dest_lat: number;
  dest_lon: number;
  mode?: 'drive' | 'truck' | 'bicycle' | 'walk' | 'transit';
  waypoints?: string; // "lat1,lon1|lat2,lon2"
}

export function getGeoapifyRoute(params: GeoapifyRouteParams): Promise<unknown> {
  const qs = new URLSearchParams(
    Object.entries(params).reduce<Record<string, string>>((acc, [k, v]) => {
      if (v !== undefined) acc[k] = String(v);
      return acc;
    }, {}),
  ).toString();
  return fetchJSON(`/geoapify/route?${qs}`);
}

export function planRoutes(body: {
  mode: string;
  agents: { start_location: [number, number]; end_location?: [number, number] }[];
  jobs: { location: [number, number]; duration?: number; priority?: number }[];
}): Promise<unknown> {
  return fetchJSON('/geoapify/routeplanner', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export function geocode(query: string): Promise<unknown> {
  return fetchJSON(`/geoapify/geocode?query=${encodeURIComponent(query)}`);
}

export function reverseGeocode(lat: number, lon: number): Promise<unknown> {
  return fetchJSON(`/geoapify/reverse-geocode?lat=${lat}&lon=${lon}`);
}
