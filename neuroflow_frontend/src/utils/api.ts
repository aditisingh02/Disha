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
export function getPredictions(): Promise<TrafficPrediction[]> {
  return fetchJSON('/predict/traffic');
}

export function getRiskHeatmap(): Promise<HeatmapPoint[]> {
  return fetchJSON('/risk/heatmap');
}

export function getTrafficSegments(): Promise<GeoFeatureCollection> {
  return fetchJSON('/traffic/segments');
}

export function getLiveTraffic(): Promise<{
  readings_count: number;
  predictions_count: number;
  risk_scores_count: number;
  heatmap_points: number;
}> {
  return fetchJSON('/traffic/live');
}

// ── Routing ────────────────────────────────────────────
export function optimizeRoute(req: RouteRequest): Promise<RouteResponse> {
  return fetchJSON('/route/optimize', {
    method: 'POST',
    body: JSON.stringify(req),
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
