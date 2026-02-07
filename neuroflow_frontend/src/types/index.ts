/* ── Shared TypeScript types for NeuroFlow BharatFlow ── */

// ── Enums ──────────────────────────────────────────────
export type RoutingMode = 'fastest' | 'eco' | 'nash';
export type VehicleType = 'car_petrol' | '3_wheeler_lpg' | 'bus_diesel' | '2_wheeler';

// ── GeoJSON ────────────────────────────────────────────
export interface GeoPoint {
  type: 'Point';
  coordinates: [number, number]; // [lng, lat]
}

export interface GeoLineString {
  type: 'LineString';
  coordinates: [number, number][];
}

export interface GeoFeature<G = GeoPoint | GeoLineString, P = Record<string, unknown>> {
  type: 'Feature';
  geometry: G;
  properties: P;
}

export interface GeoFeatureCollection<F extends GeoFeature = GeoFeature> {
  type: 'FeatureCollection';
  features: F[];
}

// ── Traffic ────────────────────────────────────────────
export interface TrafficReading {
  segment_id: string;
  speed_kmh: number;
  volume: number;
  occupancy: number;
  speed_std: number;
  timestamp: string;
  location: GeoPoint;
}

export interface TrafficPrediction {
  segment_id: string;
  predicted_speed_t15: number;
  predicted_speed_t30: number;
  predicted_speed_t60: number;
  hourly_speeds?: number[]; // 12-hour forecast (48 steps)
  confidence: number;
  timestamp: string;
}

// ── Risk ───────────────────────────────────────────────
export interface RiskScore {
  segment_id?: string;
  hex_id?: string;
  location: GeoPoint;
  risk_value: number;       // 0–1
  risk_components: Record<string, number>;
  timestamp: string;
}

export interface HeatmapPoint {
  position: [number, number]; // [lng, lat]
  weight: number;
}

// ── Routing ────────────────────────────────────────────
export interface RouteRequest {
  origin: [number, number];      // [lat, lng]
  destination: [number, number]; // [lat, lng]
  mode: RoutingMode;
  vehicle_type?: VehicleType;
  alpha?: number;
  beta?: number;
  gamma?: number;
}

export interface SingleRoute {
  path_index?: number;
  travel_time_seconds: number;
  distance_km: number;
  emission_kgco2: number;
  distribution_weight?: number;
  geometry: {
    type: 'LineString';
    coordinates: [number, number][];
  };
  segments?: string[];
  is_eco_optimal?: boolean;
}

export interface RouteResponse {
  routes: SingleRoute[];
  mode: RoutingMode;
  braess_warning: boolean;
  system_emission_saved_kg: number;
}

// ── Analytics ──────────────────────────────────────────
export interface CorridorStats {
  corridor_name?: string;
  avg_speed_kmh: number;
  avg_travel_time_min: number;
  congestion_index: number;
  total_vehicles_estimated: number;
  dominant_vehicle_type?: string;
  timestamp?: string;
}

export interface EmissionComparison {
  fastest_route_emission_kg: number;
  eco_route_emission_kg: number;
  savings_kg: number;
  savings_percent: number;
  equivalent_trees_per_year: number;
}

export interface BraessParadoxData {
  user_equilibrium_total_time: number;
  system_optimum_total_time: number;
  improvement_percent: number;
  paradox_edges: {
    edge: string;
    reason: string;
    ue_load_pct: number;
    so_load_pct: number;
  }[];
}

export interface SystemHealth {
  status: string;
  timestamp: string;
  components: {
    mongodb: string;
    graph: {
      loaded: boolean;
      nodes: number;
      edges: number;
    };
    forecaster: {
      initialized: boolean;
      device: string;
    };
    simulation: {
      running: boolean;
      tick_count: number;
      ws_clients: number;
      latest_readings: number;
    };
  };
}

// ── WebSocket ──────────────────────────────────────────
export interface WSTrafficUpdate {
  type?: 'traffic_update';
  event?: 'traffic_update';
  timestamp: string;
  tick?: number;
  summary?: {
    total_readings: number;
    avg_speed: number;
    avg_risk: number;
    total_predictions: number;
  };
  readings?: TrafficReading[];
  readings_sample?: TrafficReading[];
  predictions?: TrafficPrediction[];
  risk_scores?: RiskScore[];
  risk_sample?: RiskScore[];
  heatmap?: HeatmapPoint[];
}

// ── Map ────────────────────────────────────────────────
export interface MapViewState {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
}

// ── Google Maps Route Data ─────────────────────────────
export interface GoogleRouteData {
  distance: string;              // "1,014 km" - formatted display
  duration: string;              // "18 hours 35 mins" - formatted display
  durationInTraffic?: string;    // "18 hours 16 mins" - with traffic
  distanceMeters: number;        // Raw value for calculations
  durationSeconds: number;       // Raw value for calculations
  durationInTrafficSeconds?: number;
}
