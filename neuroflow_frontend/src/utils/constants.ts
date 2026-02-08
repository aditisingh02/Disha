/* ── Constants ────────────────────────────────────────── */

/** Google Maps API Key */
export const GOOGLE_MAPS_API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY || '';

/** Bengaluru center (Majestic / City Railway Station) */
export const BENGALURU_CENTER = {
  latitude: 12.9716,
  longitude: 77.5946,
} as const;

/** Default map view for Silk Board – Indiranagar corridor */
export const DEFAULT_VIEW_STATE = {
  longitude: 103.8198,
  latitude: 1.3521,
  zoom: 11.5,
  pitch: 45,
  bearing: 0,
} as const;

/** Default Google Maps center */
export const DEFAULT_CENTER = {
  lat: 1.3521,
  lng: 103.8198,
} as const;

/** API base – proxied by Vite in dev */
export const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api/v1';
export const WS_URL = import.meta.env.VITE_WS_URL || `ws://${window.location.host}/api/v1/ws/live`;
export const CHAT_WS_URL = import.meta.env.VITE_CHAT_WS_URL || `ws://${window.location.host}/api/v1/ws/chat`;

/** ARAI emission factors (kg CO₂ / km) */
export const EMISSION_FACTORS: Record<string, number> = {
  '2_wheeler': 0.035,
  car_petrol: 0.140,
  car_diesel: 0.165,
  auto_rickshaw: 0.080,
  bus_diesel: 0.750,
};

/** Risk color scale (green → yellow → red) */
export const RISK_COLORS: [number, number, number, number][] = [
  [34, 197, 94, 180],   // green – low risk
  [250, 204, 21, 200],  // yellow – medium
  [239, 68, 68, 220],   // red – high risk
];

/** Route mode colors for Google Maps polylines */
export const MODE_COLORS: Record<string, string> = {
  fastest: '#0ea5e9',    // sky blue
  eco: '#22c55e',        // green
  nash: '#8b5cf6',       // violet
};

/** Route mode colors as RGB arrays for compatibility */
export const MODE_COLORS_RGB: Record<string, [number, number, number]> = {
  fastest: [14, 165, 233],    // sky blue
  eco: [34, 197, 94],         // green
  nash: [139, 92, 246],       // violet
};
