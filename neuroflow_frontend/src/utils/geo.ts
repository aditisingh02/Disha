/** Geo utility helpers */

/** Haversine distance in km between two [lng, lat] points */
export function haversineKm(
  [lng1, lat1]: [number, number],
  [lng2, lat2]: [number, number],
): number {
  const R = 6371;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function toRad(deg: number): number {
  return (deg * Math.PI) / 180;
}

/** Convert path to Deck.gl-compatible positions */
export function pathToPositions(path: [number, number][]): [number, number, number][] {
  return path.map(([lng, lat]) => [lng, lat, 0]);
}

/** Interpolate color between green → yellow → red based on value 0–1 */
export function riskColor(value: number): [number, number, number, number] {
  const clamped = Math.max(0, Math.min(1, value));
  if (clamped < 0.5) {
    const t = clamped * 2;
    return [
      Math.round(34 + (250 - 34) * t),
      Math.round(197 + (204 - 197) * t),
      Math.round(94 + (21 - 94) * t),
      200,
    ];
  }
  const t = (clamped - 0.5) * 2;
  return [
    Math.round(250 + (239 - 250) * t),
    Math.round(204 + (68 - 204) * t),
    Math.round(21 + (68 - 21) * t),
    220,
  ];
}

/** Format seconds to "Xm Ys" */
export function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

/** Format km to readable */
export function formatKm(km: number): string {
  return km < 1 ? `${Math.round(km * 1000)}m` : `${km.toFixed(1)} km`;
}

/** Approximate emission factors (kg CO₂ per km) - ARAI India standards */
const EMISSION_FACTORS: Record<string, number> = {
  car_petrol: 0.14,       // Average petrol car
  '2_wheeler': 0.05,      // Motorcycle/scooter
  '3_wheeler_lpg': 0.08,  // Auto-rickshaw LPG
  bus_diesel: 0.10,       // Per passenger (assuming ~40 passengers)
};

/** Calculate CO₂ emission in kg based on distance and vehicle type */
export function calculateEmission(distanceKm: number, vehicleType: string): number {
  const factor = EMISSION_FACTORS[vehicleType] ?? 0.14;
  return distanceKm * factor;
}

/** Format emission to readable string */
export function formatEmission(emissionKg: number): string {
  if (emissionKg < 1) {
    return `${Math.round(emissionKg * 1000)} g`;
  }
  return `${emissionKg.toFixed(2)} kg`;
}
