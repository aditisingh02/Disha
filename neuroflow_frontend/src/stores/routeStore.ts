import { create } from 'zustand';
import type {
  RoutingMode,
  VehicleType,
  GoogleRouteData,
} from '@/types';

interface RouteState {
  origin: [number, number] | null;     // [lat, lng]
  destination: [number, number] | null;
  mode: RoutingMode;
  vehicleType: VehicleType;

  // Google Maps route data (real data)
  googleRoute: GoogleRouteData | null;
  isComputingRoute: boolean;
  error: string | null;

  // Actions
  setOrigin: (o: [number, number]) => void;
  setDestination: (d: [number, number]) => void;
  setMode: (m: RoutingMode) => void;
  setVehicleType: (v: VehicleType) => void;
  setGoogleRoute: (data: GoogleRouteData | null) => void;
  setComputingRoute: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearRoute: () => void;
}

export const useRouteStore = create<RouteState>((set) => ({
  origin: null,
  destination: null,
  mode: 'nash',
  vehicleType: 'car_petrol',
  googleRoute: null,
  isComputingRoute: false,
  error: null,

  setOrigin: (origin) => set({ origin, googleRoute: null, error: null }),
  setDestination: (destination) => set({ destination }),
  setMode: (mode) => set({ mode }),
  setVehicleType: (vehicleType) => set({ vehicleType }),

  setGoogleRoute: (googleRoute) => set({ googleRoute, isComputingRoute: false, error: null }),
  setComputingRoute: (isComputingRoute) => set({ isComputingRoute }),
  setError: (error) => set({ error, isComputingRoute: false }),

  clearRoute: () =>
    set({
      origin: null,
      destination: null,
      googleRoute: null,
      isComputingRoute: false,
      error: null,
    }),
}));
