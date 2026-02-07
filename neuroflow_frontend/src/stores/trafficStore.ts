import { create } from 'zustand';
import type {
  TrafficReading,
  TrafficPrediction,
  RiskScore,
  HeatmapPoint,
} from '@/types';

// Interface for route-specific forecast from backend
export interface RouteForecast {
  hourly_speeds: number[];
  peak_speed: number;
  min_speed: number;
  avg_speed: number;
  timestamp: string;
  model_version: string;
}

interface TrafficState {
  readings: TrafficReading[];
  predictions: TrafficPrediction[];
  riskScores: RiskScore[];
  heatmap: HeatmapPoint[];
  lastUpdate: string | null;
  isConnected: boolean;
  routeForecast: RouteForecast | null;  // NEW: Route-specific forecast

  setReadings: (r: TrafficReading[]) => void;
  setPredictions: (p: TrafficPrediction[]) => void;
  setRiskScores: (rs: RiskScore[]) => void;
  setHeatmap: (h: HeatmapPoint[]) => void;
  setLastUpdate: (ts: string) => void;
  setConnected: (c: boolean) => void;
  setRouteForecast: (f: RouteForecast | null) => void;  // NEW
  updateAll: (data: {
    readings?: TrafficReading[];
    readings_sample?: TrafficReading[];
    predictions?: TrafficPrediction[];
    risk_scores?: RiskScore[];
    risk_sample?: RiskScore[];
    heatmap?: HeatmapPoint[];
    timestamp: string;
    summary?: {
      total_readings: number;
      avg_speed: number;
      avg_risk: number;
      total_predictions: number;
    };
  }) => void;
}

// Generate demo prediction data for initial display
function generateDemoPredictions(): TrafficPrediction[] {
  return Array.from({ length: 20 }, (_, i) => ({
    segment_id: `S${i + 1}`,
    predicted_speed_t15: 35 + Math.random() * 25,
    predicted_speed_t30: 30 + Math.random() * 20,
    predicted_speed_t60: 25 + Math.random() * 15,
    hourly_speeds: Array.from({ length: 48 }, (_, j) => {
      // Create a realistic-ish wave
      const base = 30;
      const trend = Math.sin(j / 8) * 10;
      const noise = Math.random() * 5;
      return base + trend + noise;
    }),
    confidence: 0.85 + Math.random() * 0.1,
    timestamp: new Date().toISOString(),
  }));
}



export const useTrafficStore = create<TrafficState>((set) => ({
  readings: [],
  predictions: generateDemoPredictions(), // Start with demo data
  riskScores: [],
  heatmap: [],
  lastUpdate: null,
  isConnected: false,
  routeForecast: null,  // NEW

  setReadings: (readings) => set({ readings }),
  setPredictions: (predictions) => set({ predictions }),
  setRiskScores: (riskScores) => set({ riskScores }),
  setHeatmap: (heatmap) => set({ heatmap }),
  setLastUpdate: (lastUpdate) => set({ lastUpdate }),
  setConnected: (isConnected) => set({ isConnected }),
  setRouteForecast: (routeForecast) => set({ routeForecast }),  // NEW

  updateAll: (data) =>
    set((state) => ({
      readings: data.readings || data.readings_sample || state.readings,
      predictions: data.predictions || state.predictions,
      riskScores: data.risk_scores || data.risk_sample || state.riskScores,
      heatmap: data.heatmap || state.heatmap,
      lastUpdate: data.timestamp,
    })),
}));
