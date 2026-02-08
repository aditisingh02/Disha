/**
 * LTA DataMall Store
 * State management for Singapore's real-time traffic data
 */

import { create } from 'zustand';

// ═══════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════

export interface TrafficCamera {
    id: string;
    latitude: number;
    longitude: number;
    image_url: string;
    description: string;
    fetched_at: string;
    distance_km?: number;
}

export interface SpeedBand {
    link_id: string;
    road_name: string;
    road_category: number;
    speed_band: number;
    min_speed: number;
    max_speed: number;
    start: [number, number]; // [lng, lat]
    end: [number, number];
}

export interface TrafficIncident {
    type: string;
    latitude: number;
    longitude: number;
    message: string;
    fetched_at: string;
}

export interface LTAStatus {
    status: string;
    api_key_configured: boolean;
    cameras_count?: number;
    timestamp?: string;
    error?: string;
}

interface LTAState {
    // Data
    cameras: TrafficCamera[];
    speedBands: SpeedBand[];
    incidents: TrafficIncident[];
    status: LTAStatus | null;

    // Selected camera for viewing
    selectedCamera: TrafficCamera | null;

    // UI State
    isLoading: boolean;
    isCameraPanelOpen: boolean;
    showSpeedBands: boolean;
    showIncidents: boolean;

    // Actions
    fetchCameras: () => Promise<void>;
    fetchSpeedBands: () => Promise<void>;
    fetchIncidents: () => Promise<void>;
    fetchStatus: () => Promise<void>;
    fetchCamerasNear: (lat: number, lng: number, radius?: number) => Promise<TrafficCamera[]>;
    selectCamera: (camera: TrafficCamera | null) => void;
    toggleCameraPanel: () => void;
    toggleSpeedBands: () => void;
    toggleIncidents: () => void;
}

// ═══════════════════════════════════════════════════════════════
// API Base URL
// ═══════════════════════════════════════════════════════════════

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ═══════════════════════════════════════════════════════════════
// Store
// ═══════════════════════════════════════════════════════════════

export const useLTAStore = create<LTAState>((set) => ({
    // Initial state
    cameras: [],
    speedBands: [],
    incidents: [],
    status: null,
    selectedCamera: null,
    isLoading: false,
    isCameraPanelOpen: false,
    showSpeedBands: true,
    showIncidents: true,

    // Fetch all cameras
    fetchCameras: async () => {
        set({ isLoading: true });
        try {
            const response = await fetch(`${API_BASE}/api/v1/lta/cameras`);
            if (!response.ok) throw new Error('Failed to fetch cameras');
            const cameras = await response.json();
            set({ cameras, isLoading: false });
        } catch (error) {
            console.error('[LTA] Failed to fetch cameras:', error);
            set({ isLoading: false });
        }
    },

    // Fetch speed bands
    fetchSpeedBands: async () => {
        try {
            const response = await fetch(`${API_BASE}/api/v1/lta/speedbands`);
            if (!response.ok) throw new Error('Failed to fetch speed bands');
            const speedBands = await response.json();
            set({ speedBands });
        } catch (error) {
            console.error('[LTA] Failed to fetch speed bands:', error);
        }
    },

    // Fetch incidents
    fetchIncidents: async () => {
        try {
            const response = await fetch(`${API_BASE}/api/v1/lta/incidents`);
            if (!response.ok) throw new Error('Failed to fetch incidents');
            const incidents = await response.json();
            set({ incidents });
        } catch (error) {
            console.error('[LTA] Failed to fetch incidents:', error);
        }
    },

    // Fetch API status
    fetchStatus: async () => {
        try {
            const response = await fetch(`${API_BASE}/api/v1/lta/status`);
            if (!response.ok) throw new Error('Failed to fetch status');
            const status = await response.json();
            set({ status });
        } catch (error) {
            console.error('[LTA] Failed to fetch status:', error);
            set({ status: { status: 'error', api_key_configured: false, error: String(error) } });
        }
    },

    // Fetch cameras near a location
    fetchCamerasNear: async (lat: number, lng: number, radius = 2.0) => {
        try {
            const response = await fetch(
                `${API_BASE}/api/v1/lta/cameras/near/${lat}/${lng}?radius_km=${radius}`
            );
            if (!response.ok) throw new Error('Failed to fetch nearby cameras');
            return await response.json();
        } catch (error) {
            console.error('[LTA] Failed to fetch nearby cameras:', error);
            return [];
        }
    },

    // Select a camera for viewing
    selectCamera: (camera) => {
        set({ selectedCamera: camera, isCameraPanelOpen: camera !== null });
    },

    // Toggle camera panel
    toggleCameraPanel: () => {
        set((state) => ({ isCameraPanelOpen: !state.isCameraPanelOpen }));
    },

    // Toggle speed bands visibility
    toggleSpeedBands: () => {
        set((state) => ({ showSpeedBands: !state.showSpeedBands }));
    },

    // Toggle incidents visibility
    toggleIncidents: () => {
        set((state) => ({ showIncidents: !state.showIncidents }));
    },
}));
