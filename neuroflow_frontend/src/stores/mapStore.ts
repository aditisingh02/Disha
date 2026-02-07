import { create } from 'zustand';
import type { MapViewState } from '@/types';
import { DEFAULT_VIEW_STATE } from '@/utils/constants';

type PickMode = 'origin' | 'destination' | null;

interface MapState {
  viewState: MapViewState;
  pickMode: PickMode;
  showHeatmap: boolean;
  showTrafficFlow: boolean;
  showRoutes: boolean;
  show3D: boolean;

  setViewState: (vs: Partial<MapViewState>) => void;
  setPickMode: (pm: PickMode) => void;
  toggleHeatmap: () => void;
  toggleTrafficFlow: () => void;
  toggleRoutes: () => void;
  toggle3D: () => void;
}

export const useMapStore = create<MapState>((set) => ({
  viewState: { ...DEFAULT_VIEW_STATE },
  pickMode: null,
  showHeatmap: true,
  showTrafficFlow: true,
  showRoutes: true,
  show3D: true,

  setViewState: (vs) =>
    set((s) => ({ viewState: { ...s.viewState, ...vs } })),
  setPickMode: (pickMode) => set({ pickMode }),
  toggleHeatmap: () => set((s) => ({ showHeatmap: !s.showHeatmap })),
  toggleTrafficFlow: () => set((s) => ({ showTrafficFlow: !s.showTrafficFlow })),
  toggleRoutes: () => set((s) => ({ showRoutes: !s.showRoutes })),
  toggle3D: () =>
    set((s) => ({
      show3D: !s.show3D,
      viewState: {
        ...s.viewState,
        pitch: s.show3D ? 0 : 45,
      },
    })),
}));
