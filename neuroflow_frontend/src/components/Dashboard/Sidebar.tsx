import { useCallback } from 'react';
import { useRouteStore } from '@/stores/routeStore';
import { useMapStore } from '@/stores/mapStore';
import { useTrafficStore } from '@/stores/trafficStore';
import type { VehicleType } from '@/types';
import { calculateEmission, formatEmission } from '@/utils/geo';
import {
  Car, Bike, Truck, Bus,
  MapPin, Flag, Navigation,
  AlertTriangle, Wind, X,
  Layers, Activity, TrendingUp
} from 'lucide-react';

const VEHICLE_TYPES: { value: VehicleType; label: string; icon: React.ReactNode; emission: string }[] = [
  { value: 'car_petrol', label: 'Car (Petrol)', icon: <Car size={20} />, emission: '140 g/km' },
  { value: '2_wheeler', label: '2-Wheeler', icon: <Bike size={20} />, emission: '50 g/km' },
  { value: '3_wheeler_lpg', label: 'Auto LPG', icon: <Truck size={20} />, emission: '80 g/km' },
  { value: 'bus_diesel', label: 'Bus (Diesel)', icon: <Bus size={20} />, emission: '100 g/km' },
];

export default function Sidebar() {
  const vehicleType = useRouteStore((s) => s.vehicleType);
  const setVehicleType = useRouteStore((s) => s.setVehicleType);
  const origin = useRouteStore((s) => s.origin);
  const destination = useRouteStore((s) => s.destination);
  const googleRoute = useRouteStore((s) => s.googleRoute);
  const isComputingRoute = useRouteStore((s) => s.isComputingRoute);
  const error = useRouteStore((s) => s.error);
  const clearRoute = useRouteStore((s) => s.clearRoute);
  const setRouteForecast = useTrafficStore((s) => s.setRouteForecast);
  const setOrchestratorRouteForecast = useTrafficStore((s) => s.setOrchestratorRouteForecast);
  const orchestratorRouteForecast = useTrafficStore((s) => s.orchestratorRouteForecast);

  const handleClearRoute = useCallback(() => {
    clearRoute();
    setRouteForecast(null);
    setOrchestratorRouteForecast(null);
  }, [clearRoute, setRouteForecast, setOrchestratorRouteForecast]);

  const setPickMode = useMapStore((s) => s.setPickMode);
  const showHeatmap = useMapStore((s) => s.showHeatmap);
  const showTrafficFlow = useMapStore((s) => s.showTrafficFlow);
  const showRoutes = useMapStore((s) => s.showRoutes);
  const toggleHeatmap = useMapStore((s) => s.toggleHeatmap);
  const toggleTrafficFlow = useMapStore((s) => s.toggleTrafficFlow);
  const toggleRoutes = useMapStore((s) => s.toggleRoutes);

  const handlePickOrigin = useCallback(() => setPickMode('origin'), [setPickMode]);
  const handlePickDestination = useCallback(() => setPickMode('destination'), [setPickMode]);

  // Calculate CO₂ emission based on vehicle type and distance
  const distanceKm = googleRoute ? googleRoute.distanceMeters / 1000 : 0;
  const emissionKg = calculateEmission(distanceKm, vehicleType);

  // Determine traffic status
  const hasTrafficDelay = googleRoute?.durationInTrafficSeconds && googleRoute?.durationSeconds &&
    googleRoute.durationInTrafficSeconds > googleRoute.durationSeconds;

  const trafficDelayMinutes = hasTrafficDelay && googleRoute?.durationInTrafficSeconds && googleRoute?.durationSeconds
    ? Math.round((googleRoute.durationInTrafficSeconds - googleRoute.durationSeconds) / 60)
    : 0;

  return (
    <aside className="glass-panel w-96 max-h-full flex flex-col overflow-hidden shadow-2xl">
      {/* Header Section */}
      <div className="p-5 border-b border-white/20">
        <div className="flex items-center gap-2 text-slate-700">
          <Navigation size={20} className="text-emerald-600" />
          <h2 className="text-sm font-bold uppercase tracking-wider">Route Planner</h2>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-5 space-y-6 no-scrollbar">
        {/* Origin / Destination */}
        <div className="space-y-3">
          <LocationButton
            type="origin"
            coords={origin}
            onClick={handlePickOrigin}
          />
          <LocationButton
            type="destination"
            coords={destination}
            onClick={handlePickDestination}
          />
        </div>

        {/* Vehicle Type */}
        <div>
          <SectionHeader icon={<Car size={16} />} title="Vehicle Type" />
          <div className="grid grid-cols-2 gap-3">
            {VEHICLE_TYPES.map((v) => (
              <button
                key={v.value}
                onClick={() => setVehicleType(v.value)}
                className={`flex flex-col items-center gap-2 p-3 rounded-lg border transition-all duration-200 ${vehicleType === v.value
                  ? 'bg-emerald-50 border-emerald-500 text-emerald-800'
                  : 'bg-white/40 border-slate-200 text-slate-600 hover:bg-white/60'
                  }`}
              >
                <div className={vehicleType === v.value ? 'text-emerald-600' : 'text-slate-400'}>
                  {v.icon}
                </div>
                <div className="text-center">
                  <span className="block text-xs font-semibold">{v.label}</span>
                  <span className="block text-[10px] opacity-100">{v.emission}</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="px-4 py-3 rounded-lg bg-red-50/80 border border-red-200 text-red-700 text-sm flex items-center gap-3 backdrop-blur-sm">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        )}

        {/* Loading State */}
        {isComputingRoute && (
          <div className="px-4 py-6 rounded-lg bg-slate-50/50 border border-slate-200 flex flex-col items-center justify-center gap-3">
            <div className="w-6 h-6 border-2 border-emerald-500/30 border-t-emerald-600 rounded-full animate-spin" />
            <span className="text-xs font-medium text-slate-500">Optimizing route...</span>
          </div>
        )}

        {/* Route Results */}
        {googleRoute && !isComputingRoute && (
          <div className="animate-enter space-y-4">
            <div className="flex items-center justify-between">
              <SectionHeader icon={<Activity size={16} />} title="Route Analytics" />
              <button
                onClick={handleClearRoute}
                className="text-xs flex items-center gap-1 text-slate-400 hover:text-red-500 transition-colors"
              >
                <X size={12} /> Clear
              </button>
            </div>

            {/* Main Stats Card */}
            <div className="glass-card p-5 bg-white/60">
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="text-center">
                  <p className="text-[10px] text-slate-400 uppercase tracking-wide mb-1">Distance</p>
                  <p className="text-xl font-bold text-slate-800">{googleRoute.distance}</p>
                </div>
                <div className="text-center">
                  <p className="text-[10px] text-slate-400 uppercase tracking-wide mb-1">Duration</p>
                  <p className="text-xl font-bold text-slate-800">{googleRoute.duration}</p>
                </div>
              </div>

              <div className="h-px bg-slate-200 my-4" />

              {/* Traffic Status */}
              <div className="flex items-center gap-3 mb-3">
                <div className={`p-2 rounded-full ${hasTrafficDelay ? 'bg-amber-100 text-amber-600' : 'bg-emerald-100 text-emerald-600'}`}>
                  {hasTrafficDelay ? <AlertTriangle size={16} /> : <Activity size={16} />}
                </div>
                <div>
                  <p className="text-xs font-semibold text-slate-700">
                    {hasTrafficDelay ? 'Heavy Traffic Detected' : 'Flowing Smoothly'}
                  </p>
                  <p className="text-[10px] text-slate-500">
                    {hasTrafficDelay ? `+${trafficDelayMinutes} min delay` : 'No significant delays'}
                  </p>
                </div>
              </div>

              {/* Emission */}
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-full bg-slate-100 text-slate-600">
                  <Wind size={16} />
                </div>
                <div>
                  <p className="text-xs font-semibold text-slate-700">
                    {formatEmission(emissionKg)} CO₂
                  </p>
                  <p className="text-[10px] text-slate-500">
                    Estimated impact
                  </p>
                </div>
              </div>

              {/* Orchestrator summary (same data as terminal) — reported on dashboard */}
              {orchestratorRouteForecast && !orchestratorRouteForecast.error && orchestratorRouteForecast.congestion_classification && (
                <div className="mt-4 pt-4 border-t border-slate-200">
                  <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2">Orchestrator Forecast</p>
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs text-slate-700">
                      Congestion: <strong>{orchestratorRouteForecast.congestion_classification.level}</strong>
                      {orchestratorRouteForecast.routes?.percent_emission_reduction != null && (
                        <span className="text-slate-500 ml-1">· Eco saves {orchestratorRouteForecast.routes.percent_emission_reduction.toFixed(0)}%</span>
                      )}
                    </span>
                    <button
                      type="button"
                      onClick={() => window.dispatchEvent(new CustomEvent('open-forecast-modal'))}
                      className="text-[10px] font-semibold text-emerald-600 hover:text-emerald-700 whitespace-nowrap"
                    >
                      View full forecast
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Empty States */}
        {!googleRoute && !isComputingRoute && !origin && (
          <div className="py-10 text-center opacity-60">
            <MapPin size={48} className="mx-auto text-slate-300 mb-3" />
            <p className="text-sm font-medium text-slate-500">Select Start Point</p>
          </div>
        )}
      </div>

      {/* Layer Toggles */}
      <div className="p-5 border-t border-white/20 bg-white/30 backdrop-blur-md">
        <SectionHeader icon={<Layers size={16} />} title="Map Layers" />
        <div className="space-y-2 mt-2">
          <ToggleRow label="Risk Heatmap" checked={showHeatmap} onChange={toggleHeatmap} />
          <ToggleRow label="Traffic Flow" checked={showTrafficFlow} onChange={toggleTrafficFlow} />
          <ToggleRow label="Route Lines" checked={showRoutes} onChange={toggleRoutes} />
        </div>

        {/* Forecast Button */}
        <button
          onClick={() => window.dispatchEvent(new CustomEvent('open-forecast-modal'))}
          className="w-full mt-6 py-3 px-4 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl shadow-lg shadow-emerald-200 transition-all font-semibold text-sm flex items-center justify-center gap-2 group"
        >
          <TrendingUp size={18} className="group-hover:scale-110 transition-transform" />
          View 12-Hour Forecast
        </button>
      </div>
    </aside>
  );
}

/* ── Sub-components ────────────────────────────────────────────── */

function SectionHeader({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="flex items-center gap-2 mb-3 text-slate-400">
      {icon}
      <h2 className="text-[10px] font-bold uppercase tracking-widest">{title}</h2>
    </div>
  );
}

function LocationButton({ type, coords, onClick }: { type: 'origin' | 'destination'; coords: [number, number] | null; onClick: () => void }) {
  const isOrigin = type === 'origin';
  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-3 rounded-lg border transition-all duration-200 group relative overflow-hidden ${coords
        ? 'bg-white border-slate-200 shadow-sm'
        : 'bg-white/40 border-dashed border-slate-300 hover:bg-white/60 hover:border-emerald-400'
        }`}
    >
      <div className="flex items-center gap-3 relative z-10">
        <div className={`p-2 rounded-md ${isOrigin ? 'bg-emerald-100 text-emerald-600' : 'bg-rose-100 text-rose-600'}`}>
          {isOrigin ? <MapPin size={18} /> : <Flag size={18} />}
        </div>
        <div>
          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
            {isOrigin ? 'Start' : 'Destination'}
          </p>
          <p className={`text-sm font-medium truncate ${coords ? 'text-slate-700' : 'text-slate-400 italic'}`}>
            {coords ? `${coords[0].toFixed(4)}, ${coords[1].toFixed(4)}` : 'Click on map...'}
          </p>
        </div>
      </div>
    </button>
  );
}

function ToggleRow({ label, checked, onChange }: { label: string; checked: boolean; onChange: () => void }) {
  return (
    <label className="flex items-center justify-between px-3 py-2 rounded-lg hover:bg-white/50 cursor-pointer transition-colors group">
      <span className="text-xs font-medium text-slate-600 group-hover:text-slate-900">{label}</span>
      <div
        onClick={onChange}
        className={`w-9 h-5 rounded-full transition-all duration-300 relative ${checked ? 'bg-emerald-500' : 'bg-slate-200'}`}
      >
        <div className={`absolute top-1 w-3 h-3 rounded-full bg-white shadow-sm transition-transform duration-300 ${checked ? 'translate-x-5' : 'translate-x-1'}`} />
      </div>
    </label>
  );
}
