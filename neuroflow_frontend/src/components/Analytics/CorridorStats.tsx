import { useCorridorStats, useSystemHealth } from '@/hooks/useTrafficAPI';
import {
  Gauge, Car, Clock, Bike,
  Activity, Cpu, Database, Network, Server,
  BarChart3, TrafficCone
} from 'lucide-react';

/**
 * Professional corridor performance and system health dashboard.
 */
export default function CorridorStats() {
  const { data: corridor } = useCorridorStats();
  const { data: health } = useSystemHealth();

  return (
    <div className="glass-panel p-6 space-y-6">
      {/* Corridor Performance */}
      <div>
        <div className="flex items-center gap-3 mb-4">
          <div className="w-8 h-8 rounded-lg bg-emerald-50 border border-emerald-100 flex items-center justify-center">
            <BarChart3 size={16} className="text-emerald-600" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider">Corridor Metrics</h3>
            <p className="text-[10px] text-slate-400">
              {corridor?.corridor_name ?? 'Silk Board → Indiranagar'}
            </p>
          </div>
        </div>

        {corridor ? (
          <div className="grid grid-cols-3 gap-3">
            <StatCard
              value={`${corridor.avg_speed_kmh.toFixed(1)}`}
              unit="km/h"
              label="Avg Speed"
              icon={<Gauge size={14} />}
              variant="success"
            />
            <StatCard
              value={(corridor.total_vehicles_estimated ?? 0).toLocaleString()}
              label="Vehicles"
              icon={<Car size={14} />}
              variant="neutral"
            />
            <StatCard
              value={`${(corridor.congestion_index * 100).toFixed(0)}%`}
              label="Congestion"
              icon={<TrafficCone size={14} />}
              variant={corridor.congestion_index > 0.7 ? 'warning' : 'success'}
              highlight={corridor.congestion_index > 0.7}
            />
            <StatCard
              value={`${corridor.avg_travel_time_min.toFixed(1)}`}
              unit="min"
              label="Travel Time"
              icon={<Clock size={14} />}
              variant="neutral"
            />
            <StatCard
              value={corridor.dominant_vehicle_type === '2_wheeler' ? '2-Wheeler' : 'Car'}
              label="Dominant"
              icon={<Bike size={14} />}
              variant="neutral"
            />
            <div className="col-span-1 p-2 rounded-lg bg-slate-50 border border-slate-100 flex flex-col items-center justify-center">
              <span className="text-[10px] font-bold text-slate-400 uppercase">Live Cam</span>
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse mt-1" />
            </div>
          </div>
        ) : (
          <LoadingGrid count={6} />
        )}
      </div>

      {/* Divider */}
      <div className="h-px bg-slate-100" />

      {/* System Health */}
      <div>
        <div className="flex items-center gap-3 mb-4">
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${health?.status === 'operational'
            ? 'bg-emerald-50 border border-emerald-100'
            : 'bg-amber-50 border border-amber-100'
            }`}>
            <Activity size={16} className={health?.status === 'operational' ? 'text-emerald-600' : 'text-amber-600'} />
          </div>
          <div className="flex-1">
            <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider">System Health</h3>
            <p className="text-[10px] text-slate-400">Real-time monitoring</p>
          </div>
          {health && (
            <span className={`px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wide border ${health.status === 'operational'
              ? 'bg-emerald-50 text-emerald-600 border-emerald-100'
              : 'bg-amber-50 text-amber-600 border-amber-100'
              }`}>
              {health.status}
            </span>
          )}
        </div>

        {health ? (
          <div className="space-y-2">
            <HealthItem
              label="ST-GCN Model"
              value={health.components.forecaster.initialized ? 'Active' : 'Initializing'}
              status={health.components.forecaster.initialized ? 'success' : 'warning'}
              icon={<Cpu size={14} />}
            />
            <HealthItem
              label="Graph Engine"
              value={`${health.components.graph.nodes} Nodes • ${health.components.graph.edges} Edges`}
              status="success"
              icon={<Network size={14} />}
            />
            <HealthItem
              label="MongoDB Atlas"
              value={health.components.mongodb === 'connected' ? 'Connected' : 'Error'}
              status={health.components.mongodb === 'connected' ? 'success' : 'error'}
              icon={<Database size={14} />}
            />
            <HealthItem
              label="Sim Ticks"
              value={`${health.components.simulation.tick_count.toLocaleString()}`}
              status="success"
              icon={<Server size={14} />}
            />
          </div>
        ) : (
          <LoadingGrid count={4} />
        )}
      </div>
    </div>
  );
}

function StatCard({
  value,
  unit,
  label,
  icon,
  variant,
  highlight = false,
}: {
  value: string;
  unit?: string;
  label: string;
  icon: React.ReactNode;
  variant: 'success' | 'neutral' | 'warning';
  highlight?: boolean;
}) {
  const styles = {
    success: 'bg-emerald-50/50 border-emerald-100 text-emerald-700',
    neutral: 'bg-slate-50/50 border-slate-100 text-slate-700',
    warning: 'bg-amber-50/50 border-amber-100 text-amber-700',
  };

  return (
    <div className={`p-2 rounded-lg border flex flex-col items-center justify-center text-center ${styles[variant]} ${highlight ? 'ring-2 ring-amber-100' : ''}`}>
      <div className="mb-1 opacity-70">{icon}</div>
      <p className="text-sm font-bold leading-none">
        {value}
        {unit && <span className="text-[9px] font-normal opacity-70 ml-0.5">{unit}</span>}
      </p>
      <p className="text-[9px] font-medium uppercase tracking-wide opacity-60 mt-1">{label}</p>
    </div>
  );
}

function HealthItem({
  label,
  value,
  status,
  icon,
}: {
  label: string;
  value: string;
  status: 'success' | 'warning' | 'error';
  icon: React.ReactNode;
}) {
  const colors = {
    success: 'bg-emerald-500',
    warning: 'bg-amber-500',
    error: 'bg-rose-500',
  };

  return (
    <div className="flex items-center gap-3 p-2 rounded-lg bg-slate-50/50 border border-slate-100 hover:bg-slate-100 transition-colors">
      <div className="text-slate-400">{icon}</div>
      <div className="flex-1 flex items-center justify-between">
        <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide">{label}</span>
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-slate-700">{value}</span>
          <div className={`w-1.5 h-1.5 rounded-full ${colors[status]}`} />
        </div>
      </div>
    </div>
  );
}

function LoadingGrid({ count }: { count: number }) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {[...Array(count)].map((_, i) => (
        <div key={i} className="h-12 rounded-lg bg-slate-100/50" />
      ))}
    </div>
  );
}
