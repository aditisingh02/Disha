import { useBraessParadox } from '@/hooks/useTrafficAPI';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  Cell,
} from 'recharts';
import { Target, Hourglass, AlertTriangle, TrendingUp, GitMerge, ArrowRight } from 'lucide-react';

/**
 * Professional Braess Paradox visualizer with glass styling.
 */
export default function BraessVisualizer() {
  const { data, error } = useBraessParadox();

  if (error || !data) {
    return (
      <div className="glass-panel p-6 flex items-center justify-center min-h-[160px]">
        <div className="text-center">
          <div className="w-10 h-10 mx-auto mb-3 rounded-full bg-slate-100 flex items-center justify-center">
            <Hourglass size={20} className="text-slate-400 animate-spin-slow" />
          </div>
          <p className="text-xs font-medium text-slate-500">{error ?? 'Analyzing Network Topology...'}</p>
        </div>
      </div>
    );
  }

  const chartData = [
    { name: 'User Equilibrium', time: Math.round(data.user_equilibrium_total_time), color: '#f43f5e' }, // Rose-500
    { name: 'System Optimum', time: Math.round(data.system_optimum_total_time), color: '#10b981' }, // Emerald-500
  ];

  const paradoxDetected = data.improvement_percent > 5;

  return (
    <div className="glass-panel p-6 overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${paradoxDetected
          ? 'bg-amber-50 border border-amber-100'
          : 'bg-emerald-50 border border-emerald-100'
          }`}>
          <GitMerge size={16} className={paradoxDetected ? 'text-amber-600' : 'text-emerald-600'} />
        </div>
        <div className="flex-1">
          <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider">Nash Equilibrium</h3>
          <p className="text-[10px] text-slate-400">Game Theory Analysis</p>
        </div>
        {paradoxDetected && (
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-amber-50 border border-amber-100">
            <AlertTriangle size={10} className="text-amber-600" />
            <span className="text-[9px] font-bold text-amber-700 uppercase tracking-wide">Paradox Active</span>
          </div>
        )}
      </div>

      {/* Chart */}
      <div className="w-full h-[100px] mb-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" barCategoryGap="15%" margin={{ left: 0, right: 30, top: 0, bottom: 0 }}>
            <XAxis type="number" hide />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fontSize: 10, fill: '#64748b' }}
              width={100}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              cursor={{ fill: 'transparent' }}
              contentStyle={{
                background: 'rgba(255, 255, 255, 0.95)',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
                fontSize: '11px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                padding: '4px 8px',
              }}
              formatter={(value: number) => [`${value.toLocaleString()}s`, 'Total Time']}
            />
            <Bar dataKey="time" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="p-3 rounded-lg bg-emerald-50/50 border border-emerald-100 flex items-center gap-3">
          <div className="p-1.5 rounded-md bg-white/60">
            <TrendingUp size={14} className="text-emerald-600" />
          </div>
          <div>
            <p className="text-[9px] font-bold text-emerald-800 uppercase tracking-wide opacity-70">Efficiency Gain</p>
            <p className="text-lg font-bold text-emerald-700 leading-none">{data.improvement_percent.toFixed(1)}%</p>
          </div>
        </div>
        <div className="p-3 rounded-lg bg-slate-50/50 border border-slate-100 flex items-center gap-3">
          <div className="p-1.5 rounded-md bg-white/60">
            <Target size={14} className="text-slate-600" />
          </div>
          <div>
            <p className="text-[9px] font-bold text-slate-600 uppercase tracking-wide opacity-70">Paradox Edges</p>
            <p className="text-lg font-bold text-slate-700 leading-none">{data.paradox_edges.length}</p>
          </div>
        </div>
      </div>

      {/* Paradox Edge Details */}
      {data.paradox_edges.length > 0 && (
        <div className="space-y-2">
          <p className="text-[9px] font-bold text-slate-400 uppercase tracking-wide ml-1">Affected Segments</p>
          {data.paradox_edges.slice(0, 3).map((e, i) => (
            <div key={i} className="flex items-center justify-between px-3 py-2 rounded-lg bg-slate-50/50 border border-slate-100">
              <span className="text-xs font-semibold text-slate-600">{e.edge}</span>
              <div className="flex items-center gap-2 text-[10px] font-mono">
                <span className="text-rose-500 font-bold">{e.ue_load_pct}%</span>
                <ArrowRight size={10} className="text-slate-300" />
                <span className="text-emerald-500 font-bold">{e.so_load_pct}%</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
