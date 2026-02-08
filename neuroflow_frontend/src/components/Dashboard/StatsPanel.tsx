import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';
import { useTrafficStore } from '@/stores/trafficStore';
import { useMemo } from 'react';
import { BarChart2, Radio, Activity } from 'lucide-react';

/**
 * Professional traffic predictions chart with slate/emerald styling.
 */
export default function StatsPanel() {
  const predictions = useTrafficStore((s) => s.predictions);

  const chartData = useMemo(() => {
    if (predictions.length === 0) return [];

    // Aggregate 12-hour forecast (Average across all segments)
    // We expect hourly_speeds to be present in new data
    const validPreds = predictions.filter(p => p.hourly_speeds && p.hourly_speeds.length === 48);

    if (validPreds.length === 0) {
      // Fallback or legacy data visualization?
      // Let's just return empty to show "Waiting..." or handle legacy 3-point data
      // For now, let's try to simulate or use existing t15/t30/t60 if hourly is missing?
      // Actually, let's stick to the V2 plan.
      return [];
    }

    // Calculate average speed per step (0-47)
    const totalSteps = 48;
    const avgSpeeds = new Array(totalSteps).fill(0);

    validPreds.forEach(p => {
      p.hourly_speeds!.forEach((speed, idx) => {
        avgSpeeds[idx] += speed;
      });
    });

    const count = validPreds.length;
    const hasQ90 = validPreds.some(p => p.hourly_speeds_q90 && p.hourly_speeds_q90.length === 48);
    const avgQ50 = hasQ90 ? new Array(48).fill(0) : null;
    const avgQ90 = hasQ90 ? new Array(48).fill(0) : null;
    if (hasQ90) {
      validPreds.forEach(p => {
        p.hourly_speeds_q50?.forEach((v, i) => { if (i < 48) avgQ50![i] += v; });
        p.hourly_speeds_q90?.forEach((v, i) => { if (i < 48) avgQ90![i] += v; });
      });
      avgQ50?.forEach((_, i) => { avgQ50![i] /= count; });
      avgQ90?.forEach((_, i) => { avgQ90![i] /= count; });
    }
    return avgSpeeds.map((total, idx) => {
      const avg = total / count;
      const hour = Math.floor((idx + 1) * 15 / 60);
      const min = ((idx + 1) * 15) % 60;
      let label = "";
      if ((idx + 1) % 4 === 0) label = `+${(idx + 1) / 4}h`;
      const fullLabel = min > 0 ? `+${hour}h ${min}m` : `+${hour}h`;
      return {
        name: label,
        tooltipLabel: fullLabel,
        speed: Math.round(avg),
        speedQ50: avgQ50 ? Math.round(avgQ50[idx]) : undefined,
        speedQ90: avgQ90 ? Math.round(avgQ90[idx]) : undefined,
        idx,
      };
    });
  }, [predictions]);

  if (chartData.length === 0) {
    // Check if we have legacy data (t15/t30/t60) but no hourly_speeds
    // If so, show the OLD chart logic as fallback?
    // Or just waiting state.
    const hasLegacy = predictions.length > 0;

    if (hasLegacy) {
      // Render simple spatial summary (Legacy)
      // ... (We can keep old logic here if needed, but let's encourage V2)
    }

    return (
      <div className="glass-panel p-6 flex flex-col items-center justify-center min-h-[200px]">
        <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center mb-3">
          <Radio size={24} className="text-slate-400" />
        </div>
        <p className="text-sm font-medium text-slate-600">Awaiting V2 Forecast Stream</p>
        <p className="text-xs text-slate-400 mt-1">Connecting to NeuroFlow Brain...</p>
      </div>
    );
  }

  return (
    <div className="glass-panel p-6">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="w-8 h-8 rounded-lg bg-emerald-50 border border-emerald-100 flex items-center justify-center">
          <BarChart2 size={16} className="text-emerald-600" />
        </div>
        <div className="flex-1">
          <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider">12-Hour System Forecast</h3>
          <p className="text-[10px] text-slate-400">Median + upper bound (q90) • LightGBM + GCN</p>
        </div>
        <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-emerald-50 border border-emerald-100">
          <Activity size={12} className="text-emerald-500" />
          <span className="text-[9px] font-bold text-emerald-700 tracking-wide">LIVE</span>
        </div>
      </div>

      {/* Chart */}
      <div className="h-[180px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="gradientSpeed" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis
              dataKey="name"
              tick={{ fontSize: 9, fill: '#94a3b8' }}
              axisLine={{ stroke: '#f1f5f9' }}
              tickLine={false}
              dy={10}
              interval={0} // Show all labels that exist (we sparse them in data prep)
            />
            <YAxis
              tick={{ fontSize: 9, fill: '#94a3b8' }}
              domain={[0, 80]}
              axisLine={false}
              tickLine={false}
              dx={-10}
              label={{ value: 'km/h', angle: -90, position: 'insideLeft', style: { fontSize: '9px', fill: '#cbd5e1' } }}
            />
            <Tooltip
              contentStyle={{
                background: 'rgba(255, 255, 255, 0.9)',
                backdropFilter: 'blur(8px)',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
                fontSize: '11px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                padding: '8px 12px',
              }}
              formatter={(value: number, name: string, props: { payload?: { speedQ50?: number; speedQ90?: number } }) => {
                const p = props?.payload;
                if (p?.speedQ50 != null && p?.speedQ90 != null && p.speedQ50 !== p.speedQ90) {
                  return [`Median: ${p.speedQ50} km/h · Upper: ${p.speedQ90} km/h`, 'Speed range'];
                }
                return [`${value} km/h`, 'Avg Speed'];
              }}
              labelFormatter={(label, payload) => {
                if (payload && payload.length > 0 && payload[0]?.payload) {
                  const p = payload[0].payload as { tooltipLabel?: string };
                  return `Forecast: ${p.tooltipLabel ?? label}`;
                }
                return `Forecast: ${label}`;
              }}
            />
            <Area
              type="monotone"
              dataKey="speed"
              stroke="#10b981"
              strokeWidth={2}
              fill="url(#gradientSpeed)"
              name="Avg Speed"
              activeDot={{ r: 4, strokeWidth: 0, fill: '#10b981' }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Legend / Info */}
      <div className="flex items-center justify-between mt-4 pt-4 border-t border-slate-100">
        <div className="flex items-center gap-2">
          <LegendItem color="#10b981" label="Forecast Trend" />
        </div>
        <div className="text-[10px] text-slate-400">
          Next 12 Hours (15 min intervals)
        </div>
      </div>
    </div>
  );
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <div
        className={`w-2 h-2 rounded-full`}
        style={{
          background: color,
        }}
      />
      <span className="text-[10px] font-medium text-slate-500">{label}</span>
    </div>
  );
}
