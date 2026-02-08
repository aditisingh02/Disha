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
import { X, Clock, TrendingUp, Info, MapPin, AlertTriangle, Route } from 'lucide-react';

interface ForecastModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function ForecastModal({ isOpen, onClose }: ForecastModalProps) {
    const routeForecast = useTrafficStore((s) => s.routeForecast);
    const orchestratorRouteForecast = useTrafficStore((s) => s.orchestratorRouteForecast);
    const predictions = useTrafficStore((s) => s.predictions);

    const chartData = useMemo(() => {
        // PRIORITY 1: Use route-specific forecast from backend (q50 median + q90 when present)
        if (routeForecast && routeForecast.hourly_speeds?.length === 48) {
            const q50 = routeForecast.hourly_speeds_q50 ?? routeForecast.hourly_speeds;
            const q90 = routeForecast.hourly_speeds_q90 ?? routeForecast.hourly_speeds;
            return (routeForecast.hourly_speeds as number[]).map((speed, idx) => {
                const hour = Math.floor((idx + 1) * 15 / 60);
                const min = ((idx + 1) * 15) % 60;
                let label = "";
                if ((idx + 1) % 8 === 0) label = `+${(idx + 1) / 4}h`;
                const fullLabel = min > 0 ? `+${hour}h ${min}m` : `+${hour}h`;
                return {
                    name: label,
                    tooltipLabel: fullLabel,
                    speed: Math.round(speed),
                    speedQ50: q50 ? Math.round(q50[idx]) : Math.round(speed),
                    speedQ90: q90 ? Math.round(q90[idx]) : Math.round(speed),
                    idx,
                };
            });
        }

        // FALLBACK: Aggregate from predictions (demo data)
        if (predictions.length === 0) return [];

        const validPreds = predictions.filter(p => p.hourly_speeds && p.hourly_speeds.length === 48);
        if (validPreds.length === 0) return [];

        const totalSteps = 48;
        const avgSpeeds = new Array(totalSteps).fill(0);

        validPreds.forEach(p => {
            p.hourly_speeds!.forEach((speed, idx) => {
                avgSpeeds[idx] += speed;
            });
        });

        const count = validPreds.length;
        return avgSpeeds.map((total, idx) => {
            const avg = total / count;
            const hour = Math.floor((idx + 1) * 15 / 60);
            const min = ((idx + 1) * 15) % 60;

            let label = "";
            if ((idx + 1) % 8 === 0) {
                label = `+${(idx + 1) / 4}h`;
            }

            const fullLabel = min > 0 ? `+${hour}h ${min}m` : `+${hour}h`;

            return {
                name: label,
                tooltipLabel: fullLabel,
                speed: Math.round(avg),
                speedQ50: Math.round(avg),
                speedQ90: Math.round(avg),
                idx,
            };
        });
    }, [routeForecast, predictions]);

    const hasUncertainty = routeForecast?.hourly_speeds_q90 != null && chartData.some((d: { speedQ50?: number; speedQ90?: number }) => (d.speedQ90 ?? d.speed) !== (d.speedQ50 ?? d.speed));
    const q90Mins = chartData.length && routeForecast?.hourly_speeds_q90 ? Math.min(...routeForecast.hourly_speeds_q90) : null;
    const escalationRisk = (q90Mins != null && q90Mins < 25) || (routeForecast?.regime_active && (routeForecast?.min_speed ?? 99) < 30);

    if (!isOpen) return null;

    const peakSpeed = routeForecast?.peak_speed ?? Math.max(...chartData.map((d: { speed: number }) => d.speed), 0);
    const minSpeed = routeForecast?.min_speed ?? Math.min(...chartData.map((d: { speed: number }) => d.speed), 0);
    const avgSpeed = routeForecast?.avg_speed ?? (chartData.length > 0 ? Math.round(chartData.reduce((a: number, b: { speed: number }) => a + b.speed, 0) / chartData.length) : 0);
    const modelVersion = routeForecast?.model_version ?? 'demo';

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4 animate-in fade-in duration-200">
            <div className="bg-white/90 glass-panel w-full max-w-4xl max-h-[90vh] overflow-hidden rounded-2xl shadow-2xl flex flex-col border border-white/40">

                {/* Header */}
                <div className="p-6 border-b border-slate-200/60 flex items-center justify-between bg-white/50">
                    <div>
                        <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
                            <TrendingUp className="text-emerald-600" />
                            12-Hour Traffic Forecast
                        </h2>
                        <p className="text-sm text-slate-500 mt-1">
                            Powered by <span className="font-mono text-emerald-600">{modelVersion}</span> • Predictive Digital Twin
                        </p>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 rounded-full hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors"
                    >
                        <X size={24} />
                    </button>
                </div>

                {/* Body */}
                <div className="p-6 overflow-y-auto flex-1">

                    {escalationRisk && (
                        <div className="mb-4 p-4 rounded-xl bg-amber-50 border border-amber-200 flex items-start gap-3 text-amber-800">
                            <AlertTriangleIcon />
                            <div>
                                <span className="font-semibold block">Congestion likely to escalate</span>
                                <span className="text-sm">The 90th percentile forecast indicates possible spillover or sustained congestion. Consider alternate routes or timing.</span>
                            </div>
                        </div>
                    )}

                    {/* Stats Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                        <StatCard
                            label="Projected Peak Speed"
                            value={`${peakSpeed} km/h`}
                            icon={<TrendingUp size={18} />}
                            color="text-emerald-600"
                            bg="bg-emerald-50"
                        />
                        <StatCard
                            label="Projected Avg Speed"
                            value={`${avgSpeed} km/h`}
                            icon={<Clock size={18} />}
                            color="text-blue-600"
                            bg="bg-blue-50"
                        />
                        <StatCard
                            label="Lowest Speed (Congestion)"
                            value={`${minSpeed} km/h`}
                            icon={<AlertTriangleIcon />}
                            color="text-amber-600"
                            bg="bg-amber-50"
                        />
                    </div>

                    {/* Main Chart */}
                    <div className="h-[400px] w-full bg-white/40 rounded-xl p-4 border border-slate-100 shadow-inner">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                                <defs>
                                    <linearGradient id="gradientSpeedModal" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor="#059669" stopOpacity={0.4} />
                                        <stop offset="100%" stopColor="#059669" stopOpacity={0} />
                                    </linearGradient>
                                    <linearGradient id="gradientUncertainty" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor="#94a3b8" stopOpacity={0.2} />
                                        <stop offset="100%" stopColor="#94a3b8" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                                <XAxis
                                    dataKey="name"
                                    tick={{ fontSize: 12, fill: '#64748b' }}
                                    axisLine={{ stroke: '#e2e8f0' }}
                                    tickLine={false}
                                    dy={10}
                                    interval={0}
                                />
                                <YAxis
                                    tick={{ fontSize: 12, fill: '#64748b' }}
                                    domain={[0, 100]} // Fixed domain for better comparison
                                    axisLine={false}
                                    tickLine={false}
                                    dx={-10}
                                    label={{ value: 'Speed (km/h)', angle: -90, position: 'insideLeft', style: { fontSize: '12px', fill: '#94a3b8' } }}
                                />
                                <Tooltip
                                    contentStyle={{
                                        background: 'rgba(255, 255, 255, 0.95)',
                                        backdropFilter: 'blur(12px)',
                                        border: '1px solid #e2e8f0',
                                        borderRadius: '12px',
                                        fontSize: '13px',
                                        boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
                                        padding: '12px 16px',
                                    }}
                                    formatter={(value: number, name: string, props: { payload?: { speedQ50?: number; speedQ90?: number; speed?: number } }) => {
                                        const p = props?.payload;
                                        if (p && hasUncertainty && (p.speedQ50 != null || p.speedQ90 != null)) {
                                            const med = p.speedQ50 ?? p.speed;
                                            const upper = p.speedQ90 ?? p.speed;
                                            if (med !== upper) return [`Median: ${med} km/h · Upper bound: ${upper} km/h`, 'Speed range'];
                                        }
                                        return [`${value} km/h`, 'Projected Speed'];
                                    }}
                                    labelFormatter={(label, payload) => {
                                        if (payload && payload.length > 0 && payload[0]?.payload) {
                                            const p = payload[0].payload as { tooltipLabel?: string };
                                            return `Time Horizon: ${p.tooltipLabel ?? label}`;
                                        }
                                        return `Time: ${label}`;
                                    }}
                                />
                                {hasUncertainty && (
                                    <Area
                                        type="monotone"
                                        dataKey="speedQ90"
                                        stroke="transparent"
                                        fill="url(#gradientUncertainty)"
                                        strokeWidth={0}
                                        animationDuration={1500}
                                    />
                                )}
                                <Area
                                    type="monotone"
                                    dataKey="speed"
                                    stroke="#059669"
                                    strokeWidth={3}
                                    fill="url(#gradientSpeedModal)"
                                    activeDot={{ r: 6, strokeWidth: 0, fill: '#059669' }}
                                    animationDuration={1500}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Orchestrator output (same as terminal CLI): multi-horizon, congestion, risk, routes */}
                    {orchestratorRouteForecast && !orchestratorRouteForecast.error && (
                        <div className="mt-8 space-y-6 border-t border-slate-200 pt-6">
                            <h3 className="text-base font-bold text-slate-800 flex items-center gap-2">
                                <MapPin className="text-emerald-600" />
                                Orchestrator Route Forecast (same as terminal)
                            </h3>
                            {orchestratorRouteForecast.origin_road && orchestratorRouteForecast.destination_road && (
                                <p className="text-sm text-slate-600">
                                    {orchestratorRouteForecast.origin_road} → {orchestratorRouteForecast.destination_road}
                                </p>
                            )}
                            {/* Multi-horizon table */}
                            {orchestratorRouteForecast.multi_horizon_forecasts && Object.keys(orchestratorRouteForecast.multi_horizon_forecasts).length > 0 && (
                                <div>
                                    <h4 className="text-sm font-semibold text-slate-700 mb-2">Multi-horizon forecasts</h4>
                                    <div className="overflow-x-auto rounded-lg border border-slate-200">
                                        <table className="w-full text-sm">
                                            <thead className="bg-slate-50">
                                                <tr>
                                                    <th className="text-left p-2">Horizon</th>
                                                    <th className="text-right p-2">Speed (km/h)</th>
                                                    <th className="text-left p-2">Level</th>
                                                    <th className="text-right p-2">Score</th>
                                                    <th className="text-right p-2">Delay (min)</th>
                                                    <th className="text-right p-2">CI 80%</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {Object.entries(orchestratorRouteForecast.multi_horizon_forecasts).map(([horizon, f]) => (
                                                    <tr key={horizon} className="border-t border-slate-100">
                                                        <td className="p-2 font-medium">{horizon}</td>
                                                        <td className="p-2 text-right">{f.speed_kmh?.toFixed(1) ?? '—'}</td>
                                                        <td className="p-2">{f.level ?? '—'}</td>
                                                        <td className="p-2 text-right">{(f.score ?? 0).toFixed(2)}</td>
                                                        <td className="p-2 text-right">{f.delay_vs_freeflow_min != null ? f.delay_vs_freeflow_min.toFixed(1) : '—'}</td>
                                                        <td className="p-2 text-right">
                                                            {f.ci_80_lower != null && f.ci_80_upper != null
                                                                ? `${f.ci_80_lower.toFixed(0)}–${f.ci_80_upper.toFixed(0)}`
                                                                : '—'}
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            )}
                            {/* Congestion classification */}
                            {orchestratorRouteForecast.congestion_classification && (
                                <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                                    <h4 className="text-sm font-semibold text-slate-700 mb-2 flex items-center gap-2">
                                        <AlertTriangle className="text-amber-600" size={16} />
                                        Congestion classification
                                    </h4>
                                    <div className="flex flex-wrap gap-4 text-sm">
                                        <span><strong>Level:</strong> {orchestratorRouteForecast.congestion_classification.level}</span>
                                        <span><strong>Score:</strong> {(orchestratorRouteForecast.congestion_classification.score ?? 0).toFixed(2)}</span>
                                        {orchestratorRouteForecast.congestion_classification.delay_vs_freeflow_min != null && (
                                            <span><strong>Delay vs free flow:</strong> {orchestratorRouteForecast.congestion_classification.delay_vs_freeflow_min.toFixed(1)} min</span>
                                        )}
                                        {orchestratorRouteForecast.congestion_classification.speed_ratio != null && (
                                            <span><strong>Speed ratio:</strong> {(orchestratorRouteForecast.congestion_classification.speed_ratio * 100).toFixed(0)}%</span>
                                        )}
                                    </div>
                                </div>
                            )}
                            {/* Risk */}
                            {orchestratorRouteForecast.risk && (
                                <div className="p-4 rounded-xl bg-amber-50/50 border border-amber-200">
                                    <h4 className="text-sm font-semibold text-slate-700 mb-2">Risk</h4>
                                    <div className="flex flex-wrap gap-4 text-sm">
                                        <span>Origin risk: {(orchestratorRouteForecast.risk.origin_risk ?? 0).toFixed(2)}</span>
                                        <span>Destination risk: {(orchestratorRouteForecast.risk.destination_risk ?? 0).toFixed(2)}</span>
                                        <span>Hotspot count: {orchestratorRouteForecast.risk.hotspot_count ?? 0}</span>
                                    </div>
                                </div>
                            )}
                            {/* Routes */}
                            {orchestratorRouteForecast.routes && (
                                <div className="p-4 rounded-xl bg-emerald-50/50 border border-emerald-200">
                                    <h4 className="text-sm font-semibold text-slate-700 mb-2 flex items-center gap-2">
                                        <Route className="text-emerald-600" size={16} />
                                        Routes
                                    </h4>
                                    {orchestratorRouteForecast.routes.path?.length > 0 && (
                                        <p className="text-sm text-slate-600 mb-2">
                                            Path: {orchestratorRouteForecast.routes.path.join(' → ')}
                                        </p>
                                    )}
                                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                                        <span>Total: {orchestratorRouteForecast.routes.total_km?.toFixed(1) ?? '—'} km</span>
                                        <span>Fastest: {orchestratorRouteForecast.routes.fastest_route_time_min?.toFixed(0) ?? '—'} min</span>
                                        <span>Eco: {orchestratorRouteForecast.routes.eco_route_time_min?.toFixed(0) ?? '—'} min</span>
                                        <span>Emission reduction: {orchestratorRouteForecast.routes.percent_emission_reduction?.toFixed(0) ?? '—'}%</span>
                                    </div>
                                    <div className="mt-2 text-xs text-slate-500">
                                        Fastest emissions: {orchestratorRouteForecast.routes.fastest_route_emissions_kg?.toFixed(2) ?? '—'} kg · Eco: {orchestratorRouteForecast.routes.eco_route_emissions_kg?.toFixed(2) ?? '—'} kg
                                    </div>
                                </div>
                            )}
                            {orchestratorRouteForecast.model_version && (
                                <p className="text-xs text-slate-500">Model: {orchestratorRouteForecast.model_version}</p>
                            )}
                        </div>
                    )}

                    <div className="mt-4 flex items-start gap-3 p-4 bg-blue-50/50 rounded-lg border border-blue-100 text-sm text-blue-800">
                        <Info className="shrink-0 mt-0.5" size={18} />
                        <div>
                            <span className="font-semibold block mb-1">About this Forecast</span>
                            Median (solid) and upper-bound (q90) uncertainty band when available. Confidence reflects spread between median and tail. LightGBM baseline + regime-gated residual GCN for peak-risk awareness. Not a single deterministic number—use the range for decision support.
                        </div>
                    </div>

                </div>
            </div>
        </div>
    );
}

function StatCard({ label, value, icon, color, bg }: any) {
    return (
        <div className="p-4 rounded-xl bg-white border border-slate-100 shadow-sm flex items-center gap-4">
            <div className={`p-3 rounded-full ${bg} ${color}`}>
                {icon}
            </div>
            <div>
                <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold">{label}</p>
                <p className="text-2xl font-bold text-slate-800">{value}</p>
            </div>
        </div>
    )
}

function AlertTriangleIcon() {
    return (
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" /><path d="M12 9v4" /><path d="M12 17h.01" /></svg>
    )
}
