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
import { X, Clock, TrendingUp, Info } from 'lucide-react';

interface ForecastModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function ForecastModal({ isOpen, onClose }: ForecastModalProps) {
    const routeForecast = useTrafficStore((s) => s.routeForecast);
    const predictions = useTrafficStore((s) => s.predictions);

    const chartData = useMemo(() => {
        // PRIORITY 1: Use route-specific forecast from backend
        if (routeForecast && routeForecast.hourly_speeds?.length === 48) {
            return routeForecast.hourly_speeds.map((speed, idx) => {
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
                    speed: Math.round(speed),
                    idx: idx
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
                idx: idx
            };
        });
    }, [routeForecast, predictions]);

    if (!isOpen) return null;

    // Use routeForecast stats if available, otherwise calculate from chartData
    const peakSpeed = routeForecast?.peak_speed ?? Math.max(...chartData.map(d => d.speed), 0);
    const minSpeed = routeForecast?.min_speed ?? Math.min(...chartData.map(d => d.speed), 0);
    const avgSpeed = routeForecast?.avg_speed ?? (chartData.length > 0 ? Math.round(chartData.reduce((a, b) => a + b.speed, 0) / chartData.length) : 0);
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
                                    formatter={(value: number) => [`${value} km/h`, 'Projected Speed']}
                                    labelFormatter={(label, payload) => {
                                        if (payload && payload.length > 0 && payload[0]?.payload) {
                                            const p = payload[0].payload as { tooltipLabel?: string };
                                            return `Time Horizon: ${p.tooltipLabel ?? label}`;
                                        }
                                        return `Time: ${label}`;
                                    }}
                                />
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

                    <div className="mt-4 flex items-start gap-3 p-4 bg-blue-50/50 rounded-lg border border-blue-100 text-sm text-blue-800">
                        <Info className="shrink-0 mt-0.5" size={18} />
                        <div>
                            <span className="font-semibold block mb-1">About this Forecast</span>
                            This prediction is generated by the ST-GCN (Spatio-Temporal Graph Convolutional Network) model, analyzing real-time data from 300+ sensors across the city. It accounts for weather patterns, historical trends, and live event data.
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
