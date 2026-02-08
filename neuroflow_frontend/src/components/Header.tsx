import { useState, useEffect } from 'react';
import { useDataLive } from '@/hooks/useTrafficAPI';
import { BrainCircuit, Target, Leaf, BarChart3, Clock as ClockIcon } from 'lucide-react';

/**
 * Professional glass header with silver/slate accents.
 */
export default function Header() {
    const dataLive = useDataLive();

    return (
        <header className="fixed top-6 left-1/2 -translate-x-1/2 z-50">
            <div className="glass-panel px-6 py-3 flex items-center gap-6 rounded-full">
                {/* Logo & Branding */}
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-slate-900 flex items-center justify-center shadow-lg shadow-slate-900/20">
                        <BrainCircuit size={18} className="text-emerald-400" />
                    </div>

                    <div>
                        <h1 className="text-sm font-bold text-slate-800 tracking-tight">
                            DISHA
                        </h1>
                        <p className="text-[9px] text-slate-500 font-bold tracking-widest uppercase">
                            BharatFlow Intelligence
                        </p>
                    </div>
                </div>

                <div className="h-6 w-px bg-slate-200/50" />

                {/* Center - Feature Pills (Minimal) */}
                <div className="hidden md:flex items-center gap-1 whitespace-nowrap">
                    <FeatureItem icon={<Target size={14} />} label="Nash Eq." />
                    <FeatureItem icon={<Leaf size={14} />} label="Eco-Route" />
                    <FeatureItem icon={<BarChart3 size={14} />} label="ST-GCN" />
                </div>

                <div className="h-6 w-px bg-slate-200/50 hidden md:block" />

                {/* Right - Status & Time */}
                <div className="flex items-center gap-4">
                    {/* Live Status - reflects WebSocket or fresh REST data */}
                    <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full border ${dataLive ? 'bg-emerald-50/50 border-emerald-100' : 'bg-rose-50/50 border-rose-100'}`}>
                        <div className={`relative flex h-2 w-2`}>
                            {dataLive && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-200"></span>}
                            <span className={`relative inline-flex rounded-full h-2 w-2 ${dataLive ? 'bg-emerald-500' : 'bg-rose-500'}`}></span>
                        </div>
                        <span className={`text-[10px] font-bold tracking-wide ${dataLive ? 'text-emerald-700' : 'text-rose-700'}`}>
                            {dataLive ? 'ONLINE' : 'OFFLINE'}
                        </span>
                    </div>

                    {/* Clock */}
                    <Clock />
                </div>
            </div>
        </header>
    );
}

function FeatureItem({ icon, label }: { icon: React.ReactNode; label: string }) {
    return (
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium text-slate-500 hover:bg-slate-100/50 transition-colors cursor-default">
            {icon}
            <span>{label}</span>
        </div>
    );
}

function Clock() {
    const [timeStr, setTimeStr] = useState(() =>
        new Date().toLocaleTimeString('en-IN', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: true,
            timeZone: 'Asia/Kolkata',
        })
    );
    useEffect(() => {
        const t = setInterval(() => {
            setTimeStr(
                new Date().toLocaleTimeString('en-IN', {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: true,
                    timeZone: 'Asia/Kolkata',
                })
            );
        }, 1000);
        return () => clearInterval(t);
    }, []);

    return (
        <div className="flex items-center gap-1.5 text-slate-400">
            <ClockIcon size={14} />
            <span className="text-xs font-semibold text-slate-600 font-mono whitespace-nowrap">{timeStr}</span>
        </div>
    );
}
