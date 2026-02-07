import { useEmissionSavings } from '@/hooks/useTrafficAPI';
import { Leaf, AlertTriangle, Award } from 'lucide-react';

/**
 * Professional emission savings card with glass styling.
 */
export default function EmissionCard() {
  const { data, error } = useEmissionSavings();

  if (error) {
    return (
      <div className="glass-panel p-6 flex items-center gap-3 text-rose-500">
        <AlertTriangle size={24} />
        <p className="text-sm font-medium">Emission data unavailable</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="glass-panel p-6">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-10 h-10 rounded-xl bg-slate-100 flex items-center justify-center">
            <Leaf size={20} className="text-slate-400" />
          </div>
          <div className="flex-1 space-y-2">
            <div className="h-4 w-32 bg-slate-100 rounded" />
            <div className="h-3 w-24 bg-slate-50 rounded" />
          </div>
        </div>
        <div className="h-24 bg-slate-50/50 rounded-xl" />
      </div>
    );
  }

  return (
    <div className="glass-panel p-6 relative overflow-hidden group">
      {/* Background Decoration (Subtle) */}
      <div className="absolute -right-6 -top-6 w-32 h-32 bg-emerald-500/5 rounded-full blur-2xl group-hover:bg-emerald-500/10 transition-colors" />

      {/* Header */}
      <div className="relative flex items-center gap-3 mb-6">
        <div className="w-8 h-8 rounded-lg bg-emerald-50 border border-emerald-100 flex items-center justify-center">
          <Leaf size={16} className="text-emerald-600" />
        </div>
        <div>
          <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider">Carbon Savings</h3>
          <p className="text-[10px] text-slate-400">ARAI Certified Model</p>
        </div>
      </div>

      {/* Big Number */}
      <div className="relative mb-6">
        <div className="flex items-baseline gap-1">
          <span className="text-4xl font-bold text-slate-800 tracking-tight">
            {data.savings_percent.toFixed(1)}
          </span>
          <span className="text-xl font-bold text-emerald-500">%</span>
        </div>
        <p className="text-xs text-slate-500 font-medium">Reduction vs Standard Route</p>
      </div>

      {/* Stats Grid */}
      <div className="relative grid grid-cols-3 gap-2">
        <StatBox
          label="Emitted"
          value={data.eco_route_emission_kg.toFixed(2)}
          unit="kg"
          variant="success"
        />
        <StatBox
          label="Saved"
          value={data.savings_kg.toFixed(2)}
          unit="kg"
          variant="primary"
        />
        <StatBox
          label="Trees"
          value={data.equivalent_trees_per_year.toFixed(1)}
          unit="/yr"
          variant="accent"
        />
      </div>

      {/* Certification Badge */}
      <div className="relative mt-4 flex items-center justify-center">
        <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-50 border border-slate-100">
          <Award size={12} className="text-emerald-600" />
          <span className="text-[9px] font-bold text-slate-500 tracking-wide uppercase">Bharat Stage VI Compliant</span>
        </div>
      </div>
    </div>
  );
}

function StatBox({
  label,
  value,
  unit,
  variant,
}: {
  label: string;
  value: string;
  unit: string;
  variant: 'success' | 'primary' | 'accent';
}) {
  const styles = {
    success: 'bg-emerald-50/50 border-emerald-100 text-emerald-700',
    primary: 'bg-slate-50/50 border-slate-100 text-slate-700',
    accent: 'bg-teal-50/50 border-teal-100 text-teal-700',
  };

  return (
    <div className={`p-2.5 rounded-lg border flex flex-col items-center justify-center ${styles[variant]}`}>
      <p className="text-lg font-bold leading-none mb-1">
        {value}
        <span className="text-[9px] font-normal opacity-70 ml-0.5">{unit}</span>
      </p>
      <p className="text-[9px] font-bold uppercase tracking-wide opacity-60">{label}</p>
    </div>
  );
}
