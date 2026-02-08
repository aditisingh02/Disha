/**
 * Phase 1 Foundation — Dataset profile, baseline metrics, road network.
 * Phase 2 — Forecast metrics (ST-GCN vs baseline). Single source of truth: training_dataset_enriched.
 */
import { useCallback, useEffect, useState } from 'react';
import { Database, BarChart2, Map, TrendingUp } from 'lucide-react';
import {
  getOrchestratorDatasetProfile,
  getOrchestratorBaselineMetrics,
  getOrchestratorRoadNetworkSummary,
  getOrchestratorPhase2Metrics,
  getOrchestratorPhase3Innovation,
} from '@/utils/api';

interface Profile {
  n_rows?: number;
  temporal_coverage?: { min?: string; max?: string; span_days?: number; unique_timestamps?: number };
  geometry_summary?: { unique_road_names?: number };
  missing_value_report?: Record<string, unknown>;
}

export default function Phase1Foundation() {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [baselines, setBaselines] = useState<Record<string, { model: string; test_mae: number; test_rmse: number }> | null>(null);
  const [roadSummary, setRoadSummary] = useState<{ nodes: number; edges: number } | null>(null);
  const [phase3, setPhase3] = useState<{ dynamic_risk_fields?: { risk_tensor_summary?: { hotspot_count?: number } }; greenwave_eco_routing?: Record<string, unknown>; event_impact_encoder?: { attribution_scores?: { event_impact_on_speed_kmh?: number } } } | null>(null);
  const [phase2, setPhase2] = useState<{
    test_mae: number;
    baseline_test_mae: number;
    improvement_ratio: number;
    meets_30_percent_improvement: boolean;
    leakage_check_passed: boolean;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchAll = useCallback(() => {
    getOrchestratorDatasetProfile()
      .then((p) => setProfile(p))
      .catch(() => setError('Profile unavailable'));
    getOrchestratorBaselineMetrics()
      .then((b) => setBaselines(b))
      .catch(() => {});
    getOrchestratorRoadNetworkSummary()
      .then((r) => setRoadSummary(r))
      .catch(() => {});
    getOrchestratorPhase2Metrics()
      .then((p) => setPhase2(p))
      .catch(() => {});
    getOrchestratorPhase3Innovation()
      .then((p) => setPhase3(p))
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 60_000); // refresh every 60s
    return () => clearInterval(interval);
  }, [fetchAll]);

  if (error && !profile) {
    return (
      <div className="glass-panel p-4 rounded-xl border border-slate-200">
        <p className="text-xs text-slate-500">Orchestrator: run profiling first</p>
      </div>
    );
  }

  return (
    <div className="glass-panel p-6 rounded-xl border border-slate-200">
      <div className="flex items-center gap-2 mb-4">
        <Database className="text-emerald-600" size={18} />
        <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider">Phase 1 — Foundation</h3>
      </div>
      <div className="space-y-4 text-xs">
        {profile && (
          <div>
            <p className="text-slate-500 mb-1">Dataset (training_dataset_enriched)</p>
            <p className="font-mono text-slate-700">
              {profile.n_rows?.toLocaleString()} rows · {profile.temporal_coverage?.span_days} days · {profile.geometry_summary?.unique_road_names} roads
            </p>
            {profile.temporal_coverage?.min && (
              <p className="text-slate-400 mt-0.5">
                {profile.temporal_coverage.min.slice(0, 10)} → {profile.temporal_coverage?.max?.slice(0, 10)}
              </p>
            )}
          </div>
        )}
        {roadSummary && (
          <div className="flex items-center gap-2">
            <Map size={14} className="text-slate-400" />
            <span className="text-slate-600">Road network: {roadSummary.nodes} nodes, {roadSummary.edges} edges</span>
          </div>
        )}
        {baselines && (
          <div>
            <p className="text-slate-500 mb-2 flex items-center gap-1">
              <BarChart2 size={12} /> Baselines (temporal split)
            </p>
            <ul className="space-y-1">
              {Object.entries(baselines).map(([name, m]) => (
                <li key={name} className="font-mono text-slate-700">
                  {m.model}: Test MAE={m.test_mae.toFixed(2)} · RMSE={m.test_rmse.toFixed(2)}
                </li>
              ))}
            </ul>
          </div>
        )}
        {phase2 && (
          <div className="pt-2 border-t border-slate-100">
            <p className="text-slate-500 mb-1 flex items-center gap-1">
              <TrendingUp size={12} /> Phase 2 — ST-GCN
            </p>
            <p className="font-mono text-slate-700 text-[11px]">
              Test MAE={phase2.test_mae.toFixed(2)} · Baseline={phase2.baseline_test_mae.toFixed(2)} · Improvement={(phase2.improvement_ratio * 100).toFixed(1)}%
            </p>
            <p className="text-[10px] mt-0.5">
              {phase2.meets_30_percent_improvement ? (
                <span className="text-emerald-600">Meets ≥30% improvement</span>
              ) : (
                <span className="text-amber-600">Below 30% (tune/train longer)</span>
              )}
              {' · '}
              Leakage check: {phase2.leakage_check_passed ? 'passed' : 'failed'}
            </p>
          </div>
        )}
        {phase3 && (
          <div className="pt-2 border-t border-slate-100 text-[10px] text-slate-600">
            <p className="font-semibold text-slate-500 mb-0.5">Phase 3 — Innovation</p>
            <p>Risk hotspots: {phase3.dynamic_risk_fields?.risk_tensor_summary?.hotspot_count ?? '—'}</p>
            <p>Event impact on speed: {phase3.event_impact_encoder?.attribution_scores?.event_impact_on_speed_kmh ?? '—'} km/h</p>
          </div>
        )}
      </div>
    </div>
  );
}
