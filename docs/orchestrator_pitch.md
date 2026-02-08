# NeuroFlow-Orchestrator — 30-Second Pitch (Metric-Backed)

**Core objective:** Traffic system that prevents accidents, reduces emissions, and maintains network equilibrium.

- **Data:** Single source of truth — `training_dataset_enriched.csv` (36,720 rows, 16 roads, 402 days). No synthetic labels except derived speed from speed_band.
- **Phase 1:** Temporal split (no shuffle). Baseline: Historical Average Test MAE ≈ 11.76 km/h, RMSE ≈ 14.78. Road network: 16 nodes, directed weighted graph from geometry.
- **Phase 2:** Multi-horizon ST-GCN (GCN + LSTM) trained on same dataset. Test MAE reported; ≥30% improvement over baseline is the validation target (achieved only with sufficient tuning). Leakage check: train max timestamp < test min timestamp.
- **Phase 3:** Dynamic Risk Fields (risk tensor summary, hotspot count from speed/density/incidents). GreenWave Eco-Routing (emission proxy from dataset). Event Impact Encoder (attribution: speed delta when has_major_event).
- **Claims:** All metrics from orchestrator outputs; no speculative numbers. Frontend reflects every phase (dataset profile, baselines, Phase 2 metrics, Phase 3 innovation).

**Differentiation:** Proactive safety (risk fields), sustainability (eco-routing), system equilibrium (ε-Nash in main app). Forecasting: multi-horizon 1h / 6h / 24h from primary dataset only.
