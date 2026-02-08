# NeuroFlow-Orchestrator — Terminal Mode

Terminal-first verification when frontend visualization is unavailable. **Terminal output is the source of truth.**

## Execution mode

- **EXECUTION_ORDER:** STRICT (Phase 1 → 2 → 3 → 4)
- **Primary dataset:** `training_dataset_enriched.csv` (read-only; no modification)
- **Frontend:** Disabled; frontend inputs simulated via CLI

## Commands

From repo root (or `neuroflow_backend`):

```bash
# Run full Phase 1–4 timeline; all metrics printed to terminal
python -m app.orchestrator.terminal_mode --timeline

# Run CLI demo only (prompts for origin, destination, departure, optimization, event_context)
python -m app.orchestrator.terminal_mode --cli

# CLI demo with default inputs (no prompt)
python -m app.orchestrator.terminal_mode --cli --no-prompt

# CLI demo with explicit inputs (scriptable)
python -m app.orchestrator.terminal_mode --cli --origin PIE --destination AYE --optimization eco --event-context none
```

Default (no args): run full timeline, then ask "Run CLI demo now? [y/N]".

## Required terminal sections

- **INPUT_RECEIVED** — Echo and validate CLI inputs (origin_node_id, destination_node_id, departure_time, optimization_preference, event_context)
- **MODEL_OUTPUT** — Multi-horizon forecasts (1h, 3h, 6h, 12h, 24h: speed, congestion level, 80% CI), congestion_classification (level, score, delay_vs_freeflow_min), model_version
- **RISK_ANALYSIS** — Risk at origin/destination, hotspot count
- **ROUTING_COMPARISON** — Fastest vs eco route (time, emissions, % reduction, path)
- **UNCERTAINTY_ESTIMATES** — Speed and emission-reduction intervals
- **FINAL_OUTPUTS** — end_to_end_latency_seconds, system_consistency_checks, uncertainty_intervals

## CLI input schema

| Input | Options / format |
|-------|-------------------|
| origin_node_id | Road name (e.g. PIE, AYE); must be in graph |
| destination_node_id | Road name |
| departure_time | ISO datetime or `now` |
| optimization_preference | `fastest`, `eco`, `balanced` |
| event_context | `none`, `accident`, `weather`, `public_event` |

## Validation

- Temporal train/val/test split; no shuffling across time
- No future leakage (train_max < test_min)
- All preprocessing steps logged to terminal
- Metrics (MAE, RMSE, improvement %, leakage) printed to terminal
