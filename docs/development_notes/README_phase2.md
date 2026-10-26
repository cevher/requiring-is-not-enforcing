# Phase 2 scripts: financial systemic risk feasibility refinement

Copy these files into your existing project's `src/` folder.

Recommended run order after your existing scripts:

```powershell
python src/08_build_forward_labels.py
python src/09_build_residual_network.py
python src/10_build_tail_network_metrics.py
python src/11_build_prediction_baselines.py
python src/12_build_provenance_evidence_table_v2.py
```

Main new outputs:

- `data_processed/labels/snapshot_forward_labels.csv`
- `data_processed/labels/forward_label_summary.csv`
- `data_processed/networks/residual_corr_edges.csv`
- `data_processed/networks/residual_network_metrics.csv`
- `data_processed/networks/tail_comovement_edges.csv`
- `data_processed/networks/tail_network_metrics.csv`
- `outputs/reports/baseline_model_results.csv`
- `outputs/reports/baseline_predictions.csv`
- `data_processed/evidence/evidence_table_v2.csv`
- `outputs/reports/evidence_table_v2_summary.csv`

Purpose:

1. Replace weak same-day labels with forward external stress labels.
2. Add market-adjusted residual networks.
3. Add downside tail co-movement networks.
4. Test whether network and macro features predict next-period stress.
5. Produce a richer provenance/evidence table for the later LLM layer.
