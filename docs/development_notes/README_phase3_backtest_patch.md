# Phase 3 patch: separated networks + expanding-window backtesting

Copy these files into your existing project's `src/` folder.

New files:
- 13_build_separated_networks.py
- 14_build_stress_onset_labels.py
- 15_expanding_window_backtest.py
- 16_build_decision_evidence_v3.py

Run order after the 2005 pipeline and Phase 2 labels exist:

```powershell
python src/13_build_separated_networks.py
python src/14_build_stress_onset_labels.py
python src/15_expanding_window_backtest.py
python src/16_build_decision_evidence_v3.py
```

Main outputs:
- data_processed/networks/separated/*_metrics.csv
- data_processed/networks/separated/*_edges.csv
- data_processed/labels/snapshot_forward_labels_with_onset.csv
- data_processed/labels/onset_label_summary.csv
- outputs/reports/expanding_window_results.csv
- outputs/reports/expanding_window_predictions.csv
- outputs/reports/expanding_window_fold_summary.csv
- data_processed/evidence/decision_evidence_v3.csv
- data_processed/evidence/decision_records_v3.csv

Purpose:
1. Separate threshold networks from top-k networks.
2. Add stress-onset early-warning label.
3. Replace the weak single split with expanding-window temporal validation.
4. Produce decision-level evidence records for the future LLM layer.
