# Phase 5 patch: label-component ablation + reliability-adjusted decision layer

Copy these files into your existing project's `src/` folder.

New files:
- 21_label_component_ablation.py
- 22_build_decision_evidence_v5.py
- 23_build_llm_decision_inputs_v5.py
- 24_build_v5_method_tables.py

Run order after Phase 4:

```powershell
python src/21_label_component_ablation.py
python src/22_build_decision_evidence_v5.py
python src/23_build_llm_decision_inputs_v5.py
python src/24_build_v5_method_tables.py
```

Main outputs:
- outputs/reports/label_component_ablation_results.csv
- outputs/reports/label_component_ablation_predictions.csv
- outputs/reports/label_component_ablation_fold_summary.csv
- data_processed/evidence/decision_records_v5.csv
- data_processed/evidence/decision_evidence_v5.csv
- outputs/reports/decision_layer_v5_summary.csv
- outputs/reports/decision_layer_v5_reliability_summary.csv
- data_processed/evidence/llm_decision_inputs_v5.jsonl
- data_processed/evidence/llm_decision_inputs_v5_sample.json
- outputs/tables/table_model_vs_rule_main.csv
- outputs/tables/table_label_component_ablation_top.csv
- outputs/tables/table_decision_layer_summary.csv
- outputs/reports/v5_interpretation_notes.md

Design changes:
1. Adds label-component ablations to test whether performance depends too much on VIX/NFCI.
2. Builds a dual-score decision layer:
   - VIX-scaled rule score
   - macro-only logistic model score
   - averaged decision score
3. Adds rule/model disagreement flags.
4. Replaces naive confidence with reliability-adjusted confidence based on previous out-of-sample performance.
5. Prunes evidence for LLM inputs and removes NaN metric text.
6. Keeps networks as relational/provenance evidence, not as the main prediction engine.
