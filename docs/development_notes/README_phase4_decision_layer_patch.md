# Phase 4 patch: rule baselines + corrected decision evidence + LLM input format

Copy these files into your existing project's `src/` folder.

New files:
- 17_evaluate_rule_baselines.py
- 18_build_decision_evidence_v4.py
- 19_build_llm_decision_inputs.py
- 20_compare_model_and_rule_baselines.py

Run after Phase 3:

```powershell
python src/17_evaluate_rule_baselines.py
python src/18_build_decision_evidence_v4.py
python src/19_build_llm_decision_inputs.py
python src/20_compare_model_and_rule_baselines.py
```

Main outputs:
- outputs/reports/rule_baseline_results.csv
- outputs/reports/rule_baseline_predictions.csv
- data_processed/evidence/decision_records_v4.csv
- data_processed/evidence/decision_evidence_v4.csv
- outputs/reports/decision_layer_v4_summary.csv
- outputs/reports/decision_layer_v4_selection_metadata.txt
- data_processed/evidence/llm_decision_inputs.jsonl
- data_processed/evidence/llm_decision_inputs_sample.json
- outputs/reports/final_baseline_comparison.csv

Design choice:
The primary risk scoring layer uses macro-only logistic raw predictions by default.
Networks are not forced into the predictor. They are used as relational/provenance
evidence for firm/sector attribution and LLM decision-note generation.
