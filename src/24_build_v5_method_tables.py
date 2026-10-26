"""
24_build_v5_method_tables.py

Creates compact tables for manuscript drafting.

Inputs:
- outputs/reports/final_baseline_comparison.csv
- outputs/reports/label_component_ablation_results.csv
- data_processed/evidence/decision_records_v5.csv
- outputs/reports/decision_layer_v5_summary.csv

Outputs:
- outputs/tables/table_model_vs_rule_main.csv
- outputs/tables/table_label_component_ablation_top.csv
- outputs/tables/table_decision_layer_summary.csv
- outputs/reports/v5_interpretation_notes.md
"""

import pandas as pd
from pathlib import Path

from config import REPORT_DIR, TABLE_DIR, EVIDENCE_DIR
from common import ensure_dirs

ensure_dirs(TABLE_DIR, REPORT_DIR)

notes = []

# Main comparison
main_path = REPORT_DIR / "final_baseline_comparison.csv"
if main_path.exists():
    main = pd.read_csv(main_path)
    target_main = main[main["target"] == "next_20d_external_stress"].copy()
    cols = ["target", "system_type", "method_name", "brier", "roc_auc", "average_precision", "ece_10_bins", "balanced_accuracy", "n_oos"]
    top = target_main.sort_values(["brier", "ece_10_bins"]).head(15)
    top[[c for c in cols if c in top.columns]].to_csv(TABLE_DIR / "table_model_vs_rule_main.csv", index=False)

    notes.append("## Main rule/model comparison")
    if not top.empty:
        best = top.iloc[0]
        notes.append(
            f"- Best Brier for next_20d_external_stress: {best.get('method_name')} "
            f"({best.get('system_type')}), Brier={best.get('brier'):.4f}, "
            f"ROC-AUC={best.get('roc_auc'):.4f}, ECE={best.get('ece_10_bins'):.4f}."
        )
        if "vix" in str(best.get("method_name", "")).lower():
            notes.append("- Interpretation: VIX-based rule is a strong baseline; avoid claiming predictive superiority over volatility indicators.")
else:
    notes.append("- Missing final_baseline_comparison.csv")

# Ablation
ablation_path = REPORT_DIR / "label_component_ablation_results.csv"
if ablation_path.exists():
    abl = pd.read_csv(ablation_path)
    target_abl = abl[abl["target"] == "next_20d_external_stress"].copy()
    cols = ["target", "feature_set", "model", "calibration", "brier", "roc_auc", "average_precision", "ece_10_bins", "balanced_accuracy", "n_oos"]
    top_abl = target_abl.sort_values(["brier", "ece_10_bins"]).head(20)
    top_abl[[c for c in cols if c in top_abl.columns]].to_csv(TABLE_DIR / "table_label_component_ablation_top.csv", index=False)

    notes.append("\n## Label-component ablation")
    if not top_abl.empty:
        best_abl = top_abl.iloc[0]
        notes.append(
            f"- Best ablation model: {best_abl.get('feature_set')} | {best_abl.get('model')} | {best_abl.get('calibration')}, "
            f"Brier={best_abl.get('brier'):.4f}, ROC-AUC={best_abl.get('roc_auc'):.4f}, ECE={best_abl.get('ece_10_bins'):.4f}."
        )
        notes.append("- Use this table to discuss whether performance collapses when VIX/NFCI are removed.")
else:
    notes.append("- Missing label_component_ablation_results.csv")

# Decision summary
decision_path = EVIDENCE_DIR / "decision_records_v5.csv"
summary_path = REPORT_DIR / "decision_layer_v5_summary.csv"

if decision_path.exists():
    dec = pd.read_csv(decision_path)
    summary = {
        "n_decisions": len(dec),
        "mean_rule_score": dec["rule_score"].mean() if "rule_score" in dec else None,
        "mean_model_score": dec["model_score"].mean() if "model_score" in dec else None,
        "mean_decision_score": dec["decision_score"].mean() if "decision_score" in dec else None,
        "disagreement_rate": dec["disagreement_flag"].mean() if "disagreement_flag" in dec else None,
        "abstention_rate": dec["abstention_flag"].mean() if "abstention_flag" in dec else None,
    }
    pd.DataFrame([summary]).to_csv(TABLE_DIR / "table_decision_layer_summary.csv", index=False)

    notes.append("\n## Decision layer")
    notes.append(f"- Decision records: {summary['n_decisions']}")
    notes.append(f"- Rule/model disagreement rate: {summary['disagreement_rate']:.4f}")
    notes.append(f"- Abstention rate: {summary['abstention_rate']:.4f}")
else:
    notes.append("- Missing decision_records_v5.csv")

if summary_path.exists():
    evs = pd.read_csv(summary_path)
    notes.append("\n## Evidence layer")
    notes.append(f"- Evidence categories: {len(evs)}")
    if "mean_records_per_decision" in evs.columns and not evs.empty:
        notes.append(f"- Mean evidence records per decision: {evs['mean_records_per_decision'].iloc[0]:.2f}")

out = REPORT_DIR / "v5_interpretation_notes.md"
out.write_text("\n".join(notes), encoding="utf-8")

print(f"Saved manuscript tables to: {TABLE_DIR}")
print(f"Saved notes: {out}")
print("\n".join(notes))
