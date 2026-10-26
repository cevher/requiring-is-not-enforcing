"""
23_build_llm_decision_inputs_v5.py

Creates compact JSONL inputs for the future LLM decision-note layer.

v5 improvements:
- Compact evidence selection is used from decision_evidence_v5.
- Rule/model disagreement and reliability-adjusted confidence are explicit.
- LLM is explicitly forbidden from treating networks as the primary predictor.
- LLM must cite evidence IDs.
- LLM must disclose uncertainty when confidence is low, reliability is weak, or disagreement exists.

Inputs:
- data_processed/evidence/decision_records_v5.csv
- data_processed/evidence/decision_evidence_v5.csv

Outputs:
- data_processed/evidence/llm_decision_inputs_v5.jsonl
- data_processed/evidence/llm_decision_inputs_v5_sample.json
"""

import json
import pandas as pd

from config import EVIDENCE_DIR
from common import ensure_dirs

ensure_dirs(EVIDENCE_DIR)

decision_path = EVIDENCE_DIR / "decision_records_v5.csv"
evidence_path = EVIDENCE_DIR / "decision_evidence_v5.csv"

decisions = pd.read_csv(decision_path, parse_dates=["date"])
evidence = pd.read_csv(evidence_path, parse_dates=["date"])

out_path = EVIDENCE_DIR / "llm_decision_inputs_v5.jsonl"
sample_path = EVIDENCE_DIR / "llm_decision_inputs_v5_sample.json"

system_instruction = (
    "You are a financial systemic-risk decision-support assistant. "
    "You must generate concise, evidence-grounded decision notes. "
    "Do not provide investment advice. "
    "Do not claim causal contagion unless directly supported by the evidence. "
    "Separate the risk scoring layer from the relational evidence layer. "
    "The network evidence is for relational attribution and explanation, not the primary risk prediction engine. "
    "Every factual statement about the current decision must cite evidence_id values from the evidence list. "
    "If reliability_adjusted_confidence is low, abstention_flag is 1, or disagreement_flag is 1, explicitly state that the decision should be treated cautiously. "
    "Do not mention any metric value that is not present in the evidence list."
)

records = []

with out_path.open("w", encoding="utf-8") as f:
    for _, drow in decisions.iterrows():
        did = drow["decision_id"]
        ev = evidence[evidence["decision_id"] == did].copy()

        ev_records = ev[[
            "evidence_id",
            "entity_type",
            "entity_id",
            "source_type",
            "source_name",
            "evidence_role",
            "value_or_snippet",
        ]].to_dict("records")

        payload = {
            "decision_id": did,
            "date": str(pd.Timestamp(drow["date"]).date()),
            "task": "Generate an evidence-grounded systemic-risk decision note.",
            "system_instruction": system_instruction,
            "decision_context": {
                "target": drow["target"],
                "rule_score": None if pd.isna(drow["rule_score"]) else float(drow["rule_score"]),
                "model_score": None if pd.isna(drow["model_score"]) else float(drow["model_score"]),
                "decision_score": None if pd.isna(drow["decision_score"]) else float(drow["decision_score"]),
                "risk_level": drow["risk_level"],
                "base_confidence": drow["base_confidence"],
                "reliability_adjusted_confidence": drow["reliability_adjusted_confidence"],
                "abstention_flag": int(drow["abstention_flag"]),
                "probability_bin": drow["probability_bin"],
                "historical_bin_n": None if pd.isna(drow["historical_bin_n"]) else int(drow["historical_bin_n"]),
                "historical_bin_accuracy": None if pd.isna(drow["historical_bin_accuracy"]) else float(drow["historical_bin_accuracy"]),
                "historical_bin_error_rate": None if pd.isna(drow["historical_bin_error_rate"]) else float(drow["historical_bin_error_rate"]),
                "reliability_status": drow["reliability_status"],
                "rule_model_score_gap": None if pd.isna(drow["rule_model_score_gap"]) else float(drow["rule_model_score_gap"]),
                "disagreement_flag": int(drow["disagreement_flag"]),
                "network_role": drow["network_role"],
            },
            "required_output_schema": {
                "risk_summary": "2-3 sentences; cite evidence IDs",
                "scoring_interpretation": "explain rule/model score and reliability; cite evidence IDs",
                "macro_evidence": "2-4 bullet points with evidence_id citations",
                "relational_network_evidence": "2-4 bullet points with evidence_id citations; do not call it causal contagion",
                "firm_or_sector_attribution": "2-4 bullet points with evidence_id citations",
                "uncertainty_and_abstention_note": "state confidence, reliability, and disagreement limitations",
                "non_advice_disclaimer": "one sentence",
            },
            "evidence": ev_records,
        }

        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        records.append(payload)

if records:
    # choose a sample that has disagreement or low confidence if possible
    sample = None
    for r in records:
        ctx = r["decision_context"]
        if ctx.get("disagreement_flag") == 1 or ctx.get("reliability_adjusted_confidence") == "low":
            sample = r
            break
    if sample is None:
        sample = records[0]

    with sample_path.open("w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

print(f"Saved: {out_path} ({len(records)} records)")
print(f"Saved sample: {sample_path}")
