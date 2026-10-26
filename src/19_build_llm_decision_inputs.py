"""
19_build_llm_decision_inputs.py

Creates JSONL inputs for the future LLM decision-note layer.

Design:
The LLM is not used as a predictor. It receives:
- calibrated risk score
- macro evidence
- network evidence
- firm/sector attribution evidence
- explicit instructions to cite evidence IDs and avoid unsupported claims.

Inputs:
- data_processed/evidence/decision_records_v4.csv
- data_processed/evidence/decision_evidence_v4.csv

Outputs:
- data_processed/evidence/llm_decision_inputs.jsonl
- data_processed/evidence/llm_decision_inputs_sample.json
"""

import json
import pandas as pd

from config import EVIDENCE_DIR
from common import ensure_dirs

ensure_dirs(EVIDENCE_DIR)

decision_path = EVIDENCE_DIR / "decision_records_v4.csv"
evidence_path = EVIDENCE_DIR / "decision_evidence_v4.csv"

decisions = pd.read_csv(decision_path, parse_dates=["date"])
evidence = pd.read_csv(evidence_path, parse_dates=["date"])

out_path = EVIDENCE_DIR / "llm_decision_inputs.jsonl"
sample_path = EVIDENCE_DIR / "llm_decision_inputs_sample.json"

system_instruction = (
    "You are a financial systemic-risk decision-support assistant. "
    "You must generate concise, evidence-grounded decision notes. "
    "Do not provide investment advice. "
    "Do not claim causal contagion unless the evidence explicitly supports it. "
    "Distinguish calibrated risk scoring from relational network evidence. "
    "Every factual statement about the current decision must cite evidence_id values from the provided evidence list. "
    "If the confidence band is low or abstention_flag is 1, explicitly state that the decision should be treated cautiously."
)

records = []

with out_path.open("w", encoding="utf-8") as f:
    for _, drow in decisions.iterrows():
        did = drow["decision_id"]
        ev = evidence[evidence["decision_id"] == did].copy()

        # Keep evidence compact but sufficient.
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
            "date": str(drow["date"].date()),
            "task": "Generate an evidence-grounded systemic-risk decision note.",
            "system_instruction": system_instruction,
            "decision_context": {
                "target": drow["target"],
                "predicted_probability": float(drow["predicted_probability"]),
                "predicted_class": int(drow["predicted_class"]),
                "risk_level": drow["risk_level"],
                "confidence_band": drow["confidence_band"],
                "abstention_flag": int(drow["abstention_flag"]),
                "network_role": drow["network_role"],
            },
            "required_output_schema": {
                "risk_summary": "2-3 sentences",
                "main_macro_evidence": "bullet list with evidence_id citations",
                "relational_network_evidence": "bullet list with evidence_id citations",
                "firm_or_sector_attribution": "bullet list with evidence_id citations",
                "uncertainty_note": "state confidence/abstention limitations",
                "non_advice_disclaimer": "one sentence",
            },
            "evidence": ev_records,
        }

        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        records.append(payload)

if records:
    with sample_path.open("w", encoding="utf-8") as f:
        json.dump(records[0], f, ensure_ascii=False, indent=2)

print(f"Saved: {out_path} ({len(records)} records)")
print(f"Saved sample: {sample_path}")
