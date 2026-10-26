"""
26_generate_llm_prompt_conditions.py

HOTFIX v5: compact JSON + financial interpretation guide + safe prompt evidence IDs

Purpose:
Generates LLM prompt conditions from v5 decision inputs.

Why this version exists:
- GPT-OSS 20B followed the evidence constraints well, but sometimes corrupted
  evidence IDs containing decimal ranges, especially IDs such as:
      RELIABILITY-20110429-0.4-0.6
  into variants such as:
      RELIABILITY-20110429-0.4-6

- This version creates prompt-only safe evidence IDs. Original evidence files are
  not modified. Only the evidence IDs shown to the LLM inside prompt conditions
  are sanitized.

Example:
    RELIABILITY-20110429-0.4-0.6
    -> RELIABILITY-20110429-BIN_0P4_0P6

Other retained features:
1. Compact JSON schema.
2. Financial interpretation guide.
3. no_reliability removes reliability evidence and scrubs reliability/disagreement/
   abstention fields from SCORE evidence text.
4. full condition requires network evidence to be relational/contextual, not the
   primary prediction engine.
5. no_network prevents firm/sector network attribution.

Inputs:
- data_processed/evidence/llm_decision_inputs_v5.jsonl
- data_processed/evidence/llm_eval_selected_decisions.csv

Output:
- data_processed/evidence/llm_prompt_conditions.jsonl

Conditions:
- full
- no_network
- no_reliability
- free_form
"""

import argparse
import copy
import json
import re
import pandas as pd

from config import EVIDENCE_DIR
from common import ensure_dirs

ensure_dirs(EVIDENCE_DIR)


NETWORK_ROLES = {
    "relational_context",
    "firm_level_relational_attribution",
    "sector_level_relational_attribution",
}

NETWORK_SOURCE_TYPES = {
    "financial_network_summary",
    "network_centrality",
    "sector_network_centrality",
}

RELIABILITY_ROLES = {"reliability_context"}
RELIABILITY_SOURCE_TYPES = {"historical_oos_reliability"}

RELIABILITY_CONTEXT_KEYS = {
    "reliability_adjusted_confidence",
    "base_confidence",
    "historical_bin_n",
    "historical_bin_accuracy",
    "historical_bin_error_rate",
    "historical_bin_brier",
    "reliability_status",
    "abstention_flag",
    "disagreement_flag",
    "rule_model_score_gap",
    "probability_bin",
}

RELIABILITY_SNIPPET_FIELDS = [
    "reliability_adjusted_confidence",
    "base_confidence",
    "abstention_flag",
    "disagreement_flag",
    "historical_bin_n",
    "historical_bin_accuracy",
    "historical_bin_error_rate",
    "historical_bin_brier",
    "reliability_status",
    "rule_model_score_gap",
    "probability_bin",
]

NETWORK_CONTEXT_KEYS = {"network_role"}


FINANCIAL_INTERPRETATION_GUIDE = [
    "Do not describe VIX below 20 as high stress. Use low-to-moderate, mild, or moderately elevated volatility unless other evidence clearly supports stronger stress language.",
    "Interpret negative NFCI values as looser-than-average or less tight financial conditions, not as financial-sector deterioration or stress.",
    "Do not infer firm-level stress directly from network density, top-k centrality, residual centrality, or tail centrality.",
    "Describe network metrics as relational context, co-movement evidence, or attribution evidence.",
    "Do not claim causal contagion, spillover, causal transmission, or directional transmission from correlation, top-k, residual, or tail network evidence.",
    "Use systemic-risk monitoring or decision-support signal language rather than investment or trading language.",
]


COMPACT_OUTPUT_SCHEMA = {
    "risk_summary": {
        "text": "max 2 short sentences",
        "evidence_ids": ["1-3 IDs"]
    },
    "scoring_interpretation": {
        "text": "max 1 short sentence",
        "evidence_ids": ["1-2 IDs"]
    },
    "macro_evidence": [
        {"claim": "max 25 words", "evidence_ids": ["1-2 IDs"]}
    ],
    "relational_network_evidence": [
        {"claim": "max 25 words; explicitly say relational/context if relevant", "evidence_ids": ["1-2 IDs"]}
    ],
    "firm_or_sector_attribution": [
        {"claim": "max 25 words", "evidence_ids": ["1-2 IDs"]}
    ],
    "uncertainty_and_abstention_note": {
        "text": "max 1 short sentence",
        "evidence_ids": ["1-2 IDs if available"]
    },
    "non_advice_disclaimer": "This is systemic-risk decision support, not investment advice."
}


FREE_FORM_SCHEMA = {
    "risk_summary": "max 2 short sentences",
    "uncertainty_note": "state that limited structured evidence is available",
    "non_advice_disclaimer": "This is systemic-risk decision support, not investment advice."
}


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def make_safe_evidence_id(evidence_id):
    """
    Creates LLM-friendly evidence IDs.

    The main problem is reliability-bin suffixes such as 0.4-0.6. LLMs sometimes
    shorten or corrupt these. We replace those suffixes with BIN_0P4_0P6.

    Other IDs are also lightly sanitized to avoid spaces, slashes, and repeated
    punctuation, while preserving readability.
    """
    if not isinstance(evidence_id, str):
        return evidence_id

    safe = evidence_id.strip()

    # Specific fix for reliability IDs ending in decimal bins.
    # Example: RELIABILITY-20110429-0.4-0.6 -> RELIABILITY-20110429-BIN_0P4_0P6
    m = re.match(r"^(RELIABILITY-\d{8})-(\d+\.\d+)-(\d+\.\d+)$", safe)
    if m:
        prefix, lo, hi = m.groups()
        lo = lo.replace(".", "P")
        hi = hi.replace(".", "P")
        return f"{prefix}-BIN_{lo}_{hi}"

    # General safety cleanup for any remaining awkward characters.
    safe = safe.replace(" ", "_")
    safe = safe.replace("/", "_")
    safe = safe.replace("\\", "_")

    # Avoid decimal dots inside IDs where possible.
    safe = re.sub(r"(?<=\d)\.(?=\d)", "P", safe)

    # Collapse accidental double separators.
    safe = re.sub(r"__+", "_", safe)
    safe = re.sub(r"--+", "-", safe)

    return safe


def apply_safe_evidence_ids(payload):
    """
    Applies safe prompt-only evidence IDs inside:
    - evidence[*].evidence_id
    - allowed_evidence_ids

    Also stores original_evidence_id in each evidence item for traceability.
    """
    alias_map = {}

    for e in payload.get("evidence", []):
        original = e.get("evidence_id")
        if not original:
            continue

        safe = make_safe_evidence_id(original)

        # Ensure uniqueness if two originals map to same safe string.
        if safe in alias_map.values() and original not in alias_map:
            suffix = 2
            candidate = f"{safe}_V{suffix}"
            while candidate in alias_map.values():
                suffix += 1
                candidate = f"{safe}_V{suffix}"
            safe = candidate

        alias_map[original] = safe
        e["original_evidence_id"] = original
        e["evidence_id"] = safe

    payload["evidence_id_alias_map"] = alias_map
    payload["allowed_evidence_ids"] = [
        e.get("evidence_id")
        for e in payload.get("evidence", [])
        if e.get("evidence_id")
    ]

    return payload


def scrub_semicolon_fields(text, fields_to_remove):
    if not isinstance(text, str):
        return text

    parts = [p.strip() for p in text.split(";")]
    cleaned = []

    for part in parts:
        lower = part.lower()
        remove = False
        for field in fields_to_remove:
            if lower.startswith(field.lower() + "="):
                remove = True
                break
        if not remove and part:
            cleaned.append(part)

    return "; ".join(cleaned)


def remove_network_evidence(evidence):
    out = []
    for e in evidence:
        if e.get("evidence_role") in NETWORK_ROLES:
            continue
        if e.get("source_type") in NETWORK_SOURCE_TYPES:
            continue
        out.append(e)
    return out


def remove_reliability_evidence(evidence):
    out = []
    for e in evidence:
        if e.get("evidence_role") in RELIABILITY_ROLES:
            continue
        if e.get("source_type") in RELIABILITY_SOURCE_TYPES:
            continue

        e2 = copy.deepcopy(e)

        if e2.get("source_type") in {"dual_risk_score", "calibrated_risk_score"}:
            e2["value_or_snippet"] = scrub_semicolon_fields(
                e2.get("value_or_snippet", ""),
                RELIABILITY_SNIPPET_FIELDS,
            )
            e2["value_or_snippet"] = (
                e2.get("value_or_snippet", "")
                + "; reliability_disagreement_and_abstention_fields_withheld_for_this_condition"
            )

        out.append(e2)

    return out


def add_guide_to_instruction(payload):
    guide_text = " Financial interpretation guide: " + " ".join(
        f"({i+1}) {rule}" for i, rule in enumerate(FINANCIAL_INTERPRETATION_GUIDE)
    )
    payload["system_instruction"] = payload.get("system_instruction", "") + guide_text
    return payload


def condition_payload(base, condition):
    payload = copy.deepcopy(base)

    evidence = payload.get("evidence", [])
    context = payload.get("decision_context", {})

    if condition == "full":
        payload["condition"] = "full"
        payload["condition_description"] = (
            "Full framework: score, macro evidence, reliability evidence, and "
            "relational network evidence are provided."
        )
        payload["system_instruction"] += (
            " In relational_network_evidence, explicitly state that network evidence "
            "is for relational attribution/context, not the primary stress predictor."
        )

    elif condition == "no_network":
        payload["condition"] = "no_network"
        payload["condition_description"] = (
            "Network evidence is withheld. The note must rely on risk score, "
            "reliability, and macro evidence only."
        )
        payload["evidence"] = remove_network_evidence(evidence)

        for key in NETWORK_CONTEXT_KEYS:
            context.pop(key, None)
        payload["decision_context"] = context

        payload["system_instruction"] += (
            " Network evidence has been intentionally withheld. Do not infer firm-level "
            "or sector-level network attribution. Do not mention network centrality, "
            "graph structure, or relational network evidence unless saying it was not provided."
        )

    elif condition == "no_reliability":
        payload["condition"] = "no_reliability"
        payload["condition_description"] = (
            "Reliability evidence, disagreement indicators, and abstention/confidence "
            "fields are withheld."
        )
        payload["evidence"] = remove_reliability_evidence(evidence)

        for key in RELIABILITY_CONTEXT_KEYS:
            context.pop(key, None)
        payload["decision_context"] = context

        payload["system_instruction"] += (
            " Reliability evidence, disagreement indicators, abstention information, "
            "and reliability-adjusted confidence have been withheld. Do not invent "
            "historical reliability statistics, abstention flags, confidence bands, "
            "or rule/model disagreement claims."
        )

    elif condition == "free_form":
        payload["condition"] = "free_form"
        payload["condition_description"] = (
            "Minimal baseline: no structured evidence list is provided."
        )

        payload["decision_context"] = {
            "target": context.get("target"),
            "decision_score": context.get("decision_score"),
            "risk_level": context.get("risk_level"),
            "date": payload.get("date"),
        }

        payload["evidence"] = []
        payload["system_instruction"] = (
            "You are a financial systemic-risk decision-support assistant. "
            "Generate a concise decision note from minimal context. "
            "Do not provide investment advice. Do not claim causal contagion. "
            "Because no structured evidence is provided, do not fabricate evidence IDs, "
            "firm-specific claims, sector-specific claims, or macro indicator values."
        )
        payload = add_guide_to_instruction(payload)
        payload["required_output_schema"] = FREE_FORM_SCHEMA
        payload["allowed_evidence_ids"] = []
        payload["evidence_id_alias_map"] = {}
        return payload

    else:
        raise ValueError(f"Unknown condition: {condition}")

    payload = add_guide_to_instruction(payload)
    payload["required_output_schema"] = COMPACT_OUTPUT_SCHEMA

    payload = apply_safe_evidence_ids(payload)

    return payload


def build_messages(payload):
    user_object = {
        "decision_id": payload.get("decision_id"),
        "date": payload.get("date"),
        "condition": payload.get("condition"),
        "condition_description": payload.get("condition_description"),
        "task": payload.get("task"),
        "decision_context": payload.get("decision_context"),
        "financial_interpretation_guide": FINANCIAL_INTERPRETATION_GUIDE,
        "required_output_schema": payload.get("required_output_schema"),
        "allowed_evidence_ids": payload.get("allowed_evidence_ids", []),
        "evidence": payload.get("evidence", []),
        "strict_generation_rules": [
            "Return valid JSON only.",
            "Do not wrap the answer in markdown.",
            "Use only evidence IDs listed in allowed_evidence_ids.",
            "Do not invent evidence IDs.",
            "Copy evidence IDs exactly as shown; do not shorten, round, or reformat them.",
            "Keep the output compact.",
            "Maximum 2 macro_evidence items.",
            "Maximum 2 relational_network_evidence items.",
            "Maximum 2 firm_or_sector_attribution items.",
            "Each claim must be 25 words or fewer.",
            "Do not repeat all evidence; select the most relevant evidence only.",
            "Do not claim causal contagion.",
            "Do not provide investment advice.",
            "Follow the financial_interpretation_guide exactly."
        ],
    }

    return [
        {"role": "system", "content": payload.get("system_instruction", "")},
        {"role": "user", "content": json.dumps(user_object, ensure_ascii=False)},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(EVIDENCE_DIR / "llm_decision_inputs_v5.jsonl"))
    parser.add_argument("--selected", default=str(EVIDENCE_DIR / "llm_eval_selected_decisions.csv"))
    parser.add_argument("--output", default=str(EVIDENCE_DIR / "llm_prompt_conditions.jsonl"))
    parser.add_argument("--conditions", nargs="+", default=["full", "no_network", "no_reliability", "free_form"])
    args = parser.parse_args()

    base_records = read_jsonl(args.input)

    selected = pd.read_csv(args.selected)
    if "decision_id" not in selected.columns:
        raise ValueError(f"{args.selected} must contain a decision_id column.")

    selected_ids = set(selected["decision_id"].tolist())
    base_records = [r for r in base_records if r.get("decision_id") in selected_ids]

    n = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for base in base_records:
            for condition in args.conditions:
                payload = condition_payload(base, condition)
                messages = build_messages(payload)

                out = {
                    "prompt_id": f"{payload['decision_id']}::{condition}",
                    "decision_id": payload["decision_id"],
                    "date": payload.get("date"),
                    "condition": condition,
                    "messages": messages,
                    "payload": payload,
                    "allowed_evidence_ids": payload.get("allowed_evidence_ids", []),
                    "evidence_id_alias_map": payload.get("evidence_id_alias_map", {}),
                    "decision_context": payload.get("decision_context", {}),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                n += 1

    print(f"Saved prompt conditions: {args.output}")
    print(f"n_prompts={n}")

    diagnostic = []
    generated = read_jsonl(args.output)
    for r in generated:
        diagnostic.append({
            "condition": r.get("condition"),
            "n_allowed_evidence_ids": len(r.get("allowed_evidence_ids", [])),
            "has_reliability_context": any(
                k in r.get("decision_context", {})
                for k in RELIABILITY_CONTEXT_KEYS
            ),
            "has_network_context": any(
                k in r.get("decision_context", {})
                for k in NETWORK_CONTEXT_KEYS
            ),
            "n_aliases": len(r.get("evidence_id_alias_map", {})),
        })

    diag_df = pd.DataFrame(diagnostic)
    print(diag_df.groupby("condition").agg({
        "n_allowed_evidence_ids": "mean",
        "has_reliability_context": "sum",
        "has_network_context": "sum",
        "n_aliases": "mean",
    }))


if __name__ == "__main__":
    main()
