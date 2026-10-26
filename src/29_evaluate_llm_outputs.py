"""
29_evaluate_llm_outputs.py

Automatic evaluation for LLM decision-note outputs.

HOTFIX:
- Avoids false positives for required disclaimers such as
  "not investment advice" or "does not constitute investment advice".
- Still flags actual trading/investment instruction language such as buy/sell/short.

Checks:
- JSON validity
- cited evidence ID validity
- invented evidence IDs
- citation correctness
- forbidden causal overclaim phrases
- investment/trading advice phrases
- uncertainty compliance when required
- network role separation
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
from common import ensure_dirs

EVAL_DIR = PROJECT_ROOT / "outputs" / "llm_eval"
ensure_dirs(EVAL_DIR)

EVIDENCE_ID_RE = re.compile(r"\b[A-Z]+(?:-[A-Z0-9_\.]+)*-\d{8}(?:-[A-Za-z0-9_\.]+)*\b")

FORBIDDEN_CAUSAL_PATTERNS = [
    r"\bcaused contagion\b",
    r"\bcausal contagion\b",
    r"\bcaused systemic\b",
    r"\btriggered contagion\b",
    r"\bspread from\b",
    r"\bwill crash\b",
    r"\bwill collapse\b",
]

# Actual advice / trading instruction terms.
# "investment advice" itself is not flagged if it appears inside a disclaimer.
INVESTMENT_ADVICE_PATTERNS = [
    r"\byou should buy\b",
    r"\byou should sell\b",
    r"\brecommend buying\b",
    r"\brecommend selling\b",
    r"\bbuy this\b",
    r"\bsell this\b",
    r"\bshort\b",
    r"\blong position\b",
    r"\bportfolio allocation\b",
]

SAFE_DISCLAIMER_PATTERNS = [
    r"\bnot investment advice\b",
    r"\bdoes not constitute investment advice\b",
    r"\bnot financial advice\b",
    r"\bnot trading advice\b",
    r"\bsystemic-risk decision support, not investment advice\b",
]

UNCERTAINTY_TERMS = [
    "caution", "cautious", "uncertain", "uncertainty",
    "low confidence", "abstain", "abstention",
    "disagreement", "insufficient", "limited reliability",
    "treat cautiously", "should be treated cautiously",
]

ROLE_SEPARATION_TERMS = [
    "relational evidence",
    "not the primary",
    "not primary",
    "network evidence",
    "relational context",
    "attribution",
    "co-movement evidence",
    "contextual explanation",
]


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_json(text):
    if not isinstance(text, str):
        return None, "not_string"

    t = text.strip()

    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()

    try:
        return json.loads(t), ""
    except Exception as e:
        start = t.find("{")
        end = t.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(t[start:end+1]), ""
            except Exception as e2:
                return None, repr(e2)
        return None, repr(e)


def flatten_text(obj):
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return " ".join(flatten_text(v) for v in obj.values())
    if isinstance(obj, list):
        return " ".join(flatten_text(x) for x in obj)
    return str(obj)


def count_patterns(text, patterns):
    text_l = text.lower()
    count = 0
    hits = []
    for pat in patterns:
        if re.search(pat, text_l):
            count += 1
            hits.append(pat)
    return count, ";".join(hits)


def count_investment_advice(text):
    """
    Counts actual advice/trading instruction patterns, while ignoring required
    non-advice disclaimers.
    """
    text_l = text.lower()

    # Remove safe disclaimer phrases before scanning.
    cleaned = text_l
    for pat in SAFE_DISCLAIMER_PATTERNS:
        cleaned = re.sub(pat, "", cleaned)

    count = 0
    hits = []
    for pat in INVESTMENT_ADVICE_PATTERNS:
        if re.search(pat, cleaned):
            count += 1
            hits.append(pat)

    return count, ";".join(hits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", default="")
    args = parser.parse_args()

    in_path = Path(args.input)
    records = read_jsonl(in_path)
    prefix = args.output_prefix or in_path.stem

    rows = []

    for rec in records:
        response_text = rec.get("response_text", "")
        parsed, json_error = extract_json(response_text)
        parsed_text = flatten_text(parsed) if parsed is not None else response_text

        allowed = set(rec.get("allowed_evidence_ids", []))
        cited = set(EVIDENCE_ID_RE.findall(parsed_text))
        valid_cited = cited.intersection(allowed)
        invented = cited - allowed

        total_cited = len(cited)
        citation_correctness = len(valid_cited) / total_cited if total_cited > 0 else None

        causal_count, causal_hits = count_patterns(parsed_text, FORBIDDEN_CAUSAL_PATTERNS)
        advice_count, advice_hits = count_investment_advice(parsed_text)

        ctx = rec.get("decision_context", {})
        condition = rec.get("condition", "")

        requires_uncertainty = False
        if ctx.get("abstention_flag") == 1:
            requires_uncertainty = True
        if ctx.get("disagreement_flag") == 1:
            requires_uncertainty = True
        if str(ctx.get("reliability_adjusted_confidence", "")).lower() == "low":
            requires_uncertainty = True
        if str(ctx.get("reliability_status", "")).lower().startswith("insufficient"):
            requires_uncertainty = True

        uncertainty_present = any(term in parsed_text.lower() for term in UNCERTAINTY_TERMS)
        uncertainty_compliance = 1 if (not requires_uncertainty or uncertainty_present) else 0

        has_network_evidence = any("NETWORK" in eid or "FIRM-" in eid or "SECTOR-" in eid for eid in allowed)
        role_separation_present = any(term in parsed_text.lower() for term in ROLE_SEPARATION_TERMS)
        bad_network_prediction_phrase = bool(re.search(r"network\s+(predicted|predicts|forecast|forecasts)", parsed_text.lower()))

        role_separation_score = None
        if condition == "full" and has_network_evidence:
            role_separation_score = 1 if role_separation_present and not bad_network_prediction_phrase else 0
        elif condition == "no_network":
            role_separation_score = 1 if not bad_network_prediction_phrase else 0

        rows.append({
            "provider_name": rec.get("provider_name"),
            "model": rec.get("model"),
            "prompt_id": rec.get("prompt_id"),
            "decision_id": rec.get("decision_id"),
            "condition": condition,
            "status": rec.get("status"),
            "json_valid": int(parsed is not None),
            "json_error": json_error,
            "n_allowed_evidence_ids": len(allowed),
            "n_cited_evidence_ids": total_cited,
            "n_valid_cited_evidence_ids": len(valid_cited),
            "n_invented_evidence_ids": len(invented),
            "citation_correctness": citation_correctness,
            "invented_evidence_ids": ";".join(sorted(invented)),
            "forbidden_causal_count": causal_count,
            "forbidden_causal_hits": causal_hits,
            "investment_advice_count": advice_count,
            "investment_advice_hits": advice_hits,
            "requires_uncertainty": int(requires_uncertainty),
            "uncertainty_present": int(uncertainty_present),
            "uncertainty_compliance": uncertainty_compliance,
            "role_separation_score": role_separation_score,
            "response_chars": len(response_text or ""),
        })

    df = pd.DataFrame(rows)
    out_results = EVAL_DIR / f"{prefix}_auto_eval_results.csv"
    df.to_csv(out_results, index=False)

    summary_cols = [
        "json_valid",
        "citation_correctness",
        "n_invented_evidence_ids",
        "forbidden_causal_count",
        "investment_advice_count",
        "uncertainty_compliance",
        "role_separation_score",
    ]

    summary = (
        df.groupby(["provider_name", "model", "condition"], dropna=False)[summary_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary["n_outputs"] = df.groupby(["provider_name", "model", "condition"], dropna=False).size().values

    out_summary = EVAL_DIR / f"{prefix}_auto_eval_summary.csv"
    summary.to_csv(out_summary, index=False)

    print(f"Saved: {out_results}")
    print(f"Saved: {out_summary}")
    print(summary)


if __name__ == "__main__":
    main()
