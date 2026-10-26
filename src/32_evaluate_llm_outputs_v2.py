"""
32_evaluate_llm_outputs_v2.py

Corrected automatic evaluation for LLM decision-note outputs.

Why v2 exists (all four issues found in 29_evaluate_llm_outputs.py):

1. JSON recovery.
   v1 used t.find("{") ... t.rfind("}"), i.e. the OUTERMOST brace span. When a
   model emits a valid JSON object followed by extra commentary that itself
   contains braces (e.g. gpt-oss harmony channels), the outermost span is not
   parseable and the output is scored json_valid = 0. v2 scans for the FIRST
   well-formed balanced object, and reports three separate quantities:
     - strict_json_only : the whole response is exactly one JSON object
     - json_recoverable : a schema-shaped object can be recovered
     - extra_text_ratio : share of characters outside the recovered object
   The real failure mode is extra-schema text, not unparseable output.

2. Uncertainty compliance.
   v1 set requires_uncertainty from the PROMPT context. In the no_reliability
   and free_form conditions those fields are withheld by design, so the
   requirement was never triggered and compliance was trivially 1.0 in exactly
   the conditions being compared. v2 takes requires_uncertainty from the
   ground-truth decision record, so the requirement is identical across
   conditions.

3. Investment-advice detection.
   v1 pattern r"\bshort\b" fires on ordinary wording such as "short-term".
   v2 uses trading-instruction patterns only.

4. Causal overclaim.
   v1 counted a forbidden phrase even when the model was echoing the
   prohibition ("do not claim causal contagion"). v2 ignores a match that is
   preceded by a negation/prohibition cue inside a short left window, and
   reports echoed matches separately.

Usage:
  python src/32_evaluate_llm_outputs_v2.py --input <generations.jsonl> [--output-prefix NAME]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT, EVIDENCE_DIR
from common import ensure_dirs

EVAL_DIR = PROJECT_ROOT / "outputs" / "llm_eval"
ensure_dirs(EVAL_DIR)

EVIDENCE_ID_RE = re.compile(r"\b[A-Z]+(?:-[A-Z0-9_\.]+)*-\d{8}(?:-[A-Za-z0-9_\.]+)*\b")

HARMONY_RE = re.compile(r"<\|(channel|start|end|message|return|constrain)\|>")

FORBIDDEN_CAUSAL_PATTERNS = [
    r"caused contagion", r"causal contagion", r"caused systemic",
    r"triggered contagion", r"spread from", r"will crash", r"will collapse",
    r"causal transmission", r"directional transmission",
]

NEGATION_CUES = [
    "do not", "does not", "don't", "no ", "not ", "never", "avoid", "avoids",
    "avoiding", "without", "cannot", "can not", "must not", "should not",
    "refrain", "prohibit", "forbidden", "rather than", "instead of",
]

INVESTMENT_ADVICE_PATTERNS = [
    r"you should buy", r"you should sell", r"recommend buying",
    r"recommend selling", r"buy this", r"sell this", r"go short",
    r"short position", r"shorting", r"long position",
    r"portfolio allocation", r"we advise (buying|selling)",
]

SAFE_DISCLAIMER_PATTERNS = [
    r"not investment advice", r"does not constitute investment advice",
    r"not financial advice", r"not trading advice",
]

UNCERTAINTY_TERMS = [
    "caution", "cautious", "uncertain", "uncertainty", "low confidence",
    "abstain", "abstention", "disagreement", "insufficient",
    "limited reliability", "treat cautiously",
]

# Strict role separation: the note must state that network/relational evidence
# is contextual rather than the primary predictor. Merely using the words
# "network evidence" or "attribution" is not enough.
ROLE_SEPARATION_STRICT = [
    r"not (the )?primary (stress )?(predictor|prediction|driver|signal)",
    r"relational (context|attribution|evidence)[^.]{0,80}not",
    r"for (relational )?(attribution|context)[^.]{0,60}not (a )?(causal|predictive)",
    r"context(ual)? (only|rather than)",
    r"rather than (the )?(primary|causal)",
]

ROLE_SEPARATION_LOOSE = [
    "relational evidence", "not the primary", "not primary", "network evidence",
    "relational context", "attribution", "co-movement evidence",
    "contextual explanation",
]

BAD_NETWORK_PREDICTION = re.compile(r"network\s+(predicted|predicts|forecast|forecasts|caused|drives)")


def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def strip_fences(t):
    t = t.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    return t


def first_balanced_object(t):
    """Return (obj, start, end) for the first well-formed JSON object, else (None, -1, -1)."""
    s = t.find("{")
    while s != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(s, len(t)):
            c = t[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[s:i + 1]), s, i + 1
                    except Exception:
                        break
        s = t.find("{", s + 1)
    return None, -1, -1


SCHEMA_KEYS = {
    "risk_summary", "scoring_interpretation", "macro_evidence",
    "relational_network_evidence", "firm_or_sector_attribution",
    "uncertainty_note", "non_advice_disclaimer",
}


def best_schema_object(t):
    """Prefer the first object that actually looks like a decision note."""
    rest = t
    offset = 0
    best = (None, -1, -1)
    while True:
        obj, s, e = first_balanced_object(rest)
        if obj is None:
            break
        if isinstance(obj, dict):
            keys = {str(k).lower().replace(" ", "_") for k in obj.keys()}
            if keys & SCHEMA_KEYS:
                return obj, offset + s, offset + e
            if best[0] is None:
                best = (obj, offset + s, offset + e)
        offset += e
        rest = rest[e:]
    return best


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


def causal_counts(text):
    """Returns (genuine, echoed, hits). A match preceded by a negation cue
    within 60 characters is treated as prompt echo, not an overclaim."""
    t = text.lower()
    genuine, echoed, hits = 0, 0, []
    for pat in FORBIDDEN_CAUSAL_PATTERNS:
        for m in re.finditer(pat, t):
            left = t[max(0, m.start() - 60):m.start()]
            if any(cue in left for cue in NEGATION_CUES):
                echoed += 1
            else:
                genuine += 1
                hits.append(pat)
    return genuine, echoed, ";".join(hits)


def advice_count(text):
    t = text.lower()
    for pat in SAFE_DISCLAIMER_PATTERNS:
        t = re.sub(pat, " ", t)
    n, hits = 0, []
    for pat in INVESTMENT_ADVICE_PATTERNS:
        if re.search(pat, t):
            n += 1
            hits.append(pat)
    return n, ";".join(hits)


def load_truth():
    p = EVIDENCE_DIR / "decision_records_v5.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    truth = {}
    for _, r in df.iterrows():
        need = (
            int(r.get("abstention_flag", 0) or 0) == 1
            or int(r.get("disagreement_flag", 0) or 0) == 1
            or str(r.get("reliability_adjusted_confidence", "")).lower() == "low"
            or str(r.get("reliability_status", "")).lower().startswith("insufficient")
        )
        truth[r["decision_id"]] = bool(need)
    return truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-prefix", default="")
    args = ap.parse_args()

    in_path = Path(args.input)
    records = read_jsonl(in_path)
    prefix = args.output_prefix or (in_path.stem + "_v2")
    truth = load_truth()

    rows = []
    for rec in records:
        raw = rec.get("response_text") or ""
        t = strip_fences(raw)

        strict = None
        try:
            json.loads(t)
            strict = 1
        except Exception:
            strict = 0

        obj, s, e = best_schema_object(t)
        recoverable = int(obj is not None)
        extra_chars = (len(t) - (e - s)) if obj is not None else len(t)
        extra_ratio = (extra_chars / len(t)) if len(t) > 0 else None

        parsed_text = flatten_text(obj) if obj is not None else t

        allowed = set(rec.get("allowed_evidence_ids", []))
        cited = set(EVIDENCE_ID_RE.findall(parsed_text))
        valid_cited = cited & allowed
        invented = cited - allowed
        cc = len(valid_cited) / len(cited) if cited else None

        g_causal, e_causal, causal_hits = causal_counts(parsed_text)
        adv, adv_hits = advice_count(parsed_text)

        did = rec.get("decision_id")
        requires_unc = bool(truth.get(did, False))
        unc_present = any(term in parsed_text.lower() for term in UNCERTAINTY_TERMS)
        unc_comp = int((not requires_unc) or unc_present)

        condition = rec.get("condition", "")
        has_net = any(("NETWORK" in x) or x.startswith("FIRM-") or x.startswith("SECTOR-") for x in allowed)
        loose = any(term in parsed_text.lower() for term in ROLE_SEPARATION_LOOSE)
        strict_rs = any(re.search(p, parsed_text.lower()) for p in ROLE_SEPARATION_STRICT)
        bad_net = bool(BAD_NETWORK_PREDICTION.search(parsed_text.lower()))

        rs_loose = rs_strict = None
        if condition == "full" and has_net:
            rs_loose = int(loose and not bad_net)
            rs_strict = int(strict_rs and not bad_net)
        elif condition == "no_network":
            rs_loose = int(not bad_net)
            rs_strict = int(not bad_net)

        rows.append({
            "provider_name": rec.get("provider_name"),
            "model": rec.get("model"),
            "prompt_id": rec.get("prompt_id"),
            "decision_id": did,
            "condition": condition,
            "status": rec.get("status"),
            "strict_json_only": strict,
            "json_recoverable": recoverable,
            "extra_text_ratio": extra_ratio,
            "harmony_leak": int(bool(HARMONY_RE.search(raw))),
            "response_chars": len(raw),
            "n_allowed_evidence_ids": len(allowed),
            "n_cited_evidence_ids": len(cited),
            "n_invented_evidence_ids": len(invented),
            "citation_correctness": cc,
            "invented_evidence_ids": ";".join(sorted(invented)),
            "causal_overclaim_genuine": g_causal,
            "causal_echoed_prohibition": e_causal,
            "causal_hits": causal_hits,
            "investment_advice_count": adv,
            "investment_advice_hits": adv_hits,
            "requires_uncertainty_truth": int(requires_unc),
            "uncertainty_present": int(unc_present),
            "uncertainty_compliance": unc_comp,
            "role_separation_loose": rs_loose,
            "role_separation_strict": rs_strict,
        })

    df = pd.DataFrame(rows)
    df.to_csv(EVAL_DIR / f"{prefix}_auto_eval_results.csv", index=False)

    cols = [
        "strict_json_only", "json_recoverable", "extra_text_ratio", "harmony_leak",
        "citation_correctness", "n_invented_evidence_ids",
        "causal_overclaim_genuine", "causal_echoed_prohibition",
        "investment_advice_count", "requires_uncertainty_truth",
        "uncertainty_compliance", "role_separation_loose", "role_separation_strict",
    ]
    summ = df.groupby(["provider_name", "model", "condition"], dropna=False)[cols].mean().round(4)
    summ["n_outputs"] = df.groupby(["provider_name", "model", "condition"], dropna=False).size()
    summ.to_csv(EVAL_DIR / f"{prefix}_auto_eval_summary.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()
