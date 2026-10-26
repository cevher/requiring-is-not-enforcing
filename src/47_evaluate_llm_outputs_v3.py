"""
47_evaluate_llm_outputs_v3.py

Third revision of the automatic audit. v2 fixed four measurement defects
(outermost-span parsing, prompt-sourced uncertainty requirement, the \\bshort\\b
advice pattern, and prohibition echo). This version fixes a fifth, found in v2
itself, and adds the dedicated-field check needed by the `full_required_role`
condition.

-------------------------------------------------------------------------------
THE DEFECT IN v2
-------------------------------------------------------------------------------
v2 computed strict role separation as:

    if condition == "full" and has_net:
        rs_strict = int(strict_rs and not bad_net)     # strict regex
    elif condition == "no_network":
        rs_strict = int(not bad_net)                   # absence of one bad phrase

The two branches report DIFFERENT QUANTITIES in the same column:

  - `full` is scored on whether the note contains an explicit role statement.
  - `no_network` is scored on whether the note avoids the single phrase
    "network predicts/predicted/forecasts/caused/drives".

The second criterion is satisfiable by any note that does not mention networks
at all, which is every note in that condition by construction. It cannot fail.
Applying the strict regex uniformly to the archived generations gives:

                    full      no_network
    Llama 3          0/10         0/10
    GPT-OSS-20B      4/10         0/10

So the published 0.00-vs-1.00 contrast is a property of the branch structure,
not of the generated text. This is the same failure mode as the vacuous metrics
documented in Section 3.7.1 of the manuscript, in the audit code written to
correct them.

A second consequence of the same branch structure: `no_reliability` carries
network evidence (verified: 10/10 of its prompts contain NETWORK/FIRM/SECTOR
identifiers) but is scored None, because the test is on the condition NAME
rather than on whether network evidence is present. That is the source of the
undefined entry for that condition in the results tables.

-------------------------------------------------------------------------------
WHAT THIS VERSION REPORTS
-------------------------------------------------------------------------------
Three codings, side by side, in the naive-vs-corrected style the manuscript
already uses:

  rs_strict_published : v2's logic, reproduced exactly, for the contrast.
  rs_strict_uniform   : the strict criterion applied wherever network evidence
                        is present, by `has_net` rather than by condition name;
                        None (not applicable) where no network evidence exists,
                        because a note that makes no relational claim has no
                        role to mark.
  rs_strict_field     : for `full_required_role` only -- the strict criterion
                        applied to the dedicated output field alone.

The regex itself is UNCHANGED from v2. Widening it after seeing the outputs
would make the comparison untrustworthy; the point here is the scope of
application, not the pattern.

Usage
-----
    python src/47_evaluate_llm_outputs_v3.py --input <generations.jsonl> \
        [--output-prefix NAME]
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

# UNCHANGED from v2. Do not tune.
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

BAD_NETWORK_PREDICTION = re.compile(
    r"network\s+(predicted|predicts|forecast|forecasts|caused|drives)"
)

ROLE_FIELD_KEY = "relational_evidence_role_statement"

SCHEMA_KEYS = {
    "risk_summary", "scoring_interpretation", "macro_evidence",
    "relational_network_evidence", "firm_or_sector_attribution",
    "uncertainty_note", "non_advice_disclaimer",
}


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


def best_schema_object(t):
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


def find_role_field(obj):
    """Return the text of the dedicated role field, or None if absent/empty."""
    if not isinstance(obj, dict):
        return None
    for k, v in obj.items():
        if str(k).lower().replace(" ", "_") == ROLE_FIELD_KEY:
            txt = flatten_text(v).strip()
            if not txt or txt.lower() in {"n/a", "na", "none", "null"}:
                return ""
            return txt
    return None


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
    prefix = args.output_prefix or (in_path.stem + "_v3")
    truth = load_truth()

    rows = []
    for rec in records:
        raw = rec.get("response_text") or ""
        t = strip_fences(raw)

        try:
            json.loads(t)
            strict_json = 1
        except Exception:
            strict_json = 0

        obj, s, e = best_schema_object(t)
        recoverable = int(obj is not None)
        extra_chars = (len(t) - (e - s)) if obj is not None else len(t)
        extra_ratio = (extra_chars / len(t)) if len(t) > 0 else None

        parsed_text = flatten_text(obj) if obj is not None else t
        low = parsed_text.lower()

        allowed = set(rec.get("allowed_evidence_ids", []))
        cited = set(EVIDENCE_ID_RE.findall(parsed_text))
        valid_cited = cited & allowed
        invented = cited - allowed
        cc = len(valid_cited) / len(cited) if cited else None

        g_causal, e_causal, causal_hits = causal_counts(parsed_text)
        adv, adv_hits = advice_count(parsed_text)

        did = rec.get("decision_id")
        requires_unc = bool(truth.get(did, False))
        unc_present = any(term in low for term in UNCERTAINTY_TERMS)
        unc_comp = int((not requires_unc) or unc_present)

        condition = rec.get("condition", "")
        has_net = any(
            ("NETWORK" in x) or x.startswith("FIRM-") or x.startswith("SECTOR-")
            for x in allowed
        )

        loose = any(term in low for term in ROLE_SEPARATION_LOOSE)
        strict_rs = any(re.search(p, low) for p in ROLE_SEPARATION_STRICT)
        bad_net = bool(BAD_NETWORK_PREDICTION.search(low))

        # ---- coding 1: v2's logic, reproduced exactly ----------------------
        rs_published = None
        rs_loose_published = None
        if condition == "full" and has_net:
            rs_loose_published = int(loose and not bad_net)
            rs_published = int(strict_rs and not bad_net)
        elif condition == "no_network":
            rs_loose_published = int(not bad_net)
            rs_published = int(not bad_net)

        # ---- coding 2: uniform, scoped by evidence rather than by name -----
        # Not applicable where no relational claim can be made.
        rs_uniform = int(strict_rs and not bad_net) if has_net else None
        rs_loose_uniform = int(loose and not bad_net) if has_net else None

        # ---- coding 3: the dedicated field, for full_required_role ---------
        role_field_text = find_role_field(obj)
        role_field_present = None if role_field_text is None else int(bool(role_field_text))
        rs_field = None
        if role_field_text:
            fl = role_field_text.lower()
            rs_field = int(
                any(re.search(p, fl) for p in ROLE_SEPARATION_STRICT)
                and not BAD_NETWORK_PREDICTION.search(fl)
            )

        rows.append({
            "provider_name": rec.get("provider_name"),
            "model": rec.get("model"),
            "prompt_id": rec.get("prompt_id"),
            "decision_id": did,
            "condition": condition,
            "status": rec.get("status"),
            "strict_json_only": strict_json,
            "json_recoverable": recoverable,
            "extra_text_ratio": extra_ratio,
            "harmony_leak": int(bool(HARMONY_RE.search(raw))),
            "response_chars": len(raw),
            "has_network_evidence": int(has_net),
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
            "role_separation_loose_published": rs_loose_published,
            "role_separation_strict_published": rs_published,
            "role_separation_loose_uniform": rs_loose_uniform,
            "role_separation_strict_uniform": rs_uniform,
            "role_field_present": role_field_present,
            "role_separation_strict_field": rs_field,
            "role_field_text": role_field_text,
        })

    df = pd.DataFrame(rows)
    out_results = EVAL_DIR / f"{prefix}_auto_eval_results.csv"
    df.to_csv(out_results, index=False)

    cols = [
        "strict_json_only", "json_recoverable", "extra_text_ratio", "harmony_leak",
        "has_network_evidence", "citation_correctness", "n_invented_evidence_ids",
        "causal_overclaim_genuine", "causal_echoed_prohibition",
        "investment_advice_count", "requires_uncertainty_truth",
        "uncertainty_compliance",
        "role_separation_strict_published",
        "role_separation_strict_uniform",
        "role_field_present",
        "role_separation_strict_field",
    ]
    grouped = df.groupby(["provider_name", "model", "condition"], dropna=False)
    summ = grouped[cols].mean().round(4)
    summ["n_outputs"] = grouped.size()
    out_summary = EVAL_DIR / f"{prefix}_auto_eval_summary.csv"
    summ.to_csv(out_summary)

    print(summ.to_string())
    print(f"\nSaved: {out_results}")
    print(f"Saved: {out_summary}")

    # ---- the contrast the manuscript needs ---------------------------------
    print("\n" + "=" * 78)
    print("STRICT ROLE SEPARATION: published coding vs uniform coding")
    print("=" * 78)
    for (prov, model, cond), g in grouped:
        pub = g["role_separation_strict_published"]
        uni = g["role_separation_strict_uniform"]
        pub_s = f"{pub.sum():.0f}/{pub.notna().sum()}" if pub.notna().any() else "n/a"
        uni_s = f"{uni.sum():.0f}/{uni.notna().sum()}" if uni.notna().any() else "n/a (no network evidence)"
        print(f"  {str(model)[:28]:<30}{cond:<22} published={pub_s:<10} uniform={uni_s}")


if __name__ == "__main__":
    main()
