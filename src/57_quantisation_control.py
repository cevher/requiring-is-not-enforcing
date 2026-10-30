"""
57_quantisation_control.py

Quantisation control for the cross-model contrast of Section 4.4.

WHY THIS EXISTS
---------------
The primary model is served at Q4_0 and the secondary at MXFP4. The reported
cross-model difference -- a field marked required omitted from 4 of 10 outputs
by the primary model and from none by the secondary -- is therefore confounded
with quantisation. Q4_0 is an older symmetric 4-bit format, and omitting a
required key is the failure mode low-bit quantisation is known to produce.

This script re-runs the scoring of the required-role conditions over an
additional serving precision (Q8_0) of the SAME model and reports the two side
by side. The prompts, the decoding settings and the scoring code are unchanged:
the provider entries ollama_llama3_q8_* are clones of their q4 counterparts with
the model tag as the single manipulation.

Scoring reuses 47_evaluate_llm_outputs_v3.py, so the numbers are produced by the
same instrument that produced the reported ones. The role-separation figures
here are the LEXICAL criterion, not human coding.

Usage
-----
    python src/57_quantisation_control.py
"""

import importlib.util, json, re, sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
GEN = ROOT / "outputs" / "llm_generations"

spec = importlib.util.spec_from_file_location("ev47", SRC / "47_evaluate_llm_outputs_v3.py")
ev = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SRC))
spec.loader.exec_module(ev)

BASE = [
    ("Q4_0  schema only",     "ollama_llama_required_role_q4"),
    ("Q8_0  schema only",     "ollama_llama_required_role_q8"),
    ("Q4_0  decoder-enforced","ollama_llama_required_role_enforced_q4"),
    ("Q8_0  decoder-enforced","ollama_llama_required_role_enforced_q8"),
    ("-- secondary, schema",  "gptoss20b_required_role"),
    ("-- secondary, enforced","gptoss20b_required_role_enforced"),
]

# Repeat runs are discovered rather than listed: <stem>_rep2.jsonl, _rep3.jsonl, ...
# Run-to-run spread is the measure of what a ten-prompt design can resolve.
FILES = []
for label, stem in BASE:
    FILES.append((label, stem + ".jsonl"))
    for extra in sorted(GEN.glob(stem + "_rep*.jsonl")):
        FILES.append((label + "  [" + extra.stem.split("_")[-1] + "]", extra.name))

def score(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    n = len(rows)
    parsed = present = rs_field = rs_uniform = 0
    err = sum(1 for r in rows if r.get("status") != "ok")
    for r in rows:
        obj, _s, _e = ev.best_schema_object(r.get("response_text") or "")
        if obj is None:
            continue
        parsed += 1
        low = ev.flatten_text(obj).lower()
        allowed = r.get("allowed_evidence_ids", []) or []
        has_net = any(("NETWORK" in x) or x.startswith("FIRM-") or x.startswith("SECTOR-")
                      for x in allowed)
        bad_net = bool(ev.BAD_NETWORK_PREDICTION.search(low))
        strict_rs = any(re.search(pt, low) for pt in ev.ROLE_SEPARATION_STRICT)
        if has_net and strict_rs and not bad_net:
            rs_uniform += 1
        t = ev.find_role_field(obj)
        if t:
            present += 1
            fl = t.lower()
            if any(re.search(pt, fl) for pt in ev.ROLE_SEPARATION_STRICT) \
               and not ev.BAD_NETWORK_PREDICTION.search(fl):
                rs_field += 1
    return dict(n=n, parsed=parsed, present=present, err=err,
                rs_field=rs_field, rs_uniform=rs_uniform, model=rows[0].get("model"))

hdr = f"{'condition':30s} {'model tag':24s} {'err':>4} {'parsed':>8} {'role field':>11} {'S_NOT field':>12} {'S_NOT note':>11}"
print(hdr); print("-" * len(hdr))
for label, fn in FILES:
    p = GEN / fn
    if not p.exists():
        print(f"{label:30s} (dosya yok)")
        continue
    s = score(p)
    print(f"{label:30s} {str(s['model']):24s} {s['err']:>4} {s['parsed']:>5}/{s['n']:<2} "
          f"{s['present']:>8}/{s['n']:<2} {s['rs_field']:>9}/{s['n']:<2} {s['rs_uniform']:>8}/{s['n']:<2}")
