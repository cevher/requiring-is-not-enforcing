"""
60_interface_ladder.py

The three-level interface ladder, at both serving precisions.

Level 1  the obligation is carried by the system instruction only  (condition: full)
Level 2  the obligation is carried by a required output field      (full_required_role, schema)
Level 3  the field is compelled at the decoder                     (full_required_role, enforced)

Level 1 has no dedicated field, so its measure is the strict lexical role
criterion over the whole note, scoped to notes that received relational
evidence. Levels 2 and 3 report both that measure and the criterion applied to
the dedicated field. Scoring is imported from 47_evaluate_llm_outputs_v3.py, so
every number here comes from the instrument that produced the reported ones.

Repeat runs are listed explicitly rather than discovered, so a missing file is
visible as missing. Run-to-run spread is the measure of what a ten-prompt
design can resolve: it was +/-1-2 of 10 in the required-role conditions.

    python src/60_interface_ladder.py
"""

import importlib.util, json, re, sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
GEN = SRC.parent / "outputs" / "llm_generations"
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location("ev47", SRC / "47_evaluate_llm_outputs_v3.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

ROWS = [
    ("1 instruction   Q4 moving",  "ollama_llama_native_10decisions_4conditions_final_clean.jsonl", "full"),
    ("1 instruction   Q4 pinned",  "ollama_llama_full_rerun_q4_clean.jsonl",            "full"),
    ("1 instruction   Q8",         "ollama_llama_4conditions_q8_clean.jsonl",                 "full"),
    ("1 instruction   Q8 rep2",    "ollama_llama_full_q8_rep2_clean.jsonl",                   "full"),
    ("1 instruction   Q8 rep3",    "ollama_llama_full_q8_rep3_clean.jsonl",                   "full"),
    (None, None, None),
    ("2 required      Q4",         "ollama_llama_required_role_q4.jsonl",               None),
    ("2 required      Q8",         "ollama_llama_required_role_q8.jsonl",               None),
    ("2 required      Q8 rep2",    "ollama_llama_required_role_q8_rep2.jsonl",          None),
    ("2 required      Q8 rep3",    "ollama_llama_required_role_q8_rep3.jsonl",          None),
    (None, None, None),
    ("3 enforced      Q4",         "ollama_llama_required_role_enforced_q4.jsonl",      None),
    ("3 enforced      Q8",         "ollama_llama_required_role_enforced_q8.jsonl",      None),
    ("3 enforced      Q8 rep2",    "ollama_llama_required_role_enforced_q8_rep2.jsonl", None),
    ("3 enforced      Q8 rep3",    "ollama_llama_required_role_enforced_q8_rep3.jsonl", None),
    (None, None, None),
    ("-- secondary  1 instruction","gptoss20b_schema.jsonl",                            "full"),
    ("-- secondary  2 required",   "gptoss20b_required_role.jsonl",                     None),
    ("-- secondary  3 enforced",   "gptoss20b_required_role_enforced.jsonl",            None),
]

def score(path, cond):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if cond:
        rows = [r for r in rows if r.get("condition") == cond]
    if not rows:
        return None
    n = len(rows)
    err = sum(1 for r in rows if r.get("status") != "ok")
    parsed = present = rs_field = rs_note = 0
    for r in rows:
        obj, _s, _e = ev.best_schema_object(r.get("response_text") or "")
        if obj is None:
            continue
        parsed += 1
        low = ev.flatten_text(obj).lower()
        allowed = r.get("allowed_evidence_ids", []) or []
        has_net = any(("NETWORK" in x) or x.startswith("FIRM-") or x.startswith("SECTOR-")
                      for x in allowed)
        bad = bool(ev.BAD_NETWORK_PREDICTION.search(low))
        if has_net and any(re.search(p, low) for p in ev.ROLE_SEPARATION_STRICT) and not bad:
            rs_note += 1
        t = ev.find_role_field(obj)
        if t:
            present += 1
            fl = t.lower()
            if any(re.search(p, fl) for p in ev.ROLE_SEPARATION_STRICT) \
               and not ev.BAD_NETWORK_PREDICTION.search(fl):
                rs_field += 1
    return dict(n=n, err=err, parsed=parsed, present=present,
                rs_field=rs_field, rs_note=rs_note, model=rows[0].get("model"))

hdr = (f"{'arayuz katmani':30s} {'model tag':24s} {'err':>4} {'parsed':>7} "
       f"{'alan':>7} {'S_NOT alan':>11} {'S_NOT not':>10}")
print(hdr); print("-" * len(hdr))
for label, fn, cond in ROWS:
    if label is None:
        print(); continue
    p = GEN / fn
    if not p.exists():
        print(f"{label:30s} (dosya yok)"); continue
    s = score(p, cond)
    if s is None:
        print(f"{label:30s} (kosul bulunamadi: {cond})"); continue
    fld = f"{s['present']:>4}/{s['n']:<2}" if s["present"] else ("   -  " if cond else f"{0:>4}/{s['n']:<2}")
    fs  = f"{s['rs_field']:>6}/{s['n']:<2}" if cond is None else "     - "
    print(f"{label:30s} {str(s['model']):24s} {s['err']:>4} {s['parsed']:>4}/{s['n']:<2} "
          f"{fld} {fs} {s['rs_note']:>6}/{s['n']:<2}")
