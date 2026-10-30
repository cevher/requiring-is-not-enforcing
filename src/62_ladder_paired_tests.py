"""
62_ladder_paired_tests.py

Paired exact tests for the three-level interface ladder, at both serving
precisions and over all repeat runs.

The same ten decision snapshots are rendered at every level, so the arms are
repeated measurements on the same units and McNemar's exact test is the test
the design supports. mcnemar_exact, mcnemar_floor and holm are imported from
55_paired_tests.py so the arithmetic is the same as the reported tests.

Measure. Role separation is scored by the strict lexical criterion over the
whole note, scoped to notes that received relational evidence. The same measure
is used at every level, including level 1, which has no dedicated field. In the
Q8 arm the field-level and note-level scores coincide exactly, so the choice
does not move any number; both are printed.

This is the LEXICAL criterion. Human coding of a subset is Phase 3; these
contrasts are recomputed on it when it exists.

    python src/62_ladder_paired_tests.py
"""
import importlib.util, json, re, sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
GEN = SRC.parent / "outputs" / "llm_generations"
sys.path.insert(0, str(SRC))

def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, SRC / fn)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

ev = _load("ev47", "47_evaluate_llm_outputs_v3.py")
pt = _load("pt55", "55_paired_tests.py")

def per_item(fn, cond=None):
    """{decision_id: (rs_note, rs_field_or_None)} for status-ok rows."""
    p = GEN / fn
    if not p.exists():
        return None
    out = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        if cond and r.get("condition") != cond:
            continue
        obj, _s, _e = ev.best_schema_object(r.get("response_text") or "")
        if obj is None:
            out[r["decision_id"]] = (0, 0); continue
        low = ev.flatten_text(obj).lower()
        allowed = r.get("allowed_evidence_ids", []) or []
        has_net = any(("NETWORK" in x) or x.startswith("FIRM-") or x.startswith("SECTOR-")
                      for x in allowed)
        bad = bool(ev.BAD_NETWORK_PREDICTION.search(low))
        note = int(has_net and any(re.search(q, low) for q in ev.ROLE_SEPARATION_STRICT) and not bad)
        t = ev.find_role_field(obj)
        fld = None
        if t is not None:
            fl = (t or "").lower()
            fld = int(bool(t) and any(re.search(q, fl) for q in ev.ROLE_SEPARATION_STRICT)
                      and not ev.BAD_NETWORK_PREDICTION.search(fl))
        out[r["decision_id"]] = (note, fld)
    return out

ARMS = {
  "Q8 run1": {
    1: ("ollama_llama_4conditions_q8_clean.jsonl", "full"),
    2: ("ollama_llama_required_role_q8.jsonl", None),
    3: ("ollama_llama_required_role_enforced_q8.jsonl", None)},
  "Q8 run2": {
    1: ("ollama_llama_full_q8_rep2_clean.jsonl", "full"),
    2: ("ollama_llama_required_role_q8_rep2.jsonl", None),
    3: ("ollama_llama_required_role_enforced_q8_rep2.jsonl", None)},
  "Q8 run3": {
    1: ("ollama_llama_full_q8_rep3_clean.jsonl", "full"),
    2: ("ollama_llama_required_role_q8_rep3.jsonl", None),
    3: ("ollama_llama_required_role_enforced_q8_rep3.jsonl", None)},
  "Q4": {
    1: ("ollama_llama_full_rerun_q4_clean.jsonl", "full"),
    2: ("ollama_llama_required_role_q4.jsonl", None),
    3: ("ollama_llama_required_role_enforced_q4.jsonl", None)},
  "secondary (MXFP4)": {
    1: ("gptoss20b_schema.jsonl", "full"),
    2: ("gptoss20b_required_role.jsonl", None),
    3: ("gptoss20b_required_role_enforced.jsonl", None)},
}

LEVELS = {1: "talimat", 2: "zorunlu alan", 3: "zorlanmis"}
PAIRS = [(1, 2), (2, 3), (1, 3)]

print(f"{'kol':20s} {'kontrast':28s} {'A':>6} {'B':>6} {'kayip':>6} {'kazanc':>7} "
      f"{'McNemar p':>11} {'taban':>8}")
print("-" * 100)
rows = []
for arm, lv in ARMS.items():
    data = {k: per_item(*v) for k, v in lv.items()}
    if any(d is None for d in data.values()):
        print(f"{arm:20s} (dosya eksik)"); continue
    for a, b in PAIRS:
        ids = sorted(set(data[a]) & set(data[b]))
        lost = sum(1 for i in ids if data[a][i][0] == 1 and data[b][i][0] == 0)
        gain = sum(1 for i in ids if data[a][i][0] == 0 and data[b][i][0] == 1)
        sa = sum(data[a][i][0] for i in ids)
        sb = sum(data[b][i][0] for i in ids)
        p = pt.mcnemar_exact(lost, gain)
        fl = pt.mcnemar_floor(lost + gain)
        name = f"{LEVELS[a]} -> {LEVELS[b]}"
        print(f"{arm:20s} {name:28s} {sa:>3}/{len(ids):<2} {sb:>3}/{len(ids):<2} "
              f"{lost:>6} {gain:>7} {p:>11.6f} {fl:>8.5f}")
        rows.append((arm, name, sa, sb, len(ids), lost, gain, p, fl))
    print()

print("=" * 100)
print("Birincil aile adayi: Q8 run1 uzerinde 'talimat -> zorunlu alan' ve")
print("'zorunlu alan -> zorlanmis'. Tekrar kosulari ayni yonu dogruluyor.")
fam = [r[7] for r in rows if r[0] == "Q8 run1" and r[1] != "talimat -> zorlanmis"]
if fam:
    for (name, p), h in zip([(r[1], r[7]) for r in rows
                             if r[0] == "Q8 run1" and r[1] != "talimat -> zorlanmis"],
                            pt.holm(fam)):
        print(f"   {name:30s} McNemar p = {p:.6f}   Holm p = {h:.6f}")
print(f"\non cift icin iki yonlu McNemar tabani: {pt.mcnemar_floor(10):.5f}")
