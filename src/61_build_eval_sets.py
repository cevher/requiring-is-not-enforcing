"""
61_build_eval_sets.py

Assembles one evaluation input per serving precision so that
47_evaluate_llm_outputs_v3.py can be run once over a complete condition set.

Q8 set: all five conditions at llama3:8b-instruct-q8_0 (first run of each;
run-to-run spread is reported separately by 60_interface_ladder.py).

Q4 set: the counterpart at Q4_0, every condition generated under the pinned tag
llama3:8b-instruct-q4_0, so the two arms differ only in serving precision.

    python src/61_build_eval_sets.py
"""
import json
from pathlib import Path

GEN = Path(__file__).resolve().parent.parent / "outputs" / "llm_generations"

SETS = {
    "eval_set_q8.jsonl": [
        ("ollama_llama_4conditions_q8_clean.jsonl", None),
        ("ollama_llama_required_role_q8.jsonl", None),
        ("ollama_llama_required_role_enforced_q8.jsonl", None),
    ],
    "eval_set_secondary.jsonl": [
        ("gptoss20b_schema.jsonl", None),
        ("gptoss20b_required_role.jsonl", None),
        ("gptoss20b_required_role_enforced.jsonl", None),
    ],
    "eval_set_q4.jsonl": [
        ("ollama_llama_full_rerun_q4_clean.jsonl", None),
        ("ollama_llama_no_network_q4pin.jsonl", None),
        ("ollama_llama_no_reliability_q4pin.jsonl", None),
        ("ollama_llama_free_form_q4pin.jsonl", None),
        ("ollama_llama_required_role_q4.jsonl", None),
        ("ollama_llama_required_role_enforced_q4.jsonl", None),
    ],
}

for out, parts in SETS.items():
    rows = []
    for fn, keep in parts:
        p = GEN / fn
        if not p.exists():
            print(f"  ATLANDI (yok): {fn}"); continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if keep and r.get("condition") not in keep:
                continue
            rows.append(r)
    with open(GEN / out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    conds = {}
    for r in rows:
        conds[r["condition"]] = conds.get(r["condition"], 0) + 1
    bad = sum(1 for r in rows if r.get("status") != "ok")
    print(f"{out}: n={len(rows)} hata={bad} kosullar={conds}")
