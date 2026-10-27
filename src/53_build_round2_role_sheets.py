"""Round-2 coding sheets: the decoder-enforced Llama 3 condition.

Why a second round at all. The decoder-enforced run on the primary model was
added after the first coding round, in response to the objection that decoder
enforcement had only ever been tested on the model that already complied. Its
notes therefore have no human coding, and scoring them with the lexical
criterion is exactly what Section 4.3 shows to be unsafe.

Why 30 items and not 10. Ten items from a single condition are not blind in
practice: a homogeneous set tells the reader what it is. The sheet therefore
carries

  * 10 new items   L-OCT-RR-ENF, decoder-enforced required role, Llama 3
  * 10 repeats     L-OCT-RR, the same ten decisions WITHOUT decoder enforcement
  * 10 repeats     stratified across the four remaining round-1 blocks

The ten repeats of L-OCT-RR matter beyond blinding: the enforced and unenforced
conditions are then coded in one pass, by the same readers, on the same ten
decisions, which is what the paired test for that contrast requires. The other
ten repeats give a test-retest estimate against round 1 at no extra cost.

Everything else is inherited from 49_build_role_statement_sheets.py: the two
binary questions, the worked example, the blinding-leak defusal and the
workbook writer. Nothing about the instrument changes between rounds.

Writes
    outputs/llm_eval/role_statement_r2_sheet_rater1.xlsx
    outputs/llm_eval/role_statement_r2_sheet_rater2.xlsx
    outputs/llm_eval/role_statement_r2_key_rater1.csv
    outputs/llm_eval/role_statement_r2_key_rater2.csv

The key files must not be opened before both sheets are complete.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent


def load49():
    spec = importlib.util.spec_from_file_location("sheets49", SRC / "49_build_role_statement_sheets.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SRC))
    spec.loader.exec_module(mod)
    return mod


S = load49()

NEW_BLOCK = ("L-OCT-RR-ENF",
             "ollama_llama_required_role_enforced_q4.jsonl",
             "full_required_role")

PAIR_BLOCK = "L-OCT-RR"          # all ten, for the within-pass paired contrast
DISTRACTOR_BLOCKS = ["L-APR-FULL", "L-OCT-FULL", "G-FULL", "G-RR", "G-RR-ENF"]
N_DISTRACTOR = 10
SEED_SAMPLE = 20261027


_RENDER = None


def render_note(text):
    """Load 43's renderer once. Importing it executes that script's main(),
    which rewrites the round-1 blank templates, so it must not run per block."""
    global _RENDER
    if _RENDER is None:
        _RENDER = S.load_render_note()
    return _RENDER(text)


def collect_block(block, fname, cond):
    path = S.GEN / fname
    if not path.exists():
        raise SystemExit(f"Uretim dosyasi yok: {path}")
    out = []
    for rec in S.read_jsonl(path):
        if rec.get("status") != "ok" or rec.get("condition") != cond:
            continue
        if not S.has_network(rec):
            continue
        out.append({
            "block": block,
            "prompt_id": rec.get("prompt_id"),
            "decision_id": rec.get("decision_id"),
            "condition": cond,
            "model": rec.get("model"),
            "source_file": fname,
            "round1_coded": block != NEW_BLOCK[0],
            "text": S.defuse_blinding_leak(render_note(rec.get("response_text", ""))),
        })
    return out


def main():
    print("Maddeler toplaniyor:\n")

    new_items = collect_block(*NEW_BLOCK)
    print(f"  {NEW_BLOCK[0]:<14} {NEW_BLOCK[1]:<48} {len(new_items)} yeni madde")

    old = []
    for block, fname, cond in S.BLOCKS:
        old += collect_block(block, fname, cond)

    pair = [x for x in old if x["block"] == PAIR_BLOCK]
    print(f"  {PAIR_BLOCK:<14} {'(ayni on karar, zorlamasiz)':<48} {len(pair)} eslesme maddesi")

    rng = np.random.default_rng(SEED_SAMPLE)
    pool = [x for x in old if x["block"] in DISTRACTOR_BLOCKS]
    per = {b: [x for x in pool if x["block"] == b] for b in DISTRACTOR_BLOCKS}
    picks = []
    b = 0
    while len(picks) < N_DISTRACTOR:
        blk = DISTRACTOR_BLOCKS[b % len(DISTRACTOR_BLOCKS)]
        avail = [x for x in per[blk] if x not in picks]
        if avail:
            picks.append(avail[int(rng.integers(len(avail)))])
        b += 1
        if b > 200:
            break
    print(f"  {'dagitici':<14} {'(dort bloktan tabakali)':<48} {len(picks)} tekrar maddesi")

    items = new_items + pair + picks
    if len(new_items) != 10:
        print(f"\n  UYARI: yeni blokta {len(new_items)} madde var, 10 bekleniyordu.")
    print(f"\nToplam {len(items)} madde.\n")

    for rater, seed in [("rater1", 20261101), ("rater2", 20261102)]:
        body, key = S.build(items, seed, prefix="R2")
        flag = {(it["block"], it["prompt_id"]): it["round1_coded"] for it in items}
        key["round1_coded"] = [flag[(b, p)] for b, p in zip(key["block"], key["prompt_id"])]
        ok = S.write_workbook(S.EVAL / f"role_statement_r2_sheet_{rater}.xlsx", body)
        key.to_csv(S.EVAL / f"role_statement_r2_key_{rater}.csv", index=False)
        if ok:
            print(f"  yazildi: role_statement_r2_sheet_{rater}.xlsx  ({len(body)} madde + 1 ornek satir)")
        print(f"  yazildi: role_statement_r2_key_{rater}.csv")

    print("\nDagilim:")
    print(pd.DataFrame(items).groupby(["block", "model"]).size().to_string())
    print("\nAnahtar dosyalarini iki sayfa da bitmeden acmayin.")


if __name__ == "__main__":
    main()
