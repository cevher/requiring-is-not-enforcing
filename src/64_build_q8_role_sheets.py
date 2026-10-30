"""Coding sheets for the three-level interface ladder at Q8_0.

Why this round. Section 4.4 reports the ladder at the eight-bit serving
precision, where the requirement is met at the middle interface level and at
neither end. Those notes have no human coding, and Section 4.3 shows that
scoring this construct with the lexical criterion alone is unsafe. The sheet
therefore covers all three levels in one pass, on the same ten decisions, which
is what the paired tests for both contrasts require.

  * 10 items  Q8-INSTR     requirement in the system instruction
  * 10 items  Q8-RR        requirement as a required output field
  * 10 items  Q8-RR-ENF    required field enforced at the decoder

Blinding. Two cues in the rendered output identify the condition on sight and
both are removed here. The dedicated role field produces a heading that appears
in one condition only and names the construct under coding; it is folded into
the relational-evidence section by the inherited defusal, unchanged. Separately,
the model echoes blocks of its own prompt back as output keys in the
instruction condition and in no other, so any section rendered from a key that
is not part of the declared output schema is dropped. Neither step alters a
word of what the model asserted about its evidence.

Everything else is inherited from 49_build_role_statement_sheets.py: the two
binary questions, the worked example and the workbook writer. Nothing about the
instrument changes between rounds.

Writes
    outputs/llm_eval/role_statement_q8_sheet_rater1.xlsx
    outputs/llm_eval/role_statement_q8_sheet_rater2.xlsx
    outputs/llm_eval/role_statement_q8_key_rater1.csv
    outputs/llm_eval/role_statement_q8_key_rater2.csv

The key files must not be opened before both sheets are complete.
"""
import collections
import importlib.util
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent


def load49():
    spec = importlib.util.spec_from_file_location("sheets49", SRC / "49_build_role_statement_sheets.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SRC))
    spec.loader.exec_module(mod)
    return mod


S = load49()

BLOCKS = [
    ("Q8-INSTR",   "ollama_llama_4conditions_q8_clean.jsonl",        "full"),
    ("Q8-RR",      "ollama_llama_required_role_q8.jsonl",            "full_required_role"),
    ("Q8-RR-ENF",  "ollama_llama_required_role_enforced_q8.jsonl",   "full_required_role"),
]

# Keys the model echoes back from its own prompt. They are not part of the
# declared output schema, they carry no claim about the evidence, and they
# appear in one condition only, so a section rendered from them is a condition
# cue and nothing else. They are removed from the object BEFORE rendering, so
# the removal does not depend on how the renderer labels a heading.
ECHOED_KEYS = {
    "strict_generation_rules",
    "financial_interpretation_guide",
    "required_output_schema",
    "decision_context",
    "task",
    "condition",
    "condition_description",
}

_RENDER = None


def render_note(text):
    global _RENDER
    if _RENDER is None:
        _RENDER = S.load_render_note()
    return _RENDER(text)


def strip_echoed_keys(raw):
    """Delete echoed prompt blocks from the model's JSON, keeping everything else
    byte-identical. Returns the original string when no such key is present."""
    t = (raw or "").strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        return raw
    try:
        obj = json.loads(t[a:b + 1])
    except Exception:
        return raw
    if not isinstance(obj, dict):
        return raw
    drop = [k for k in obj if str(k).strip().lower().replace(" ", "_") in ECHOED_KEYS]
    if not drop:
        return raw
    for k in drop:
        del obj[k]
    return json.dumps(obj, ensure_ascii=False)


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
        txt = S.defuse_blinding_leak(render_note(strip_echoed_keys(rec.get("response_text", ""))))
        out.append({
            "block": block,
            "prompt_id": rec.get("prompt_id"),
            "decision_id": rec.get("decision_id"),
            "condition": cond,
            "model": rec.get("model"),
            "source_file": fname,
            "text": txt,
        })
    return out


def main():
    items = []
    print("Maddeler toplaniyor:\n")
    for block, fname, cond in BLOCKS:
        got = collect_block(block, fname, cond)
        print(f"  {block:<12} {fname:<48} {len(got)} madde")
        if len(got) != 10:
            print(f"    UYARI: 10 bekleniyordu, {len(got)} bulundu.")
        items += got
    print(f"\nToplam {len(items)} madde.\n")

    for rater, seed in [("rater1", 20261030), ("rater2", 20261031)]:
        body, key = S.build(items, seed, prefix="Q8")
        ok = S.write_workbook(S.EVAL / f"role_statement_q8_sheet_{rater}.xlsx", body)
        key.to_csv(S.EVAL / f"role_statement_q8_key_{rater}.csv", index=False)
        if ok:
            print(f"  yazildi: role_statement_q8_sheet_{rater}.xlsx  ({len(body)} madde + 1 ornek satir)")
        print(f"  yazildi: role_statement_q8_key_{rater}.csv")

    # Korleme denetimi: hicbir kosul, metninden taninabilir olmamali.
    print("\nKorleme denetimi:")
    heads = collections.defaultdict(collections.Counter)
    for it in items:
        for b in it["text"].split("\n\n"):
            h = b.lstrip("■ ").split("\n", 1)[0].strip()
            if h:
                heads[h][it["block"]] += 1
    bad = False
    for h, c in sorted(heads.items()):
        blocks_present = [b for b in c if c[b] > 0]
        marker = ""
        if len(blocks_present) < len(BLOCKS):
            marker = "   <-- yalnizca " + ", ".join(blocks_present)
            bad = True
        print(f"  {h[:52]:<54}{dict(c)}{marker}")
    print("\n  SONUC:", "bir veya daha fazla baslik kosula ozgu - incele." if bad
          else "her baslik her kosulda var, yapisal ipucu yok.")


if __name__ == "__main__":
    main()
