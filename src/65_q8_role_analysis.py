"""Q8 ladder coding: agreement, consensus, and the criterion check.

Reads the two filled Q8 sheets and their keys and reports

  1. inter-rater agreement over the 30 items, per question
  2. S_CTX and S_NOT by interface level, on reader consensus
  3. agreement between the consensus and the strict lexical criterion
  4. paired exact tests for the two ladder contrasts, on consensus codings

Merging is on (block, item_id): the same decision appears once per block, so a
merge on decision_id alone would multiply the rows.

Writes outputs/llm_eval/role_statement_q8_items.csv and ..._q8_summary.csv
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EVAL = ROOT / "outputs" / "llm_eval"
GEN = ROOT / "outputs" / "llm_generations"
sys.path.insert(0, str(SRC))

def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, SRC / fn)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

ev = _load("ev47", "47_evaluate_llm_outputs_v3.py")
pt = _load("pt55", "55_paired_tests.py")

LEVELS = {"Q8-INSTR": "instruction", "Q8-RR": "required field", "Q8-RR-ENF": "decoder-enforced"}
ORDER = ["Q8-INSTR", "Q8-RR", "Q8-RR-ENF"]


def yn(v):
    s = str(v).strip().upper()
    if s in ("E", "EVET", "Y", "YES", "1", "TRUE"): return 1
    if s in ("H", "HAYIR", "N", "NO", "0", "FALSE"): return 0
    return None


def cohen_kappa(x, y):
    n = len(x)
    if n == 0: return float("nan")
    po = sum(1 for a, b in zip(x, y) if a == b) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def load_rater(tag):
    d = pd.read_excel(EVAL / f"role_statement_q8_sheet_{tag}.xlsx", sheet_name="KODLAMA")
    d = d[d["Sıra"] != "ÖRNEK"].copy()
    cols = [c for c in d.columns if c.startswith("S_CTX")] + [c for c in d.columns if c.startswith("S_NOT")]
    d = d.rename(columns={cols[0]: "S_CTX", cols[1]: "S_NOT"})
    d["item_id"] = d["Sıra"].astype(str)
    for q in ("S_CTX", "S_NOT"):
        d[q] = d[q].map(yn)
    k = pd.read_csv(EVAL / f"role_statement_q8_key_{tag}.csv")
    m = d[["item_id", "S_CTX", "S_NOT"]].merge(k, on="item_id", validate="one_to_one")
    return m[["block", "decision_id", "S_CTX", "S_NOT"]]


def lexical(block_file, cond):
    out = {}
    for line in open(GEN / block_file, encoding="utf-8"):
        r = json.loads(line)
        if r.get("status") != "ok" or r.get("condition") != cond: continue
        obj, _s, _e = ev.best_schema_object(r.get("response_text") or "")
        low = ev.flatten_text(obj).lower() if obj else ""
        bad = bool(ev.BAD_NETWORK_PREDICTION.search(low))
        out[r["decision_id"]] = int(any(re.search(p, low) for p in ev.ROLE_SEPARATION_STRICT) and not bad)
    return out


FILES = {"Q8-INSTR": ("ollama_llama_4conditions_q8_clean.jsonl", "full"),
         "Q8-RR": ("ollama_llama_required_role_q8.jsonl", "full_required_role"),
         "Q8-RR-ENF": ("ollama_llama_required_role_enforced_q8.jsonl", "full_required_role")}


def main():
    a = load_rater("rater1").rename(columns={"S_CTX": "ctx_a", "S_NOT": "not_a"})
    b = load_rater("rater2").rename(columns={"S_CTX": "ctx_b", "S_NOT": "not_b"})
    d = a.merge(b, on=["block", "decision_id"], validate="one_to_one")
    assert len(d) == 30, len(d)

    print("1. OKUYUCULAR ARASI ANLASMA (30 madde)\n")
    for q, ca, cb in (("S_CTX", "ctx_a", "ctx_b"), ("S_NOT", "not_a", "not_b")):
        x, y = list(d[ca]), list(d[cb])
        ag = sum(1 for p, r in zip(x, y) if p == r)
        print(f"  {q}: {ag}/30 ayni  (Cohen kappa = {cohen_kappa(x, y):.3f})")

    for q, ca, cb in (("S_CTX", "ctx_a", "ctx_b"), ("S_NOT", "not_a", "not_b")):
        d[q] = [p if p == r else None for p, r in zip(d[ca], d[cb])]

    print("\n2. ARAYUZ KATMANINA GORE UZLASI KODLAMASI\n")
    print(f"  {'katman':<22}{'S_CTX':>12}{'S_NOT':>12}{'sozluksel':>12}")
    lex = {bk: lexical(*FILES[bk]) for bk in ORDER}
    rows = []
    for bk in ORDER:
        g = d[d.block == bk]
        s = {}
        for q in ("S_CTX", "S_NOT"):
            ok = g[g[q].notna()]
            s[q] = (int(ok[q].sum()), len(ok))
        lx = sum(lex[bk].get(x, 0) for x in g.decision_id)
        print(f"  {LEVELS[bk]:<22}{s['S_CTX'][0]:>5}/{s['S_CTX'][1]:<6}{s['S_NOT'][0]:>5}/{s['S_NOT'][1]:<6}{lx:>8}/{len(g):<4}")
        rows.append({"block": bk, "level": LEVELS[bk],
                     "S_CTX_k": s["S_CTX"][0], "S_CTX_n": s["S_CTX"][1],
                     "S_NOT_k": s["S_NOT"][0], "S_NOT_n": s["S_NOT"][1],
                     "lexical_k": lx, "n": len(g)})

    print("\n3. UZLASI ILE SOZLUKSEL KRITER ARASINDAKI ANLASMA (S_NOT)\n")
    cons, lexv = [], []
    for _, r in d.iterrows():
        if pd.isna(r["S_NOT"]): continue
        cons.append(int(r["S_NOT"])); lexv.append(lex[r.block].get(r.decision_id, 0))
    ag = sum(1 for p, q in zip(cons, lexv) if p == q)
    print(f"  {ag}/{len(cons)} ayni  (Cohen kappa = {cohen_kappa(cons, lexv):.3f})")
    mism = [(r.block, r.decision_id, int(r['S_NOT']), lex[r.block].get(r.decision_id, 0))
            for _, r in d.iterrows() if pd.notna(r['S_NOT'])
            and int(r['S_NOT']) != lex[r.block].get(r.decision_id, 0)]
    for m in mism: print(f"    uyusmazlik: {m[0]:<12} {m[1]:<34} uzlasi={m[2]} sozluksel={m[3]}")

    print("\n4. ESLI TESTLER (uzlasi kodlamasi, S_NOT)\n")
    piv = d.pivot(index="decision_id", columns="block", values="S_NOT")
    for x, y in (("Q8-INSTR", "Q8-RR"), ("Q8-RR", "Q8-RR-ENF"), ("Q8-INSTR", "Q8-RR-ENF")):
        sub = piv[[x, y]].dropna()
        lost = int(((sub[x] == 1) & (sub[y] == 0)).sum())
        gain = int(((sub[x] == 0) & (sub[y] == 1)).sum())
        p = pt.mcnemar_exact(lost, gain)
        print(f"  {LEVELS[x]:<18} -> {LEVELS[y]:<18} {int(sub[x].sum())}/{len(sub)} -> {int(sub[y].sum())}/{len(sub)}"
              f"  kayip={lost} kazanc={gain}  McNemar p = {p:.6f}")

    d.to_csv(EVAL / "role_statement_q8_items.csv", index=False)
    pd.DataFrame(rows).to_csv(EVAL / "role_statement_q8_summary.csv", index=False)
    print(f"\n  yazildi: {EVAL/'role_statement_q8_items.csv'}")
    print(f"  yazildi: {EVAL/'role_statement_q8_summary.csv'}")


if __name__ == "__main__":
    main()
