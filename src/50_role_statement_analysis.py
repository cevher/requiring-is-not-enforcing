# -*- coding: utf-8 -*-
"""
50_role_statement_analysis.py

Analyses the blind role-statement coding and validates the regex against it.

The human coding is the PRIMARY measurement of role separation. The regex in
47_evaluate_llm_outputs_v3.py is a secondary instrument whose agreement with the
human coding is reported, in the same way 45_instrument_validation.py treats the
citation and causal-assertion checks.

This ordering is not a convenience. The strict regex was shown to split
semantically identical sentences ("not primary risk prediction" vs "not the
primary driver"), so a condition comparison computed from it is not trustworthy.
Widening the pattern after seeing the outputs would be fitting the instrument to
the data; letting humans define the construct and then measuring how well the
pattern tracks them does not have that problem.

Two coded questions:
  S_CTX  the note calls relational evidence context or attribution
  S_NOT  the note states explicitly that it is NOT the primary predictor

S_NOT is the construct the system instruction demands and the one the strict
regex claims to measure. S_CTX separates partial compliance - asserting the
positive half without the contrast - which a single criterion hides.

Inputs
------
outputs/llm_eval/role_statement_sheet_rater{1,2}.xlsx   (filled in)
outputs/llm_eval/role_statement_key_rater{1,2}.csv
outputs/llm_eval/*_auto_eval_results.csv                (for the regex coding)

Usage
-----
    python src/50_role_statement_analysis.py
"""

import glob
import math
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
from common import ensure_dirs

EVAL = PROJECT_ROOT / "outputs" / "llm_eval"
ensure_dirs(EVAL)

BLOCK_LABEL = {
    "L-APR-FULL": "Llama 3 full (April, llama3:latest)",
    "L-OCT-FULL": "Llama 3 full (October, pinned q4_0)",
    "L-OCT-RR": "Llama 3 full_required_role",
    "G-FULL": "GPT-OSS full",
    "G-RR": "GPT-OSS full_required_role",
    "G-RR-ENF": "GPT-OSS full_required_role, decoder-enforced",
}
BLOCK_ORDER = ["L-APR-FULL", "L-OCT-FULL", "L-OCT-RR", "G-FULL", "G-RR", "G-RR-ENF"]

AUDIT_PREFIX = {
    "L-APR-FULL": "llama3_main_v3",
    "L-OCT-FULL": "llama3_full_q4",
    "L-OCT-RR": "llama3_rr_q4",
    "G-FULL": "gptoss20b_v3",
    "G-RR": "gptoss_rr",
    "G-RR-ENF": "gptoss_rr_enforced",
}


def yn(v):
    s = str(v).strip().upper()
    if s in {"E", "EVET", "Y", "YES", "1", "TRUE"}:
        return 1
    if s in {"H", "HAYIR", "N", "NO", "0", "FALSE"}:
        return 0
    return None


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def cohen_kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in pairs if x == y) / n
    pa1 = sum(x for x, _ in pairs) / n
    pb1 = sum(y for _, y in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def load_reader(rater):
    sheet = EVAL / f"role_statement_sheet_{rater}.xlsx"
    key = EVAL / f"role_statement_key_{rater}.csv"
    if not sheet.exists():
        raise FileNotFoundError(f"Eksik: {sheet}")
    df = pd.read_excel(sheet, sheet_name="KODLAMA")
    df = df[df["Sıra"].astype(str) != "ÖRNEK"].copy()

    ctx_col = next(c for c in df.columns if str(c).startswith("S_CTX"))
    not_col = next(c for c in df.columns if str(c).startswith("S_NOT"))

    out = pd.DataFrame({
        "item_id": df["Sıra"],
        "S_CTX": df[ctx_col].map(yn),
        "S_NOT": df[not_col].map(yn),
    })
    return out.merge(pd.read_csv(key), on="item_id", how="left")


def load_regex_coding():
    rows = []
    for block, prefix in AUDIT_PREFIX.items():
        hits = glob.glob(str(EVAL / f"{prefix}_auto_eval_results.csv"))
        if not hits:
            continue
        d = pd.read_csv(hits[0])
        d = d[d["role_separation_strict_uniform"].notna()]
        for _, r in d.iterrows():
            rows.append({"block": block, "prompt_id": r["prompt_id"],
                         "regex_S_NOT": int(r["role_separation_strict_uniform"])})
    return pd.DataFrame(rows)


def main():
    r1 = load_reader("rater1").rename(columns={"S_CTX": "S_CTX_r1", "S_NOT": "S_NOT_r1"})
    r2 = load_reader("rater2").rename(columns={"S_CTX": "S_CTX_r2", "S_NOT": "S_NOT_r2"})

    # prompt_id is NOT unique across blocks: the same prompt appears in the
    # April and October runs. Merging on it alone produces a cartesian product,
    # which silently inflates every n. Join on the (block, prompt_id) pair.
    items = r1.drop(columns=["item_id"]).merge(
        r2[["block", "prompt_id", "S_CTX_r2", "S_NOT_r2"]],
        on=["block", "prompt_id"], how="inner")
    assert len(items) == len(r1), (
        f"birlesme hatali: {len(items)} satir, beklenen {len(r1)}")

    missing = {"rater1": int(r1[["S_CTX_r1", "S_NOT_r1"]].isna().sum().sum()),
               "rater2": int(r2[["S_CTX_r2", "S_NOT_r2"]].isna().sum().sum())}
    if any(missing.values()):
        print(f"UYARI - bos ya da okunamayan hucre: {missing}\n")

    regex = load_regex_coding()
    items = items.merge(regex, on=["block", "prompt_id"], how="left")

    # consensus: both readers agree
    items["S_NOT_cons"] = items.apply(
        lambda r: r.S_NOT_r1 if r.S_NOT_r1 == r.S_NOT_r2 else None, axis=1)
    items["S_CTX_cons"] = items.apply(
        lambda r: r.S_CTX_r1 if r.S_CTX_r1 == r.S_CTX_r2 else None, axis=1)

    items.to_csv(EVAL / "role_statement_items.csv", index=False)

    print("=" * 96)
    print("1. ROLE SEPARATION BY CONDITION - human coding is primary")
    print("=" * 96)
    print(f"{'condition':<46}{'S_CTX':>12}{'S_NOT':>12}{'95% CI (S_NOT)':>22}")
    summary = []
    for b in BLOCK_ORDER:
        g = items[items.block == b]
        if g.empty:
            continue
        ctx = g["S_CTX_cons"].dropna()
        nt = g["S_NOT_cons"].dropna()
        lo, hi = wilson(int(nt.sum()), len(nt)) if len(nt) else (float("nan"),) * 2
        print(f"{BLOCK_LABEL[b]:<46}"
              f"{f'{int(ctx.sum())}/{len(ctx)}':>12}"
              f"{f'{int(nt.sum())}/{len(nt)}':>12}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>22}")
        summary.append({"block": b, "label": BLOCK_LABEL[b],
                        "S_CTX_k": int(ctx.sum()), "S_CTX_n": len(ctx),
                        "S_NOT_k": int(nt.sum()), "S_NOT_n": len(nt),
                        "S_NOT_lo": lo, "S_NOT_hi": hi})
    pd.DataFrame(summary).to_csv(EVAL / "role_statement_summary.csv", index=False)

    print("\n" + "=" * 96)
    print("2. READER AGREEMENT")
    print("=" * 96)
    for q in ["S_CTX", "S_NOT"]:
        a = items[f"{q}_r1"].tolist()
        b = items[f"{q}_r2"].tolist()
        pairs = [(x, y) for x, y in zip(a, b) if pd.notna(x) and pd.notna(y)]
        agree = sum(1 for x, y in pairs if x == y)
        print(f"  {q}: raw agreement {agree}/{len(pairs)} = {agree/len(pairs):.3f}   "
              f"Cohen kappa = {cohen_kappa(a, b):.3f}")

    print("\n" + "=" * 96)
    print("3. REGEX VALIDATED AGAINST THE HUMAN CODING (S_NOT)")
    print("=" * 96)
    v = items[items.regex_S_NOT.notna() & items.S_NOT_cons.notna()]
    if v.empty:
        print("  (karsilastirilacak madde yok)")
    else:
        tp = int(((v.regex_S_NOT == 1) & (v.S_NOT_cons == 1)).sum())
        fp = int(((v.regex_S_NOT == 1) & (v.S_NOT_cons == 0)).sum())
        fn = int(((v.regex_S_NOT == 0) & (v.S_NOT_cons == 1)).sum())
        tn = int(((v.regex_S_NOT == 0) & (v.S_NOT_cons == 0)).sum())
        n = len(v)
        print(f"  n = {n}   agreement = {(tp+tn)/n:.3f}   kappa = "
              f"{cohen_kappa(v.regex_S_NOT.tolist(), v.S_NOT_cons.tolist()):.3f}")
        print(f"  regex MISSED (human E, regex 0): {fn}")
        print(f"  regex FALSE HIT (human H, regex 1): {fp}")
        if fn:
            print("\n  Kacirilan maddeler, kosullara gore:")
            print(v[(v.regex_S_NOT == 0) & (v.S_NOT_cons == 1)]
                  .groupby("block").size().to_string())
        miss = v[(v.regex_S_NOT == 0) & (v.S_NOT_cons == 1)]
        miss.to_csv(EVAL / "role_statement_regex_misses.csv", index=False)

    print("\n" + "=" * 96)
    print("4. PARTIAL COMPLIANCE: context asserted, contrast absent")
    print("=" * 96)
    for b in BLOCK_ORDER:
        g = items[items.block == b]
        if g.empty:
            continue
        part = int(((g.S_CTX_cons == 1) & (g.S_NOT_cons == 0)).sum())
        print(f"  {BLOCK_LABEL[b]:<46}{part}/{len(g)}")

    print(f"\nYazildi: {EVAL/'role_statement_items.csv'}")
    print(f"Yazildi: {EVAL/'role_statement_summary.csv'}")


if __name__ == "__main__":
    main()
