"""Round-2 coding: agreement, test-retest, and the new condition's result.

Reads the two filled round-2 sheets and their keys, merges on (block, prompt_id)
rather than prompt_id alone -- the same prompt appears in several blocks, so
merging on prompt_id produces a cartesian product that silently inflates every
n. Reports:

  1. inter-rater agreement on the round-2 pass
  2. test-retest against round 1 on the 20 repeated items
  3. S_CTX and S_NOT by block, including the decoder-enforced condition

Writes outputs/llm_eval/role_statement_r2_items.csv and ..._r2_summary.csv
"""
import csv
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "llm_eval"


def yn(v):
    s = str(v).strip().upper()
    if s in ("E", "EVET", "Y", "YES", "1", "TRUE"):
        return 1
    if s in ("H", "HAYIR", "N", "NO", "0", "FALSE"):
        return 0
    return None


def cohen_kappa(x, y):
    n = len(x)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in zip(x, y) if a == b) / n
    labels = set(x) | set(y)
    pe = sum((sum(1 for a in x if a == k) / n) * (sum(1 for b in y if b == k) / n)
             for k in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def load_rater(tag):
    sheet = pd.read_excel(EVAL / f"role_statement_r2_sheet_{tag}.xlsx", sheet_name="KODLAMA")
    key = pd.read_csv(EVAL / f"role_statement_r2_key_{tag}.csv")
    ctx = next(c for c in sheet.columns if c.startswith("S_CTX"))
    nt = next(c for c in sheet.columns if c.startswith("S_NOT"))
    body = sheet[sheet["Sıra"].astype(str).str.startswith("R2-")].copy()
    body = body.rename(columns={"Sıra": "item_id", ctx: "S_CTX", nt: "S_NOT"})
    body["S_CTX"] = body["S_CTX"].map(yn)
    body["S_NOT"] = body["S_NOT"].map(yn)
    m = body[["item_id", "S_CTX", "S_NOT"]].merge(key, on="item_id", how="inner")
    assert len(m) == len(body), f"{tag}: anahtar eslesmesi eksik"
    return m


def main():
    r1 = load_rater("rater1").rename(columns={"S_CTX": "S_CTX_r1", "S_NOT": "S_NOT_r1"})
    r2 = load_rater("rater2").rename(columns={"S_CTX": "S_CTX_r2", "S_NOT": "S_NOT_r2"})

    items = r1.drop(columns=["item_id"]).merge(
        r2[["block", "prompt_id", "S_CTX_r2", "S_NOT_r2"]],
        on=["block", "prompt_id"], how="inner")
    assert len(items) == len(r1), (
        f"birlesme hatali: {len(items)} satir, beklenen {len(r1)}. "
        "prompt_id tek basina benzersiz degil; (block, prompt_id) ciftiyle birlestir.")

    for q in ("S_CTX", "S_NOT"):
        items[f"{q}_cons"] = [a if a == b else None
                              for a, b in zip(items[f"{q}_r1"], items[f"{q}_r2"])]

    print("=" * 78)
    print("1. DEGERLENDIRICILER ARASI UYUM (ikinci tur, 30 madde)")
    print("=" * 78)
    for q in ("S_CTX", "S_NOT"):
        a, b = list(items[f"{q}_r1"]), list(items[f"{q}_r2"])
        agree = sum(1 for x, y in zip(a, b) if x == y)
        print(f"  {q}: {agree}/{len(a)} ayni  (kappa = {cohen_kappa(a, b):.3f})")

    print("\n" + "=" * 78)
    print("2. TEST-TEKRAR: birinci turda da kodlanan 20 madde")
    print("=" * 78)
    p1 = EVAL / "role_statement_items.csv"
    if p1.exists():
        old = pd.read_csv(p1)
        old = old[["block", "prompt_id", "S_CTX_cons", "S_NOT_cons"]].rename(
            columns={"S_CTX_cons": "S_CTX_r1pass", "S_NOT_cons": "S_NOT_r1pass"})
        j = items.merge(old, on=["block", "prompt_id"], how="inner")
        print(f"  eslesen madde: {len(j)}")
        for q in ("S_CTX", "S_NOT"):
            k = j[j[f"{q}_cons"].notna() & j[f"{q}_r1pass"].notna()]
            if len(k):
                same = int((k[f"{q}_cons"].astype(float) == k[f"{q}_r1pass"].astype(float)).sum())
                print(f"  {q}: iki turda ayni sonuc {same}/{len(k)}"
                      f"  (kappa = {cohen_kappa(list(k[f'{q}_cons'].astype(int)), list(k[f'{q}_r1pass'].astype(int))):.3f})")
    else:
        print("  birinci tur dosyasi yok, atlandi")

    print("\n" + "=" * 78)
    print("3. BLOK BAZINDA SONUC (uzlasi; uzlasmazlik None sayilir)")
    print("=" * 78)
    print(f'  {"block":16s} {"n":>3s} {"S_CTX":>9s} {"S_NOT":>9s}')
    rows = []
    for blk, g in items.groupby("block"):
        c = g["S_CTX_cons"].dropna()
        s = g["S_NOT_cons"].dropna()
        print(f'  {blk:16s} {len(g):3d} {int(c.sum()):4d}/{len(c):<4d} {int(s.sum()):4d}/{len(s):<4d}')
        rows.append({"block": blk, "n": len(g),
                     "S_CTX_k": int(c.sum()), "S_CTX_n": len(c),
                     "S_NOT_k": int(s.sum()), "S_NOT_n": len(s)})

    items.to_csv(EVAL / "role_statement_r2_items.csv", index=False)
    pd.DataFrame(rows).to_csv(EVAL / "role_statement_r2_summary.csv", index=False)
    print(f"\nyazildi: outputs/llm_eval/role_statement_r2_items.csv")
    print(f"yazildi: outputs/llm_eval/role_statement_r2_summary.csv")


if __name__ == "__main__":
    main()
