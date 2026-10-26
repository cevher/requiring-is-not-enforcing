# -*- coding: utf-8 -*-
"""
45_instrument_validation.py

Compares two human readers against the automatic instrument on the same items.

The reported quantity is READER-SCRIPT agreement, not reader-reader agreement.
Reader-reader agreement is also printed, but it is secondary: two people running
the same lookup should agree, so a high value there demonstrates little. What
needs demonstrating is that the automatic checks, on which every result in the
paper now depends, read the generated notes the way a careful human reader does.

For each item the script recomputes exactly the task the reader was given:

  S1a  how many distinct evidence identifiers appear in the first three cited
  S1b  how many of those appear in the allowed evidence list
  S3   E if the note states that network evidence is not the primary predictor,
       H if it discusses network evidence without saying so,
       YOK if it never mentions network evidence
  S6   E if the note asserts transmission/causation as fact, H otherwise
       (a match preceded by a negation cue is an echo of the prompt, not a claim)

Divergences are listed individually, because that is where the instrument gets
corrected: during calibration a reader caught "is not a primary predictor",
which the earlier pattern missed by requiring "not the primary".

Outputs:
  outputs/llm_eval/instrument_validation_items.csv
  outputs/llm_eval/instrument_validation_summary.csv
"""

import json
import re

import numpy as np
import pandas as pd

from config import PROJECT_ROOT
from common import ensure_dirs

EVAL = PROJECT_ROOT / "outputs" / "llm_eval"
GEN = PROJECT_ROOT / "outputs" / "llm_generations"
ensure_dirs(EVAL)

SOURCE = GEN / "ollama_llama_native_10decisions_4conditions_final_clean.jsonl"

ID_RE = re.compile(r"\b[A-Z]+(?:-[A-Z0-9_\.]+)*-\d{8}(?:-[A-Za-z0-9_\.]+)*\b")
NET_RE = re.compile(r"network|centrality|co-?movement|density", re.I)
NOT_PRIMARY_RE = re.compile(
    r"not (a|the)? ?primary|not (a|the)? ?(stress )?predictor|"
    r"rather than (a|the)? ?primary|context(ual)? (only|rather than)|"
    r"not (a )?causal", re.I)
TRANSMIT_RE = re.compile(
    r"caused|causing|contagion|spillover|transmitted|transmission|"
    r"spread (from|to)|led to|triggered", re.I)
NEG_CUES = ["do not", "does not", "not ", "no ", "avoid", "without", "cannot",
            "must not", "should not", "rather than", "never"]


def values_only(raw):
    """Claim text only: field names and evidence_ids lists are not the note."""
    s, e = raw.find("{"), raw.rfind("}")
    try:
        obj = json.loads(raw[s:e + 1])
    except Exception:
        return raw
    out = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "evidence_ids":
                    continue
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        else:
            out.append(str(o))
    walk(obj)
    return " ".join(out)


def script_answers(rec):
    raw = rec.get("response_text") or ""
    body = values_only(raw)
    allowed = set(rec.get("allowed_evidence_ids") or [])

    seen = []
    for m in ID_RE.findall(raw):
        if m not in seen:
            seen.append(m)
    first3 = seen[:3]

    if not NET_RE.search(body):
        s3 = "YOK"
    else:
        s3 = "E" if NOT_PRIMARY_RE.search(body) else "H"

    genuine = 0
    for m in TRANSMIT_RE.finditer(body):
        left = body[max(0, m.start() - 60):m.start()].lower()
        if not any(c in left for c in NEG_CUES):
            genuine += 1

    return {"S1a": len(first3),
            "S1b": sum(1 for x in first3 if x in allowed),
            "S3": s3,
            "S6": "E" if genuine else "H"}


def tag(name):
    m = re.match(r"^(S\d[ab]?)\.", str(name))
    return m.group(1) if m else None


def norm(v, numeric=False):
    if pd.isna(v):
        return np.nan
    if numeric:
        try:
            return int(float(v))
        except Exception:
            return np.nan
    s = str(v).strip().upper()
    return {"EVET": "E", "Y": "E", "YES": "E", "HAYIR": "H", "N": "H", "NO": "H",
            "NA": "YOK", "N/A": "YOK"}.get(s, s)


def load_reader(sheet, key):
    df = pd.read_excel(EVAL / sheet, sheet_name="PUANLAMA")
    cols = {tag(c): c for c in df.columns if tag(c)}
    df = df[df["Sıra"].astype(str).str.upper() != "ÖRNEK"]
    out = df[["Sıra"]].rename(columns={"Sıra": "item_id"}).copy()
    for q, c in cols.items():
        out[q] = df[c].map(lambda v: norm(v, numeric=q.startswith("S1")))
    return out.merge(pd.read_csv(EVAL / key), on="item_id", how="left")


gens = {json.loads(l)["prompt_id"]: json.loads(l)
        for l in open(SOURCE, encoding="utf-8") if l.strip()}

r1 = load_reader("binary_rating_sheet_rater1.xlsx", "binary_rating_key_rater1.csv")
r2 = load_reader("binary_rating_sheet_rater2.xlsx", "binary_rating_key_rater2.csv")

QS = ["S1a", "S1b", "S3", "S6"]

missing = {n: int(d[QS].isna().sum().sum()) for n, d in [("rater1", r1), ("rater2", r2)]}
print("=" * 78)
print("0. COMPLETENESS")
print("=" * 78)
for n, m in missing.items():
    print(f"  {n}: {m} unanswered cell(s) of {len(r1) * len(QS)}")
if any(missing.values()):
    print("\n  Fill the empty cells before quoting the numbers below.")
print()

rows = []
for _, a in r1.iterrows():
    b = r2[r2["prompt_id"] == a["prompt_id"]]
    b = b.iloc[0] if len(b) else None
    ref = script_answers(gens[a["prompt_id"]])
    rec = {"prompt_id": a["prompt_id"], "condition": a["condition"]}
    for q in QS:
        rec[f"{q}_script"] = ref[q]
        rec[f"{q}_r1"] = a[q]
        rec[f"{q}_r2"] = b[q] if b is not None else np.nan
    rows.append(rec)
items = pd.DataFrame(rows)
items.to_csv(EVAL / "instrument_validation_items.csv", index=False)

summary = []
for q in QS:
    sc = items[f"{q}_script"]
    for reader in ["r1", "r2"]:
        h = items[f"{q}_{reader}"]
        m = h.notna()
        if m.sum() == 0:
            continue
        summary.append({"check": q, "reader": reader, "n": int(m.sum()),
                        "agreement_with_script": round(float((h[m] == sc[m]).mean()), 3),
                        "n_divergent": int((h[m] != sc[m]).sum())})
    m = items[f"{q}_r1"].notna() & items[f"{q}_r2"].notna()
    if m.sum():
        summary.append({"check": q, "reader": "r1_vs_r2", "n": int(m.sum()),
                        "agreement_with_script": round(
                            float((items.loc[m, f"{q}_r1"] == items.loc[m, f"{q}_r2"]).mean()), 3),
                        "n_divergent": int((items.loc[m, f"{q}_r1"] != items.loc[m, f"{q}_r2"]).sum())})
summ = pd.DataFrame(summary)
summ.to_csv(EVAL / "instrument_validation_summary.csv", index=False)

print("=" * 78)
print("1. READER-SCRIPT AGREEMENT (primary)   /   reader-reader (secondary)")
print("=" * 78)
print(summ.to_string(index=False))

print("\n" + "=" * 78)
print("2. DIVERGENT ITEMS - inspect each one; the script may be what needs fixing")
print("=" * 78)
any_div = False
for _, r in items.iterrows():
    for q in QS:
        for reader in ["r1", "r2"]:
            h = r[f"{q}_{reader}"]
            if pd.notna(h) and h != r[f"{q}_script"]:
                any_div = True
                print(f"  {r['prompt_id'][-28:]:28} [{r['condition']:14}] {q}: "
                      f"script={r[f'{q}_script']}  {reader}={h}")
if not any_div:
    print("  none")

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
prim = summ[summ["reader"] != "r1_vs_r2"]
if len(prim):
    worst = prim.loc[prim["agreement_with_script"].idxmin()]
    print(f"  Lowest reader-script agreement: {worst['check']} "
          f"({worst['reader']}, {worst['agreement_with_script']:.3f}).")
    print("  Report the agreement per check, and describe every divergence above,")
    print("  including any case where the reader was right and the script was not:")
    print("  that is the evidence that the instrument was validated rather than")
    print("  assumed. Do not silently adjust the script to match the readers after")
    print("  the fact without saying so in the paper.")
