"""
44_binary_rating_analysis.py

Analyses the second-round binary rating passes.

Reports, per question:
  - raw percentage agreement between the two annotators
  - Cohen's kappa, and PABAK (prevalence-adjusted bias-adjusted kappa)
  - the marginal distribution of each annotator's answers

Why PABAK is reported alongside kappa: several of these checks are expected to
have skewed marginals (for example, a causal assertion is rare). With a skewed
marginal, Cohen's kappa can be near zero even when the annotators agree on 95%
of items, and it is undefined when one annotator uses a single category. PABAK
is the agreement statistic that stays interpretable in that situation, so both
are reported and neither alone is used to accept or reject the protocol.

It then compares conditions on each check, and repeats the convergent-validity
tests from the first round: whether the annotators' answers track the objective
automatic measures of the same property.

Outputs:
  outputs/llm_eval/binary_agreement.csv
  outputs/llm_eval/binary_condition_summary.csv
  outputs/llm_eval/binary_validity_checks.csv
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from config import PROJECT_ROOT
from common import ensure_dirs

EVAL = PROJECT_ROOT / "outputs" / "llm_eval"
ensure_dirs(EVAL)

ORDER = ["full", "no_network", "no_reliability", "free_form"]

CATEGORICAL = ["q2_value_match", "q3_role_statement", "q4_uncertainty_reason_named",
               "q5_entity_not_in_evidence", "q6_causal_assertion"]
NUMERIC = ["q1_n_ids_checked", "q1_n_ids_found"]

AGREEMENT_FLOOR = 0.70   # raw agreement below this means the question is not mechanical enough
PABAK_FLOOR = 0.40


def norm(v):
    """Accept E/H/YOK in any case, and the English Y/N/NA."""
    s = str(v).strip().upper()
    if s in {"E", "EVET", "Y", "YES", "1", "TRUE"}:
        return "E"
    if s in {"H", "HAYIR", "HAYIR.", "N", "NO", "0", "FALSE"}:
        return "H"
    if s in {"YOK", "NA", "N/A"}:
        return "YOK"
    if s in {"", "-"} or s == "NAN":
        # An unanswered cell is missing data, not "not applicable". Collapsing the
        # two would silently manufacture agreement between two blank sheets.
        return np.nan
    return s


def load(sheet, key):
    # The workbook is written with full Turkish questions as column headers so the
    # annotator can see what is being asked while answering. 43_ stores the
    # header -> machine-name mapping next to the sheets.
    colmap = json.load(open(EVAL / "binary_sheet_colmap.json", encoding="utf-8"))
    df = pd.read_excel(EVAL / sheet, sheet_name="PUANLAMA")
    df = df.rename(columns=colmap)
    # The first row is the worked example shipped with the sheet, not data.
    df = df[df["item_id"].astype(str).str.upper() != "ÖRNEK"].copy()
    df = df.merge(pd.read_csv(EVAL / key), on="item_id", how="left")
    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].map(norm)
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pabak(a, b):
    """PABAK = 2 * p_observed - 1, for the observed agreement proportion."""
    po = float((a == b).mean())
    return 2 * po - 1


r1 = load("binary_rating_sheet_rater1.xlsx", "binary_rating_key_rater1.csv")
r2 = load("binary_rating_sheet_rater2.xlsx", "binary_rating_key_rater2.csv")

# Completeness first: an analysis of a half-filled sheet is worse than no analysis,
# because the missing cells are invisible in the summary statistics.
print("=" * 78)
print("0. COMPLETENESS")
print("=" * 78)
incomplete = False
for name, df in [("rater1", r1), ("rater2", r2)]:
    missing = {q: int(df[q].isna().sum()) for q in CATEGORICAL + NUMERIC if q in df.columns}
    total = sum(missing.values())
    print(f"  {name}: {len(df)} item, {total} unanswered cell(s)"
          + (f"  -> {', '.join(f'{k}={v}' for k, v in missing.items() if v)}" if total else ""))
    incomplete = incomplete or total > 0
if incomplete:
    print("\n  Some cells are still empty. Fill them before quoting any number below;")
    print("  the statistics are computed on the answered subset only.")
print()

pair = r1.merge(r2, on="prompt_id", suffixes=("_1", "_2"))

rows = []
for q in CATEGORICAL + NUMERIC:
    a, b = pair.get(f"{q}_1"), pair.get(f"{q}_2")
    if a is None or b is None:
        continue
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    if len(a) == 0:
        continue
    po = float((a == b).mean())
    try:
        k = cohen_kappa_score(a.astype(str), b.astype(str))
    except Exception:
        k = np.nan
    rows.append({
        "question": q, "n": int(len(a)),
        "raw_agreement": round(po, 3),
        "cohens_kappa": None if pd.isna(k) else round(float(k), 3),
        "pabak": round(pabak(a, b), 3),
        "rater1_distribution": dict(a.value_counts().sort_index()),
        "rater2_distribution": dict(b.value_counts().sort_index()),
    })
agree = pd.DataFrame(rows)
agree.to_csv(EVAL / "binary_agreement.csv", index=False)

# ------------------------------------------------------------ condition summary
cond_rows = []
for name, df in [("rater1", r1), ("rater2", r2)]:
    for cond in ORDER:
        g = df[df["condition"] == cond]
        rec = {"annotator": name, "condition": cond, "n": len(g)}
        for q in CATEGORICAL:
            if q in g.columns:
                rec[f"{q}__E"] = round(float((g[q] == "E").mean()), 3)
                rec[f"{q}__YOK"] = round(float((g[q] == "YOK").mean()), 3)
        if {"q1_n_ids_checked", "q1_n_ids_found"} <= set(g.columns):
            chk = g["q1_n_ids_checked"].sum()
            rec["q1_id_hit_rate"] = round(float(g["q1_n_ids_found"].sum() / chk), 3) if chk else None
        cond_rows.append(rec)
cond = pd.DataFrame(cond_rows)
cond.to_csv(EVAL / "binary_condition_summary.csv", index=False)

# ------------------------------------------------------------- validity checks
auto = pd.read_csv(EVAL / "llama3_main_v2_auto_eval_results.csv")
strict = auto[auto["role_separation_strict"].notna()][["prompt_id", "role_separation_strict"]]
inv = auto[["prompt_id", "n_invented_evidence_ids"]]

vrows = []
for name, df in [("rater1", r1), ("rater2", r2)]:
    m = df.merge(strict, on="prompt_id")
    if "q3_role_statement" in m.columns and len(m):
        e = (m["q3_role_statement"] == "E").astype(int)
        y = m["role_separation_strict"].astype(int)
        acc = float((e == y).mean())
    else:
        acc = np.nan
    m2 = df.merge(inv, on="prompt_id")
    has = m2["n_invented_evidence_ids"].fillna(0) > 0
    if {"q1_n_ids_checked", "q1_n_ids_found"} <= set(m2.columns):
        miss = (m2["q1_n_ids_found"] < m2["q1_n_ids_checked"])
        detect = float(miss[has].mean()) if has.any() else np.nan
        false_alarm = float(miss[~has].mean()) if (~has).any() else np.nan
    else:
        detect = false_alarm = np.nan
    vrows.append({
        "annotator": name,
        "q3_matches_objective_role_separation": None if pd.isna(acc) else round(acc, 3),
        "q1_flags_missing_id_when_invented_present": None if pd.isna(detect) else round(detect, 3),
        "q1_flags_missing_id_when_none_present": None if pd.isna(false_alarm) else round(false_alarm, 3),
    })
valid = pd.DataFrame(vrows)
valid.to_csv(EVAL / "binary_validity_checks.csv", index=False)

print("=" * 78)
print("1. AGREEMENT BETWEEN THE TWO ANNOTATORS")
print("=" * 78)
print(agree.to_string(index=False))

print("\n" + "=" * 78)
print("2. CONDITION SUMMARY (share answering 'E', by annotator)")
print("=" * 78)
print(cond.to_string(index=False))

print("\n" + "=" * 78)
print("3. CONVERGENT VALIDITY AGAINST THE AUTOMATIC MEASURES")
print("=" * 78)
print(valid.to_string(index=False))

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
weak = agree[(agree["raw_agreement"] < AGREEMENT_FLOOR) |
             (agree["pabak"].astype(float) < PABAK_FLOOR)]["question"].tolist()
strong = [q for q in agree["question"] if q not in weak]
if strong:
    print(f"Adequately reproducible, may carry a reported finding: {', '.join(strong)}")
if weak:
    print(f"Not reproducible enough to report as a finding: {', '.join(weak)}")
    print("Report these as an instrument limitation rather than dropping them silently:")
    print("a check that two annotators cannot apply identically is a result about the")
    print("check, and it belongs in the paper.")
if not weak:
    print("Every check cleared both thresholds. The human layer can be reported as a")
    print("confirmation of the automatic audit rather than as an independent claim.")
