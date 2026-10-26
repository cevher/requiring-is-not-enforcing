"""
42_human_review_analysis.py

Full analysis of the two blind human rating passes and the LLM judge.

This script deliberately computes, and reports, three things in a fixed order:

  1. Inter-annotator agreement. If agreement is inadequate, nothing downstream
     is interpretable, so this is computed first and printed first.
  2. Condition means per annotator, and whether the condition ORDERING
     replicates across annotators. Weak item-level agreement can still permit a
     condition-level claim if both annotators order the conditions the same way;
     that is checked rather than assumed.
  3. Convergent validity: whether each annotator's judgement tracks the
     objective automatic measures of the same construct (strict role separation,
     invented evidence identifiers). An annotator whose ratings are unrelated to
     the objective properties of the text is not measuring the construct.

Only if (1) and (2) pass may condition contrasts be reported as findings. The
script prints an explicit verdict rather than leaving that decision implicit.

Outputs:
  outputs/llm_eval/human_interrater_agreement.csv
  outputs/llm_eval/human_condition_means_by_rater.csv
  outputs/llm_eval/human_validity_checks.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pointbiserialr
from sklearn.metrics import cohen_kappa_score

from config import PROJECT_ROOT
from common import ensure_dirs

EVAL = PROJECT_ROOT / "outputs" / "llm_eval"
ensure_dirs(EVAL)

CRIT = ["evidence_faithfulness_score_1_5", "macro_network_role_separation_1_5",
        "uncertainty_quality_1_5", "decision_usefulness_1_5",
        "unsupported_claim_count_manual", "causal_overclaim_manual_0_1"]
ORDER = ["full", "no_network", "no_reliability", "free_form"]

# Conventional interpretation of weighted kappa (Landis & Koch).
KAPPA_FLOOR = 0.40


def load(sheet, key, is_csv=False):
    df = pd.read_csv(sheet) if is_csv else pd.read_excel(sheet, sheet_name="rate_these")
    df.columns = [c.replace("judge_", "") for c in df.columns]
    return df.merge(pd.read_csv(key), on="item_id", how="left")


r1 = load(EVAL / "blind_rating_sheet_rater1.xlsx", EVAL / "blind_rating_key_rater1.csv")
r2 = load(EVAL / "blind_rating_sheet_rater2.xlsx", EVAL / "blind_rating_key_rater2.csv")
judge = load(EVAL / "llm_judge_ratings.csv", EVAL / "blind_rating_key_rater1.csv", is_csv=True)
auto = pd.read_csv(EVAL / "llama3_main_v2_auto_eval_results.csv")

# ---------------------------------------------------------------- 1. agreement
pair = r1.merge(r2, on="prompt_id", suffixes=("_1", "_2"))
rows = []
for c in CRIT:
    a = pd.to_numeric(pair[f"{c}_1"], errors="coerce")
    b = pd.to_numeric(pair[f"{c}_2"], errors="coerce")
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    try:
        kappa = cohen_kappa_score(a.round().astype(int), b.round().astype(int), weights="quadratic")
    except Exception:
        kappa = np.nan
    rho = spearmanr(a, b).statistic if a.nunique() > 1 and b.nunique() > 1 else np.nan
    rows.append({
        "criterion": c, "n": int(len(a)),
        "mean_rater1": round(float(a.mean()), 3), "mean_rater2": round(float(b.mean()), 3),
        "quadratic_weighted_kappa": None if pd.isna(kappa) else round(float(kappa), 3),
        "spearman_rho": None if pd.isna(rho) else round(float(rho), 3),
        "exact_agreement": round(float((a == b).mean()), 3),
        "within_one_agreement": round(float((abs(a - b) <= 1).mean()), 3),
        "mean_abs_diff": round(float(abs(a - b).mean()), 3),
    })
agree = pd.DataFrame(rows)
agree.to_csv(EVAL / "human_interrater_agreement.csv", index=False)

# ------------------------------------------------------- 2. condition ordering
means, rank_rows = [], []
for name, df in [("rater1", r1), ("rater2", r2), ("judge", judge)]:
    g = df.groupby("condition")[CRIT].mean().reindex(ORDER)
    n = df.groupby("condition")[CRIT].nunique().reindex(ORDER)
    for cond in ORDER:
        rec = {"annotator": name, "condition": cond}
        for c in CRIT:
            rec[c] = round(float(g.loc[cond, c]), 2)
            rec[f"{c}__distinct"] = int(n.loc[cond, c])
        means.append(rec)
cond_means = pd.DataFrame(means)
cond_means.to_csv(EVAL / "human_condition_means_by_rater.csv", index=False)

for c in CRIT:
    a = r1.groupby("condition")[c].mean().reindex(ORDER)
    b = r2.groupby("condition")[c].mean().reindex(ORDER)
    rho = spearmanr(a, b).statistic if a.nunique() > 1 and b.nunique() > 1 else np.nan
    rank_rows.append({"criterion": c,
                      "condition_rank_rho_r1_r2": None if pd.isna(rho) else round(float(rho), 3)})
ranks = pd.DataFrame(rank_rows)

# ----------------------------------------------------------- 3. validity check
strict = auto[auto["role_separation_strict"].notna()][
    ["prompt_id", "role_separation_strict", "n_invented_evidence_ids"]]
inv = auto[["prompt_id", "n_invented_evidence_ids"]]

vrows = []
for name, df in [("rater1", r1), ("rater2", r2), ("judge", judge)]:
    m = df.merge(strict, on="prompt_id")
    x = pd.to_numeric(m["macro_network_role_separation_1_5"], errors="coerce")
    y = m["role_separation_strict"].astype(int)
    r = pointbiserialr(y, x).statistic if x.nunique() > 1 and y.nunique() > 1 else np.nan

    m2 = df.merge(inv, on="prompt_id")
    has = m2["n_invented_evidence_ids"].fillna(0) > 0
    f = pd.to_numeric(m2["evidence_faithfulness_score_1_5"], errors="coerce")
    vrows.append({
        "annotator": name,
        "role_sep_rating_vs_objective_r": None if pd.isna(r) else round(float(r), 3),
        "role_sep_mean_when_objective_0": round(float(x[y == 0].mean()), 2),
        "role_sep_mean_when_objective_1": round(float(x[y == 1].mean()), 2),
        "faithfulness_with_invented_ids": round(float(f[has].mean()), 2),
        "faithfulness_without_invented_ids": round(float(f[~has].mean()), 2),
        "faithfulness_penalises_invented_ids": bool(f[has].mean() < f[~has].mean()),
    })
valid = pd.DataFrame(vrows)
valid.to_csv(EVAL / "human_validity_checks.csv", index=False)

# ------------------------------------------------------------------- printout
print("=" * 78)
print("1. INTER-ANNOTATOR AGREEMENT (two blind passes, 40 items)")
print("=" * 78)
print(agree.to_string(index=False))

print("\n" + "=" * 78)
print("2. CONDITION MEANS BY ANNOTATOR, AND WHETHER THE ORDERING REPLICATES")
print("=" * 78)
for name in ["rater1", "rater2", "judge"]:
    sub = cond_means[cond_means["annotator"] == name].set_index("condition")[CRIT]
    print(f"\n--- {name} ---")
    print(sub.to_string())
print("\nSpearman correlation of condition means between the two human annotators:")
print(ranks.to_string(index=False))

print("\n" + "=" * 78)
print("3. CONVERGENT VALIDITY AGAINST THE OBJECTIVE AUTOMATIC MEASURES")
print("=" * 78)
print(valid.to_string(index=False))

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
k = agree["quadratic_weighted_kappa"].astype(float)
bad = agree.loc[k.notna() & (k < KAPPA_FLOOR), "criterion"].tolist()
neg_rank = ranks.loc[ranks["condition_rank_rho_r1_r2"].astype(float) < 0.5, "criterion"].tolist()

if bad:
    print(f"Weighted kappa is below {KAPPA_FLOOR:.2f} for: {', '.join(bad)}.")
    print("Item-level agreement is inadequate, so condition contrasts computed from")
    print("these ratings are not interpretable as findings.")
if neg_rank:
    print(f"\nThe condition ordering does not replicate across annotators for: "
          f"{', '.join(neg_rank)}.")
    print("A weakly-agreeing pair of annotators can still support a condition-level")
    print("claim if they order the conditions the same way. They do not.")
print("\nReport the agreement failure. Do not select the annotator whose ratings")
print("happen to match the hypothesis: choosing an annotator after seeing which one")
print("supports the expected result is the practice that makes findings")
print("irreproducible, and a reviewer will ask why two passes were run and one")
print("reported.")
