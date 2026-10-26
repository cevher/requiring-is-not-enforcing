"""
37_rating_agreement.py

Computes agreement between the original human ratings and a second rating pass
(either a blind re-rating by the same rater, or the LLM judge from
36_run_llm_judge.py), and re-runs the condition comparison on the second pass.

Reports, per criterion:
  - quadratic-weighted Cohen's kappa (ordinal agreement)
  - Spearman rho
  - exact agreement and within-one agreement
  - mean absolute difference
  - whether the second pass shows within-condition variance where the first did not

Usage:
  python src/37_rating_agreement.py --second outputs/llm_eval/llm_judge_ratings.csv --prefix judge
  python src/37_rating_agreement.py --second outputs/llm_eval/blind_rerating_completed.xlsx --prefix rerate
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from config import PROJECT_ROOT
from common import ensure_dirs

EVAL_DIR = PROJECT_ROOT / "outputs" / "llm_eval"
ensure_dirs(EVAL_DIR)

DEFAULT_FIRST = PROJECT_ROOT / "ollama_llama_human_review_completed.xlsx"

CRITERIA = [
    "evidence_faithfulness_score_1_5",
    "macro_network_role_separation_1_5",
    "uncertainty_quality_1_5",
    "decision_usefulness_1_5",
    "unsupported_claim_count_manual",
    "causal_overclaim_manual_0_1",
]


def load_second(path):
    p = str(path)
    df = pd.read_excel(p, sheet_name=0) if p.endswith((".xlsx", ".xlsm")) else pd.read_csv(p)
    df.columns = [c.replace("judge_", "") for c in df.columns]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", default=str(DEFAULT_FIRST),
                    help="pass 1 ratings: the original review sheet, or a blind sheet "
                         "completed by rater 1")
    ap.add_argument("--first-key", default="",
                    help="key file for --first when it is a blind sheet (item_id -> prompt_id)")
    ap.add_argument("--second", required=True)
    ap.add_argument("--second-key", default=str(EVAL_DIR / "blind_rating_key_rater2.csv"))
    ap.add_argument("--prefix", default="second")
    args = ap.parse_args()

    first = load_second(args.first)
    if args.first_key:
        first = first.merge(pd.read_csv(args.first_key), on="item_id", how="left")
    second = load_second(args.second).merge(pd.read_csv(args.second_key), on="item_id", how="left")

    # Both passes are aligned on prompt_id, so the two raters may see different
    # item orders without breaking the pairing.
    merged = second.merge(
        first.drop(columns=[c for c in ("item_id", "condition", "decision_id", "original_row")
                            if c in first.columns]),
        on="prompt_id", how="left", suffixes=("_2", "_1"))

    rows = []
    for c in CRITERIA:
        a = pd.to_numeric(merged.get(f"{c}_1"), errors="coerce")
        b = pd.to_numeric(merged.get(f"{c}_2"), errors="coerce")
        if a is None or b is None:
            continue
        m = a.notna() & b.notna()
        a, b = a[m], b[m]
        if len(a) < 3:
            continue
        try:
            kappa = cohen_kappa_score(a.round().astype(int), b.round().astype(int),
                                      weights="quadratic")
        except Exception:
            kappa = np.nan
        rho = spearmanr(a, b).statistic if a.nunique() > 1 and b.nunique() > 1 else np.nan
        rows.append({
            "criterion": c, "n": int(len(a)),
            "mean_pass1": round(float(a.mean()), 3),
            "mean_pass2": round(float(b.mean()), 3),
            "quadratic_weighted_kappa": None if pd.isna(kappa) else round(float(kappa), 3),
            "spearman_rho": None if pd.isna(rho) else round(float(rho), 3),
            "exact_agreement": round(float((a == b).mean()), 3),
            "within_one_agreement": round(float((abs(a - b) <= 1).mean()), 3),
            "mean_abs_diff": round(float(abs(a - b).mean()), 3),
        })

    agree = pd.DataFrame(rows)
    agree.to_csv(EVAL_DIR / f"rating_agreement_{args.prefix}.csv", index=False)

    var_rows = []
    for c in CRITERIA:
        b = pd.to_numeric(merged.get(f"{c}_2"), errors="coerce")
        a = pd.to_numeric(merged.get(f"{c}_1"), errors="coerce")
        if b is None:
            continue
        for cond, g in merged.groupby("condition"):
            i = g.index
            var_rows.append({
                "condition": cond, "criterion": c,
                "distinct_pass1": int(a.loc[i].nunique()) if a is not None else None,
                "distinct_pass2": int(b.loc[i].nunique()),
                "mean_pass2": round(float(b.loc[i].mean()), 3),
                "sd_pass2": round(float(b.loc[i].std(ddof=1)), 3) if b.loc[i].notna().sum() > 1 else None,
            })
    var = pd.DataFrame(var_rows)
    var.to_csv(EVAL_DIR / f"condition_stats_{args.prefix}.csv", index=False)

    print("=== Agreement between pass 1 (human) and pass 2 ===")
    print(agree.to_string(index=False))
    print("\n=== Pass 2 condition means and within-condition variance ===")
    print(var.sort_values(["criterion", "condition"]).to_string(index=False))

    flat = var[var["distinct_pass2"] <= 1]
    if len(flat):
        print("\nNOTE: pass 2 still has zero within-condition variance in these cells:")
        print(flat[["condition", "criterion", "mean_pass2"]].to_string(index=False))


if __name__ == "__main__":
    main()
