"""
34_human_review_statistics.py

Adds inferential structure to the human-review comparison, which in the
manuscript is reported as bare condition means over n = 10 per cell.

Produces, for every human-rated criterion:
  - per-condition mean, median, SD and the number of DISTINCT values used
    (a criterion with one distinct value per condition cannot be reported as
    an item-level rating and is marked as non-discriminating),
  - BCa-free percentile bootstrap 95% CI of the condition mean,
  - paired Wilcoxon signed-rank tests against the full-evidence condition,
    paired by decision_id, with Holm correction and rank-biserial effect size.

Outputs:
  outputs/llm_eval/human_review_condition_stats.csv
  outputs/llm_eval/human_review_paired_tests.csv
"""

import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import PROJECT_ROOT
from common import ensure_dirs

EVAL_DIR = PROJECT_ROOT / "outputs" / "llm_eval"
ensure_dirs(EVAL_DIR)

SHEET = PROJECT_ROOT / "ollama_llama_human_review_completed.xlsx"

CRITERIA = [
    "evidence_faithfulness_score_1_5",
    "macro_network_role_separation_1_5",
    "uncertainty_quality_1_5",
    "decision_usefulness_1_5",
    "unsupported_claim_count_manual",
]

RNG = np.random.default_rng(20260919)
B = 20000


def boot_ci(x, stat=np.mean, alpha=0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0 or np.allclose(x, x[0]):
        v = float(x[0]) if len(x) else np.nan
        return v, v
    idx = RNG.integers(0, len(x), size=(B, len(x)))
    draws = stat(x[idx], axis=1)
    return float(np.percentile(draws, 100 * alpha / 2)), float(np.percentile(draws, 100 * (1 - alpha / 2)))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        prev = max(prev, val)
        adj[i] = min(1.0, prev)
    return adj


df = pd.read_excel(SHEET)
conditions = ["full", "no_network", "no_reliability", "free_form"]

rows = []
for c in conditions:
    g = df[df["condition"] == c]
    for crit in CRITERIA:
        x = g[crit].astype(float)
        lo, hi = boot_ci(x)
        rows.append({
            "condition": c,
            "criterion": crit,
            "n": int(x.notna().sum()),
            "mean": round(float(x.mean()), 3),
            "median": float(x.median()),
            "sd": round(float(x.std(ddof=1)), 3) if x.notna().sum() > 1 else np.nan,
            "distinct_values": int(x.nunique()),
            "constant_within_condition": bool(x.nunique() <= 1),
            "ci95_low": round(lo, 3),
            "ci95_high": round(hi, 3),
        })

stats = pd.DataFrame(rows)
stats.to_csv(EVAL_DIR / "human_review_condition_stats.csv", index=False)

# ---------------------------------------------------------- paired contrasts
wide = {c: df[df["condition"] == c].set_index("decision_id") for c in conditions}
tests = []
for crit in CRITERIA:
    pairs = []
    for a, b in itertools.combinations(conditions, 2):
        ids = wide[a].index.intersection(wide[b].index)
        xa = wide[a].loc[ids, crit].astype(float)
        xb = wide[b].loc[ids, crit].astype(float)
        d = (xa - xb).dropna()
        n_nonzero = int((d != 0).sum())
        if n_nonzero == 0:
            p, stat, rb = np.nan, np.nan, 0.0
        else:
            try:
                stat, p = wilcoxon(xa, xb, zero_method="wilcox", alternative="two-sided")
            except Exception:
                stat, p = np.nan, np.nan
            pos = float((d > 0).sum()); neg = float((d < 0).sum())
            rb = (pos - neg) / n_nonzero
        pairs.append({
            "criterion": crit, "condition_a": a, "condition_b": b,
            "n_pairs": len(d), "n_nonzero_diffs": n_nonzero,
            "mean_diff": round(float(d.mean()), 3) if len(d) else np.nan,
            "wilcoxon_p": p, "rank_biserial": round(rb, 3),
        })
    ps = [x["wilcoxon_p"] for x in pairs]
    valid = [i for i, v in enumerate(ps) if not (v is None or (isinstance(v, float) and np.isnan(v)))]
    if valid:
        adj = holm([ps[i] for i in valid])
        for k, i in enumerate(valid):
            pairs[i]["p_holm"] = round(float(adj[k]), 4)
    for x in pairs:
        x.setdefault("p_holm", np.nan)
        if not (isinstance(x["wilcoxon_p"], float) and np.isnan(x["wilcoxon_p"])):
            x["wilcoxon_p"] = round(float(x["wilcoxon_p"]), 4)
    tests.extend(pairs)

tests = pd.DataFrame(tests)
tests.to_csv(EVAL_DIR / "human_review_paired_tests.csv", index=False)

print("=== Condition statistics (n = 10 per cell) ===")
print(stats.to_string(index=False))
print("\n=== Paired Wilcoxon contrasts (Holm-corrected) ===")
print(tests.to_string(index=False))

nd = stats[stats["constant_within_condition"]]
if len(nd):
    print("\nWARNING: criteria with a single distinct value inside a condition "
          "(cannot be presented as independent item-level ratings):")
    print(nd[["condition", "criterion", "mean"]].to_string(index=False))
