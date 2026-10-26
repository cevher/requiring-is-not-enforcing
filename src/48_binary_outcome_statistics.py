"""
48_binary_outcome_statistics.py

Inferential statistics for the binary audit outcomes.

WHY THIS EXISTS
---------------
Every proportion in the results tables is reported as a bare point estimate over
ten outputs per cell. At that sample size a reader cannot tell which differences
are real. Two of the manuscript's reported contrasts have very different
standing once this is computed:

  - role separation 0/10 vs 10/10 is significant under Fisher's exact test;
  - citation correctness 0.9732 vs 1.0000 is not distinguishable from noise.

Both are currently written in the same register. This script supplies the
interval and the test so the distinction can be stated.

No SciPy dependency: Fisher's exact test and the Wilson interval are computed
directly, so the script runs wherever the rest of the pipeline runs.

Inputs
------
One or more `*_auto_eval_results.csv` files produced by
47_evaluate_llm_outputs_v3.py.

Usage
-----
    python src/48_binary_outcome_statistics.py \
        --input outputs/llm_eval/llama3_main_v3_auto_eval_results.csv \
                outputs/llm_eval/gptoss20b_v3_auto_eval_results.csv \
        --baseline full \
        --compare full_required_role

    # all pairwise comparisons against `full`:
    python src/48_binary_outcome_statistics.py --input <files...>
"""

import argparse
import math
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
from common import ensure_dirs

EVAL_DIR = PROJECT_ROOT / "outputs" / "llm_eval"
ensure_dirs(EVAL_DIR)

BINARY_METRICS = [
    "role_separation_strict_uniform",
    "role_separation_strict_published",
    "role_separation_strict_field",
    "uncertainty_compliance",
    "json_recoverable",
]


def wilson_interval(successes, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / denom, (centre + margin) / denom)


def _hypergeom_pmf(a, row1, row2, col1):
    """P(X = a) for a 2x2 table with the given margins."""
    total = row1 + row2
    b = row1 - a
    c = col1 - a
    d = row2 - c
    if min(a, b, c, d) < 0:
        return 0.0
    return (
        math.comb(row1, a) * math.comb(row2, c) / math.comb(total, col1)
    )


def fisher_exact_two_sided(a, b, c, d):
    """
    Two-sided Fisher's exact test for the table [[a, b], [c, d]].

    Returns (odds_ratio, p_value). The p-value sums the probability of every
    table with the same margins whose probability does not exceed the observed
    one, which is the conventional two-sided definition.
    """
    row1, row2 = a + b, c + d
    col1 = a + c
    total = row1 + row2
    if total == 0 or row1 == 0 or row2 == 0 or col1 == 0 or (b + d) == 0:
        return (float("nan"), 1.0)

    p_obs = _hypergeom_pmf(a, row1, row2, col1)
    lo = max(0, col1 - row2)
    hi = min(row1, col1)

    tol = 1e-12
    p_value = sum(
        p for k in range(lo, hi + 1)
        if (p := _hypergeom_pmf(k, row1, row2, col1)) <= p_obs + tol
    )

    if b == 0 or c == 0:
        odds = float("inf") if (a and d) else float("nan")
    else:
        odds = (a * d) / (b * c)

    return (odds, min(1.0, p_value))


def summarise_cell(series):
    """Return (successes, n) over non-null binary values."""
    s = series.dropna()
    return int(s.sum()), int(s.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--baseline", default="full")
    ap.add_argument("--compare", nargs="*", default=None,
                    help="Conditions to test against the baseline. Default: all others.")
    ap.add_argument("--output", default=str(EVAL_DIR / "binary_outcome_statistics.csv"))
    args = ap.parse_args()

    frames = []
    for path in args.input:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing results file: {p}")
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)

    if "model" not in df.columns or "condition" not in df.columns:
        raise ValueError("Input must be a v3 auto_eval_results.csv (needs model and condition columns).")

    # ---------------- proportions with Wilson intervals ---------------------
    print("=" * 94)
    print("PROPORTIONS WITH 95% WILSON INTERVALS")
    print("=" * 94)
    print(f"{'model':<26}{'condition':<24}{'metric':<36}{'k/n':>8}  95% CI")

    prop_rows = []
    for (model, cond), g in df.groupby(["model", "condition"], dropna=False):
        for metric in BINARY_METRICS:
            if metric not in g.columns:
                continue
            k, n = summarise_cell(g[metric])
            if n == 0:
                continue
            lo, hi = wilson_interval(k, n)
            print(f"{str(model)[:24]:<26}{str(cond):<24}{metric:<36}{k:>3}/{n:<4}  [{lo:.3f}, {hi:.3f}]")
            prop_rows.append({
                "model": model, "condition": cond, "metric": metric,
                "successes": k, "n": n, "proportion": k / n,
                "wilson_lo": lo, "wilson_hi": hi,
            })

    # ---------------- Fisher's exact tests vs the baseline ------------------
    print("\n" + "=" * 94)
    print(f"FISHER'S EXACT TESTS (two-sided), baseline = '{args.baseline}'")
    print("=" * 94)
    print(f"{'model':<26}{'metric':<36}{'comparison':<24}{'p':>10}")

    test_rows = []
    for model, g_model in df.groupby("model", dropna=False):
        conds = [c for c in g_model["condition"].dropna().unique() if c != args.baseline]
        if args.compare:
            conds = [c for c in conds if c in args.compare]

        base_g = g_model[g_model["condition"] == args.baseline]
        if base_g.empty:
            print(f"{str(model)[:24]:<26}(baseline condition '{args.baseline}' absent -- skipped)")
            continue

        for metric in BINARY_METRICS:
            if metric not in g_model.columns:
                continue
            a, n1 = summarise_cell(base_g[metric])
            if n1 == 0:
                continue
            b = n1 - a

            for cond in conds:
                cmp_g = g_model[g_model["condition"] == cond]
                c, n2 = summarise_cell(cmp_g[metric])
                if n2 == 0:
                    continue
                d = n2 - c

                odds, p = fisher_exact_two_sided(a, b, c, d)
                label = f"{args.baseline} vs {cond}"
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"{str(model)[:24]:<26}{metric:<36}{label:<24}{p:>10.4g}  {stars}"
                      f"   ({a}/{n1} vs {c}/{n2})")
                test_rows.append({
                    "model": model, "metric": metric,
                    "baseline": args.baseline, "comparison": cond,
                    "baseline_k": a, "baseline_n": n1,
                    "comparison_k": c, "comparison_n": n2,
                    "odds_ratio": odds, "p_value": p, "significance": stars,
                })

    out = Path(args.output)
    pd.DataFrame(prop_rows).to_csv(out.with_name(out.stem + "_proportions.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(out.with_name(out.stem + "_tests.csv"), index=False)
    print(f"\nSaved: {out.with_name(out.stem + '_proportions.csv')}")
    print(f"Saved: {out.with_name(out.stem + '_tests.csv')}")

    print(
        "\nNote. With ten outputs per cell the Wilson interval on a proportion of 1.00 "
        "still runs down to about 0.72, so only large contrasts are separable. Report "
        "the intervals next to the point estimates and state which differences are "
        "tested rather than described."
    )


if __name__ == "__main__":
    main()
