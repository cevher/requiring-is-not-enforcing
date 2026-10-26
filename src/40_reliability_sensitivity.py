"""
40_reliability_sensitivity.py

One-at-a-time sensitivity analysis of the reliability layer's design constants.

The reliability layer in 22_build_decision_evidence_v5.py is a rule set with six
hand-set cut-offs. They are design parameters, not estimates, so the paper must
show that its qualitative output does not hinge on their exact values.

Parameters examined:
  conf_high      base confidence = high when |s - 0.5| >= 0.30
  conf_med       base confidence = medium when |s - 0.5| >= 0.15
  gap            disagreement when |s_rule - s_model| >= 0.35
  err_cut        downgrade when historical bin error rate >= 0.35
  brier_cut      downgrade when historical bin Brier >= 0.25
  abstain_band   abstain when s in [0.5 - w, 0.5 + w], w = 0.05
  min_bin_n      a bin with fewer than 10 prior observations is not estimated
                 (evaluated upward only: lowering it would require recomputing
                 bin statistics that were never stored for small bins)

Reported per perturbation:
  abstention rate, disagreement rate, confidence distribution, and agreement
  with the baseline configuration (share of the 144 decision records whose
  abstention flag and confidence class are unchanged), plus the effect on the
  ten snapshots used in the LLM experiment.

Outputs:
  outputs/reports/reliability_sensitivity.csv
"""

import numpy as np
import pandas as pd

from config import EVIDENCE_DIR, REPORT_DIR
from common import ensure_dirs

ensure_dirs(REPORT_DIR)

BASE = dict(conf_high=0.30, conf_med=0.15, gap=0.35,
            err_cut=0.35, brier_cut=0.25, abstain_band=0.05, min_bin_n=10)

ORDER = ["low", "medium", "high"]


def downgrade(c, k=1):
    i = ORDER.index(c) if c in ORDER else 0
    return ORDER[max(0, i - k)]


def evaluate(df, p):
    s = df["decision_score"].astype(float)

    base_conf = np.where(abs(s - 0.5) >= p["conf_high"], "high",
                  np.where(abs(s - 0.5) >= p["conf_med"], "medium", "low"))

    dis = ((df["rule_model_class_disagreement"].fillna(0).astype(int) == 1) |
           (df["rule_model_score_gap"].astype(float) >= p["gap"])).astype(int)

    # reliability_status is recomputed only for the upward min_bin_n direction:
    # a bin that had enough history under the baseline may fall below a higher cut-off.
    n = df["historical_bin_n"].fillna(0).astype(float)
    estimated = (df["reliability_status"] == "estimated") & (n >= p["min_bin_n"])

    err = df["historical_bin_error_rate"].astype(float)
    bri = df["historical_bin_brier"].astype(float)

    conf = []
    for i in range(len(df)):
        c = base_conf[i]
        if dis.iloc[i] == 1:
            c = downgrade(c)
        if not estimated.iloc[i]:
            c = downgrade(c)
        else:
            if pd.notna(err.iloc[i]) and err.iloc[i] >= p["err_cut"]:
                c = downgrade(c)
            if pd.notna(bri.iloc[i]) and bri.iloc[i] >= p["brier_cut"]:
                c = downgrade(c)
        conf.append(c)
    conf = pd.Series(conf, index=df.index)

    w = p["abstain_band"]
    abst = (((s >= 0.5 - w) & (s <= 0.5 + w)) |
            ((dis == 1) & (conf == "low"))).astype(int)

    return pd.DataFrame({"confidence": conf, "abstention": abst, "disagreement": dis})


df = pd.read_csv(EVIDENCE_DIR / "decision_records_v5.csv", parse_dates=["date"])
sample = pd.read_csv(EVIDENCE_DIR / "llm_eval_selected_decisions.csv")
sample_ids = set(sample["decision_id"])
in_sample = df["decision_id"].isin(sample_ids)

base = evaluate(df, BASE)

# Validation: the reimplementation must reproduce the stored layer.
match_conf = (base["confidence"] == df["reliability_adjusted_confidence"]).mean()
match_abst = (base["abstention"] == df["abstention_flag"]).mean()
print(f"Reimplementation check vs stored decision layer: "
      f"confidence {match_conf:.3f}, abstention {match_abst:.3f}")
if min(match_conf, match_abst) < 0.99:
    print("  WARNING: the reimplementation does not reproduce the stored layer; "
          "the sensitivity table below is not comparable to the published records.")

GRID = {
    "conf_high": [0.25, 0.30, 0.35],
    "conf_med": [0.10, 0.15, 0.20],
    "gap": [0.30, 0.35, 0.40],
    "err_cut": [0.30, 0.35, 0.40],
    "brier_cut": [0.20, 0.25, 0.30],
    "abstain_band": [0.025, 0.05, 0.075],
    "min_bin_n": [10, 15, 20],
}

rows = []
for param, values in GRID.items():
    for v in values:
        p = dict(BASE)
        p[param] = v
        r = evaluate(df, p)
        rows.append({
            "parameter": param,
            "value": v,
            "is_baseline": v == BASE[param],
            "abstention_rate": round(float(r["abstention"].mean()), 4),
            "disagreement_rate": round(float(r["disagreement"].mean()), 4),
            "conf_low": round(float((r["confidence"] == "low").mean()), 4),
            "conf_medium": round(float((r["confidence"] == "medium").mean()), 4),
            "conf_high": round(float((r["confidence"] == "high").mean()), 4),
            "abstention_agreement_with_base": round(float((r["abstention"] == base["abstention"]).mean()), 4),
            "confidence_agreement_with_base": round(float((r["confidence"] == base["confidence"]).mean()), 4),
            "llm_sample_flags_changed": int(
                ((r.loc[in_sample, "abstention"] != base.loc[in_sample, "abstention"]) |
                 (r.loc[in_sample, "confidence"] != base.loc[in_sample, "confidence"])).sum()),
        })

out = pd.DataFrame(rows)
out.to_csv(REPORT_DIR / "reliability_sensitivity.csv", index=False)
print("\n" + out.to_string(index=False))

pert = out[~out["is_baseline"]]
print("\nSummary over all non-baseline perturbations:")
print(f"  abstention rate range      : {pert['abstention_rate'].min():.4f} - {pert['abstention_rate'].max():.4f} "
      f"(baseline {base['abstention'].mean():.4f})")
print(f"  min confidence agreement   : {pert['confidence_agreement_with_base'].min():.4f}")
print(f"  min abstention agreement   : {pert['abstention_agreement_with_base'].min():.4f}")
print(f"  max LLM-sample flag changes: {pert['llm_sample_flags_changed'].max()} of 10")
