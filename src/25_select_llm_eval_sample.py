"""
25_select_llm_eval_sample.py

Selects a stratified decision sample for LLM decision-note evaluation.

Input:
- data_processed/evidence/decision_records_v5.csv

Output:
- data_processed/evidence/llm_eval_selected_decisions.csv
"""

import argparse
import pandas as pd
import numpy as np

from config import EVIDENCE_DIR
from common import ensure_dirs

ensure_dirs(EVIDENCE_DIR)


def pick_group(df, mask, n, label, used_ids):
    g = df[mask].copy()
    g = g[~g["decision_id"].isin(used_ids)]
    if g.empty:
        return pd.DataFrame(columns=df.columns)

    g = g.sort_values("date")
    if len(g) <= n:
        out = g.copy()
    else:
        idx = np.linspace(0, len(g) - 1, n).round().astype(int)
        out = g.iloc[idx].copy()

    out["sample_reason"] = label
    used_ids.update(out["decision_id"].tolist())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-size", type=int, default=40)
    parser.add_argument("--input", default=str(EVIDENCE_DIR / "decision_records_v5.csv"))
    parser.add_argument("--output", default=str(EVIDENCE_DIR / "llm_eval_selected_decisions.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["date"]).sort_values("date")

    required = {"decision_id", "date", "error_type", "disagreement_flag", "abstention_flag", "risk_level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in decision records: {missing}")

    used = set()
    picks = []

    group_plan = [
        ("true_positive", df["error_type"].eq("true_positive"), 6),
        ("true_negative", df["error_type"].eq("true_negative"), 6),
        ("false_positive", df["error_type"].eq("false_positive"), 6),
        ("false_negative", df["error_type"].eq("false_negative"), 6),
        ("disagreement", df["disagreement_flag"].eq(1), 6),
        ("abstention", df["abstention_flag"].eq(1), 6),
        ("low_risk", df["risk_level"].eq("low"), 3),
        ("moderate_risk", df["risk_level"].eq("moderate"), 3),
        ("high_risk", df["risk_level"].eq("high"), 3),
    ]

    for label, mask, n in group_plan:
        picks.append(pick_group(df, mask, n, label, used))

    selected = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame(columns=df.columns)

    if len(selected) < args.target_size:
        remaining = df[~df["decision_id"].isin(used)].sort_values("date")
        need = args.target_size - len(selected)
        if not remaining.empty:
            if len(remaining) <= need:
                extra = remaining.copy()
            else:
                idx = np.linspace(0, len(remaining) - 1, need).round().astype(int)
                extra = remaining.iloc[idx].copy()
            extra["sample_reason"] = "chronological_fill"
            selected = pd.concat([selected, extra], ignore_index=True)

    priority = {
        "false_positive": 1,
        "false_negative": 1,
        "disagreement": 2,
        "abstention": 2,
        "true_positive": 3,
        "true_negative": 3,
        "high_risk": 4,
        "moderate_risk": 4,
        "low_risk": 4,
        "chronological_fill": 5,
    }
    selected["priority"] = selected["sample_reason"].map(priority).fillna(9)
    selected = selected.sort_values(["priority", "date"]).drop_duplicates("decision_id").head(args.target_size)
    selected = selected.sort_values("date")

    selected.to_csv(args.output, index=False)
    summary = selected["sample_reason"].value_counts().reset_index()
    summary.columns = ["sample_reason", "n"]

    print(f"Saved selected decisions: {args.output}")
    print(f"n_selected={len(selected)}")
    print(summary)


if __name__ == "__main__":
    main()
