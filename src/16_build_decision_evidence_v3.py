"""
16_build_decision_evidence_v3.py

Builds decision-level evidence records by combining:
- expanding-window model predictions,
- macro evidence,
- top firm/sector network evidence from separated networks.

This prepares the substrate for a later LLM decision-note generator.

Inputs:
- outputs/reports/expanding_window_predictions.csv
- data_processed/networks/separated/*_metrics.csv
- data_processed/labels/stress_labels.csv

Outputs:
- data_processed/evidence/decision_evidence_v3.csv
- data_processed/evidence/decision_records_v3.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import NETWORK_DIR, LABEL_DIR, EVIDENCE_DIR, REPORT_DIR
from common import ensure_dirs

ensure_dirs(EVIDENCE_DIR)

SEPARATED_DIR = NETWORK_DIR / "separated"

pred_path = REPORT_DIR / "expanding_window_predictions.csv"
if not pred_path.exists():
    raise FileNotFoundError(f"Run 15_expanding_window_backtest.py first. Missing: {pred_path}")

pred = pd.read_csv(pred_path, parse_dates=["date"])
daily = pd.read_csv(LABEL_DIR / "stress_labels.csv", parse_dates=["date"]).sort_values("date")

# Choose a default decision model:
# Prefer next_20d_external_stress, macro_plus_all_networks, random_forest, platt.
preferred = pred[
    (pred["target"] == "next_20d_external_stress") &
    (pred["feature_set"] == "macro_plus_all_networks") &
    (pred["model"] == "random_forest") &
    (pred["calibration"] == "platt")
].copy()

if preferred.empty:
    # Fallback to best available grouped by date using first prediction.
    preferred = pred[pred["target"] == "next_20d_external_stress"].copy()

if preferred.empty:
    raise RuntimeError("No prediction records available for next_20d_external_stress.")

preferred = preferred.sort_values("date").copy()
preferred["decision_id"] = preferred["date"].dt.strftime("DEC-%Y%m%d") + "-" + preferred["feature_set"].astype(str)

records = []
decision_records = []

macro_cols = [c for c in ["vix", "nfci", "term_spread_10y_2y", "high_yield_oas", "ofr_fsi", "universe_market_return"] if c in daily.columns]
daily = daily.sort_values("date")

def risk_level(p):
    if p >= 0.70:
        return "high"
    if p >= 0.40:
        return "moderate"
    return "low"

def abstain_flag(p):
    # uncertainty zone around 0.5
    return int(0.45 <= p <= 0.55)

network_names = ["raw_topk", "raw_threshold", "residual_topk", "tail_topk"]

network_tables = {}
for name in network_names:
    path = SEPARATED_DIR / f"{name}_metrics.csv"
    if path.exists():
        network_tables[name] = pd.read_csv(path, parse_dates=["date"])

for _, row in preferred.iterrows():
    d = row["date"]
    decision_id = row["decision_id"]
    p = float(row["y_prob"])
    rl = risk_level(p)
    abst = abstain_flag(p)

    decision_records.append({
        "decision_id": decision_id,
        "date": d,
        "target": row["target"],
        "feature_set": row["feature_set"],
        "model": row["model"],
        "calibration": row["calibration"],
        "predicted_probability": p,
        "predicted_class": int(row["y_pred"]),
        "actual_label": int(row["y_true"]),
        "risk_level": rl,
        "abstention_flag": abst,
    })

    records.append({
        "decision_id": decision_id,
        "evidence_id": f"PRED-{d:%Y%m%d}-{row['feature_set']}-{row['model']}-{row['calibration']}",
        "date": d,
        "entity_id": "MARKET",
        "entity_type": "decision_output",
        "source_type": "model_prediction",
        "source_name": "expanding_window_backtest",
        "variable_or_text_field": "predicted_probability",
        "value_or_snippet": f"Predicted next-20-day external stress probability={p:.4f}; risk_level={rl}; abstention={abst}.",
        "used_in_decision": "yes",
    })

    # Macro evidence nearest previous day
    tmp = daily[daily["date"] <= d]
    if not tmp.empty:
        mrow = tmp.iloc[-1]
        for col in macro_cols:
            val = mrow.get(col, np.nan)
            if pd.notna(val):
                records.append({
                    "decision_id": decision_id,
                    "evidence_id": f"MACRO-{d:%Y%m%d}-{col}",
                    "date": d,
                    "entity_id": "MARKET",
                    "entity_type": "market",
                    "source_type": "macro_financial_stress",
                    "source_name": "open_macro_stress_data",
                    "variable_or_text_field": col,
                    "value_or_snippet": f"{col}={val}",
                    "used_in_decision": "yes",
                })

    # Network firm evidence
    for net_name, table in network_tables.items():
        g = table[table["date"] == d].copy()
        if g.empty:
            # nearest previous snapshot
            prev_dates = table.loc[table["date"] <= d, "date"]
            if prev_dates.empty:
                continue
            prev = prev_dates.max()
            g = table[table["date"] == prev].copy()

        g = g.sort_values("weighted_degree", ascending=False).head(5)
        for rank, nrow in enumerate(g.itertuples(index=False), start=1):
            records.append({
                "decision_id": decision_id,
                "evidence_id": f"{net_name.upper()}-{pd.Timestamp(d):%Y%m%d}-{nrow.ticker}-rank{rank}",
                "date": d,
                "entity_id": nrow.ticker,
                "entity_type": "firm",
                "source_type": "network_metric",
                "source_name": net_name,
                "variable_or_text_field": "weighted_degree/eigenvector/betweenness",
                "value_or_snippet": (
                    f"{nrow.ticker} ranked {rank} in {net_name}; "
                    f"weighted_degree={nrow.weighted_degree:.4f}; "
                    f"eigenvector={nrow.eigenvector:.4f}; "
                    f"betweenness={nrow.betweenness:.4f}; sector={nrow.sector}."
                ),
                "used_in_decision": "yes",
            })

evidence = pd.DataFrame(records)
decisions = pd.DataFrame(decision_records)

evidence.to_csv(EVIDENCE_DIR / "decision_evidence_v3.csv", index=False)
decisions.to_csv(EVIDENCE_DIR / "decision_records_v3.csv", index=False)

print(f"Saved: {EVIDENCE_DIR / 'decision_evidence_v3.csv'} ({len(evidence)} rows)")
print(f"Saved: {EVIDENCE_DIR / 'decision_records_v3.csv'} ({len(decisions)} rows)")
