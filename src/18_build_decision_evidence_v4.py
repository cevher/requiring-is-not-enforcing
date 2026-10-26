"""
18_build_decision_evidence_v4.py

Builds the corrected decision-evidence layer.

Design correction:
- Networks are NOT forced into the primary risk scorer.
- Primary systemic-stress score comes from the best calibrated/scoring model selected
  from expanding-window results, by default:
      target = next_20d_external_stress
      feature_set = macro_only
      model = logistic
      calibration = raw

- Networks are used as relational/provenance evidence:
      raw_threshold  -> systemic co-movement intensity
      raw_topk       -> stable firm/sector centrality
      residual_topk  -> market-adjusted relational evidence
      tail_topk      -> downside co-movement evidence

Inputs:
- outputs/reports/expanding_window_predictions.csv
- outputs/reports/expanding_window_results.csv
- outputs/reports/rule_baseline_results.csv, optional
- data_processed/networks/separated/*_metrics.csv
- data_processed/labels/stress_labels.csv

Outputs:
- data_processed/evidence/decision_records_v4.csv
- data_processed/evidence/decision_evidence_v4.csv
- outputs/reports/decision_layer_v4_summary.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import NETWORK_DIR, LABEL_DIR, EVIDENCE_DIR, REPORT_DIR
from common import ensure_dirs

ensure_dirs(EVIDENCE_DIR, REPORT_DIR)

SEPARATED_DIR = NETWORK_DIR / "separated"

TARGET = "next_20d_external_stress"
PREFERRED_FEATURE_SET = "macro_only"
PREFERRED_MODEL = "logistic"
PREFERRED_CALIBRATION = "raw"

NETWORKS_FOR_EVIDENCE = [
    "raw_threshold",
    "raw_topk",
    "residual_topk",
    "tail_topk",
]

TOP_FIRMS_PER_NETWORK = 5
TOP_SECTORS_PER_NETWORK = 3


def risk_level(p: float):
    if p >= 0.70:
        return "high"
    if p >= 0.40:
        return "moderate"
    return "low"


def confidence_band(p: float):
    if p <= 0.20 or p >= 0.80:
        return "high_confidence"
    if p <= 0.35 or p >= 0.65:
        return "medium_confidence"
    return "low_confidence"


def abstain_flag(p: float):
    # uncertainty zone around 0.5
    return int(0.45 <= p <= 0.55)


def error_type(y_true, y_pred):
    if y_true == 1 and y_pred == 1:
        return "true_positive"
    if y_true == 0 and y_pred == 0:
        return "true_negative"
    if y_true == 0 and y_pred == 1:
        return "false_positive"
    if y_true == 1 and y_pred == 0:
        return "false_negative"
    return "unknown"


def nearest_previous_rows(table, date):
    g = table[table["date"] == date].copy()
    if not g.empty:
        return g

    prev = table.loc[table["date"] <= date, "date"]
    if prev.empty:
        return pd.DataFrame(columns=table.columns)

    d0 = prev.max()
    return table[table["date"] == d0].copy()


def select_prediction(pred):
    preferred = pred[
        (pred["target"] == TARGET) &
        (pred["feature_set"] == PREFERRED_FEATURE_SET) &
        (pred["model"] == PREFERRED_MODEL) &
        (pred["calibration"] == PREFERRED_CALIBRATION)
    ].copy()

    if not preferred.empty:
        return preferred, {
            "selection_mode": "preferred_default",
            "target": TARGET,
            "feature_set": PREFERRED_FEATURE_SET,
            "model": PREFERRED_MODEL,
            "calibration": PREFERRED_CALIBRATION,
        }

    # Fallback: choose best Brier among available results for target
    results_path = REPORT_DIR / "expanding_window_results.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
        subset = results[results["target"] == TARGET].sort_values(["brier", "ece_10_bins"])
        if not subset.empty:
            r = subset.iloc[0]
            chosen = pred[
                (pred["target"] == r["target"]) &
                (pred["feature_set"] == r["feature_set"]) &
                (pred["model"] == r["model"]) &
                (pred["calibration"] == r["calibration"])
            ].copy()
            if not chosen.empty:
                return chosen, {
                    "selection_mode": "fallback_best_brier",
                    "target": r["target"],
                    "feature_set": r["feature_set"],
                    "model": r["model"],
                    "calibration": r["calibration"],
                }

    raise RuntimeError("Could not select prediction records for decision evidence.")


pred_path = REPORT_DIR / "expanding_window_predictions.csv"
if not pred_path.exists():
    raise FileNotFoundError(f"Missing {pred_path}. Run 15_expanding_window_backtest.py first.")

pred = pd.read_csv(pred_path, parse_dates=["date"])
selected_pred, selection_info = select_prediction(pred)
selected_pred = selected_pred.sort_values("date").copy()

daily = pd.read_csv(LABEL_DIR / "stress_labels.csv", parse_dates=["date"]).sort_values("date")
macro_cols = [
    c for c in [
        "vix", "nfci", "term_spread_10y_2y", "high_yield_oas",
        "ofr_fsi", "universe_market_return", "external_stress_label"
    ]
    if c in daily.columns
]

network_tables = {}
for net in NETWORKS_FOR_EVIDENCE:
    path = SEPARATED_DIR / f"{net}_metrics.csv"
    if path.exists():
        network_tables[net] = pd.read_csv(path, parse_dates=["date"])
    else:
        print(f"Warning: missing network table: {path}")

decision_records = []
evidence_records = []

for _, row in selected_pred.iterrows():
    d = pd.Timestamp(row["date"])
    p = float(row["y_prob"])
    y_pred = int(row["y_pred"])
    y_true = int(row["y_true"])
    rl = risk_level(p)
    cb = confidence_band(p)
    abst = abstain_flag(p)
    et = error_type(y_true, y_pred)

    decision_id = f"DEC-{d:%Y%m%d}-{PREFERRED_FEATURE_SET}-{PREFERRED_MODEL}-{PREFERRED_CALIBRATION}"

    decision_records.append({
        "decision_id": decision_id,
        "date": d,
        "target": row["target"],
        "risk_score_source": "macro_financial_calibrated_scoring_layer",
        "feature_set": row["feature_set"],
        "model": row["model"],
        "calibration": row["calibration"],
        "predicted_probability": p,
        "predicted_class": y_pred,
        "actual_label": y_true,
        "risk_level": rl,
        "confidence_band": cb,
        "abstention_flag": abst,
        "error_type": et,
        "network_role": "relational_provenance_and_explanation_not_primary_prediction",
    })

    evidence_records.append({
        "decision_id": decision_id,
        "evidence_id": f"PRED-{d:%Y%m%d}-{row['feature_set']}-{row['model']}-{row['calibration']}",
        "date": d,
        "entity_id": "MARKET",
        "entity_type": "decision_output",
        "source_type": "calibrated_risk_score",
        "source_name": "expanding_window_macro_financial_model",
        "variable_or_text_field": "predicted_probability/risk_level/confidence_band",
        "value_or_snippet": (
            f"Predicted next-20-day external stress probability={p:.4f}; "
            f"predicted_class={y_pred}; risk_level={rl}; "
            f"confidence_band={cb}; abstention_flag={abst}."
        ),
        "used_in_decision": "yes",
        "evidence_role": "risk_score",
    })

    # Macro evidence
    tmp = daily[daily["date"] <= d]
    if not tmp.empty:
        mrow = tmp.iloc[-1]
        for col in macro_cols:
            val = mrow.get(col, np.nan)
            if pd.isna(val):
                continue
            evidence_records.append({
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
                "evidence_role": "risk_score_input",
            })

    # Network evidence
    for net_name, table in network_tables.items():
        g = nearest_previous_rows(table, d)
        if g.empty:
            continue

        # Date-level summary evidence
        density = g["network_density"].dropna().iloc[0] if "network_density" in g.columns and g["network_density"].notna().any() else np.nan
        edges = g["network_edges"].dropna().iloc[0] if "network_edges" in g.columns and g["network_edges"].notna().any() else np.nan
        nodes = g["network_nodes"].dropna().iloc[0] if "network_nodes" in g.columns and g["network_nodes"].notna().any() else np.nan

        evidence_records.append({
            "decision_id": decision_id,
            "evidence_id": f"NETSUMMARY-{net_name.upper()}-{d:%Y%m%d}",
            "date": d,
            "entity_id": "MARKET_NETWORK",
            "entity_type": "network",
            "source_type": "financial_network_summary",
            "source_name": net_name,
            "variable_or_text_field": "network_density/network_edges/network_nodes",
            "value_or_snippet": (
                f"{net_name}: density={density}; edges={edges}; nodes={nodes}. "
                f"Used as relational evidence, not as the primary risk-score driver."
            ),
            "used_in_decision": "yes",
            "evidence_role": "relational_context",
        })

        # Top firms
        if "weighted_degree" in g.columns:
            top_firms = g.sort_values("weighted_degree", ascending=False).head(TOP_FIRMS_PER_NETWORK)
            for rank, nrow in enumerate(top_firms.itertuples(index=False), start=1):
                evidence_records.append({
                    "decision_id": decision_id,
                    "evidence_id": f"FIRM-{net_name.upper()}-{d:%Y%m%d}-{nrow.ticker}-rank{rank}",
                    "date": d,
                    "entity_id": nrow.ticker,
                    "entity_type": "firm",
                    "source_type": "network_centrality",
                    "source_name": net_name,
                    "variable_or_text_field": "weighted_degree/eigenvector/betweenness",
                    "value_or_snippet": (
                        f"{nrow.ticker} ranked {rank} in {net_name}; "
                        f"weighted_degree={nrow.weighted_degree:.4f}; "
                        f"eigenvector={nrow.eigenvector:.4f}; "
                        f"betweenness={nrow.betweenness:.4f}; sector={nrow.sector}."
                    ),
                    "used_in_decision": "yes",
                    "evidence_role": "firm_level_relational_attribution",
                })

            # Top sectors
            sector = (
                g.groupby("sector", as_index=False)
                .agg(
                    sector_mean_weighted_degree=("weighted_degree", "mean"),
                    sector_max_weighted_degree=("weighted_degree", "max"),
                    n_firms=("ticker", "nunique"),
                )
                .sort_values("sector_mean_weighted_degree", ascending=False)
                .head(TOP_SECTORS_PER_NETWORK)
            )
            for rank, srow in enumerate(sector.itertuples(index=False), start=1):
                evidence_records.append({
                    "decision_id": decision_id,
                    "evidence_id": f"SECTOR-{net_name.upper()}-{d:%Y%m%d}-{str(srow.sector).replace(' ', '_')}-rank{rank}",
                    "date": d,
                    "entity_id": srow.sector,
                    "entity_type": "sector",
                    "source_type": "sector_network_centrality",
                    "source_name": net_name,
                    "variable_or_text_field": "sector_mean_weighted_degree/sector_max_weighted_degree",
                    "value_or_snippet": (
                        f"{srow.sector} ranked {rank} by average weighted degree in {net_name}; "
                        f"mean_degree={srow.sector_mean_weighted_degree:.4f}; "
                        f"max_degree={srow.sector_max_weighted_degree:.4f}; n_firms={srow.n_firms}."
                    ),
                    "used_in_decision": "yes",
                    "evidence_role": "sector_level_relational_attribution",
                })

decisions = pd.DataFrame(decision_records)
evidence = pd.DataFrame(evidence_records)

decision_path = EVIDENCE_DIR / "decision_records_v4.csv"
evidence_path = EVIDENCE_DIR / "decision_evidence_v4.csv"

decisions.to_csv(decision_path, index=False)
evidence.to_csv(evidence_path, index=False)

summary = {
    "selected_prediction": selection_info,
    "n_decisions": int(len(decisions)),
    "n_evidence_records": int(len(evidence)),
    "mean_evidence_per_decision": float(len(evidence) / max(len(decisions), 1)),
}

summary_df = (
    evidence
    .groupby(["source_type", "evidence_role"], as_index=False)
    .size()
    .rename(columns={"size": "n_records"})
)

summary_out = REPORT_DIR / "decision_layer_v4_summary.csv"
summary_df.to_csv(summary_out, index=False)

meta_out = REPORT_DIR / "decision_layer_v4_selection_metadata.txt"
meta_out.write_text(str(summary), encoding="utf-8")

print(f"Saved: {decision_path} ({len(decisions)} rows)")
print(f"Saved: {evidence_path} ({len(evidence)} rows)")
print(f"Saved: {summary_out}")
print(f"Saved: {meta_out}")
print(summary_df)
