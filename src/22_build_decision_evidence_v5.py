"""
22_build_decision_evidence_v5.py

HOTFIX v1:
- Replaces deprecated/removed pandas fillna(method="ffill") usage with ffill().
- Improves y_true handling after merging rule and model predictions.
- Keeps the original v5 design:
  rule/model dual score, disagreement flags, reliability-adjusted confidence,
  compact evidence selection, and NaN-safe evidence text.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import NETWORK_DIR, LABEL_DIR, EVIDENCE_DIR, REPORT_DIR
from common import ensure_dirs

ensure_dirs(EVIDENCE_DIR, REPORT_DIR)

SEPARATED_DIR = NETWORK_DIR / "separated"

TARGET = "next_20d_external_stress"
RULE_BASELINE = "vix_scaled_score"
MODEL_FEATURE_SET = "macro_only"
MODEL_NAME = "logistic"
MODEL_CALIBRATION = "raw"

NETWORKS_FOR_EVIDENCE = [
    "raw_threshold",
    "raw_topk",
    "residual_topk",
    "tail_topk",
]

TOP_FIRMS_TOTAL = 5
TOP_SECTORS_TOTAL = 3
MAX_NETWORK_SUMMARIES = 3

BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]


def prob_bin(p):
    if pd.isna(p):
        return "missing"
    for i in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[i] <= p < BIN_EDGES[i + 1]:
            return f"{BIN_EDGES[i]:.1f}-{BIN_EDGES[i+1]:.1f}"
    return "missing"


def risk_level(p):
    if pd.isna(p):
        return "unknown"
    if p >= 0.70:
        return "high"
    if p >= 0.40:
        return "moderate"
    return "low"


def base_confidence_from_probability(p):
    if pd.isna(p):
        return "low"
    d = abs(p - 0.5)
    if d >= 0.30:
        return "high"
    if d >= 0.15:
        return "medium"
    return "low"


def downgrade(conf, levels=1):
    order = ["low", "medium", "high"]
    idx = order.index(conf) if conf in order else 0
    idx = max(0, idx - levels)
    return order[idx]


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def fmt_metric(name, value, digits=4):
    val = safe_float(value)
    if val is None:
        return None
    return f"{name}={val:.{digits}f}"


def clean_join(parts):
    return "; ".join([p for p in parts if p is not None and str(p).lower() != "nan"])


def error_type(y_true, y_pred):
    if pd.isna(y_true) or pd.isna(y_pred):
        return "unknown"
    y_true = int(y_true)
    y_pred = int(y_pred)
    if y_true == 1 and y_pred == 1:
        return "true_positive"
    if y_true == 0 and y_pred == 0:
        return "true_negative"
    if y_true == 0 and y_pred == 1:
        return "false_positive"
    if y_true == 1 and y_pred == 0:
        return "false_negative"
    return "unknown"


def historical_reliability(pred_history, current_date, score_col, y_col="y_true", min_n=10):
    p = pred_history.loc[pred_history["date"] == current_date, score_col]
    if p.empty or pd.isna(p.iloc[0]):
        return {
            "bin": "missing",
            "historical_bin_n": 0,
            "historical_bin_accuracy": np.nan,
            "historical_bin_error_rate": np.nan,
            "historical_bin_brier": np.nan,
            "reliability_status": "missing_score",
        }

    score = float(p.iloc[0])
    b = prob_bin(score)

    hist = pred_history[pred_history["date"] < current_date].copy()
    if hist.empty:
        return {
            "bin": b,
            "historical_bin_n": 0,
            "historical_bin_accuracy": np.nan,
            "historical_bin_error_rate": np.nan,
            "historical_bin_brier": np.nan,
            "reliability_status": "insufficient_history",
        }

    hist["bin"] = hist[score_col].apply(prob_bin)
    hbin = hist[hist["bin"] == b].dropna(subset=[score_col, y_col])

    if len(hbin) < min_n:
        return {
            "bin": b,
            "historical_bin_n": int(len(hbin)),
            "historical_bin_accuracy": np.nan,
            "historical_bin_error_rate": np.nan,
            "historical_bin_brier": np.nan,
            "reliability_status": "insufficient_bin_history",
        }

    hbin = hbin.copy()
    hbin["pred"] = (hbin[score_col] >= 0.5).astype(int)
    acc = float((hbin["pred"] == hbin[y_col]).mean())
    err = 1.0 - acc
    brier = float(np.mean((hbin[score_col] - hbin[y_col]) ** 2))

    return {
        "bin": b,
        "historical_bin_n": int(len(hbin)),
        "historical_bin_accuracy": acc,
        "historical_bin_error_rate": err,
        "historical_bin_brier": brier,
        "reliability_status": "estimated",
    }


def reliability_adjusted_confidence(score, disagreement_flag, reliability):
    base = base_confidence_from_probability(score)
    conf = base

    if disagreement_flag == 1:
        conf = downgrade(conf, 1)

    status = reliability.get("reliability_status", "unknown")
    err = reliability.get("historical_bin_error_rate", np.nan)
    brier = reliability.get("historical_bin_brier", np.nan)

    if status != "estimated":
        conf = downgrade(conf, 1)
    else:
        if pd.notna(err) and err >= 0.35:
            conf = downgrade(conf, 1)
        if pd.notna(brier) and brier >= 0.25:
            conf = downgrade(conf, 1)

    return conf


def abstain(score, disagreement_flag, confidence):
    if pd.isna(score):
        return 1
    if 0.45 <= score <= 0.55:
        return 1
    if disagreement_flag == 1 and confidence == "low":
        return 1
    return 0


def nearest_previous_rows(table, date):
    g = table[table["date"] == date].copy()
    if not g.empty:
        return g

    prev = table.loc[table["date"] <= date, "date"]
    if prev.empty:
        return pd.DataFrame(columns=table.columns)

    d0 = prev.max()
    return table[table["date"] == d0].copy()


rule_path = REPORT_DIR / "rule_baseline_predictions.csv"
model_path = REPORT_DIR / "expanding_window_predictions.csv"

if not rule_path.exists():
    raise FileNotFoundError(f"Missing {rule_path}. Run 17_evaluate_rule_baselines.py first.")
if not model_path.exists():
    raise FileNotFoundError(f"Missing {model_path}. Run 15_expanding_window_backtest.py first.")

rule = pd.read_csv(rule_path, parse_dates=["date"])
model = pd.read_csv(model_path, parse_dates=["date"])

rule_sel = rule[
    (rule["target"] == TARGET) &
    (rule["baseline"] == RULE_BASELINE)
][["date", "target", "y_true", "y_prob", "y_pred"]].copy()
rule_sel = rule_sel.rename(columns={"y_prob": "rule_score", "y_pred": "rule_pred", "y_true": "y_true_rule"})

model_sel = model[
    (model["target"] == TARGET) &
    (model["feature_set"] == MODEL_FEATURE_SET) &
    (model["model"] == MODEL_NAME) &
    (model["calibration"] == MODEL_CALIBRATION)
][["date", "target", "y_true", "y_prob", "y_pred"]].copy()
model_sel = model_sel.rename(columns={"y_prob": "model_score", "y_pred": "model_pred", "y_true": "y_true_model"})

dec = pd.merge(
    rule_sel,
    model_sel[["date", "y_true_model", "model_score", "model_pred"]],
    on="date",
    how="outer",
).sort_values("date")

dec["target"] = TARGET
dec["y_true"] = dec["y_true_rule"].combine_first(dec["y_true_model"])

# HOTFIX: do not use fillna(method="ffill") because it fails in some pandas versions.
# In principle y_true should be available from at least one side of the merge.
# If a rare missing value exists, forward/backward fill as a last-resort safety fallback.
dec["y_true"] = dec["y_true"].ffill().bfill()

dec["rule_model_score_gap"] = (dec["rule_score"] - dec["model_score"]).abs()
dec["rule_model_class_disagreement"] = (dec["rule_pred"] != dec["model_pred"]).astype(int)
dec["rule_model_large_gap"] = (dec["rule_model_score_gap"] >= 0.35).astype(int)
dec["disagreement_flag"] = ((dec["rule_model_class_disagreement"] == 1) | (dec["rule_model_large_gap"] == 1)).astype(int)
dec["decision_score"] = dec[["rule_score", "model_score"]].mean(axis=1)
dec["decision_pred"] = (dec["decision_score"] >= 0.5).astype(int)

history = dec[["date", "decision_score", "y_true"]].dropna().copy()

rel_records = []
decision_records = []

for _, row in dec.iterrows():
    d = pd.Timestamp(row["date"])
    rel = historical_reliability(history, d, "decision_score", y_col="y_true", min_n=10)
    score = safe_float(row["decision_score"])
    dis = int(row["disagreement_flag"]) if pd.notna(row["disagreement_flag"]) else 0
    conf = reliability_adjusted_confidence(score, dis, rel)
    abst = abstain(score, dis, conf)

    decision_id = f"DEC-{d:%Y%m%d}-dual-rule-model-v5"

    rel_records.append({
        "decision_id": decision_id,
        "date": d,
        **rel,
    })

    decision_records.append({
        "decision_id": decision_id,
        "date": d,
        "target": TARGET,
        "rule_score_source": RULE_BASELINE,
        "model_score_source": f"{MODEL_FEATURE_SET}|{MODEL_NAME}|{MODEL_CALIBRATION}",
        "rule_score": row.get("rule_score", np.nan),
        "model_score": row.get("model_score", np.nan),
        "decision_score": score,
        "rule_pred": row.get("rule_pred", np.nan),
        "model_pred": row.get("model_pred", np.nan),
        "decision_pred": int(score >= 0.5) if score is not None else np.nan,
        "actual_label": int(row["y_true"]) if pd.notna(row["y_true"]) else np.nan,
        "risk_level": risk_level(score),
        "base_confidence": base_confidence_from_probability(score),
        "reliability_adjusted_confidence": conf,
        "abstention_flag": abst,
        "probability_bin": rel["bin"],
        "historical_bin_n": rel["historical_bin_n"],
        "historical_bin_accuracy": rel["historical_bin_accuracy"],
        "historical_bin_error_rate": rel["historical_bin_error_rate"],
        "historical_bin_brier": rel["historical_bin_brier"],
        "reliability_status": rel["reliability_status"],
        "rule_model_score_gap": row.get("rule_model_score_gap", np.nan),
        "rule_model_class_disagreement": row.get("rule_model_class_disagreement", np.nan),
        "rule_model_large_gap": row.get("rule_model_large_gap", np.nan),
        "disagreement_flag": dis,
        "error_type": error_type(row.get("y_true", np.nan), int(score >= 0.5) if score is not None else np.nan),
        "network_role": "relational_provenance_and_explanation_not_primary_prediction",
    })

decisions = pd.DataFrame(decision_records)
rel_df = pd.DataFrame(rel_records)

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

evidence_records = []

for _, row in decisions.iterrows():
    d = pd.Timestamp(row["date"])
    decision_id = row["decision_id"]

    evidence_records.append({
        "decision_id": decision_id,
        "evidence_id": f"SCORE-{d:%Y%m%d}-dual-rule-model",
        "date": d,
        "entity_id": "MARKET",
        "entity_type": "decision_output",
        "source_type": "dual_risk_score",
        "source_name": "vix_rule_plus_macro_model",
        "variable_or_text_field": "rule_score/model_score/decision_score/confidence/disagreement",
        "value_or_snippet": clean_join([
            fmt_metric("rule_score", row["rule_score"]),
            fmt_metric("model_score", row["model_score"]),
            fmt_metric("decision_score", row["decision_score"]),
            f"risk_level={row['risk_level']}",
            f"reliability_adjusted_confidence={row['reliability_adjusted_confidence']}",
            f"abstention_flag={row['abstention_flag']}",
            f"disagreement_flag={row['disagreement_flag']}",
        ]),
        "used_in_decision": "yes",
        "evidence_role": "risk_score",
    })

    evidence_records.append({
        "decision_id": decision_id,
        "evidence_id": f"RELIABILITY-{d:%Y%m%d}-{row['probability_bin']}",
        "date": d,
        "entity_id": "MODEL_RELIABILITY",
        "entity_type": "calibration",
        "source_type": "historical_oos_reliability",
        "source_name": "previous_out_of_sample_predictions",
        "variable_or_text_field": "historical_bin_accuracy/error_rate/brier",
        "value_or_snippet": clean_join([
            f"probability_bin={row['probability_bin']}",
            f"reliability_status={row['reliability_status']}",
            f"historical_bin_n={row['historical_bin_n']}",
            fmt_metric("historical_bin_accuracy", row["historical_bin_accuracy"]),
            fmt_metric("historical_bin_error_rate", row["historical_bin_error_rate"]),
            fmt_metric("historical_bin_brier", row["historical_bin_brier"]),
        ]),
        "used_in_decision": "yes",
        "evidence_role": "reliability_context",
    })

    tmp = daily[daily["date"] <= d]
    if not tmp.empty:
        mrow = tmp.iloc[-1]
        preferred_macro = ["vix", "nfci", "term_spread_10y_2y", "universe_market_return", "external_stress_label"]
        selected_macro = [c for c in preferred_macro if c in macro_cols]
        for col in selected_macro[:5]:
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

    for net_name in NETWORKS_FOR_EVIDENCE[:MAX_NETWORK_SUMMARIES]:
        if net_name not in network_tables:
            continue
        g = nearest_previous_rows(network_tables[net_name], d)
        if g.empty:
            continue

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
            "value_or_snippet": clean_join([
                f"{net_name}",
                fmt_metric("density", density),
                fmt_metric("edges", edges, digits=0),
                fmt_metric("nodes", nodes, digits=0),
                "role=relational_evidence_not_primary_predictor",
            ]),
            "used_in_decision": "yes",
            "evidence_role": "relational_context",
        })

    firm_candidates = []
    for net_name, table in network_tables.items():
        g = nearest_previous_rows(table, d)
        if g.empty or "weighted_degree" not in g.columns:
            continue
        top = g.sort_values("weighted_degree", ascending=False).head(3).copy()
        top["source_network"] = net_name
        firm_candidates.append(top)

    if firm_candidates:
        firms = pd.concat(firm_candidates, ignore_index=True)
        firms = firms.sort_values("weighted_degree", ascending=False)
        seen = set()
        selected = []
        for _, fr in firms.iterrows():
            key = fr["ticker"]
            if key in seen:
                continue
            selected.append(fr)
            seen.add(key)
            if len(selected) >= TOP_FIRMS_TOTAL:
                break

        for rank, fr in enumerate(selected, start=1):
            parts = [
                f"{fr['ticker']} selected as firm-level relational evidence",
                f"network={fr['source_network']}",
                fmt_metric("weighted_degree", fr.get("weighted_degree")),
                fmt_metric("eigenvector", fr.get("eigenvector")),
                fmt_metric("betweenness", fr.get("betweenness")),
                f"sector={fr.get('sector')}",
            ]
            evidence_records.append({
                "decision_id": decision_id,
                "evidence_id": f"FIRM-{str(fr['source_network']).upper()}-{d:%Y%m%d}-{fr['ticker']}-rank{rank}",
                "date": d,
                "entity_id": fr["ticker"],
                "entity_type": "firm",
                "source_type": "network_centrality",
                "source_name": fr["source_network"],
                "variable_or_text_field": "weighted_degree/eigenvector/betweenness",
                "value_or_snippet": clean_join(parts),
                "used_in_decision": "yes",
                "evidence_role": "firm_level_relational_attribution",
            })

    sector_candidates = []
    for net_name in ["raw_topk", "tail_topk", "residual_topk"]:
        if net_name not in network_tables:
            continue
        g = nearest_previous_rows(network_tables[net_name], d)
        if g.empty or "weighted_degree" not in g.columns:
            continue
        sec = (
            g.groupby("sector", as_index=False)
            .agg(
                sector_mean_weighted_degree=("weighted_degree", "mean"),
                sector_max_weighted_degree=("weighted_degree", "max"),
                n_firms=("ticker", "nunique"),
            )
        )
        sec["source_network"] = net_name
        sector_candidates.append(sec)

    if sector_candidates:
        sectors = pd.concat(sector_candidates, ignore_index=True)
        sectors = sectors.sort_values("sector_mean_weighted_degree", ascending=False)
        seen = set()
        selected_sec = []
        for _, sr in sectors.iterrows():
            key = str(sr["sector"])
            if key in seen:
                continue
            selected_sec.append(sr)
            seen.add(key)
            if len(selected_sec) >= TOP_SECTORS_TOTAL:
                break

        for rank, sr in enumerate(selected_sec, start=1):
            evidence_records.append({
                "decision_id": decision_id,
                "evidence_id": f"SECTOR-{str(sr['source_network']).upper()}-{d:%Y%m%d}-{str(sr['sector']).replace(' ', '_')}-rank{rank}",
                "date": d,
                "entity_id": sr["sector"],
                "entity_type": "sector",
                "source_type": "sector_network_centrality",
                "source_name": sr["source_network"],
                "variable_or_text_field": "sector_mean_weighted_degree/sector_max_weighted_degree",
                "value_or_snippet": clean_join([
                    f"{sr['sector']} selected as sector-level relational evidence",
                    f"network={sr['source_network']}",
                    fmt_metric("mean_degree", sr["sector_mean_weighted_degree"]),
                    fmt_metric("max_degree", sr["sector_max_weighted_degree"]),
                    f"n_firms={int(sr['n_firms'])}",
                ]),
                "used_in_decision": "yes",
                "evidence_role": "sector_level_relational_attribution",
            })

evidence = pd.DataFrame(evidence_records)

decision_path = EVIDENCE_DIR / "decision_records_v5.csv"
evidence_path = EVIDENCE_DIR / "decision_evidence_v5.csv"
reliability_path = REPORT_DIR / "decision_layer_v5_reliability_summary.csv"
summary_path = REPORT_DIR / "decision_layer_v5_summary.csv"

decisions.to_csv(decision_path, index=False)
evidence.to_csv(evidence_path, index=False)
rel_df.to_csv(reliability_path, index=False)

summary = (
    evidence
    .groupby(["source_type", "evidence_role"], as_index=False)
    .size()
    .rename(columns={"size": "n_records"})
)
summary["n_decisions"] = len(decisions)
summary["mean_records_per_decision"] = len(evidence) / max(len(decisions), 1)
summary.to_csv(summary_path, index=False)

print(f"Saved: {decision_path} ({len(decisions)} rows)")
print(f"Saved: {evidence_path} ({len(evidence)} rows)")
print(f"Saved: {reliability_path}")
print(f"Saved: {summary_path}")
print(summary)
