"""
12_build_provenance_evidence_table_v2.py

Builds a richer provenance/evidence table for later LLM decision support.

Each evidence record is a structured, citable item:
- network evidence
- firm-level centrality evidence
- sector-level aggregate evidence
- macro/stress evidence
- forward label record for evaluation

This is not the LLM layer yet. It creates the retrieval/provenance substrate.
"""

import pandas as pd
import numpy as np

from config import NETWORK_DIR, LABEL_DIR, EVIDENCE_DIR, REPORT_DIR
from common import ensure_dirs

ensure_dirs(EVIDENCE_DIR)

metrics = pd.read_csv(NETWORK_DIR / "network_metrics.csv", parse_dates=["date"])
labels = pd.read_csv(LABEL_DIR / "snapshot_forward_labels.csv", parse_dates=["date"])
daily = pd.read_csv(LABEL_DIR / "stress_labels.csv", parse_dates=["date"])

records = []

# Top firm-level network evidence per snapshot
for date, g in metrics.groupby("date"):
    g = g.sort_values("weighted_degree", ascending=False).head(10)

    for rank, row in enumerate(g.itertuples(index=False), start=1):
        records.append({
            "evidence_id": f"FIRMNET-{pd.Timestamp(date):%Y%m%d}-{row.ticker}-rank{rank}",
            "date": date,
            "entity_id": row.ticker,
            "entity_type": "firm",
            "source_type": "rolling_correlation_network",
            "source_name": "open_market_price_network",
            "variable_or_text_field": "weighted_degree/eigenvector/betweenness",
            "value_or_snippet": (
                f"{row.ticker} ranked {rank} by weighted degree. "
                f"weighted_degree={row.weighted_degree:.4f}; "
                f"eigenvector={row.eigenvector:.4f}; "
                f"betweenness={row.betweenness:.4f}; "
                f"sector={row.sector}."
            ),
            "timestamp": date,
            "used_in_decision": "",
            "decision_id": "",
            "provenance_granularity": "firm_snapshot_metric",
        })

# Sector evidence
sector = (
    metrics
    .groupby(["date", "sector"], as_index=False)
    .agg(
        sector_avg_weighted_degree=("weighted_degree", "mean"),
        sector_max_weighted_degree=("weighted_degree", "max"),
        sector_avg_eigenvector=("eigenvector", "mean"),
        n_firms=("ticker", "nunique"),
    )
)

for row in sector.itertuples(index=False):
    records.append({
        "evidence_id": f"SECTORNET-{pd.Timestamp(row.date):%Y%m%d}-{str(row.sector).replace(' ', '_')}",
        "date": row.date,
        "entity_id": row.sector,
        "entity_type": "sector",
        "source_type": "sector_aggregated_network",
        "source_name": "open_market_price_network",
        "variable_or_text_field": "sector_avg_weighted_degree/sector_avg_eigenvector",
        "value_or_snippet": (
            f"Sector {row.sector}: avg_weighted_degree={row.sector_avg_weighted_degree:.4f}; "
            f"max_weighted_degree={row.sector_max_weighted_degree:.4f}; "
            f"avg_eigenvector={row.sector_avg_eigenvector:.4f}; "
            f"n_firms={row.n_firms}."
        ),
        "timestamp": row.date,
        "used_in_decision": "",
        "decision_id": "",
        "provenance_granularity": "sector_snapshot_metric",
    })

# Macro evidence at snapshot dates
daily = daily.sort_values("date")
snapshot_dates = sorted(metrics["date"].dropna().unique())
macro_cols = [c for c in ["vix", "nfci", "high_yield_oas", "term_spread_10y_2y", "treasury_10y", "treasury_2y", "ofr_fsi"] if c in daily.columns]

for d in snapshot_dates:
    d = pd.Timestamp(d)
    tmp = daily[daily["date"] <= d]
    if tmp.empty:
        continue
    row = tmp.iloc[-1]

    for col in macro_cols:
        val = row.get(col, np.nan)
        if pd.isna(val):
            continue
        records.append({
            "evidence_id": f"MACRO-{d:%Y%m%d}-{col}",
            "date": d,
            "entity_id": "MARKET",
            "entity_type": "market",
            "source_type": "macro_financial_stress",
            "source_name": "FRED_or_open_stress_source",
            "variable_or_text_field": col,
            "value_or_snippet": f"{col}={val}",
            "timestamp": d,
            "used_in_decision": "",
            "decision_id": "",
            "provenance_granularity": "market_snapshot_indicator",
        })

# Forward label evidence, for evaluation only
for row in labels.itertuples(index=False):
    records.append({
        "evidence_id": f"LABEL-{pd.Timestamp(row.date):%Y%m%d}-next20external",
        "date": row.date,
        "entity_id": "MARKET",
        "entity_type": "market",
        "source_type": "evaluation_label",
        "source_name": "external_stress_forward_label",
        "variable_or_text_field": "next_20d_external_stress",
        "value_or_snippet": f"next_20d_external_stress={getattr(row, 'next_20d_external_stress', '')}",
        "timestamp": row.date,
        "used_in_decision": "no_training_target_only",
        "decision_id": "",
        "provenance_granularity": "evaluation_label",
    })

evidence = pd.DataFrame(records)
out = EVIDENCE_DIR / "evidence_table_v2.csv"
evidence.to_csv(out, index=False)

summary = (
    evidence
    .groupby(["source_type", "entity_type"], as_index=False)
    .size()
    .rename(columns={"size": "n_records"})
)
summary.to_csv(REPORT_DIR / "evidence_table_v2_summary.csv", index=False)

print(f"Saved: {out} ({len(evidence)} rows)")
print(f"Saved: {REPORT_DIR / 'evidence_table_v2_summary.csv'}")
print(summary)
