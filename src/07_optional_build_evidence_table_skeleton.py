"""Create initial provenance/evidence table skeleton from market, network, and label outputs."""
import pandas as pd
from config import NETWORK_DIR, LABEL_DIR, EVIDENCE_DIR
from common import ensure_dirs

ensure_dirs(EVIDENCE_DIR)
metrics = pd.read_csv(NETWORK_DIR / "network_metrics.csv", parse_dates=["date"])
labels = pd.read_csv(LABEL_DIR / "stress_labels.csv", parse_dates=["date"])
records = []

for _, row in metrics.iterrows():
    date = row["date"]
    ticker = row["ticker"]
    for field in ["weighted_degree", "eigenvector", "betweenness", "clustering", "network_density"]:
        records.append({
            "evidence_id": f"NET-{date:%Y%m%d}-{ticker}-{field}",
            "date": date,
            "entity_id": ticker,
            "entity_type": "firm",
            "source_type": "network_metric",
            "source_name": "rolling_correlation_network",
            "variable_or_text_field": field,
            "value_or_snippet": row.get(field),
            "timestamp": date,
            "used_in_decision": "",
            "decision_id": ""
        })

for _, row in labels.iterrows():
    date = row["date"]
    for field in ["vix", "high_yield_oas", "term_spread_10y_2y", "high_stress_label"]:
        if field in row and pd.notna(row[field]):
            records.append({
                "evidence_id": f"MACRO-{date:%Y%m%d}-{field}",
                "date": date,
                "entity_id": "MARKET",
                "entity_type": "market",
                "source_type": "macro_stress",
                "source_name": "FRED_or_user_provided_stress_data",
                "variable_or_text_field": field,
                "value_or_snippet": row[field],
                "timestamp": date,
                "used_in_decision": "",
                "decision_id": ""
            })

pd.DataFrame(records).to_csv(EVIDENCE_DIR / "evidence_table_skeleton.csv", index=False)
print(f"Saved evidence table skeleton: {EVIDENCE_DIR / 'evidence_table_skeleton.csv'}")
