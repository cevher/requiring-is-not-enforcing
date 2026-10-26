"""
08_build_forward_labels.py

Creates snapshot-level forward stress labels for the financial systemic risk project.

Purpose:
- Avoid weak "same-day return explains same-day stress" logic.
- Align network snapshots at date t with future stress windows.
- Prefer external stress proxies where available: VIX, NFCI, OFR FSI if present.

Inputs:
- data_processed/labels/stress_labels.csv
- data_processed/networks/network_metrics.csv

Outputs:
- data_processed/labels/snapshot_forward_labels.csv
- data_processed/labels/forward_label_summary.csv

Important:
This is still a feasibility/research engineering script. For the final paper,
thresholds should be selected inside the training period during temporal validation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import LABEL_DIR, NETWORK_DIR
from common import ensure_dirs

ensure_dirs(LABEL_DIR)

labels_path = LABEL_DIR / "stress_labels.csv"
network_metrics_path = NETWORK_DIR / "network_metrics.csv"

daily = pd.read_csv(labels_path, parse_dates=["date"]).sort_values("date")
metrics = pd.read_csv(network_metrics_path, parse_dates=["date"]).sort_values("date")

snapshot_dates = sorted(metrics["date"].dropna().unique())

# Build external stress proxy from VIX/NFCI/OFR if available.
# VIX high: top 20 percent
# NFCI high: positive or top 20 percent; positive is a common interpretation of tighter-than-average conditions
# OFR high: positive when available

daily = daily.set_index("date").sort_index()

if "vix" in daily.columns and daily["vix"].notna().sum() > 30:
    vix_thr = daily["vix"].quantile(0.80)
    daily["vix_external_stress"] = (daily["vix"] >= vix_thr).astype(int)
else:
    vix_thr = np.nan
    daily["vix_external_stress"] = 0

if "nfci" in daily.columns and daily["nfci"].notna().sum() > 30:
    # positive NFCI means tighter-than-average financial conditions.
    daily["nfci_external_stress"] = (daily["nfci"] > 0).astype(int)
else:
    daily["nfci_external_stress"] = 0

if "ofr_fsi" in daily.columns and daily["ofr_fsi"].notna().sum() > 30:
    daily["ofr_external_stress"] = (daily["ofr_fsi"] > 0).astype(int)
else:
    daily["ofr_external_stress"] = 0

# Combined external stress. If OFR exists, it contributes. If not, VIX/NFCI still work.
daily["external_stress_today"] = (
    (daily["vix_external_stress"] == 1) |
    (daily["nfci_external_stress"] == 1) |
    (daily["ofr_external_stress"] == 1)
).astype(int)

# Market-return based stress retained only as a diagnostic, not primary final label.
if "return_downside_stress" in daily.columns:
    daily["market_return_stress_today"] = daily["return_downside_stress"].fillna(0).astype(int)
else:
    daily["market_return_stress_today"] = 0

records = []

for d in snapshot_dates:
    d = pd.Timestamp(d)
    if d not in daily.index:
        # Use nearest previous daily observation
        prev = daily.index[daily.index <= d]
        if len(prev) == 0:
            continue
        d0 = prev.max()
    else:
        d0 = d

    current_row = daily.loc[d0]

    # Future windows: next 20 trading observations and next 60 trading observations
    future = daily.loc[daily.index > d0]
    next_20 = future.head(20)
    next_60 = future.head(60)

    def any_stress(frame, col):
        if frame.empty or col not in frame.columns:
            return np.nan
        return int(frame[col].max() >= 1)

    def mean_stress(frame, col):
        if frame.empty or col not in frame.columns:
            return np.nan
        return float(frame[col].mean())

    records.append({
        "date": d0,
        "snapshot_date": d,
        "current_external_stress": int(current_row.get("external_stress_today", 0)),
        "current_vix_stress": int(current_row.get("vix_external_stress", 0)),
        "current_nfci_stress": int(current_row.get("nfci_external_stress", 0)),
        "current_ofr_stress": int(current_row.get("ofr_external_stress", 0)),
        "current_market_return_stress": int(current_row.get("market_return_stress_today", 0)),
        "next_20d_external_stress": any_stress(next_20, "external_stress_today"),
        "next_60d_external_stress": any_stress(next_60, "external_stress_today"),
        "next_20d_market_return_stress": any_stress(next_20, "market_return_stress_today"),
        "next_60d_market_return_stress": any_stress(next_60, "market_return_stress_today"),
        "next_20d_external_stress_share": mean_stress(next_20, "external_stress_today"),
        "next_60d_external_stress_share": mean_stress(next_60, "external_stress_today"),
        "vix_threshold_80pct": vix_thr,
    })

out = pd.DataFrame(records)
out.to_csv(LABEL_DIR / "snapshot_forward_labels.csv", index=False)

summary = []
for col in [
    "current_external_stress",
    "next_20d_external_stress",
    "next_60d_external_stress",
    "next_20d_market_return_stress",
    "next_60d_market_return_stress",
]:
    if col in out.columns:
        summary.append({
            "label": col,
            "n": int(out[col].notna().sum()),
            "positive_count": int(out[col].fillna(0).sum()),
            "positive_share": float(out[col].mean(skipna=True)),
        })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(LABEL_DIR / "forward_label_summary.csv", index=False)

print(f"Saved: {LABEL_DIR / 'snapshot_forward_labels.csv'}")
print(f"Saved: {LABEL_DIR / 'forward_label_summary.csv'}")
print(summary_df)
