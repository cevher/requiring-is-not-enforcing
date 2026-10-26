"""
04_build_labels.py

Builds daily stress labels from VIX/NFCI/OFR if available and a market-return diagnostic.

2005 revision:
- Equal-weighted market return is computed only when enough tickers are active.
- Market-return stress is retained as diagnostic, not as the primary final target.
"""

import numpy as np
import pandas as pd

from config import (
    MACRO_RAW_DIR, RETURNS_DIR, LABEL_DIR,
    VIX_HIGH_QUANTILE, RETURN_STRESS_QUANTILE,
    MIN_MARKET_TICKERS_PER_DAY
)
from common import ensure_dirs

ensure_dirs(LABEL_DIR)

macro = pd.read_csv(MACRO_RAW_DIR / "fred_stress_data.csv", parse_dates=["date"]).set_index("date").sort_index()
returns = pd.read_csv(RETURNS_DIR / "daily_log_returns.csv", parse_dates=["date"]).set_index("date").sort_index()

active_counts = returns.notna().sum(axis=1)
market_return = returns.mean(axis=1, skipna=True)
market_return = market_return.where(active_counts >= MIN_MARKET_TICKERS_PER_DAY)
market_return = market_return.rename("universe_market_return")

df = macro.join(market_return, how="outer").sort_index().ffill()
df["active_tickers_for_market_return"] = active_counts.reindex(df.index).ffill()

df = df.dropna(subset=["universe_market_return"], how="all")

if "vix" in df.columns and df["vix"].notna().sum() > 20:
    vix_thr = df["vix"].quantile(VIX_HIGH_QUANTILE)
    df["vix_high_stress"] = (df["vix"] >= vix_thr).astype(int)
else:
    vix_thr = np.nan
    df["vix_high_stress"] = 0

ret_thr = df["universe_market_return"].quantile(RETURN_STRESS_QUANTILE)
df["return_downside_stress"] = (df["universe_market_return"] <= ret_thr).astype(int)

if "ofr_fsi" in df.columns and df["ofr_fsi"].notna().sum() > 20:
    df["ofr_high_stress"] = (df["ofr_fsi"] > 0).astype(int)
else:
    df["ofr_high_stress"] = 0

if "nfci" in df.columns and df["nfci"].notna().sum() > 20:
    df["nfci_high_stress"] = (df["nfci"] > 0).astype(int)
else:
    df["nfci_high_stress"] = 0

df["external_stress_label"] = (
    (df["vix_high_stress"] == 1) |
    (df["nfci_high_stress"] == 1) |
    (df["ofr_high_stress"] == 1)
).astype(int)

# Legacy label kept for continuity. Do not use as the primary final target.
df["high_stress_label"] = (
    (df["vix_high_stress"] == 1) |
    (df["return_downside_stress"] == 1) |
    (df["ofr_high_stress"] == 1)
).astype(int)

out = LABEL_DIR / "stress_labels.csv"
df.reset_index().to_csv(out, index=False)

summary = {
    "vix_threshold": vix_thr,
    "return_downside_threshold": ret_thr,
    "total_days": int(df.shape[0]),
    "external_stress_days": int(df["external_stress_label"].sum()),
    "external_stress_share": float(df["external_stress_label"].mean()),
    "legacy_high_stress_days": int(df["high_stress_label"].sum()),
    "legacy_high_stress_share": float(df["high_stress_label"].mean()),
    "mean_active_tickers_for_market_return": float(df["active_tickers_for_market_return"].mean()),
}
pd.Series(summary).to_csv(LABEL_DIR / "stress_label_summary.csv")

print(f"Saved stress labels: {out}")
print(summary)
