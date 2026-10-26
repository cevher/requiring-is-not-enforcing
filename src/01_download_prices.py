"""
01_download_prices.py

Downloads open market price data through yfinance and creates daily log returns.

2005 revision:
- START_DATE is controlled from config.py.
- Does NOT globally drop tickers that lack full-period coverage.
- Keeps the return matrix with dynamic availability.
- Rolling-network scripts decide which tickers are valid inside each window.
- Saves ticker coverage diagnostics for transparency.

This matters because a 2005 start date would otherwise drop firms that have
shorter histories even if they are usable in later rolling windows.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    START_DATE, END_DATE, UNIVERSE_PATH,
    PRICE_RAW_DIR, RETURNS_DIR
)
from common import ensure_dirs, load_universe

ensure_dirs(PRICE_RAW_DIR, RETURNS_DIR)

universe = load_universe(UNIVERSE_PATH)
tickers = universe["ticker"].dropna().unique().tolist()

print(f"Downloading {len(tickers)} tickers from {START_DATE} to {END_DATE}...")

data = yf.download(
    tickers=tickers,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
    group_by="ticker",
    threads=True,
    progress=True,
)

if data.empty:
    raise RuntimeError("No price data downloaded. Check internet connection, tickers, or yfinance availability.")

raw_path = PRICE_RAW_DIR / "yfinance_raw_prices.parquet"
data.to_parquet(raw_path)
print(f"Saved raw price panel: {raw_path}")

records = []

if isinstance(data.columns, pd.MultiIndex):
    available_tickers = sorted(set(data.columns.get_level_values(0)))
    for t in tickers:
        if t not in available_tickers:
            print(f"Warning: {t} missing from downloaded data.")
            continue
        sub = data[t].copy()
        sub["ticker"] = t
        sub["date"] = sub.index
        records.append(sub.reset_index(drop=True))
else:
    # Single ticker fallback
    t = tickers[0]
    sub = data.copy()
    sub["ticker"] = t
    sub["date"] = sub.index
    records.append(sub.reset_index(drop=True))

if not records:
    raise RuntimeError("No ticker-level records could be constructed.")

prices_long = pd.concat(records, ignore_index=True)
prices_long.columns = [str(c).lower().replace(" ", "_") for c in prices_long.columns]

if "adj_close" not in prices_long.columns and "adjusted_close" in prices_long.columns:
    prices_long = prices_long.rename(columns={"adjusted_close": "adj_close"})

required_cols = {"date", "ticker", "close", "volume"}
missing = required_cols - set(prices_long.columns)
if missing:
    raise RuntimeError(f"Downloaded price data missing required columns: {missing}")

price_col = "adj_close" if "adj_close" in prices_long.columns else "close"

prices_long = prices_long.sort_values(["ticker", "date"])
prices_long = prices_long.dropna(subset=[price_col])

prices_long["log_price"] = np.log(prices_long[price_col])
prices_long["log_return"] = prices_long.groupby("ticker")["log_price"].diff()
prices_long["simple_return"] = prices_long.groupby("ticker")[price_col].pct_change()
prices_long["dollar_volume"] = prices_long[price_col] * prices_long["volume"]

prices_long = prices_long.merge(universe, on="ticker", how="left")

prices_csv = PRICE_RAW_DIR / "prices_long.csv"
prices_long.to_csv(prices_csv, index=False)
print(f"Saved long price file: {prices_csv}")

returns = prices_long.pivot(index="date", columns="ticker", values="log_return").sort_index()
returns = returns.dropna(how="all")

# IMPORTANT:
# Do not globally drop tickers by full-period coverage.
# Dynamic rolling-window scripts will filter tickers per window.
returns_path = RETURNS_DIR / "daily_log_returns.csv"
returns.to_csv(returns_path)
print(f"Saved dynamic-availability returns matrix: {returns_path}")
print(f"Return matrix shape: {returns.shape}")

coverage = []
for t in returns.columns:
    s = returns[t]
    first_valid = s.first_valid_index()
    last_valid = s.last_valid_index()
    n_obs = int(s.notna().sum())
    coverage.append({
        "ticker": t,
        "first_valid_date": first_valid,
        "last_valid_date": last_valid,
        "n_return_obs": n_obs,
        "coverage_share_over_global_index": float(s.notna().mean()),
    })

coverage_df = pd.DataFrame(coverage)
coverage_df = coverage_df.merge(universe, on="ticker", how="left")
coverage_path = PRICE_RAW_DIR / "ticker_coverage_report.csv"
coverage_df.to_csv(coverage_path, index=False)
print(f"Saved ticker coverage report: {coverage_path}")

daily_active = returns.notna().sum(axis=1).rename("active_tickers").to_frame()
daily_active_path = RETURNS_DIR / "daily_active_ticker_count.csv"
daily_active.to_csv(daily_active_path)
print(f"Saved daily active ticker count: {daily_active_path}")

print("Coverage summary:")
print(coverage_df[["ticker", "first_valid_date", "last_valid_date", "n_return_obs", "coverage_share_over_global_index"]].head(20))
