"""
02_download_macro_stress.py

Downloads free macro-financial stress proxies from FRED without pandas_datareader.

This version avoids pandas_datareader because some pandas/pandas-datareader
version combinations produce deprecate_kwarg-related errors.

If you later obtain OFR FSI manually, put it into:
data_raw/macro/ofr_fsi.csv

Expected OFR columns:
date, ofr_fsi
"""

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from config import START_DATE, END_DATE, MACRO_RAW_DIR
from common import ensure_dirs


ensure_dirs(MACRO_RAW_DIR)


FRED_SERIES = {
    "VIXCLS": "vix",
    "DGS10": "treasury_10y",
    "DGS2": "treasury_2y",
    "T10Y2Y": "term_spread_10y_2y",
    "BAMLH0A0HYM2": "high_yield_oas",
    "NFCI": "nfci",
}


def download_fred_csv(series_id: str, renamed_col: str) -> pd.DataFrame:
    """
    Download a FRED series using the public fredgraph CSV endpoint.

    Returns a DataFrame indexed by date with one column named renamed_col.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "financial-systemic-risk-research/0.1"},
    )
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))

    # FRED CSV usually returns: observation_date, SERIES_ID
    date_col = "observation_date"
    if date_col not in df.columns:
        # fallback: use first column as date
        date_col = df.columns[0]

    value_col = series_id
    if value_col not in df.columns:
        # fallback: use the first non-date column
        value_col = [c for c in df.columns if c != date_col][0]

    df = df[[date_col, value_col]].copy()
    df = df.rename(columns={date_col: "date", value_col: renamed_col})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[renamed_col] = pd.to_numeric(df[renamed_col].replace(".", pd.NA), errors="coerce")

    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    # Filter date range
    df = df.loc[(df.index >= pd.to_datetime(START_DATE)) & (df.index <= pd.to_datetime(END_DATE))]

    return df


frames = []

for series_id, renamed_col in FRED_SERIES.items():
    try:
        print(f"Downloading FRED series: {series_id} -> {renamed_col}")
        s = download_fred_csv(series_id, renamed_col)
        if s.empty:
            print(f"Warning: {series_id} returned empty data after date filtering.")
        else:
            frames.append(s)
    except Exception as e:
        print(f"Warning: failed to download {series_id}: {e}")


if not frames:
    raise RuntimeError("No macro/stress data downloaded from FRED.")

macro = pd.concat(frames, axis=1).sort_index()
macro.index.name = "date"

# Forward-fill non-daily/holiday gaps.
# This does not backfill before first valid observation.
macro = macro.ffill()


# Optional OFR FSI merge if user provides it manually
ofr_path = MACRO_RAW_DIR / "ofr_fsi.csv"

if ofr_path.exists():
    try:
        ofr = pd.read_csv(ofr_path)
        ofr.columns = [c.lower().strip() for c in ofr.columns]

        if "date" not in ofr.columns:
            print("Warning: ofr_fsi.csv found but no date column. Skipping OFR merge.")
        else:
            ofr["date"] = pd.to_datetime(ofr["date"], errors="coerce")
            ofr = ofr.dropna(subset=["date"]).set_index("date").sort_index()

            if "ofr_fsi" not in ofr.columns:
                candidate_cols = [c for c in ofr.columns if c != "date"]
                if not candidate_cols:
                    print("Warning: ofr_fsi.csv has no usable value column. Skipping OFR merge.")
                else:
                    ofr = ofr.rename(columns={candidate_cols[0]: "ofr_fsi"})

            if "ofr_fsi" in ofr.columns:
                macro = macro.join(ofr[["ofr_fsi"]], how="left").ffill()
                print("Merged user-provided OFR FSI.")

    except Exception as e:
        print(f"Warning: failed to merge OFR FSI: {e}")


out = MACRO_RAW_DIR / "fred_stress_data.csv"
macro.to_csv(out)

print(f"Saved macro/stress data: {out}")
print(macro.tail())