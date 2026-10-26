import re
from pathlib import Path
import pandas as pd

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def load_universe(path):
    df = pd.read_csv(path)
    required = {"ticker", "company_name", "sector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Universe file missing columns: {missing}")
    return df

def last_business_day_each_month(index):
    idx = pd.DatetimeIndex(index).sort_values().unique()
    s = pd.Series(idx, index=idx)
    return s.groupby([idx.year, idx.month]).max().tolist()
