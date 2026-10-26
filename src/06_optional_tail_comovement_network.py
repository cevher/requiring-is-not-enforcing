"""Optional downside co-movement network based on negative-tail event co-occurrence."""
import pandas as pd
from config import RETURNS_DIR, NETWORK_DIR
from common import ensure_dirs, last_business_day_each_month

ensure_dirs(NETWORK_DIR)
TAIL_QUANTILE = 0.10
ROLLING_WINDOW_DAYS = 120
MIN_COEXCEEDANCE_RATE = 0.10

returns = pd.read_csv(RETURNS_DIR / "daily_log_returns.csv", parse_dates=["date"]).set_index("date").sort_index()
snapshot_dates = [d for d in last_business_day_each_month(returns.index) if returns.index.get_loc(d) >= ROLLING_WINDOW_DAYS]
records = []

for snap_date in snapshot_dates:
    loc = returns.index.get_loc(snap_date)
    window = returns.iloc[loc - ROLLING_WINDOW_DAYS + 1: loc + 1]
    window = window.dropna(axis=1, thresh=int(0.8 * ROLLING_WINDOW_DAYS))
    if window.shape[1] < 5:
        continue

    thresholds = window.quantile(TAIL_QUANTILE)
    tail_events = window.le(thresholds, axis=1).astype(int)
    tickers = tail_events.columns.tolist()

    for i, a in enumerate(tickers):
        for j, b in enumerate(tickers):
            if j <= i:
                continue
            both = ((tail_events[a] == 1) & (tail_events[b] == 1)).mean()
            if both >= MIN_COEXCEEDANCE_RATE:
                records.append({
                    "date": snap_date,
                    "source": a,
                    "target": b,
                    "tail_coexceedance_rate": both
                })

pd.DataFrame(records).to_csv(NETWORK_DIR / "tail_comovement_edges.csv", index=False)
print(f"Saved tail co-movement edges: {NETWORK_DIR / 'tail_comovement_edges.csv'}")
