"""
10_build_tail_network_metrics.py

Builds downside tail co-movement networks and node-level metrics.

2005 revision:
- Dynamic window-level ticker availability.
- Uses MIN_TICKERS_PER_WINDOW from config.
"""

import numpy as np
import pandas as pd
import networkx as nx

from config import RETURNS_DIR, NETWORK_DIR, UNIVERSE_PATH, MIN_WINDOW_OBS_RATIO, MIN_TICKERS_PER_WINDOW
from common import ensure_dirs, load_universe, last_business_day_each_month

ensure_dirs(NETWORK_DIR)

TAIL_QUANTILE = 0.10
ROLLING_WINDOW_DAYS = 120
MIN_COEXCEEDANCE_RATE = 0.08
TOP_K_NEIGHBORS = 5

returns = pd.read_csv(RETURNS_DIR / "daily_log_returns.csv", parse_dates=["date"]).set_index("date").sort_index()
universe = load_universe(UNIVERSE_PATH)
sector_map = dict(zip(universe["ticker"], universe["sector"]))

snapshot_dates = last_business_day_each_month(returns.index)
snapshot_dates = [d for d in snapshot_dates if returns.index.get_loc(d) >= ROLLING_WINDOW_DAYS]

edge_records = []
metric_records = []
snapshot_records = []

for snap_date in snapshot_dates:
    loc = returns.index.get_loc(snap_date)
    window = returns.iloc[loc - ROLLING_WINDOW_DAYS + 1: loc + 1]

    min_obs = int(MIN_WINDOW_OBS_RATIO * ROLLING_WINDOW_DAYS)
    window = window.dropna(axis=1, thresh=min_obs)

    if window.shape[1] < MIN_TICKERS_PER_WINDOW:
        snapshot_records.append({
            "date": snap_date,
            "status": "skipped_too_few_tickers",
            "active_tickers": int(window.shape[1]),
        })
        continue

    thresholds = window.quantile(TAIL_QUANTILE)
    tail_events = window.le(thresholds, axis=1).astype(int)
    tickers = tail_events.columns.tolist()

    weights = pd.DataFrame(index=tickers, columns=tickers, dtype=float)

    for a in tickers:
        for b in tickers:
            if a == b:
                weights.loc[a, b] = np.nan
                continue
            both = ((tail_events[a] == 1) & (tail_events[b] == 1)).mean()
            weights.loc[a, b] = both

    edge_dict = {}

    for i, a in enumerate(tickers):
        for j, b in enumerate(tickers):
            if j <= i:
                continue
            w = weights.loc[a, b]
            if pd.notna(w) and w >= MIN_COEXCEEDANCE_RATE:
                edge_dict[tuple(sorted([a, b]))] = float(w)

    for a in tickers:
        vals = weights.loc[a].dropna().sort_values(ascending=False).head(TOP_K_NEIGHBORS)
        for b, w in vals.items():
            key = tuple(sorted([a, b]))
            edge_dict[key] = max(edge_dict.get(key, -np.inf), float(w))

    G = nx.Graph()
    for ticker in tickers:
        G.add_node(ticker, sector=sector_map.get(ticker, "Unknown"))

    for (a, b), w in edge_dict.items():
        G.add_edge(a, b, weight=w)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G) if n > 1 else np.nan

    degree = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight=None) if n > 2 else {node: np.nan for node in G.nodes}
    try:
        eigen = nx.eigenvector_centrality_numpy(G, weight="weight") if m > 0 else {node: 0 for node in G.nodes}
    except Exception:
        eigen = {node: np.nan for node in G.nodes}

    clustering = nx.clustering(G, weight="weight") if m > 0 else {node: 0 for node in G.nodes}

    snapshot_records.append({
        "date": snap_date,
        "status": "built",
        "active_tickers": int(len(tickers)),
        "tail_network_nodes": n,
        "tail_network_edges": m,
        "tail_network_density": density,
    })

    for a, b, data in G.edges(data=True):
        edge_records.append({
            "date": snap_date,
            "source": a,
            "target": b,
            "weight": data["weight"],
            "source_sector": sector_map.get(a, "Unknown"),
            "target_sector": sector_map.get(b, "Unknown"),
            "network_type": "downside_tail_comovement",
        })

    for node in G.nodes:
        metric_records.append({
            "date": snap_date,
            "ticker": node,
            "sector": sector_map.get(node, "Unknown"),
            "tail_weighted_degree": degree.get(node, 0),
            "tail_betweenness": betweenness.get(node, np.nan),
            "tail_eigenvector": eigen.get(node, np.nan),
            "tail_clustering": clustering.get(node, np.nan),
            "tail_network_nodes": n,
            "tail_network_edges": m,
            "tail_network_density": density,
            "active_tickers_in_window": int(len(tickers)),
        })

edges = pd.DataFrame(edge_records)
metrics = pd.DataFrame(metric_records)
snapshots = pd.DataFrame(snapshot_records)

edges.to_csv(NETWORK_DIR / "tail_comovement_edges.csv", index=False)
metrics.to_csv(NETWORK_DIR / "tail_network_metrics.csv", index=False)
snapshots.to_csv(NETWORK_DIR / "tail_network_snapshot_coverage.csv", index=False)

print(f"Saved: {NETWORK_DIR / 'tail_comovement_edges.csv'} ({len(edges)} rows)")
print(f"Saved: {NETWORK_DIR / 'tail_network_metrics.csv'} ({len(metrics)} rows)")
print(f"Saved: {NETWORK_DIR / 'tail_network_snapshot_coverage.csv'}")
print(snapshots["status"].value_counts(dropna=False))
