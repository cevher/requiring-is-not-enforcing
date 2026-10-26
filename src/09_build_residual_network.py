"""
09_build_residual_network.py

Builds market-adjusted residual correlation networks.

2005 revision:
- Dynamic window-level ticker availability.
- Skips snapshots with too few active tickers.
"""

import numpy as np
import pandas as pd
import networkx as nx

from config import (
    RETURNS_DIR, NETWORK_DIR, UNIVERSE_PATH,
    ROLLING_WINDOW_DAYS, CORR_THRESHOLD, TOP_K_NEIGHBORS,
    MIN_WINDOW_OBS_RATIO, MIN_TICKERS_PER_WINDOW
)
from common import ensure_dirs, load_universe, last_business_day_each_month

ensure_dirs(NETWORK_DIR)

returns = pd.read_csv(RETURNS_DIR / "daily_log_returns.csv", parse_dates=["date"]).set_index("date").sort_index()
universe = load_universe(UNIVERSE_PATH)
sector_map = dict(zip(universe["ticker"], universe["sector"]))

snapshot_dates = last_business_day_each_month(returns.index)
snapshot_dates = [d for d in snapshot_dates if returns.index.get_loc(d) >= ROLLING_WINDOW_DAYS]

edge_records = []
metric_records = []
snapshot_records = []

def regress_residuals(window: pd.DataFrame) -> pd.DataFrame:
    market = window.mean(axis=1, skipna=True)
    residuals = pd.DataFrame(index=window.index)

    x = market.values.astype(float)
    valid_x = np.isfinite(x)

    for ticker in window.columns:
        y = window[ticker].values.astype(float)
        valid = valid_x & np.isfinite(y)
        if valid.sum() < max(20, int(0.5 * len(window))):
            continue
        X = np.column_stack([np.ones(valid.sum()), x[valid]])
        beta = np.linalg.lstsq(X, y[valid], rcond=None)[0]
        y_hat = beta[0] + beta[1] * x
        residuals[ticker] = y - y_hat

    return residuals

def add_topk_edges(corr, k):
    edges = set()
    for i in corr.index:
        vals = corr.loc[i].drop(labels=[i], errors="ignore").dropna()
        vals = vals[vals > 0]
        top = vals.sort_values(ascending=False).head(k)
        for j, w in top.items():
            a, b = sorted([i, j])
            edges.add((a, b, float(w)))
    return list(edges)

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

    residuals = regress_residuals(window).dropna(axis=1, thresh=min_obs)

    if residuals.shape[1] < MIN_TICKERS_PER_WINDOW:
        snapshot_records.append({
            "date": snap_date,
            "status": "skipped_too_few_residual_tickers",
            "active_tickers": int(residuals.shape[1]),
        })
        continue

    corr = residuals.corr(min_periods=min_obs)

    threshold_edges = []
    for i, src in enumerate(corr.columns):
        for j, dst in enumerate(corr.columns):
            if j <= i:
                continue
            w = corr.loc[src, dst]
            if pd.notna(w) and w >= CORR_THRESHOLD:
                threshold_edges.append((src, dst, float(w)))

    topk_edges = add_topk_edges(corr, TOP_K_NEIGHBORS)

    edge_dict = {}
    for a, b, w in threshold_edges + topk_edges:
        key = tuple(sorted([a, b]))
        edge_dict[key] = max(edge_dict.get(key, -np.inf), w)

    G = nx.Graph()
    for ticker in residuals.columns:
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
        "active_tickers": int(residuals.shape[1]),
        "resid_network_nodes": n,
        "resid_network_edges": m,
        "resid_network_density": density,
    })

    for a, b, data in G.edges(data=True):
        edge_records.append({
            "date": snap_date,
            "source": a,
            "target": b,
            "weight": data["weight"],
            "source_sector": sector_map.get(a, "Unknown"),
            "target_sector": sector_map.get(b, "Unknown"),
            "network_type": "market_adjusted_residual_correlation",
        })

    for node in G.nodes:
        metric_records.append({
            "date": snap_date,
            "ticker": node,
            "sector": sector_map.get(node, "Unknown"),
            "resid_weighted_degree": degree.get(node, 0),
            "resid_betweenness": betweenness.get(node, np.nan),
            "resid_eigenvector": eigen.get(node, np.nan),
            "resid_clustering": clustering.get(node, np.nan),
            "resid_network_nodes": n,
            "resid_network_edges": m,
            "resid_network_density": density,
            "active_tickers_in_window": int(residuals.shape[1]),
        })

edges = pd.DataFrame(edge_records)
metrics = pd.DataFrame(metric_records)
snapshots = pd.DataFrame(snapshot_records)

edges.to_csv(NETWORK_DIR / "residual_corr_edges.csv", index=False)
metrics.to_csv(NETWORK_DIR / "residual_network_metrics.csv", index=False)
snapshots.to_csv(NETWORK_DIR / "residual_network_snapshot_coverage.csv", index=False)

print(f"Saved: {NETWORK_DIR / 'residual_corr_edges.csv'} ({len(edges)} rows)")
print(f"Saved: {NETWORK_DIR / 'residual_network_metrics.csv'} ({len(metrics)} rows)")
print(f"Saved: {NETWORK_DIR / 'residual_network_snapshot_coverage.csv'}")
print(snapshots["status"].value_counts(dropna=False))
