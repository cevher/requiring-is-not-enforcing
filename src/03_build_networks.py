"""
03_build_networks.py

Builds monthly rolling correlation networks from daily log returns.

2005 revision:
- Uses dynamic window-level ticker availability.
- Does not require firms to exist for the whole 2005-2025 period.
- Skips snapshots with too few active tickers.
- Records active universe size per snapshot.
"""

import numpy as np
import pandas as pd
import networkx as nx

from config import (
    UNIVERSE_PATH, RETURNS_DIR, NETWORK_DIR,
    ROLLING_WINDOW_DAYS, CORR_THRESHOLD, TOP_K_NEIGHBORS,
    MIN_WINDOW_OBS_RATIO, MIN_TICKERS_PER_WINDOW
)
from common import ensure_dirs, load_universe, last_business_day_each_month

ensure_dirs(NETWORK_DIR)

returns_path = RETURNS_DIR / "daily_log_returns.csv"
returns = pd.read_csv(returns_path, parse_dates=["date"]).set_index("date").sort_index()

universe = load_universe(UNIVERSE_PATH)
sector_map = dict(zip(universe["ticker"], universe["sector"]))

snapshot_dates = last_business_day_each_month(returns.index)
snapshot_dates = [d for d in snapshot_dates if returns.index.get_loc(d) >= ROLLING_WINDOW_DAYS]

edge_records = []
metric_records = []
sector_edge_records = []
snapshot_records = []

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

    active_tickers = window.columns.tolist()
    n_active = len(active_tickers)

    if n_active < MIN_TICKERS_PER_WINDOW:
        snapshot_records.append({
            "date": snap_date,
            "status": "skipped_too_few_tickers",
            "active_tickers": n_active,
            "window_days": ROLLING_WINDOW_DAYS,
            "min_obs_required": min_obs,
        })
        continue

    corr = window.corr(min_periods=min_obs)

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
    for ticker in active_tickers:
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
        "active_tickers": n_active,
        "window_days": ROLLING_WINDOW_DAYS,
        "min_obs_required": min_obs,
        "network_nodes": n,
        "network_edges": m,
        "network_density": density,
    })

    for a, b, data in G.edges(data=True):
        edge_records.append({
            "date": snap_date,
            "source": a,
            "target": b,
            "weight": data["weight"],
            "source_sector": sector_map.get(a, "Unknown"),
            "target_sector": sector_map.get(b, "Unknown"),
        })

    for node in G.nodes:
        metric_records.append({
            "date": snap_date,
            "ticker": node,
            "sector": sector_map.get(node, "Unknown"),
            "weighted_degree": degree.get(node, 0),
            "betweenness": betweenness.get(node, np.nan),
            "eigenvector": eigen.get(node, np.nan),
            "clustering": clustering.get(node, np.nan),
            "network_nodes": n,
            "network_edges": m,
            "network_density": density,
            "active_tickers_in_window": n_active,
        })

    if edge_dict:
        edges_df = pd.DataFrame([
            {
                "source_sector": sector_map.get(a, "Unknown"),
                "target_sector": sector_map.get(b, "Unknown"),
                "weight": w
            }
            for (a, b), w in edge_dict.items()
        ])
        sec = (
            edges_df
            .groupby(["source_sector", "target_sector"], as_index=False)
            .agg(mean_weight=("weight", "mean"), edge_count=("weight", "size"))
        )
        sec["date"] = snap_date
        sector_edge_records.extend(sec.to_dict("records"))

edges = pd.DataFrame(edge_records)
metrics = pd.DataFrame(metric_records)
sector_edges = pd.DataFrame(sector_edge_records)
snapshots = pd.DataFrame(snapshot_records)

edges.to_csv(NETWORK_DIR / "rolling_corr_edges.csv", index=False)
metrics.to_csv(NETWORK_DIR / "network_metrics.csv", index=False)
sector_edges.to_csv(NETWORK_DIR / "sector_network_edges.csv", index=False)
snapshots.to_csv(NETWORK_DIR / "network_snapshot_coverage.csv", index=False)

print(f"Saved edges: {NETWORK_DIR / 'rolling_corr_edges.csv'} ({len(edges)} rows)")
print(f"Saved metrics: {NETWORK_DIR / 'network_metrics.csv'} ({len(metrics)} rows)")
print(f"Saved sector edges: {NETWORK_DIR / 'sector_network_edges.csv'} ({len(sector_edges)} rows)")
print(f"Saved snapshot coverage: {NETWORK_DIR / 'network_snapshot_coverage.csv'}")
print(snapshots["status"].value_counts(dropna=False))
