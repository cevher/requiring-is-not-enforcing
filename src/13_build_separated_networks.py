"""
13_build_separated_networks.py

Builds separated network variants for final experiments.

Why this script exists:
Earlier feasibility scripts mixed threshold edges and top-k edges in the same network.
That is acceptable for exploration but weak for final paper claims.

This script separates:

1. raw_threshold:
   rolling return correlation network with corr >= threshold
   Best for density/stress interpretation.

2. raw_topk:
   rolling return correlation network with top-k positive neighbors per node
   Best for stable centrality/provenance ranking.

3. residual_threshold:
   market-adjusted residual correlation network with corr >= threshold

4. residual_topk:
   market-adjusted residual correlation top-k network

5. tail_threshold:
   downside tail co-movement network with co-exceedance >= threshold

6. tail_topk:
   downside tail co-movement top-k network

Inputs:
- data_processed/returns/daily_log_returns.csv
- data_raw/metadata/universe.csv

Outputs:
- data_processed/networks/separated/{network_name}_edges.csv
- data_processed/networks/separated/{network_name}_metrics.csv
- data_processed/networks/separated/{network_name}_snapshot_coverage.csv
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

from config import (
    RETURNS_DIR, NETWORK_DIR, UNIVERSE_PATH,
    ROLLING_WINDOW_DAYS, CORR_THRESHOLD, TOP_K_NEIGHBORS,
    MIN_WINDOW_OBS_RATIO, MIN_TICKERS_PER_WINDOW
)
from common import ensure_dirs, load_universe, last_business_day_each_month


OUT_DIR = NETWORK_DIR / "separated"
ensure_dirs(OUT_DIR)

TAIL_WINDOW_DAYS = 120
TAIL_QUANTILE = 0.10
TAIL_COEXCEEDANCE_THRESHOLD = 0.08


returns = pd.read_csv(
    RETURNS_DIR / "daily_log_returns.csv",
    parse_dates=["date"]
).set_index("date").sort_index()

universe = load_universe(UNIVERSE_PATH)
sector_map = dict(zip(universe["ticker"], universe["sector"]))


def get_snapshot_dates(window_days: int):
    dates = last_business_day_each_month(returns.index)
    return [d for d in dates if returns.index.get_loc(d) >= window_days]


def filter_window(window: pd.DataFrame, window_days: int):
    min_obs = int(MIN_WINDOW_OBS_RATIO * window_days)
    return window.dropna(axis=1, thresh=min_obs), min_obs


def market_adjusted_residuals(window: pd.DataFrame) -> pd.DataFrame:
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


def corr_weight_matrix(window: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    return window.corr(min_periods=min_obs)


def tail_weight_matrix(window: pd.DataFrame) -> pd.DataFrame:
    thresholds = window.quantile(TAIL_QUANTILE)
    tail_events = window.le(thresholds, axis=1).astype(int)
    tickers = tail_events.columns.tolist()
    weights = pd.DataFrame(index=tickers, columns=tickers, dtype=float)

    for a in tickers:
        for b in tickers:
            if a == b:
                weights.loc[a, b] = np.nan
            else:
                weights.loc[a, b] = ((tail_events[a] == 1) & (tail_events[b] == 1)).mean()

    return weights


def edges_from_matrix(weights: pd.DataFrame, mode: str, threshold: float, top_k: int):
    edge_dict = {}

    if mode == "threshold":
        for i, a in enumerate(weights.columns):
            for j, b in enumerate(weights.columns):
                if j <= i:
                    continue
                w = weights.loc[a, b]
                if pd.notna(w) and w >= threshold:
                    edge_dict[tuple(sorted([a, b]))] = float(w)

    elif mode == "topk":
        for a in weights.index:
            vals = weights.loc[a].dropna()
            vals = vals[vals > 0]
            vals = vals.sort_values(ascending=False).head(top_k)
            for b, w in vals.items():
                key = tuple(sorted([a, b]))
                edge_dict[key] = max(edge_dict.get(key, -np.inf), float(w))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return edge_dict


def graph_metrics_from_edges(tickers, edge_dict, snap_date, network_name):
    G = nx.Graph()

    for ticker in tickers:
        G.add_node(ticker, sector=sector_map.get(ticker, "Unknown"))

    for (a, b), w in edge_dict.items():
        G.add_edge(a, b, weight=float(w))

    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G) if n > 1 else np.nan

    degree = dict(G.degree(weight="weight"))

    if n > 2 and m > 0:
        betweenness = nx.betweenness_centrality(G, weight=None)
    else:
        betweenness = {node: np.nan for node in G.nodes}

    try:
        eigen = nx.eigenvector_centrality_numpy(G, weight="weight") if m > 0 else {node: 0 for node in G.nodes}
    except Exception:
        eigen = {node: np.nan for node in G.nodes}

    clustering = nx.clustering(G, weight="weight") if m > 0 else {node: 0 for node in G.nodes}

    edge_records = []
    metric_records = []

    for a, b, data in G.edges(data=True):
        edge_records.append({
            "date": snap_date,
            "source": a,
            "target": b,
            "weight": data["weight"],
            "source_sector": sector_map.get(a, "Unknown"),
            "target_sector": sector_map.get(b, "Unknown"),
            "network_name": network_name,
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
            "network_name": network_name,
        })

    snapshot_record = {
        "date": snap_date,
        "network_name": network_name,
        "status": "built",
        "active_tickers": len(tickers),
        "network_nodes": n,
        "network_edges": m,
        "network_density": density,
    }

    return edge_records, metric_records, snapshot_record


def build_network_family(
    network_name: str,
    window_days: int,
    weight_builder,
    mode: str,
    threshold: float,
):
    snapshot_dates = get_snapshot_dates(window_days)

    all_edges = []
    all_metrics = []
    all_snapshots = []

    for snap_date in snapshot_dates:
        loc = returns.index.get_loc(snap_date)
        window = returns.iloc[loc - window_days + 1: loc + 1]
        window, min_obs = filter_window(window, window_days)

        if window.shape[1] < MIN_TICKERS_PER_WINDOW:
            all_snapshots.append({
                "date": snap_date,
                "network_name": network_name,
                "status": "skipped_too_few_tickers",
                "active_tickers": int(window.shape[1]),
                "network_nodes": np.nan,
                "network_edges": np.nan,
                "network_density": np.nan,
            })
            continue

        weights = weight_builder(window, min_obs)
        weights = weights.dropna(axis=0, how="all").dropna(axis=1, how="all")
        tickers = sorted(set(weights.index).intersection(weights.columns))

        if len(tickers) < MIN_TICKERS_PER_WINDOW:
            all_snapshots.append({
                "date": snap_date,
                "network_name": network_name,
                "status": "skipped_too_few_weighted_tickers",
                "active_tickers": int(len(tickers)),
                "network_nodes": np.nan,
                "network_edges": np.nan,
                "network_density": np.nan,
            })
            continue

        weights = weights.loc[tickers, tickers]
        edge_dict = edges_from_matrix(
            weights=weights,
            mode=mode,
            threshold=threshold,
            top_k=TOP_K_NEIGHBORS,
        )

        edge_records, metric_records, snapshot_record = graph_metrics_from_edges(
            tickers=tickers,
            edge_dict=edge_dict,
            snap_date=snap_date,
            network_name=network_name,
        )

        all_edges.extend(edge_records)
        all_metrics.extend(metric_records)
        all_snapshots.append(snapshot_record)

    pd.DataFrame(all_edges).to_csv(OUT_DIR / f"{network_name}_edges.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(OUT_DIR / f"{network_name}_metrics.csv", index=False)
    pd.DataFrame(all_snapshots).to_csv(OUT_DIR / f"{network_name}_snapshot_coverage.csv", index=False)

    print(f"{network_name}: edges={len(all_edges)}, metrics={len(all_metrics)}, snapshots={len(all_snapshots)}")


def raw_corr_builder(window, min_obs):
    return corr_weight_matrix(window, min_obs)


def residual_corr_builder(window, min_obs):
    residuals = market_adjusted_residuals(window)
    residuals = residuals.dropna(axis=1, thresh=min_obs)
    return corr_weight_matrix(residuals, min_obs)


def tail_builder(window, min_obs):
    return tail_weight_matrix(window)


if __name__ == "__main__":
    jobs = [
        ("raw_threshold", ROLLING_WINDOW_DAYS, raw_corr_builder, "threshold", CORR_THRESHOLD),
        ("raw_topk", ROLLING_WINDOW_DAYS, raw_corr_builder, "topk", CORR_THRESHOLD),
        ("residual_threshold", ROLLING_WINDOW_DAYS, residual_corr_builder, "threshold", CORR_THRESHOLD),
        ("residual_topk", ROLLING_WINDOW_DAYS, residual_corr_builder, "topk", CORR_THRESHOLD),
        ("tail_threshold", TAIL_WINDOW_DAYS, tail_builder, "threshold", TAIL_COEXCEEDANCE_THRESHOLD),
        ("tail_topk", TAIL_WINDOW_DAYS, tail_builder, "topk", TAIL_COEXCEEDANCE_THRESHOLD),
    ]

    for args in jobs:
        build_network_family(*args)
