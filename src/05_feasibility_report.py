"""Create feasibility diagnostics linking network structure and stress labels."""
import pandas as pd
import matplotlib.pyplot as plt
from config import NETWORK_DIR, LABEL_DIR, FIGURE_DIR, REPORT_DIR
from common import ensure_dirs

ensure_dirs(FIGURE_DIR, REPORT_DIR)
metrics = pd.read_csv(NETWORK_DIR / "network_metrics.csv", parse_dates=["date"])
labels = pd.read_csv(LABEL_DIR / "stress_labels.csv", parse_dates=["date"])

date_metrics = metrics.groupby("date", as_index=False).agg(
    network_density=("network_density", "mean"),
    mean_weighted_degree=("weighted_degree", "mean"),
    max_weighted_degree=("weighted_degree", "max"),
    mean_eigenvector=("eigenvector", "mean"),
    max_eigenvector=("eigenvector", "max"),
    mean_clustering=("clustering", "mean"),
    network_nodes=("network_nodes", "mean"),
    network_edges=("network_edges", "mean")
)

df = pd.merge_asof(
    date_metrics.sort_values("date"),
    labels.sort_values("date"),
    on="date",
    direction="nearest",
    tolerance=pd.Timedelta("5D")
)
df.to_csv(REPORT_DIR / "network_stress_diagnostics.csv", index=False)

if "vix" in df.columns:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df["date"], df["network_density"])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Network density")
    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["vix"])
    ax2.set_ylabel("VIX")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "network_density_vs_vix.png", dpi=200)
    plt.close(fig)

if "high_stress_label" in df.columns:
    grouped = df.groupby("high_stress_label")["mean_weighted_degree"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped.plot(kind="bar", ax=ax)
    ax.set_xlabel("High stress label")
    ax.set_ylabel("Mean weighted degree")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "avg_centrality_vs_stress.png", dpi=200)
    plt.close(fig)

top_firms = metrics.groupby(["ticker", "sector"], as_index=False).agg(
    avg_weighted_degree=("weighted_degree", "mean"),
    avg_eigenvector=("eigenvector", "mean"),
    avg_betweenness=("betweenness", "mean")
).sort_values(["avg_weighted_degree", "avg_eigenvector"], ascending=False)

top_firms.to_csv(REPORT_DIR / "top_firms_by_network_centrality.csv", index=False)

sector_diag = metrics.groupby(["date", "sector"], as_index=False).agg(
    sector_avg_weighted_degree=("weighted_degree", "mean"),
    sector_max_weighted_degree=("weighted_degree", "max"),
    sector_avg_eigenvector=("eigenvector", "mean")
)
sector_diag.to_csv(REPORT_DIR / "sector_network_diagnostics.csv", index=False)

summary = [
    "# Feasibility Summary\n",
    f"- Network snapshots: {df.shape[0]}",
    f"- Firms in metrics table: {metrics['ticker'].nunique()}",
    f"- Sectors in metrics table: {metrics['sector'].nunique()}",
]
if "high_stress_label" in df.columns:
    summary += [
        f"- High-stress snapshots: {int(df['high_stress_label'].sum())}",
        f"- High-stress share: {df['high_stress_label'].mean():.3f}",
    ]

summary += [
    f"- Average network density: {df['network_density'].mean():.4f}",
    f"- Average weighted degree: {df['mean_weighted_degree'].mean():.4f}",
    "\n## Top 10 firms by average weighted degree\n"
]

for _, r in top_firms.head(10).iterrows():
    summary.append(
        f"- {r['ticker']} ({r['sector']}): weighted_degree={r['avg_weighted_degree']:.4f}, "
        f"eigenvector={r['avg_eigenvector']:.4f}"
    )

summary.append(
    "\n## Interpretation note\n"
    "This is a feasibility report. It does not establish causal contagion. "
    "It checks whether open price/stress data can produce interpretable rolling financial networks. "
    "If this looks meaningful, proceed to provenance-aware evidence construction and calibrated LLM-based decision support."
)

(REPORT_DIR / "feasibility_summary.md").write_text("\n".join(summary), encoding="utf-8")
print(f"Saved report: {REPORT_DIR / 'feasibility_summary.md'}")
