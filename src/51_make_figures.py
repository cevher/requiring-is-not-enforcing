"""Build Figures 1-3 at the journal's full-page width (190 mm).

Single source for every figure in the paper. Sizes and font sizes are set so
that no rescaling happens at typesetting: each figure is emitted at 190 mm wide
with body text at 7.2 pt or larger, which is the publisher's floor. PDFs carry
TrueType-subset fonts (pdf.fonttype 42), not matplotlib's default Type 3.

Writes figures/Figure_1..3 as PDF and 600 dpi PNG. The prompt-condition
matrix that an earlier draft carried as a figure is Table 2 of the paper and is
set as text, so it is not generated here.
"""
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.linewidth": 0.9,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

INK = "#000000"
MUTED = "#555555"
RULE = "#aaaaaa"


def save(fig, stem):
    fig.savefig(f"{stem}.png", dpi=600)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print("written:", stem)


def figure1():


    TOP = [
        ("Open financial\ndata", ["Equity prices", "Macro-financial indicators", "Firm metadata"]),
        ("Risk-monitoring\nsubstrate", ["Rule-based monitoring score", "Macro-financial model score",
                                        "Combined monitoring score"]),
        ("Reliability and\ncalibration layer", ["Rule/model disagreement", "Historical reliability",
                                                "Reliability-adjusted\n   confidence", "Abstention flag"]),
        ("Relational\nevidence layer", ["Raw correlation networks", "Residual co-movement\n   networks",
                                        "Downside tail networks", "Firm and sector attribution"]),
    ]
    BOT = [
        ("Provenance\nevidence layer", ["Evidence identifiers", "Source types", "Evidence roles",
                                        "Value snippets", "Entity-level records"]),
        ("Evidence-grounded\nLLM explanation", ["Cited risk explanation", "Uncertainty-aware summary",
                                                "Relational attribution",
                                                "No unsupported causal or\n   investment claims"]),
        ("Audit and\nevaluation", ["Span-aware output parsing", "Citation correctness",
                                   "Invented evidence identifiers",
                                   "Uncertainty against the\n   decision record",
                                   "Role statement, human-coded"]),
    ]

    fig = plt.figure(figsize=(7.44, 3.95))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


    def box(x, y, w, h, title, bullets):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.006,rounding_size=0.014",
                     fc="white", ec=INK, lw=1.2))
        ax.text(x + w / 2, y + h - 0.055, title, fontsize=8.6, fontweight="bold",
                ha="center", va="center", linespacing=1.3)
        ax.text(x + 0.012, y + h - 0.125, "\n".join("\u2022 " + b for b in bullets),
                fontsize=7.3, va="top", linespacing=1.75)


    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14,
                                     lw=1.2, color=INK, shrinkA=0, shrinkB=0))


    G1 = 0.016
    W1 = (0.965 - 3 * G1) / 4
    Y1, H1 = 0.613, 0.367
    for i, (t, b) in enumerate(TOP):
        box(0.010 + i * (W1 + G1), Y1, W1, H1, t, b)
    for i in range(3):
        x = 0.010 + i * (W1 + G1) + W1
        arrow((x, Y1 + H1 / 2), (x + G1, Y1 + H1 / 2))

    G2 = 0.034
    W2 = 0.268
    X2 = 0.120
    Y2, H2 = 0.030, 0.413
    for i, (t, b) in enumerate(BOT):
        box(X2 + i * (W2 + G2), Y2, W2, H2, t, b)
    for i in range(2):
        x = X2 + i * (W2 + G2) + W2
        arrow((x, Y2 + H2 / 2), (x + G2, Y2 + H2 / 2))

    # elbow: last top box -> first bottom box, routed inside the axes
    xr = 0.010 + 3 * (W1 + G1) + W1
    xe = 0.988
    xl = X2 - 0.045
    ym = 0.530
    ax.plot([xr, xe, xe, xl, xl], [Y1 + H1 / 2, Y1 + H1 / 2, ym, ym, Y2 + H2 / 2],
            color=INK, lw=1.2, solid_joinstyle="miter")
    arrow((xl, Y2 + H2 / 2), (X2, Y2 + H2 / 2))

    save(fig, "Figure_1")






# ---------------------------------------------------------------- Figure 3
def figure_workflow():
    fig = plt.figure(figsize=(7.44, 3.13))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def box(x, y, w, h, title, lines, fs=7.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.006,rounding_size=0.012",
                     fc="white", ec=INK, lw=1.1))
        ax.text(x + w / 2, y + h - 0.055, title, fontsize=8.2, fontweight="bold",
                ha="center", va="center", linespacing=1.25)
        ax.text(x + 0.016, y + h - 0.1074,
                "\n".join("• " + s for s in lines), fontsize=fs, va="top", linespacing=1.55)

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13,
                                     lw=1.1, color=INK, shrinkA=0, shrinkB=0))

    box(0.012, 0.656, 0.180, 0.279, "Decision\nsnapshots",
        ["10 monthly contexts", "all monitoring errors", "structured evidence"])
    box(0.212, 0.554, 0.170, 0.381, "Prompt\nconditions",
        ["full", "no-network", "no-reliability", "free-form", "required role"])
    box(0.402, 0.605, 0.198, 0.330, "Generation",
        ["10 × 5 design", "two model families", "matched decoding",
         "pinned model build"])
    box(0.640, 0.605, 0.348, 0.330, "Automatic audit",
        ["span-aware parsing", "citation correctness and invented IDs",
         "uncertainty from the decision record", "criteria scoped by evidence"])
    box(0.640, 0.166, 0.348, 0.330, "Human coding",
        ["16 items: instrument validation",
         "60 + 30 items: role statement, blind",
         "two binary questions per note",
         "primary measure of role separation"])
    box(0.340, 0.106, 0.278, 0.279, "Configuration\nstudy",
        ["three parser and decoder settings", "naive variants retained",
         "model-version replication"])

    arrow((0.192, 0.765), (0.212, 0.765))
    arrow((0.382, 0.765), (0.402, 0.765))
    arrow((0.600, 0.804), (0.640, 0.830))
    arrow((0.600, 0.699), (0.640, 0.437))
    arrow((0.480, 0.605), (0.480, 0.392))
    # the human coding validates the automatic instrument, not the configuration study
    ax.add_patch(FancyArrowPatch((0.800, 0.496), (0.800, 0.600), arrowstyle="-|>",
                 mutation_scale=13, lw=1.1, color=INK))
    ax.text(0.812, 0.550, "validates", fontsize=7.2, style="italic",
            color=MUTED, ha="left", va="center")
    save(fig, "Figure_2")


# ---------------------------------------------------------------- Figure 4
def figure_role():
    """The two halves of the role constraint across three interface levels.

    Reader consensus from the coding pass over 30 notes, two readers external to
    the institution. S_CTX asks whether the note calls relational evidence
    context or attribution; S_NOT asks whether it states that the evidence is
    not the primary predictor. The system instruction demands both. Drawing them
    separately is the point of the figure: the two halves do not move together,
    and the level that secures S_CTX completely is the level at which S_NOT
    fails.
    """
    def wilson(k, n, z=1.959963985):
        if n == 0:
            return 0.0, 0.0
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
        return max(0.0, c - h), min(1.0, c + h)

    LEVELS = ["Requirement in the\nsystem instruction",
              "Requirement as a\nrequired output field",
              "Required field enforced\nat the decoder"]
    S_CTX = [(0, 10), (9, 9), (10, 10)]
    S_NOT = [(0, 10), (10, 10), (0, 10)]

    fig, ax = plt.subplots(figsize=(7.44, 4.0))
    w = 0.34
    xs = np.arange(3.0)
    for off, series, fill in ((-w / 2 - 0.02, S_CTX, "#bfbfbf"),
                              (w / 2 + 0.02, S_NOT, "#4d4d4d")):
        for x, (k, n) in zip(xs, series):
            p_ = k / n
            lo, hi = wilson(k, n)
            ax.bar(x + off, p_, width=w, color=fill, edgecolor=INK, lw=1.1, zorder=2)
            ax.errorbar(x + off, p_, yerr=[[p_ - lo], [hi - p_]], fmt="none",
                        ecolor=INK, elinewidth=1.1, capsize=4, capthick=1.1, zorder=3)
            ax.text(x + off, hi + 0.035, f"{k}/{n}", ha="center", va="bottom",
                    fontsize=8.0, fontweight="bold", zorder=5)

    ax.set_xticks(xs)
    ax.set_xticklabels(LEVELS, fontsize=8.2, linespacing=1.4)
    ax.set_ylim(0, 1.16)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_ylabel("Share of notes (reader consensus)", fontsize=8.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=RULE, lw=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=7)

    from matplotlib.patches import Patch
    handles = [
        Patch(fc="#bfbfbf", ec=INK, lw=1.1,
              label="S_CTX: evidence called context or attribution"),
        Patch(fc="#4d4d4d", ec=INK, lw=1.1,
              label="S_NOT: evidence stated not to be the primary predictor"),
    ]
    ax.legend(handles=handles, loc="upper center", fontsize=7.9, frameon=False,
              ncol=2, bbox_to_anchor=(0.5, -0.19), handlelength=1.8, columnspacing=2.2)
    save(fig, "Figure_3")


figure1(); figure_workflow(); figure_role()
