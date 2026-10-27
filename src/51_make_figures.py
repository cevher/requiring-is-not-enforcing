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
    """S_NOT as bars, S_CTX as dashes, the lexical criterion as open circles.

    The fourth bar in the primary group is the decoder-enforced run added after
    the first submission draft. It is the point of the figure: enforcement lifts
    S_CTX in that model and leaves S_NOT on the floor, so the two halves of the
    construct have to be drawn separately or the result is invisible.
    """
    groups = [
        ("Primary model", [
            # label, S_NOT k, n, wilson lo, hi, S_CTX k, S_CTX n, lexical k, level
            ("Instruction\n(run 1)",  0, 10, 0.000, 0.278,  0, 10,  0, 0),
            ("Instruction\n(run 2)",  1, 10, 0.018, 0.404,  2,  9,  0, 0),
            ("Required\nfield",       0, 10, 0.000, 0.278,  2,  9,  0, 1),
            ("Decoder-\nenforced",    0, 10, 0.000, 0.278,  7,  7,  0, 2),
        ]),
        ("Secondary model", [
            ("Instruction",            3, 10, 0.108, 0.603,  5, 10,  4, 0),
            ("Required\nfield",      10, 10, 0.722, 1.000, 10, 10,  3, 1),
            ("Decoder-\nenforced",   10, 10, 0.722, 1.000, 10, 10, 10, 2),
        ]),
    ]
    fills = ["white", "#bfbfbf", "#4d4d4d"]

    fig, ax = plt.subplots(figsize=(7.44, 4.55))
    xs, labels, gap, pos = [], [], 0.9, 0.0
    for _, bars in groups:
        for b in bars:
            xs.append(pos); labels.append(b[0]); pos += 1.0
        pos += gap

    i = 0
    for _, bars in groups:
        for lab, k, n, lo, hi, ck, cn, rg, lvl in bars:
            x, p_ = xs[i], k / n
            ax.bar(x, p_, width=0.72, color=fills[lvl], edgecolor=INK, lw=1.1, zorder=2)
            ax.errorbar(x, p_, yerr=[[p_ - lo], [hi - p_]], fmt="none", ecolor=INK,
                        elinewidth=1.1, capsize=5, capthick=1.1, zorder=3)
            ax.text(x, hi + 0.04, f"{k}/{n}", ha="center", va="bottom",
                    fontsize=8.0, fontweight="bold", zorder=4)
            ax.plot([x - 0.43, x + 0.43], [ck / cn] * 2, color=INK, lw=2.6,
                    solid_capstyle="butt", zorder=5)
            ax.plot([x], [rg / n], marker="o", ms=8, mfc="white", mec=INK,
                    mew=1.5, ls="none", zorder=6)
            i += 1

    centres = [np.mean(xs[0:4]), np.mean(xs[4:7])]
    for c, (name, _) in zip(centres, groups):
        ax.text(c, -0.175, name, ha="center", va="center", fontsize=9.2,
                fontweight="bold", transform=ax.get_xaxis_transform())
    ax.plot([np.mean([xs[3], xs[4]])] * 2, [0, 1.10], color=RULE, lw=0.8,
            ls=(0, (4, 4)), zorder=1)

    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.6, linespacing=1.3)
    ax.set_ylim(0, 1.12); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_ylabel("Share of notes (human coding)", fontsize=8.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=RULE, lw=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=8)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(fc=fills[0], ec=INK, lw=1.1, label="Requirement in the system instruction"),
        Patch(fc=fills[1], ec=INK, lw=1.1, label="Requirement as a required output field"),
        Patch(fc=fills[2], ec=INK, lw=1.1, label="Required field enforced at the decoder"),
        Line2D([], [], color=INK, lw=2.6, label="S_CTX: evidence called context"),
        Line2D([], [], marker="o", ms=8, mfc="white", mec=INK, mew=1.5, ls="none",
               label="Lexical criterion on the same notes"),
    ]
    ax.legend(handles=handles, loc="upper center", fontsize=7.5, frameon=False,
              ncol=2, bbox_to_anchor=(0.5, -0.27), handlelength=1.8, columnspacing=2.0)
    save(fig, "Figure_3")



figure1(); figure_workflow(); figure_role()
