"""Rebuild Figures 2, 3 and 4 as separate files.

The journal does not accept several images combined in one file, and the old
Figure 2 carried three panels. Panel C is also obsolete: it reported the strict
role-separation contrast that Section 4.3 shows to be an artefact of the scoring
branch. All three are therefore rebuilt rather than cropped.

Grayscale only: the figures must read in print and under colour-vision
deficiency, so level is encoded by fill and reinforced by a direct label.
"""
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
    "savefig.pad_inches": 0.08,
})

INK = "#000000"
MUTED = "#555555"
RULE = "#aaaaaa"


def save(fig, stem):
    for ext, dpi in (("png", 600), ("pdf", None)):
        fig.savefig(f"{stem}.{ext}", dpi=dpi) if dpi else fig.savefig(f"{stem}.{ext}")
    plt.close(fig)
    print("yazildi:", stem)


# ---------------------------------------------------------------- Figure 2
def figure2():
    rows = [
        ("Full evidence",  "✓", "✓", "✓", "instruction",
         "Full evidence-grounded explanation"),
        ("No-network",     "✓", "✓", "✗", "not applicable",
         "No firm or sector relational attribution"),
        ("No-reliability", "✓", "✗", "✓", "instruction",
         "No confidence, disagreement or abstention cues"),
        ("Free-form",      "score\nonly", "✗", "✗", "not applicable",
         "No evidence list supplied"),
        ("Required role",  "✓", "✓", "✓", "instruction +\nschema field",
         "As full evidence, plus one required field"),
    ]
    headers = ["Condition", "Risk score\n& macro", "Reliability", "Network",
               "Role requirement\ncarried by", "Output role"]
    xs = [0.012, 0.185, 0.285, 0.375, 0.465, 0.645]

    fig = plt.figure(figsize=(11.6, 4.5))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(0.012, 0.955, "Prompt-condition design", fontsize=14, fontweight="bold", va="center")

    y = 0.845
    for x, hh in zip(xs, headers):
        ax.text(x, y, hh, fontsize=10.5, fontweight="bold", va="center", linespacing=1.35)
    ax.plot([0.012, 0.988], [0.775, 0.775], color=INK, lw=1.3)

    y = 0.700
    step = 0.113
    for i, r in enumerate(rows):
        for x, cell in zip(xs, r):
            bold = (x == xs[0])
            ax.text(x, y, cell, fontsize=10.5, va="center",
                    fontweight="bold" if bold else "normal", linespacing=1.3)
        if i < len(rows) - 1:
            ax.plot([0.012, 0.988], [y - step / 2, y - step / 2], color=RULE, lw=0.6)
        y -= step

    ax.plot([0.012, 0.988], [0.108, 0.108], color=INK, lw=1.0)
    ax.text(0.012, 0.062,
            "All five conditions receive the same financial interpretation guide, the same non-advice constraint and the same required output fields.",
            fontsize=9.2, style="italic", va="center", color=MUTED)
    ax.text(0.012, 0.022,
            "The required-role condition is byte-identical to full evidence except for one added output field; the parity check in Section 3.6 verifies this.",
            fontsize=9.2, style="italic", va="center", color=MUTED)
    save(fig, "Figure_2")


# ---------------------------------------------------------------- Figure 3
def figure3():
    fig = plt.figure(figsize=(11.6, 5.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.012, 0.955, "Evaluation workflow", fontsize=14, fontweight="bold", va="center")

    def box(x, y, w, h, title, lines, fs=9.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.006,rounding_size=0.012",
                     fc="white", ec=INK, lw=1.1))
        ax.text(x + w / 2, y + h - 0.052, title, fontsize=10.2, fontweight="bold",
                ha="center", va="center", linespacing=1.25)
        ax.text(x + 0.016, y + h - 0.105,
                "\n".join("• " + s for s in lines), fontsize=fs, va="top", linespacing=1.55)

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13,
                                     lw=1.1, color=INK, shrinkA=0, shrinkB=0))

    box(0.012, 0.47, 0.175, 0.30, "Decision\nsnapshots",
        ["10 monthly contexts", "all monitoring errors", "structured evidence"])
    box(0.225, 0.47, 0.175, 0.30, "Prompt\nconditions",
        ["full", "no-network", "no-reliability", "free-form", "required role"])
    box(0.438, 0.47, 0.205, 0.30, "Generation",
        ["10 × 5 design", "two model families", "matched decoding",
         "pinned model build"])
    box(0.683, 0.615, 0.305, 0.275, "Automatic audit",
        ["span-aware parsing", "citation correctness and invented IDs",
         "uncertainty from the decision record", "criteria scoped by evidence"])
    box(0.683, 0.285, 0.305, 0.275, "Human coding",
        ["16 items: instrument validation",
         "60 items: role statement, blind",
         "two binary questions per note",
         "primary measure of role separation"])
    box(0.438, 0.10, 0.205, 0.265, "Configuration\nstudy",
        ["three parser and decoder settings", "naive variants retained",
         "model-version replication"])

    arrow((0.187, 0.62), (0.225, 0.62))
    arrow((0.400, 0.62), (0.438, 0.62))
    arrow((0.643, 0.68), (0.683, 0.73))
    arrow((0.643, 0.56), (0.683, 0.47))
    arrow((0.540, 0.47), (0.540, 0.365))
    # the human coding validates the automatic instrument, not the configuration study
    ax.add_patch(FancyArrowPatch((0.835, 0.560), (0.835, 0.612), arrowstyle="-|>",
                 mutation_scale=13, lw=1.1, color=INK))
    ax.text(0.848, 0.586, "validates", fontsize=9.0, style="italic",
            color=MUTED, ha="left", va="center")
    save(fig, "Figure_3")


# ---------------------------------------------------------------- Figure 4
def figure4():
    groups = [
        ("Primary model", [
            ("Instruction\n(run 1)", 0, 10, 0.000, 0.278, 0, 0),
            ("Instruction\n(run 2)", 1, 10, 0.018, 0.404, 0, 0),
            ("Required\nfield",      0, 10, 0.000, 0.278, 0, 1),
        ]),
        ("Secondary model", [
            ("Instruction",          3, 10, 0.108, 0.603, 4, 0),
            ("Required\nfield",     10, 10, 0.722, 1.000, 3, 1),
            ("Decoder-\nenforced",  10, 10, 0.722, 1.000, 10, 2),
        ]),
    ]
    fills = ["white", "#bfbfbf", "#4d4d4d"]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    xs, labels, gap = [], [], 0.9
    pos = 0.0
    for gi, (_, bars) in enumerate(groups):
        for bi, b in enumerate(bars):
            xs.append(pos); labels.append(b[0]); pos += 1.0
        pos += gap

    i = 0
    for gi, (_, bars) in enumerate(groups):
        for bi, (lab, k, n, lo, hi, rg, lvl) in enumerate(bars):
            p = k / n
            x = xs[i]
            ax.bar(x, p, width=0.74, color=fills[lvl], edgecolor=INK, lw=1.1, zorder=2)
            ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="none",
                        ecolor=INK, elinewidth=1.1, capsize=5, capthick=1.1, zorder=3)
            ax.text(x, hi + 0.045, f"{k}/{n}", ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", zorder=4)
            ax.plot([x], [rg / n], marker="o", ms=8.5, mfc="white",
                    mec=INK, mew=1.6, ls="none", zorder=5)
            i += 1

    centres = [np.mean(xs[0:3]), np.mean(xs[3:6])]
    for c, (name, _) in zip(centres, groups):
        ax.text(c, -0.255, name, ha="center", va="center",
                fontsize=11.5, fontweight="bold", transform=ax.get_xaxis_transform())
    ax.plot([np.mean([xs[2], xs[3]])] * 2, [0, 1.10], color=RULE, lw=0.8, ls=(0, (4, 4)), zorder=1)

    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10, linespacing=1.3)
    ax.set_ylim(0, 1.12); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_ylabel("Role statement present (human coding)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=RULE, lw=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=8)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    leg_handles = [
        Patch(fc=fills[0], ec=INK, lw=1.1, label="Requirement in the system instruction"),
        Patch(fc=fills[1], ec=INK, lw=1.1, label="Requirement as a required output field"),
        Patch(fc=fills[2], ec=INK, lw=1.1, label="Required field enforced at the decoder"),
        Line2D([], [], marker="o", ms=8.5, mfc="white", mec=INK, mew=1.6, ls="none",
               label="Lexical criterion on the same notes"),
    ]
    ax.legend(handles=leg_handles, loc="upper center", fontsize=9.6, frameon=False,
              ncol=2, bbox_to_anchor=(0.5, -0.40), handlelength=1.6, columnspacing=2.2)
    save(fig, "Figure_4")


figure2(); figure3(); figure4()
