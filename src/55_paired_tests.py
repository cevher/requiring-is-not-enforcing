"""Paired exact tests for every binary contrast the paper reports.

Why this replaces Fisher's exact test. The same ten decision snapshots are
rendered under every condition and by both models, so the two arms of each
contrast are not independent samples: they are repeated measurements on the
same units. Fisher's exact test conditions on both margins and assumes
independence, which here understates the p-value because the within-pair
correlation is positive. McNemar's exact test conditions on the discordant
pairs instead, which is what a paired binary design supports.

Both are reported. The point is not that one number replaces another but that
the design determines which test is readable, and the difference is large
enough to matter: 0.011 against 0.031 for the uncertainty contrast.

A floor worth stating. With ten pairs the smallest attainable two-sided
McNemar p is 2 / 2^10 = 0.00195. A p of 0.002 is therefore the limit of the
sample, not a measure of how large the effect is.

Multiplicity. Holm is applied within a PRE-SPECIFIED family of three primary
hypotheses (PRIMARY_FAMILY below), not across every contrast computed. Two
choices there could each be made to flatter the result, so both are stated.
Null contrasts -- decoder enforcement adding nothing, the two runs agreeing --
are reported but not entered into the family: they are not tests the paper
needs to win, and including them would inflate the correction for no
inferential reason. The uncertainty contrast is one hypothesis tested in two
model families, a replication rather than two hypotheses, so it enters once at
the less favourable of its two p-values. Holm over every contrast is printed as
a sensitivity line so neither choice is hidden.

Writes outputs/llm_eval/paired_tests.csv
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "llm_eval"

TRUE = ("1", "True", "true")

# Blocks whose coding comes from round 2: the new condition and the one it is
# contrasted against, both coded in the same pass.
ROUND2_BLOCKS = {"L-OCT-RR-ENF", "L-OCT-RR"}


def fisher_two_sided(a, b, c, d):
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2

    def pmf(k):
        return math.comb(r1, k) * math.comb(r2, c1 - k) / math.comb(n, c1)

    p_obs = pmf(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return sum(p for k in range(lo, hi + 1) if (p := pmf(k)) <= p_obs + 1e-12)


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def mcnemar_floor(n_pairs):
    return 2 / 2 ** n_pairs if n_pairs else 1.0


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out = [1.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        adj = max(running, min(1.0, pvals[i] * (len(pvals) - rank)))
        running = adj
        out[i] = adj
    return out


def paired(arm_a, arm_b):
    """arm_* : dict unit -> 0/1. Returns a, b, c, d over shared units."""
    units = sorted(set(arm_a) & set(arm_b))
    t = {"a": 0, "b": 0, "c": 0, "d": 0}
    for u in units:
        x, y = arm_a[u], arm_b[u]
        t["a" if (x and y) else "b" if x else "c" if y else "d"] += 1
    return t["a"], t["b"], t["c"], t["d"], len(units)


# ----------------------------------------------------------- human coding
def load_role_separation():
    """S_NOT consensus by block and decision.

    Round 1 covers the six original blocks. Round 2 adds the decoder-enforced
    primary-model block and re-codes the unenforced one, so that the contrast
    between them is coded in a single pass by the same readers. Where a block
    appears in both rounds, round 2 is used for that reason; test-retest
    agreement on S_NOT across the two rounds is 20/20, so the choice does not
    move any number.
    """
    out = defaultdict(dict)
    for fname in ("role_statement_items.csv", "role_statement_r2_items.csv"):
        path = EVAL / fname
        if not path.exists():
            continue
        round2 = fname.endswith("r2_items.csv")
        for r in csv.DictReader(path.open(encoding="utf8")):
            v = r.get("S_NOT_cons", "")
            if v in ("", "None", "nan"):
                continue           # readers disagreed: no consensus value
            blk, did = r["block"], r["decision_id"]
            if round2 and blk not in ROUND2_BLOCKS:
                continue           # distractor repeats stay with round 1
            out[blk][did] = int(float(v))
    return out


# ------------------------------------------------------- automatic audit
def load_uncertainty(fname):
    """Returns (compliance, required) keyed by decision, per condition."""
    comp = defaultdict(dict)
    req = defaultdict(dict)
    for r in csv.DictReader((EVAL / fname).open(encoding="utf8")):
        cond, did = r["condition"], r["decision_id"]
        comp[cond][did] = int(r["uncertainty_present"] in TRUE)
        req[cond][did] = r["requires_uncertainty_truth"] in TRUE
    return comp, req


CONTRASTS = []

# Pre-specified primary family; reasoning in the module docstring.
PRIMARY_FAMILY = {
    "Secondary model: instruction -> required field",
    "Across models under a required field",
    "Primary model: evidence list -> none, where required",
}

# One hypothesis, two model families: enters the correction once.
UNCERTAINTY_REPLICATION = {
    "Primary model: evidence list -> none, where required",
    "Secondary model: evidence list -> none, where required",
}


def add(name, family, arm_a, arm_b, label_a, label_b):
    a, b, c, d, n = paired(arm_a, arm_b)
    CONTRASTS.append({
        "contrast": name, "family": family,
        "arm_a": label_a, "arm_b": label_b,
        "k_a": a + b, "k_b": a + c, "n_pairs": n,
        "both": a, "only_a": b, "only_b": c, "neither": d,
        "fisher_p": fisher_two_sided(a + b, n - (a + b), a + c, n - (a + c)),
        "mcnemar_p": mcnemar_exact(b, c),
        "mcnemar_floor": mcnemar_floor(b + c),
    })


def main():
    S = load_role_separation()
    have = set(S)

    pairs = [
        ("Secondary model: instruction -> required field", "role separation",
         "G-FULL", "G-RR", "instruction", "required field"),
        ("Secondary model: required field -> decoder enforced", "role separation",
         "G-RR", "G-RR-ENF", "required field", "decoder enforced"),
        ("Primary model: instruction -> required field", "role separation",
         "L-OCT-FULL", "L-OCT-RR", "instruction", "required field"),
        ("Primary model: required field -> decoder enforced", "role separation",
         "L-OCT-RR", "L-OCT-RR-ENF", "required field", "decoder enforced"),
        ("Primary model: run 1 -> run 2 (pinned build)", "replication",
         "L-APR-FULL", "L-OCT-FULL", "moving tag", "pinned build"),
        ("Across models under a required field", "role separation",
         "L-OCT-RR", "G-RR", "primary", "secondary"),
        ("Across models under decoder enforcement", "role separation",
         "L-OCT-RR-ENF", "G-RR-ENF", "primary", "secondary"),
        ("Primary model: instruction -> decoder enforced", "role separation",
         "L-OCT-FULL", "L-OCT-RR-ENF", "instruction", "decoder enforced"),
    ]
    for name, fam, A, B, la, lb in pairs:
        if A in have and B in have:
            add(name, fam, S[A], S[B], la, lb)
        else:
            missing = [x for x in (A, B) if x not in have]
            print(f"  atlandi ({', '.join(missing)} yok): {name}")

    for model, fname in (("Primary", "llama3_main_v3_auto_eval_results.csv"),
                         ("Secondary", "gptoss20b_v3_auto_eval_results.csv")):
        comp, req = load_uncertainty(fname)
        if "full" in comp and "free_form" in comp:
            add(f"{model} model: evidence list -> none, all snapshots",
                "uncertainty", comp["full"], comp["free_form"], "full", "free-form")
            need = {d for d, v in req["full"].items() if v}
            add(f"{model} model: evidence list -> none, where required",
                "uncertainty",
                {d: v for d, v in comp["full"].items() if d in need},
                {d: v for d, v in comp["free_form"].items() if d in need},
                "full", "free-form")

    for c in CONTRASTS:
        c["in_primary_family"] = c["contrast"] in PRIMARY_FAMILY
        c["holm_p"] = ""

    fam = [c for c in CONTRASTS if c["in_primary_family"]]
    repl = [c["mcnemar_p"] for c in CONTRASTS
            if c["contrast"] in UNCERTAINTY_REPLICATION]
    fam_p = [max(repl) if (repl and c["contrast"] in UNCERTAINTY_REPLICATION)
             else c["mcnemar_p"] for c in fam]
    for c, p in zip(fam, holm(fam_p)):
        c["holm_p"] = p

    w = max(len(c["contrast"]) for c in CONTRASTS)
    print(f'\n  {"contrast":{w}s} {"k_a":>5s} {"k_b":>5s} {"n":>3s} {"b":>3s} {"c":>3s} '
          f'{"Fisher":>9s} {"McNemar":>9s} {"Holm":>9s}')
    print("-" * (w + 58))
    for c in CONTRASTS:
        mark = "* " if c["in_primary_family"] else "  "
        hp = f'{c["holm_p"]:9.6f}' if c["holm_p"] != "" else f'{"-":>9s}'
        print(f'{mark}{c["contrast"]:{w}s} {c["k_a"]:5d} {c["k_b"]:5d} {c["n_pairs"]:3d} '
              f'{c["only_a"]:3d} {c["only_b"]:3d} {c["fisher_p"]:9.6f} '
              f'{c["mcnemar_p"]:9.6f} {hp}')
    print("\n* onceden belirlenmis birincil aile; Holm yalnizca onun icinde.")

    wide = holm([c["mcnemar_p"] for c in CONTRASTS])
    sig = [(c["contrast"], p) for c, p in zip(CONTRASTS, wide) if p < 0.05]
    print(f'duyarlilik: Holm {len(CONTRASTS)} karsilastirmanin hepsine uygulansaydi, '
          f'p < 0.05 kalan {len(sig)}:')
    for n, pv in sig:
        print(f'    {n}  ({pv:.6f})')

    out = EVAL / "paired_tests.csv"
    with out.open("w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=list(CONTRASTS[0]))
        wr.writeheader()
        wr.writerows(CONTRASTS)
    print(f"\nyazildi: {out.relative_to(ROOT)}")
    print(f"on cift icin McNemar tabani: {mcnemar_floor(10):.5f}")


if __name__ == "__main__":
    main()
