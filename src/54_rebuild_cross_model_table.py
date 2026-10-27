"""Rebuild the cross-model table from the corrected coding, and audit every
other reported table for values carried over from the defective one.

Why this exists. Section 4.3 reports that the published role-separation branch
could not fail in the no-network condition and returned nothing in the
no-reliability condition, because it keyed on the condition NAME rather than on
whether network evidence was present. The corrected branch is scoped by
evidence. Table 3 was regenerated from the corrected branch; the cross-model
table was not, and still carries 1.00 for no-network -- the artefact value, in a
paper whose argument is that this value is an artefact.

Writes outputs/tables/cross_model_corrected.csv and prints a side-by-side audit.
"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "llm_eval"
TABLES = ROOT / "outputs" / "tables"

RUNS = [("Llama 3", "llama3_main_v3_auto_eval_results.csv"),
        ("GPT-OSS", "gptoss20b_v3_auto_eval_results.csv")]
ORDER = ["full", "no_network", "no_reliability", "free_form"]
PRETTY = {"full": "Full evidence", "no_network": "No-network",
          "no_reliability": "No-reliability", "free_form": "Free-form"}

PUBLISHED_TABLE7 = {            # what the manuscript currently prints
    ("Llama 3", "no_network"): "1.00", ("GPT-OSS", "no_network"): "1.00",
    ("Llama 3", "no_reliability"): "—", ("GPT-OSS", "no_reliability"): "—",
    ("Llama 3", "full"): "0.00", ("GPT-OSS", "full"): "0.40",
    ("Llama 3", "free_form"): "—", ("GPT-OSS", "free_form"): "—",
}


def frac(rows, col):
    vals = [r[col] for r in rows if r[col] not in ("", "None", "nan")]
    if not vals:
        return None
    return sum(int(float(v)) for v in vals), len(vals)


def fmt(f):
    return "n.a." if f is None else f"{f[0] / f[1]:.2f}"


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    data = {}
    for model, fname in RUNS:
        rows = list(csv.DictReader((EVAL / fname).open(encoding="utf8")))
        g = defaultdict(list)
        for r in rows:
            g[r["condition"]].append(r)
        for cond in ORDER:
            rs = g.get(cond, [])
            data[(model, cond)] = {
                "n": len(rs),
                "has_net": sum(1 for r in rs if r["has_network_evidence"] in ("1", "True", "true")),
                "published": frac(rs, "role_separation_strict_published"),
                "uniform": frac(rs, "role_separation_strict_uniform"),
                "citation": [float(r["citation_correctness"]) for r in rs
                             if r["citation_correctness"] not in ("", "None", "nan")],
                "invented": [int(float(r["n_invented_evidence_ids"])) for r in rs],
                "json": frac(rs, "json_recoverable"),
                "unc": frac(rs, "uncertainty_compliance"),
                "unc_req": sum(1 for r in rs
                               if r["requires_uncertainty_truth"] in ("1", "True", "true")),
                "unc_req_met": sum(1 for r in rs
                                   if r["requires_uncertainty_truth"] in ("1", "True", "true")
                                   and r["uncertainty_present"] in ("1", "True", "true")),
            }

    print("=" * 94)
    print("ROLE SEPARATION: what the manuscript prints vs the corrected branch")
    print("=" * 94)
    print(f'{"Condition":16s} {"Model":9s} {"net ev.":>8s} {"published":>10s} '
          f'{"in Table 7":>11s} {"CORRECTED":>10s}   verdict')
    print("-" * 94)
    drift = 0
    for cond in ORDER:
        for model, _ in RUNS:
            d = data[(model, cond)]
            cur = PUBLISHED_TABLE7[(model, cond)]
            corrected = fmt(d["uniform"])
            bad = (cur != corrected) and not (cur == "—" and corrected == "n.a.")
            drift += bad
            print(f'{PRETTY[cond]:16s} {model:9s} {d["has_net"]:>3d}/{d["n"]:<4d} '
                  f'{fmt(d["published"]):>10s} {cur:>11s} {corrected:>10s}   '
                  f'{"<-- STALE" if bad else "ok"}')

    print("\n" + "=" * 94)
    print("UNCERTAINTY: the metric is vacuous where the decision record requires nothing")
    print("=" * 94)
    print(f'{"Condition":16s} {"Model":9s} {"reported":>9s} {"required":>9s} {"met|required":>13s}')
    print("-" * 94)
    for cond in ORDER:
        for model, _ in RUNS:
            d = data[(model, cond)]
            print(f'{PRETTY[cond]:16s} {model:9s} {fmt(d["unc"]):>9s} '
                  f'{d["unc_req"]:>4d}/{d["n"]:<4d} {d["unc_req_met"]:>6d}/{d["unc_req"]:<6d}')

    out = TABLES / "cross_model_corrected.csv"
    with out.open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "model", "n", "has_network_evidence",
                    "json_recoverable", "citation_correctness", "invented_ids_mean",
                    "uncertainty_reported", "uncertainty_required",
                    "uncertainty_met_of_required",
                    "role_separation_published", "role_separation_corrected"])
        for cond in ORDER:
            for model, _ in RUNS:
                d = data[(model, cond)]
                cc = f"{sum(d['citation']) / len(d['citation']):.4f}" if d["citation"] else "n.d."
                w.writerow([PRETTY[cond], model, d["n"], d["has_net"], fmt(d["json"]), cc,
                            f"{sum(d['invented']) / d['n']:.2f}",
                            fmt(d["unc"]), f"{d['unc_req']}/{d['n']}",
                            f"{d['unc_req_met']}/{d['unc_req']}",
                            fmt(d["published"]), fmt(d["uniform"])])
    print(f"\nyazildi: {out.relative_to(ROOT)}")
    print(f"\nTable 7 cells carrying the defective coding: {drift}")
    return drift


if __name__ == "__main__":
    main()
