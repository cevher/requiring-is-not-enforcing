# Requiring is not enforcing — replication package

Code, data and evaluation artefacts for the paper *Requiring is not enforcing:
interface constraints and audit validity in evidence-grounded LLM explanation for
financial risk monitoring*.

The study builds a provenance-aware framework in which a language model converts
precomputed risk scores, reliability indicators, macro-financial signals and
network-derived evidence into citable risk notes, and then audits those notes
against the evidence they were given. Everything reported in the paper is
reproducible from this archive.

## What is here

| Path | Contents |
|---|---|
| `src/` | The pipeline, numbered in execution order (`01`–`51`). |
| `data_raw/` | Prices (parquet), macro series from FRED, universe metadata. |
| `data_processed/` | Returns, networks, labels, decision records, evidence records. |
| `outputs/llm_generations/` | Model outputs for every reported condition and run. |
| `outputs/llm_eval/` | Audit results, human coding sheets, item-level keys, statistics. |
| `outputs/tables/`, `figures/` | The tables and figures as they appear in the paper. |
| `docs/` | Model provenance, the file map, rater instructions. |
| `archive/` | Superseded runs and pilot models. Kept, not used. |

## Reproducing the reported results

Install `requirements.txt` (and `requirements_llm.txt` for the generation steps).
Scripts run in numeric order from the repository root. The generation steps
(`28`) need a local model server; everything downstream runs from the archived
generations without one.

The five scripts added for the required-role condition and the corrected audit:

- `46_generate_required_role_condition.py` — builds the fifth prompt condition and
  runs a parity check that the condition differs from full evidence in exactly one
  output-schema key.
- `47_evaluate_llm_outputs_v3.py` — the audit used in the paper. Reports the
  published coding and the corrected coding side by side.
- `48_binary_outcome_statistics.py` — Wilson intervals and Fisher's exact tests.
- `49_build_role_statement_sheets.py` — builds the blind human coding workbooks.
- `50_role_statement_analysis.py` — the human coding, and the lexical criterion
  validated against it.

`docs/FILE_MAP.md` says which file produces which table and figure.

## Three things stated plainly

**The naive variants are retained.** Section 4 of the paper reports what an
uncorrected audit returns on the same outputs. Those scorings are in
`archive/superseded/` and in the `*_published` columns of the v3 audit, so the
naive-variant numbers remain reproducible.

**One model tag was not pinned.** The first generation run requested the primary
model under the moving tag `llama3:latest`, whose digest was not recorded and is
no longer recoverable. The full-evidence condition was regenerated under the
pinned tag `llama3:8b-instruct-q4_0` before the required-role condition was run,
and both runs are reported. `docs/MODEL_PROVENANCE.md` has the digests and
quantisation settings.

**Two pilot models are in the archive.** `google/gemma-4-e4b` (24 outputs) and
`qwen/qwen3-coder-30b` (4 outputs) were prompt- and schema-debugging runs made
while the compact output schema was being settled. They support no reported
result and are kept so the record is complete.

## Data sources

Daily adjusted closing prices from Yahoo Finance; VIX, NFCI and Treasury yields
from FRED. `data_raw/prices/prices_long.csv` (72 MB) is excluded from version
control and is regenerated from the tracked parquet file by
`src/01_download_prices.py`.

## Licence

MIT, see `LICENSE`.
