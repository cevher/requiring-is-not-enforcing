# Which file produces which table and figure

Table and figure numbers follow the submitted manuscript.

| Manuscript | Produced by | Artefact |
|---|---|---|
| Table 1 — design constants of the reliability layer | `src/40_reliability_sensitivity.py` | `outputs/tables/` sensitivity grid; full 21-row version is supplementary Table S4 |
| Table 2 — out-of-sample monitoring under five label definitions | `src/33_leakfree_label_variants.py` | `outputs/tables/`; L2 and the no-VIX ablation rows are supplementary Table S1 |
| Table 3 — automatic audit, primary model | `src/47_evaluate_llm_outputs_v3.py` | `outputs/llm_eval/llama3_main_v3_auto_eval_summary.csv` |
| Table 4 — reader agreement and validation of the lexical criterion | `src/50_role_statement_analysis.py` | `outputs/llm_eval/role_statement_items.csv` |
| Table 5 — role separation by condition under human coding | `src/50_role_statement_analysis.py`, intervals from `src/48_binary_outcome_statistics.py` | `outputs/llm_eval/role_statement_summary.csv`, `binary_outcome_statistics_proportions.csv` |
| Table 6 — secondary model under three evaluation configurations | `src/47_evaluate_llm_outputs_v3.py` on the three archived runs | configuration A `archive/superseded/lmstudio_gptoss20b_10decisions_4conditions_v3_*`; B re-scored from the same generations; C `outputs/llm_eval/gptoss20b_v3_*` |
| Table 7 — cross-model comparison | `src/47_evaluate_llm_outputs_v3.py` | `llama3_main_v3_*` and `gptoss20b_v3_*` |
| Figure 1 — framework architecture | drawn, not generated | `figures/Figure_1.pdf` |
| Figures 2–4 — conditions, workflow, role separation | `src/51_make_figures.py` | `figures/Figure_2.pdf`, `Figure_3.pdf`, `Figure_4.pdf` |

## Generations by condition

| Condition | Primary model | Secondary model |
|---|---|---|
| full, no-network, no-reliability, free-form | `ollama_llama_native_10decisions_4conditions_final_clean.jsonl` (run 1) | `gptoss20b_schema.jsonl` |
| full, regenerated under the pinned tag (run 2) | `ollama_llama_full_rerun_q4_clean.jsonl` | — |
| required role | `ollama_llama_required_role_q4.jsonl` | `gptoss20b_required_role.jsonl` |
| required role, decoder-enforced | — | `gptoss20b_required_role_enforced.jsonl` |

`ollama_llama_full_rerun_q4.jsonl` is run 2 before the one transport-level failure
was retried; `..._retry.jsonl` is that retry; `..._clean.jsonl` is the merged set
used in the paper.

## The naive variants

The paper reports what an uncorrected audit returns on the same outputs. Those
numbers come from:

- the `*_published` columns written by `src/47_evaluate_llm_outputs_v3.py`, which
  reproduce the earlier scoring branch exactly;
- `src/29_evaluate_llm_outputs.py` and `src/32_evaluate_llm_outputs_v2.py`, the
  earlier audits, kept unchanged;
- `archive/superseded/`, which holds the runs and scorings they were applied to.

## The human rounds

| Round | Sheets | Keys | Analysis |
|---|---|---|---|
| First, unanchored 1–5 rating (failed) | `archive/superseded/` | — | `src/34_human_review_statistics.py`, `src/42_human_review_analysis.py` |
| Second, instrument validation (16 items) | `binary_rating_sheet_rater{1,2}_filled.xlsx` | `binary_rating_key_rater{1,2}.csv` | `src/45_instrument_validation.py` |
| Second, role statement coding (60 items) | `role_statement_sheet_rater{1,2}.xlsx` | `role_statement_key_rater{1,2}.csv` | `src/50_role_statement_analysis.py` |
