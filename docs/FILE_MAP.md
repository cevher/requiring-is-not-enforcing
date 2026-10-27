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
| Third, decoder-enforced condition (30 items: 10 new, 10 paired, 10 repeats) | `role_statement_r2_sheet_rater{1,2}.xlsx` | `role_statement_r2_key_rater{1,2}.csv` | `src/56_round2_role_analysis.py` |


## Added after the first submission draft

The decoder-enforced condition on the primary model, the paired tests, and two
corrections to the monitoring layer. Each is a separate script so the change is
auditable against the earlier state.

| Script | What it does |
|---|---|
| `src/52_check_ollama_schema.py` | Pre-flight: verifies that this Ollama build actually enforces a JSON schema passed in `format`, rather than silently ignoring it |
| `src/53_build_round2_role_sheets.py` | Builds the third coding round: the new condition, the same ten decisions unenforced, and ten repeats from round two |
| `src/54_rebuild_cross_model_table.py` | Rebuilds the cross-model table from the corrected coding and audits every reported table for values carried over from the defective branch |
| `src/55_paired_tests.py` | McNemar's exact test for every binary contrast, with the discordant counts, Fisher alongside, and Holm within a pre-specified family |
| `src/56_round2_role_analysis.py` | Round-two agreement, test-retest against round one, and the new condition's result |

`src/28_run_llm_generation.py` gained one branch: the Ollama adapter now accepts
a JSON schema in `format`. The branch fires only for a provider that declares
`format_json_schema`, so every earlier run is byte-identical to before.

### Two corrections to the monitoring layer

`src/33_leakfree_label_variants.py` shifted the forward-outcome thresholds by one
day rather than by the full horizon, so the threshold itself carried about twenty
days of forward information. `src/15_expanding_window_backtest.py` split train
from test on the calendar year with no gap, so a training label dated 31 December
resolved over January of the test year.

Both are corrected. The effect is reported in Section 5.5 and measured here:

- 12 of 4,947 daily L3 labels change, 16 of 4,947 L4 labels, and **no monthly
  snapshot**;
- one early snapshot (2006-12-29) leaves the training set to the longer burn-in,
  which moves the L3 and L4 rows of Table 2 by at most 0.052 and leaves the
  method ordering at the top unchanged;
- the decision layer's flag rates move by under 1.5 points, and **none of the ten
  snapshots in the generation experiment changes any categorical field** -- only
  its continuous scores, by at most 0.018.

`archive/pre_embargo/` holds the labels, backtest outputs and decision records as
they stood before these corrections. They are kept because the generation
experiment was run from those decision records: the prompts in
`data_processed/evidence/llm_prompt_conditions*.jsonl` and every file in
`outputs/llm_generations/` derive from `archive/pre_embargo/evidence/`, while the
decision-layer statistics reported in Section 4.1 derive from the corrected files
in `data_processed/evidence/`.
