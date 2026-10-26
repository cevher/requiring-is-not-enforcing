# Which evaluation file produces which table

| Manuscript | File | Notes |
|---|---|---|
| Table 2 | `llama3_main_v2_auto_eval_summary.csv` | Corrected audit (script `32`), Llama 3 main set |
| Table 3 | `instrument_validation_summary.csv`, `instrument_validation_items.csv` | Reader–script agreement (script `45`) |
| Table 4, configuration A | `lmstudio_gptoss20b_10decisions_4conditions_v3_auto_eval_summary.csv` | Naive parser, 1,600-token budget |
| Table 4, configuration B | `gptoss20b_robust_v2_auto_eval_summary.csv` | Same generations, span-aware parser |
| Table 4, configuration C | `gptoss20b_schema_auto_eval_summary.csv` | Constrained decoding, 4,096-token budget |
| Table 5 | `llama3_main_v2_...` and `gptoss20b_schema_...` | Cross-model comparison |
| Section 4.3, first protocol | `human_interrater_agreement.csv`, `human_validity_checks.csv`, `human_condition_means_by_rater.csv` | The 1–5 rating protocol that failed |

Files beginning `_superseded_` are kept only so that the record is complete and are
not used anywhere in the paper. In particular,
`_superseded_gptoss20b_jsonmode_failed_run_*.csv` is the first attempt at the
constrained-decoding run, in which every request failed at the API level
(`status = error` on all rows) because LM Studio rejected
`response_format.type = "json_object"`. The successful rerun, after the provider
configuration was changed to `json_schema`, is `gptoss20b_schema_*`.
