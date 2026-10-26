# Hotfix 26 v3: compact JSON output

Copy `src/26_generate_llm_prompt_conditions.py` into your project `src/` folder.

Then run:

```powershell
python src/26_generate_llm_prompt_conditions.py

python src/28_run_llm_generation.py --provider lmstudio_gemma --condition full --max-records 4 --output outputs/llm_generations/lmstudio_gemma_compact_full.jsonl

python src/29_evaluate_llm_outputs.py --input outputs/llm_generations/lmstudio_gemma_compact_full.jsonl
```

This version reduces output length:
- max 2 macro claims
- max 2 network claims
- max 2 firm/sector claims
- max 25 words per claim
- evidence IDs are returned in arrays

This should prevent Gemma full-condition JSON truncation.
