# Phase 6 final hotfix: financial interpretation guide + evaluator false-positive fix

Copy these files into your project's `src/` folder and overwrite old versions:

- src/26_generate_llm_prompt_conditions.py
- src/29_evaluate_llm_outputs.py

What changed:

1. `26_generate_llm_prompt_conditions.py`
   - Adds a financial interpretation guide:
     - Do not call VIX below 20 high stress.
     - Interpret negative NFCI as looser-than-average / less tight conditions.
     - Do not infer firm-level stress directly from network density or centrality.
     - Treat network evidence as relational context / co-movement / attribution evidence.
     - Do not claim causal contagion/spillover/transmission.
   - Keeps compact JSON schema.

2. `29_evaluate_llm_outputs.py`
   - Stops counting “not investment advice” disclaimers as investment-advice violations.
   - Still flags actual buy/sell/short/trading instruction language.

Run:

```powershell
python src/26_generate_llm_prompt_conditions.py

python src/28_run_llm_generation.py --provider lmstudio_gptoss20b --output outputs/llm_generations/lmstudio_gptoss20b_10decisions_4conditions_v2.jsonl

python src/29_evaluate_llm_outputs.py --input outputs/llm_generations/lmstudio_gptoss20b_10decisions_4conditions_v2.jsonl
```

Optional Llama3 rerun:

```powershell
python src/28_run_llm_generation.py --provider ollama_local_llama --output outputs/llm_generations/ollama_llama_native_10decisions_4conditions_v2.jsonl

python src/29_evaluate_llm_outputs.py --input outputs/llm_generations/ollama_llama_native_10decisions_4conditions_v2.jsonl
```
