# Phase 6 patch: LLM decision-note generation and evaluation

This patch adds the LLM decision-note experiment layer.

## Model decision

Recommended first run:
- Primary local model: LM Studio Qwen3.6 35B A3B
- Lightweight local robustness model: LM Studio Gemma 4 E4B
- Optional later: Ollama local/cloud models

The LLM is not used as a predictor. It generates evidence-grounded systemic-risk decision notes from v5 evidence records.

## Copy files

Copy the `src/` files into your project `src/` folder.

Copy `llm_provider_config.example.json` into the project root and rename it:

```powershell
copy llm_provider_config.example.json llm_provider_config.json
```

Edit the model IDs.

## Step 0: Start LM Studio server

In LM Studio:
1. Load Qwen or Gemma.
2. Start local server.
3. Usually endpoint is:

```text
http://localhost:1234/v1
```

Check exact model IDs:

```powershell
python src/27_list_openai_compatible_models.py --base-url http://localhost:1234/v1
```

Copy the exact Qwen model ID into `llm_provider_config.json`.

## Step 1: select evaluation decisions

Pilot:

```powershell
python src/25_select_llm_eval_sample.py --target-size 10
```

Main:

```powershell
python src/25_select_llm_eval_sample.py --target-size 40
```

## Step 2: generate prompt conditions

```powershell
python src/26_generate_llm_prompt_conditions.py
```

Creates:

```text
data_processed/evidence/llm_prompt_conditions.jsonl
```

Conditions:
- full
- no_network
- no_reliability
- free_form

## Step 3: run pilot generation

Qwen:

```powershell
python src/28_run_llm_generation.py --provider lmstudio_qwen --max-records 8
```

Gemma:

```powershell
python src/28_run_llm_generation.py --provider lmstudio_gemma --max-records 8
```

Only full condition:

```powershell
python src/28_run_llm_generation.py --provider lmstudio_qwen --condition full --max-records 10
```

Resume:

```powershell
python src/28_run_llm_generation.py --provider lmstudio_qwen --resume
```

## Step 4: automatic evaluation

```powershell
python src/29_evaluate_llm_outputs.py --input outputs/llm_generations/lmstudio_qwen_generations.jsonl
```

## Step 5: human review sheet

```powershell
python src/30_prepare_human_review_sheet.py ^
  --generations outputs/llm_generations/lmstudio_qwen_generations.jsonl ^
  --auto-eval outputs/llm_eval/lmstudio_qwen_generations_auto_eval_results.csv
```

## Ollama Cloud

Set API key:

```powershell
setx OLLAMA_API_KEY "your_key_here"
```

Reopen terminal, then:

```powershell
python src/28_run_llm_generation.py --provider ollama_cloud_gptoss20b --condition full --max-records 10
```

## Outputs

Generations:
```text
outputs/llm_generations/<provider>_generations.jsonl
```

Automatic evaluation:
```text
outputs/llm_eval/<provider>_auto_eval_results.csv
outputs/llm_eval/<provider>_auto_eval_summary.csv
```

Human review:
```text
outputs/llm_eval/<provider>_human_review_sheet.csv
```

## Design rule

Networks are relational/provenance evidence, not the primary prediction engine.
Risk scoring and relational explanation must remain separated.
