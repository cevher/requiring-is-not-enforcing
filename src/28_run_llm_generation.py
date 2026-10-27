"""
28_run_llm_generation.py

Provider-agnostic LLM generation runner.

Supports:
1. OpenAI-compatible chat/completions endpoint:
   - LM Studio local server
   - Ollama local /v1
   - OpenAI API

2. Ollama native /api/chat endpoint:
   - Ollama local native
   - Ollama Cloud native
"""

import argparse
import json
import os
import time
from pathlib import Path
import requests

from config import EVIDENCE_DIR, PROJECT_ROOT
from common import ensure_dirs

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "llm_generations"
ensure_dirs(OUTPUT_DIR)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "llm_provider_config.json"


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# LM Studio (>=0.4.x) rejects response_format.type = "json_object" and accepts only
# "json_schema" or "text". Ollama's `format: "json"`, used for the Llama3 run,
# constrains the decoder to emit *some* valid JSON object, not a particular shape.
# The faithful LM Studio equivalent is therefore an open object schema: structural
# constraint, no field requirements. A strict per-field schema would be a STRONGER
# constraint than the Llama3 run received and would make the two models
# non-comparable in the opposite direction.
PERMISSIVE_OBJECT_SCHEMA = {
    "name": "open_object",
    "strict": False,
    "schema": {"type": "object", "additionalProperties": True},
}

# Optional stricter variant, for a separate run only. Do not mix it into the
# main comparison.
DECISION_NOTE_SCHEMA = {
    "name": "risk_decision_note",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "risk_summary": {"type": "string"},
            "scoring_interpretation": {"type": "string"},
            "macro_evidence": {"type": "string"},
            "relational_network_evidence": {"type": "string"},
            "firm_or_sector_attribution": {"type": "string"},
            "uncertainty_note": {"type": "string"},
            "non_advice_disclaimer": {"type": "string"},
            "cited_evidence_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["risk_summary", "uncertainty_note", "non_advice_disclaimer"],
        "additionalProperties": True,
    },
}

# Decoder-enforced variant of the required-role condition. Identical to
# PERMISSIVE_OBJECT_SCHEMA except that one field is enforced by the decoder
# rather than merely described in the prompt. Single manipulation: everything
# else stays unconstrained, as in the main run.
REQUIRED_ROLE_SCHEMA = {
    "name": "required_role_object",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "relational_evidence_role_statement": {"type": "string"},
        },
        "required": ["relational_evidence_role_statement"],
        "additionalProperties": True,
    },
}

NAMED_SCHEMAS = {
    "permissive_object": PERMISSIVE_OBJECT_SCHEMA,
    "decision_note": DECISION_NOTE_SCHEMA,
    "required_role": REQUIRED_ROLE_SCHEMA,
}


def resolve_schema(value):
    if isinstance(value, str):
        if value not in NAMED_SCHEMAS:
            raise ValueError(f"Unknown named schema: {value}. "
                             f"Known: {sorted(NAMED_SCHEMAS)}")
        return NAMED_SCHEMAS[value]
    return value


def load_provider(config_path, provider_name):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    for p in config.get("providers", []):
        if p.get("provider_name") == provider_name:
            return p
    raise ValueError(f"Provider not found in {config_path}: {provider_name}")


def call_openai_compatible(provider, messages):
    base_url = provider["base_url"].rstrip("/")
    url = base_url + "/chat/completions"

    headers = {"Content-Type": "application/json"}
    api_key_env = provider.get("api_key_env")
    if api_key_env:
        key = os.environ.get(api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": provider["model"],
        "messages": messages,
        "temperature": provider.get("temperature", 0.1),
        "max_tokens": provider.get("max_tokens", 1200),
        "stream": False,
    }

    schema = provider.get("response_format_json_schema")
    if schema:
        payload["response_format"] = {"type": "json_schema",
                                      "json_schema": resolve_schema(schema)}
    elif provider.get("response_format_json", False):
        payload["response_format"] = {"type": "json_object"}

    r = requests.post(url, headers=headers, json=payload, timeout=provider.get("timeout", 180))
    if not r.ok:
        # Without the body an HTTP 400 from a local server is undiagnosable.
        raise requests.HTTPError(f"{r.status_code} {r.reason} for {url} :: {r.text[:800]}")
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    return text, data


def call_ollama_native(provider, messages):
    base_url = provider["base_url"].rstrip("/")
    url = base_url + "/chat"

    headers = {"Content-Type": "application/json"}
    api_key_env = provider.get("api_key_env")
    if api_key_env:
        key = os.environ.get(api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": provider["model"],
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": provider.get("temperature", 0.1),
            "num_predict": provider.get("max_tokens", 1200),
        },
    }

    # Ollama >= 0.5 accepts a JSON schema in `format`, not only the string "json".
    # This branch is additive: it fires only when a provider declares
    # format_json_schema, so every earlier run is byte-identical to before.
    # The schema object is the same one the LM Studio runs receive, so decoder
    # enforcement is the single manipulation across the two model families.
    if provider.get("format_json_schema"):
        payload["format"] = resolve_schema(provider["format_json_schema"])["schema"]
    elif provider.get("format_json", False):
        payload["format"] = "json"

    r = requests.post(url, headers=headers, json=payload, timeout=provider.get("timeout", 240))
    r.raise_for_status()
    data = r.json()
    text = data.get("message", {}).get("content", "")
    return text, data


def generate(provider, messages):
    adapter = provider.get("adapter", "openai_compatible")
    if adapter == "openai_compatible":
        return call_openai_compatible(provider, messages)
    if adapter == "ollama_native":
        return call_ollama_native(provider, messages)
    raise ValueError(f"Unknown adapter: {adapter}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--input", default=str(EVIDENCE_DIR / "llm_prompt_conditions.jsonl"))
    parser.add_argument("--output", default="")
    parser.add_argument("--condition", default="")
    parser.add_argument("--decision-id", default="")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    provider = load_provider(args.config, args.provider)
    prompts = read_jsonl(args.input)

    if args.condition:
        prompts = [p for p in prompts if p.get("condition") == args.condition]
    if args.decision_id:
        prompts = [p for p in prompts if p.get("decision_id") == args.decision_id]
    if args.max_records and args.max_records > 0:
        prompts = prompts[:args.max_records]

    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"{args.provider}_generations.jsonl"

    done_prompt_ids = set()
    if args.resume and output_path.exists():
        for r in read_jsonl(output_path):
            done_prompt_ids.add(r.get("prompt_id"))

    print(f"Provider: {args.provider}")
    print(f"Model: {provider.get('model')}")
    print(f"Adapter: {provider.get('adapter')}")
    print(f"Prompts to process: {len(prompts)}")
    print(f"Output: {output_path}")

    for i, prompt in enumerate(prompts, start=1):
        if prompt["prompt_id"] in done_prompt_ids:
            print(f"[{i}/{len(prompts)}] skip existing {prompt['prompt_id']}")
            continue

        print(f"[{i}/{len(prompts)}] {prompt['prompt_id']}")
        start = time.time()

        try:
            response_text, raw_response = generate(provider, prompt["messages"])
            status = "ok"
            error = ""
        except Exception as e:
            response_text = ""
            raw_response = {}
            status = "error"
            error = repr(e)

        elapsed = time.time() - start

        record = {
            "provider_name": args.provider,
            "model": provider.get("model"),
            "adapter": provider.get("adapter"),
            "prompt_id": prompt.get("prompt_id"),
            "decision_id": prompt.get("decision_id"),
            "date": prompt.get("date"),
            "condition": prompt.get("condition"),
            "status": status,
            "error": error,
            "elapsed_seconds": elapsed,
            "response_text": response_text,
            "allowed_evidence_ids": prompt.get("allowed_evidence_ids", []),
            "decision_context": prompt.get("decision_context", {}),
            "prompt_payload": prompt.get("payload", {}),
        }

        append_jsonl(output_path, record)
        if args.sleep > 0:
            time.sleep(args.sleep)

    print("Done.")


if __name__ == "__main__":
    main()
