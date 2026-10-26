"""
39_probe_lmstudio_endpoint.py

Diagnoses a local OpenAI-compatible endpoint (LM Studio / Ollama) before a full
generation run, so a 400 does not cost 40 wasted calls.

It (1) lists the models the server reports as available, and (2) sends one short
prompt under four decoding configurations, reporting which are accepted:

  A  no response_format                      (always supported)
  B  response_format = {"type":"json_object"}
  C  response_format = {"type":"json_schema", ...}   (LM Studio structured output)
  D  no response_format, large max_tokens

The full-evidence run needs a configuration that both constrains structure and
allows a long enough completion. Whichever of B/C is accepted should be used for
the GPT-OSS re-run so that its decoding constraint matches the Ollama
`format: json` used for Llama3.

Usage:
  python src/39_probe_lmstudio_endpoint.py
  python src/39_probe_lmstudio_endpoint.py --base-url http://127.0.0.1:1234/v1 --model openai/gpt-oss-20b
"""

import argparse
import json

import requests

OPEN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "open_object", "strict": False,
                    "schema": {"type": "object", "additionalProperties": True}},
}

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "risk_note",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "risk_summary": {"type": "string"},
                "uncertainty_note": {"type": "string"},
                "non_advice_disclaimer": {"type": "string"},
            },
            "required": ["risk_summary", "uncertainty_note", "non_advice_disclaimer"],
            "additionalProperties": False,
        },
    },
}

MESSAGES = [
    {"role": "system", "content": "You return only JSON."},
    {"role": "user", "content": 'Return {"risk_summary":"test","uncertainty_note":"test",'
                                '"non_advice_disclaimer":"test"} and nothing else.'},
]


def probe(base_url, model, label, extra, max_tokens, timeout=120):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": MESSAGES, "temperature": 0.0,
               "max_tokens": max_tokens, "stream": False}
    payload.update(extra)
    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except Exception as exc:
        print(f"  {label:34} CONNECTION FAILED  {exc!r}")
        return False
    if r.ok:
        try:
            txt = r.json()["choices"][0]["message"]["content"]
        except Exception:
            txt = "<no content field>"
        print(f"  {label:34} OK                 {str(txt)[:70]!r}")
        return True
    print(f"  {label:34} HTTP {r.status_code}           {r.text[:220]}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    args = ap.parse_args()

    print(f"Endpoint: {args.base_url}\nRequested model id: {args.model}\n")

    try:
        r = requests.get(args.base_url.rstrip("/") + "/models", timeout=30)
        ids = [m.get("id") for m in r.json().get("data", [])]
        print("Models the server reports:")
        for i in ids:
            print(f"  - {i}{'   <-- requested' if i == args.model else ''}")
        if args.model not in ids:
            print(f"\n  WARNING: '{args.model}' is not in the list above. "
                  "A wrong or unloaded model id is itself a 400/404 cause.")
    except Exception as exc:
        print(f"  Could not list models: {exc!r}")

    print("\nDecoding configurations:")
    results = {
        "A no response_format":       probe(args.base_url, args.model, "A no response_format", {}, 512),
        "B json_object":              probe(args.base_url, args.model, "B json_object",
                                            {"response_format": {"type": "json_object"}}, 512),
        "C json_schema (strict)":     probe(args.base_url, args.model, "C json_schema (strict)",
                                            {"response_format": SCHEMA}, 512),
        "E json_schema (open object)": probe(args.base_url, args.model, "E json_schema (open object)",
                                            {"response_format": OPEN_SCHEMA}, 512),
        "D plain, max_tokens=4096":   probe(args.base_url, args.model, "D plain, max_tokens=4096", {}, 4096),
    }

    print("\nRecommendation:")
    if results.get("E json_schema (open object)"):
        print("  Use json_schema with an OPEN object schema: provider setting")
        print("  \"response_format_json_schema\": \"permissive_object\". This matches the")
        print("  Ollama `format: json` constraint used for Llama3 without adding field")
        print("  requirements that run never had.")
    elif results.get("C json_schema (strict)"):
        print("  Only the strict schema is accepted. Use \"decision_note\" and state in the")
        print("  paper that the two models could not be given identical constraints.")
    elif results["B json_object"]:
        print("  Use response_format = json_object, as configured in")
        print("  provider 'lmstudio_gptoss20b_jsonmode'.")
    elif results["D plain, max_tokens=4096"]:
        print("  Neither constraint is accepted for this model. Run unconstrained with a large")
        print("  token budget and report in the paper that the two models could not be given")
        print("  equivalent decoding constraints, with the span-aware parser absorbing the")
        print("  difference. Do NOT present JSON validity as a model-quality result.")
    else:
        print("  The endpoint rejects even a plain request: the model is probably not loaded")
        print("  in LM Studio, or the model id differs from the one configured.")


if __name__ == "__main__":
    main()
