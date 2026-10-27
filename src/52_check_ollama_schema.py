"""Pre-flight check for the decoder-enforced Llama 3 run.

Run this BEFORE the real generation. It answers one question: does this Ollama
build actually enforce a JSON schema passed in `format`, or does it silently
ignore it and fall back to free-form JSON?

A silent fallback would be exactly the kind of configuration artefact this paper
is about, so it is checked rather than assumed.

Usage:
    python src/52_check_ollama_schema.py
"""
import importlib.util
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
CONFIG = ROOT / "llm_provider_config.json"
PROMPTS = ROOT / "data_processed" / "evidence" / "llm_prompt_conditions_required_role.jsonl"
FIELD = "relational_evidence_role_statement"
PROVIDER = "ollama_llama3_q4_required_role_enforced"


def load_gen():
    spec = importlib.util.spec_from_file_location("gen28", SRC / "28_run_llm_generation.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SRC))
    spec.loader.exec_module(mod)
    return mod


def ok(msg):   print("  [OK]   " + msg)
def bad(msg):  print("  [FAIL] " + msg)


def main():
    gen = load_gen()
    failures = 0

    print("\n1. Ollama reachable and model present")
    prov = gen.load_provider(str(CONFIG), PROVIDER)
    import requests
    base = prov["base_url"].rsplit("/api", 1)[0]
    try:
        tags = requests.get(base + "/api/tags", timeout=10).json()
        names = [m["name"] for m in tags.get("models", [])]
        if prov["model"] in names:
            ok(f"{prov['model']} is installed")
        else:
            bad(f"{prov['model']} NOT installed. Present: {names}")
            failures += 1
    except Exception as e:
        bad(f"cannot reach Ollama at {base}: {e}")
        return 1

    print("\n2. Schema is the same object the secondary model received")
    sch = gen.resolve_schema(prov["format_json_schema"])
    same = sch is gen.REQUIRED_ROLE_SCHEMA
    (ok if same else bad)("REQUIRED_ROLE_SCHEMA" + ("" if same else " MISMATCH"))
    failures += 0 if same else 1
    print("         " + json.dumps(sch["schema"]))

    print("\n3. Decoder actually enforces the field")
    msgs = [{"role": "user",
             "content": "Reply with a JSON object containing only the key \"colour\" "
                        "set to \"blue\". Do not add any other key."}]
    try:
        text, _ = gen.call_ollama_native(prov, msgs)
        obj = json.loads(text)
        if FIELD in obj:
            ok(f"field present although the prompt asked against it -> decoder enforces")
            print("         value: " + json.dumps(obj[FIELD])[:120])
        else:
            bad(f"field ABSENT. This build ignores the schema. keys={list(obj)}")
            failures += 1
    except json.JSONDecodeError:
        bad("response was not valid JSON: " + text[:200])
        failures += 1
    except Exception as e:
        bad(f"request failed: {e}")
        failures += 1

    print("\n4. Unenforced path unchanged (regression guard)")
    base_prov = gen.load_provider(str(CONFIG), "ollama_llama3_q4_pinned")
    if base_prov.get("format_json") and not base_prov.get("format_json_schema"):
        ok("ollama_llama3_q4_pinned still uses format: json, no schema")
    else:
        bad("the pinned provider was modified")
        failures += 1

    print("\n5. Prompt file parity")
    if PROMPTS.exists():
        recs = [json.loads(l) for l in PROMPTS.open(encoding="utf8")]
        ok(f"{PROMPTS.name}: {len(recs)} records "
           f"(the same file both models' required-role runs used)")
    else:
        bad(f"missing: {PROMPTS}")
        failures += 1

    print("\n" + ("ALL CHECKS PASSED - safe to run the generation"
                  if not failures else f"{failures} CHECK(S) FAILED - do not run yet"))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
