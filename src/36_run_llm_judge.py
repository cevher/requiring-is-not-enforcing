"""
36_run_llm_judge.py

Runs an independent automated annotator ("LLM judge") over the 40 generated
decision notes, using the same rubric as the human review.

Design constraints that make this defensible rather than circular:
  - The judge MUST be a different model family from the generators
    (generators: llama3 and gpt-oss). Default target: qwen3-coder-30b.
    The script refuses to run if the judge model name looks like a generator.
  - The judge sees the decision context, the allowed evidence IDs and the note,
    but never the prompt condition, and items are presented in the randomised
    order produced by 35_build_blind_rerating_sheet.py.
  - The judge is an ADDITIONAL annotator, not a replacement for the human pass.
    Report human-judge agreement (37_rating_agreement.py); do not report judge
    scores alone as "human review".

Usage:
  python src/36_run_llm_judge.py --provider lmstudio_qwen3coder30b
  python src/36_run_llm_judge.py --provider lmstudio_qwen3coder30b --limit 2   # smoke test
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from config import PROJECT_ROOT
from common import ensure_dirs

EVAL_DIR = PROJECT_ROOT / "outputs" / "llm_eval"
ensure_dirs(EVAL_DIR)

DEFAULT_CONFIG = PROJECT_ROOT / "llm_provider_config.json"
SHEET = EVAL_DIR / "blind_rating_sheet_rater1.xlsx"   # any blind sheet works: the items are identical, only the order differs

GENERATOR_MARKERS = ["llama", "gpt-oss", "gptoss"]

SYSTEM = (
    "You are a strict evaluation annotator for financial risk decision notes. "
    "You do not write decision notes and you do not give financial advice. "
    "You rate one note at a time against a fixed rubric and you return only JSON. "
    "Judge the note only against the evidence and context given to you. "
    "Do not reward fluency. Do not guess which system produced the note. "
    "Use the full range of each scale."
)

RUBRIC = """Rate the note on these criteria.

evidence_faithfulness_score_1_5:
 1 = claims contradict or ignore the cited evidence
 3 = claims are consistent with the evidence but imprecise or partly unsupported
 5 = every substantive claim is traceable to a cited record and stated at the strength that record supports

macro_network_role_separation_1_5:
 1 = relational/network evidence is used as a stress predictor or as a cause
 3 = the layers are mixed but no causal claim is made
 5 = the note explicitly marks relational evidence as context/attribution and not the primary predictor
 If the note contains no relational evidence at all, score 5 only if it makes no relational claim.

uncertainty_quality_1_5:
 1 = no uncertainty language although the context requires it
 3 = generic caution not tied to this decision context
 5 = uncertainty stated and tied to the specific driver (low confidence, rule/model disagreement, abstention, thin history)

decision_usefulness_1_5:
 1 = an analyst learns nothing beyond the score
 3 = readable but generic
 5 = an analyst could act on it: what the signal is, how reliable it is, what to inspect next

unsupported_claim_count_manual:
 integer count of substantive statements with no supporting evidence record

causal_overclaim_manual_0_1:
 1 if the note asserts causal transmission, contagion or spillover as fact; else 0.
 Echoing a prohibition such as "this is not causal contagion" is 0.
"""

# LM Studio rejects response_format.type = "json_object" and accepts only
# "json_schema" or "text". Unlike the generation experiment, a strict field schema
# is appropriate here: the judge must return a fixed set of integer ratings, and
# constraining it does not affect the conditions being compared.
RATING_JSON_SCHEMA = {
    "name": "rating",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "evidence_faithfulness_score_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
            "macro_network_role_separation_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
            "uncertainty_quality_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
            "decision_usefulness_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
            "unsupported_claim_count_manual": {"type": "integer", "minimum": 0},
            "causal_overclaim_manual_0_1": {"type": "integer", "minimum": 0, "maximum": 1},
            "reviewer_notes": {"type": "string"},
        },
        "required": [
            "evidence_faithfulness_score_1_5", "macro_network_role_separation_1_5",
            "uncertainty_quality_1_5", "decision_usefulness_1_5",
            "unsupported_claim_count_manual", "causal_overclaim_manual_0_1",
        ],
        "additionalProperties": True,
    },
}

OUT_SCHEMA = (
    'Return exactly this JSON object and nothing else: '
    '{"evidence_faithfulness_score_1_5": int, '
    '"macro_network_role_separation_1_5": int, '
    '"uncertainty_quality_1_5": int, '
    '"decision_usefulness_1_5": int, '
    '"unsupported_claim_count_manual": int, '
    '"causal_overclaim_manual_0_1": int, '
    '"reviewer_notes": "one short sentence"}'
)


def load_provider(config_path, name):
    cfg = json.load(open(config_path, encoding="utf-8"))
    for p in cfg.get("providers", []):
        if p.get("provider_name") == name:
            return p
    raise ValueError(f"Provider not found: {name}")


def call_openai_compatible(provider, messages):
    url = provider["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    env = provider.get("api_key_env")
    if env and os.environ.get(env):
        headers["Authorization"] = f"Bearer {os.environ[env]}"
    payload = {
        "model": provider["model"],
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": provider.get("max_tokens", 600),
        "stream": False,
        "response_format": {"type": "json_schema", "json_schema": RATING_JSON_SCHEMA},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=provider.get("timeout", 300))
    if not r.ok:
        raise requests.HTTPError(f"{r.status_code} {r.reason} for {url} :: {r.text[:800]}")
    return r.json()["choices"][0]["message"]["content"]


def call_ollama_native(provider, messages):
    url = provider["base_url"].rstrip("/") + "/chat"
    headers = {"Content-Type": "application/json"}
    env = provider.get("api_key_env")
    if env and os.environ.get(env):
        headers["Authorization"] = f"Bearer {os.environ[env]}"
    payload = {
        "model": provider["model"], "messages": messages, "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": provider.get("max_tokens", 600)},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=provider.get("timeout", 300))
    if not r.ok:
        raise requests.HTTPError(f"{r.status_code} {r.reason} for {url} :: {r.text[:800]}")
    return r.json().get("message", {}).get("content", "")


def call(provider, messages):
    if provider.get("adapter", "openai_compatible") == "ollama_native":
        return call_ollama_native(provider, messages)
    return call_openai_compatible(provider, messages)


def parse_scores(text):
    t = text.strip()
    s, e = t.find("{"), t.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--sheet", default=str(SHEET))
    ap.add_argument("--output", default=str(EVAL_DIR / "llm_judge_ratings.csv"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-generator-as-judge", action="store_true")
    args = ap.parse_args()

    provider = load_provider(args.config, args.provider)
    model = str(provider.get("model", "")).lower()
    if any(m in model for m in GENERATOR_MARKERS) and not args.allow_generator_as_judge:
        raise SystemExit(
            f"Refusing to run: '{model}' is one of the generator families. "
            "A judge from the same family as the generator makes the agreement "
            "statistic uninterpretable. Choose a different provider, or pass "
            "--allow-generator-as-judge and disclose it in the paper."
        )

    sheet_path = Path(args.sheet)
    if not sheet_path.exists():
        candidates = sorted(EVAL_DIR.glob("blind_rating_sheet_*.xlsx"))
        raise SystemExit(
            f"Rating sheet not found: {sheet_path}\n"
            + ("Available sheets:\n  " + "\n  ".join(str(c) for c in candidates)
               if candidates else
               "No blind rating sheet exists yet. Build one first:\n"
               "  python src/35_build_blind_rerating_sheet.py --rater rater1")
        )
    items = pd.read_excel(sheet_path, sheet_name="rate_these")
    if args.limit:
        items = items.head(args.limit)

    rows = []
    for k, it in items.iterrows():
        user = (
            f"{RUBRIC}\n\nDECISION CONTEXT:\n{it['decision_context']}\n\n"
            f"ALLOWED EVIDENCE IDS ({it['n_allowed_evidence_ids']}):\n{it['allowed_evidence_ids']}\n\n"
            f"NOTE TO RATE:\n{it['generated_note']}\n\n{OUT_SCHEMA}"
        )
        t0 = time.time()
        try:
            text = call(provider, [{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": user}])
            scores = parse_scores(text)
            status = "ok" if scores else "parse_error"
            err = "" if scores else text[:300]
        except Exception as exc:
            scores, status, err = None, "error", repr(exc)[:300]

        row = {"item_id": it["item_id"], "judge_model": provider.get("model"),
               "status": status, "error": err,
               "elapsed_seconds": round(time.time() - t0, 2)}
        if scores:
            for k2, v in scores.items():
                row[f"judge_{k2}"] = v
        rows.append(row)
        print(f"[{k+1}/{len(items)}] {it['item_id']} {status} ({row['elapsed_seconds']}s)")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {args.output}")
    print(df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
