"""
30_prepare_human_review_sheet.py

Creates a human-review CSV sheet for LLM decision-note evaluation.
"""

import argparse
import json
from pathlib import Path
import pandas as pd

from config import PROJECT_ROOT
from common import ensure_dirs

EVAL_DIR = PROJECT_ROOT / "outputs" / "llm_eval"
ensure_dirs(EVAL_DIR)


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    parser.add_argument("--auto-eval", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    gen_path = Path(args.generations)
    auto_path = Path(args.auto_eval)

    gens = pd.DataFrame(read_jsonl(gen_path))
    auto = pd.read_csv(auto_path)

    cols_gen = [
        "prompt_id", "provider_name", "model", "decision_id", "date",
        "condition", "response_text", "decision_context"
    ]
    gens = gens[[c for c in cols_gen if c in gens.columns]]

    df = auto.merge(gens, on=["prompt_id", "provider_name", "model", "decision_id", "condition"], how="left")

    if "response_text" in df.columns:
        df["response_text_preview"] = df["response_text"].astype(str).str.slice(0, 3000)

    review_cols = {
        "evidence_faithfulness_score_1_5": "",
        "macro_network_role_separation_1_5": "",
        "uncertainty_quality_1_5": "",
        "decision_usefulness_1_5": "",
        "unsupported_claim_count_manual": "",
        "causal_overclaim_manual_0_1": "",
        "reviewer_notes": "",
    }

    for c, default in review_cols.items():
        df[c] = default

    preferred_order = [
        "provider_name", "model", "condition", "decision_id", "date",
        "json_valid", "citation_correctness", "n_invented_evidence_ids",
        "forbidden_causal_count", "investment_advice_count",
        "requires_uncertainty", "uncertainty_compliance", "role_separation_score",
        "response_text_preview",
    ] + list(review_cols.keys())

    remaining = [c for c in df.columns if c not in preferred_order and c != "response_text"]
    df = df[[c for c in preferred_order if c in df.columns] + remaining]

    output = Path(args.output) if args.output else EVAL_DIR / f"{gen_path.stem}_human_review_sheet.csv"
    df.to_csv(output, index=False)

    print(f"Saved: {output}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
