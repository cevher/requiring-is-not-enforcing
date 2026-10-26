"""
31_clean_llm_generation_results.py

Cleans LLM generation JSONL files before automatic evaluation and human review.

Why:
If a generation run is interrupted or retried without using a fresh output file,
the JSONL may contain duplicate prompt_id rows. For example, a prompt may first
appear with status="error" and then later with status="ok". If the auto-eval and
human-review scripts are run directly on that file, summaries and review sheets
can be inflated or duplicated.

Cleaning rule:
- Group by prompt_id.
- Prefer the latest status="ok" record.
- If no ok record exists, keep the latest record.
- Preserve original row order as much as possible.
- Write a clean JSONL.

Example:
python src/31_clean_llm_generation_results.py ^
  --input outputs/llm_generations/ollama_llama_native_10decisions_4conditions_final.jsonl ^
  --output outputs/llm_generations/ollama_llama_native_10decisions_4conditions_final_clean.jsonl
"""

import argparse
import json
from pathlib import Path
from collections import OrderedDict


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                rec = json.loads(line)
                rec["_original_row_number"] = i
                rows.append(rec)
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in rows:
            rec = dict(rec)
            rec.pop("_original_row_number", None)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def choose_record(records):
    ok_records = [r for r in records if r.get("status") == "ok"]
    if ok_records:
        return ok_records[-1]
    return records[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input)

    grouped = OrderedDict()
    for rec in rows:
        pid = rec.get("prompt_id")
        if not pid:
            pid = f"__missing_prompt_id_row_{rec['_original_row_number']}"
        grouped.setdefault(pid, []).append(rec)

    cleaned = []
    duplicate_prompt_count = 0
    dropped_count = 0
    kept_error_count = 0

    for pid, group in grouped.items():
        if len(group) > 1:
            duplicate_prompt_count += 1
            dropped_count += len(group) - 1

        chosen = choose_record(group)
        if chosen.get("status") != "ok":
            kept_error_count += 1
        cleaned.append(chosen)

    # Sort by original order of the kept record so outputs remain interpretable.
    cleaned = sorted(cleaned, key=lambda r: r.get("_original_row_number", 0))

    write_jsonl(args.output, cleaned)

    n_ok = sum(1 for r in cleaned if r.get("status") == "ok")
    n_error = sum(1 for r in cleaned if r.get("status") != "ok")

    print(f"Input rows: {len(rows)}")
    print(f"Unique prompt_id rows kept: {len(cleaned)}")
    print(f"Duplicate prompt_ids found: {duplicate_prompt_count}")
    print(f"Rows dropped: {dropped_count}")
    print(f"Kept ok rows: {n_ok}")
    print(f"Kept non-ok rows: {n_error}")
    print(f"Clean output: {args.output}")


if __name__ == "__main__":
    main()
