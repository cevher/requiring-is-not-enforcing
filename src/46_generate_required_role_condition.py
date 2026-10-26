"""
46_generate_required_role_condition.py

Adds a fifth prompt condition: `full_required_role`.

WHY THIS CONDITION EXISTS
-------------------------
In the four-condition design, the requirement that the note declare relational
evidence to be context rather than the primary predictor is carried by the
SYSTEM INSTRUCTION only. The `full` condition shows that the instruction is not
honoured once network evidence is actually supplied.

The `no_network` condition cannot serve as the control for that effect. It
withholds network evidence, so the note makes no relational claim at all and
there is nothing for the role statement to attach to. Scoring that condition as
compliant compares two different things.

`full_required_role` is the correct control. It is byte-for-byte identical to
`full` -- same evidence records, same evidence IDs, same interpretation guide,
same system instruction, same generation rules -- except that the output schema
carries one additional REQUIRED field in which the model must state the role of
the relational evidence. The requirement therefore moves from the instruction
into the interface, and nothing else changes.

This isolates the claim in Section 5.6 of the manuscript: that structured output
interfaces which make the marking a required field, rather than an instruction,
repair the failure.

DESIGN NOTE ON NEUTRALITY
-------------------------
The field prompt does NOT tell the model which role the evidence plays. It asks
the model to state the role. A field worded as "confirm that network evidence is
contextual" would make compliance trivial and the test vacuous -- the same class
of error the audit protocol in Section 3.7.1 was written to avoid. The audit then
checks whether the stated role is the correct one.

Inputs
------
- data_processed/evidence/llm_decision_inputs_v5.jsonl
- data_processed/evidence/llm_eval_selected_decisions.csv

Output
------
- data_processed/evidence/llm_prompt_conditions_required_role.jsonl

Usage
-----
    python src/46_generate_required_role_condition.py

Then generate with the SAME provider settings used for the main run, e.g.

    python src/28_run_llm_generation.py \
        --input data_processed/evidence/llm_prompt_conditions_required_role.jsonl \
        --provider ollama_local_llama \
        --output ollama_llama_required_role.jsonl

    python src/28_run_llm_generation.py \
        --input data_processed/evidence/llm_prompt_conditions_required_role.jsonl \
        --provider lmstudio_gptoss20b_jsonmode \
        --output gptoss20b_required_role.jsonl

(Provider names must match the main run
exactly, or the condition is not comparable.)
"""

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import pandas as pd

from config import EVIDENCE_DIR
from common import ensure_dirs

ensure_dirs(EVIDENCE_DIR)

SRC_DIR = Path(__file__).resolve().parent
GEN_MODULE_PATH = SRC_DIR / "26_generate_llm_prompt_conditions.py"

CONDITION_NAME = "full_required_role"

# The required field. Neutral by design: it asks for the role, it does not supply it.
ROLE_FIELD_KEY = "relational_evidence_role_statement"
ROLE_FIELD_SPEC = (
    "REQUIRED. One sentence, max 30 words. State what role the relational/network "
    "evidence plays in this note. This field must not be omitted or left empty."
)

EXTRA_GENERATION_RULES = [
    f"The field '{ROLE_FIELD_KEY}' is required and must be present in the output.",
    f"'{ROLE_FIELD_KEY}' must be a single sentence of at most 30 words.",
    f"Do not leave '{ROLE_FIELD_KEY}' empty, null, or set to 'N/A'.",
]


def load_generator_module():
    """Load 26_generate_llm_prompt_conditions.py by path (module name starts with a digit)."""
    if not GEN_MODULE_PATH.exists():
        raise FileNotFoundError(
            f"Expected the condition builder at {GEN_MODULE_PATH}. "
            "Run this script from inside src/ so the relative import resolves."
        )
    spec = importlib.util.spec_from_file_location("gen26", GEN_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_required_role_payload(base, gen):
    """
    Build the `full_required_role` payload.

    Derived from `full` so that the evidence list, the evidence-ID sanitisation,
    the interpretation guide and the system instruction are IDENTICAL. The only
    difference is the added required output field.
    """
    payload = gen.condition_payload(base, "full")

    payload["condition"] = CONDITION_NAME
    payload["condition_description"] = (
        "Identical to the full condition in every respect except one: the role of "
        "relational evidence must be stated in a required output field rather than "
        "only in the system instruction."
    )

    schema = copy.deepcopy(payload["required_output_schema"])

    # Insert the required field directly after relational_network_evidence so that
    # it sits where the relational claims are made.
    ordered = {}
    for key, value in schema.items():
        ordered[key] = value
        if key == "relational_network_evidence":
            ordered[ROLE_FIELD_KEY] = ROLE_FIELD_SPEC
    if ROLE_FIELD_KEY not in ordered:
        ordered[ROLE_FIELD_KEY] = ROLE_FIELD_SPEC

    payload["required_output_schema"] = ordered
    return payload


def build_messages_with_extra_rules(payload, gen):
    """Reuse 26's message builder, then append the field-requirement rules."""
    messages = gen.build_messages(payload)

    user_object = json.loads(messages[1]["content"])
    user_object["strict_generation_rules"] = (
        user_object.get("strict_generation_rules", []) + EXTRA_GENERATION_RULES
    )
    user_object["required_output_schema"] = payload["required_output_schema"]
    messages[1]["content"] = json.dumps(user_object, ensure_ascii=False)

    return messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(EVIDENCE_DIR / "llm_decision_inputs_v5.jsonl"))
    ap.add_argument("--selected", default=str(EVIDENCE_DIR / "llm_eval_selected_decisions.csv"))
    ap.add_argument(
        "--output",
        default=str(EVIDENCE_DIR / "llm_prompt_conditions_required_role.jsonl"),
    )
    args = ap.parse_args()

    gen = load_generator_module()

    base_records = gen.read_jsonl(args.input)

    selected = pd.read_csv(args.selected)
    if "decision_id" not in selected.columns:
        raise ValueError(f"{args.selected} must contain a decision_id column.")
    selected_ids = set(selected["decision_id"].tolist())
    base_records = [r for r in base_records if r.get("decision_id") in selected_ids]

    if not base_records:
        raise ValueError(
            "No decision records matched the selected decision_ids. "
            "Check that llm_eval_selected_decisions.csv corresponds to llm_decision_inputs_v5.jsonl."
        )

    n = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for base in base_records:
            payload = build_required_role_payload(base, gen)
            messages = build_messages_with_extra_rules(payload, gen)

            out = {
                "prompt_id": f"{payload['decision_id']}::{CONDITION_NAME}",
                "decision_id": payload["decision_id"],
                "date": payload.get("date"),
                "condition": CONDITION_NAME,
                "messages": messages,
                "payload": payload,
                "allowed_evidence_ids": payload.get("allowed_evidence_ids", []),
                "evidence_id_alias_map": payload.get("evidence_id_alias_map", {}),
                "decision_context": payload.get("decision_context", {}),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1

    print(f"Saved required-role prompt conditions: {args.output}")
    print(f"n_prompts={n}")

    # ---- Parity check against the `full` condition --------------------------
    # The two conditions must differ in exactly one place: the output schema.
    # Anything else that differs invalidates the comparison, so verify it here
    # rather than assuming it.
    print("\nParity check (full vs full_required_role):")
    probe_base = base_records[0]
    full_payload = gen.condition_payload(probe_base, "full")
    rr_payload = build_required_role_payload(probe_base, gen)

    checks = {
        "system_instruction identical": (
            full_payload.get("system_instruction") == rr_payload.get("system_instruction")
        ),
        "evidence list identical": (
            json.dumps(full_payload.get("evidence"), sort_keys=True)
            == json.dumps(rr_payload.get("evidence"), sort_keys=True)
        ),
        "allowed_evidence_ids identical": (
            full_payload.get("allowed_evidence_ids") == rr_payload.get("allowed_evidence_ids")
        ),
        "decision_context identical": (
            json.dumps(full_payload.get("decision_context"), sort_keys=True)
            == json.dumps(rr_payload.get("decision_context"), sort_keys=True)
        ),
        "output schema differs (expected)": (
            full_payload.get("required_output_schema") != rr_payload.get("required_output_schema")
        ),
        "schema differs ONLY by the role field": (
            set(rr_payload["required_output_schema"].keys())
            - set(full_payload["required_output_schema"].keys())
            == {ROLE_FIELD_KEY}
        ),
    }

    ok = True
    for label, passed in checks.items():
        print(f"  [{'OK ' if passed else 'FAIL'}] {label}")
        ok = ok and passed

    if not ok:
        raise SystemExit(
            "\nParity check FAILED. The two conditions differ in more than the output "
            "schema, so any measured difference would be uninterpretable. Fix before running."
        )

    print("\nParity check passed: the conditions differ only in the required output field.")


if __name__ == "__main__":
    main()
