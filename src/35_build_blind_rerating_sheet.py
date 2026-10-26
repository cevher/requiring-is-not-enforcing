"""
35_build_blind_rerating_sheet.py

Builds a condition-blind, order-randomised re-rating sheet for the 40 generated
decision notes, plus a separate key file.

Why this exists:
The completed human-review file scores nine of twenty condition x criterion
cells with zero within-condition variance (e.g. uncertainty_quality is 5 for
all ten full-evidence items and 3 for all ten no-reliability items). Whatever
the intention, a rating with no within-condition variance cannot be presented
as an independent item-level judgement: it is indistinguishable from a score
assigned per condition. A second independent rater is the textbook remedy. If
one is not available, the minimum defensible substitute is a blind re-rating by
the same rater, separated in time, with the condition labels removed and the
item order randomised, reporting intra-rater agreement against the first pass.

Honest limitation to state in the paper: full blinding is impossible, because
no-network outputs structurally lack network fields and free-form outputs lack
citations. The procedure is condition-label-blind and order-randomised, not
fully blind. Say this explicitly rather than claiming a blind protocol.

Outputs:
  outputs/llm_eval/blind_rerating_sheet.xlsx   -> give this to the rater
  outputs/llm_eval/blind_rerating_key.csv      -> do NOT open before rating
"""

import argparse
import json
import numpy as np
import pandas as pd

from config import PROJECT_ROOT, EVIDENCE_DIR
from common import ensure_dirs

EVAL_DIR = PROJECT_ROOT / "outputs" / "llm_eval"
ensure_dirs(EVAL_DIR)

GEN = PROJECT_ROOT / "outputs" / "llm_generations" / \
      "ollama_llama_native_10decisions_4conditions_final_clean.jsonl"

DEFAULT_SEED = 20260919
RATING_COLUMNS = [
    "evidence_faithfulness_score_1_5",
    "macro_network_role_separation_1_5",
    "uncertainty_quality_1_5",
    "decision_usefulness_1_5",
    "unsupported_claim_count_manual",
    "causal_overclaim_manual_0_1",
    "reviewer_notes",
]

RUBRIC = {
    "evidence_faithfulness_score_1_5":
        "1 = claims contradict or ignore the cited evidence; "
        "3 = claims are consistent with the evidence but imprecise or partly unsupported; "
        "5 = every substantive claim is traceable to a cited record and stated at the strength the record supports.",
    "macro_network_role_separation_1_5":
        "1 = network/relational evidence is used as a stress predictor or cause; "
        "3 = the two layers are mixed but no causal claim is made; "
        "5 = the note explicitly marks relational evidence as context/attribution and not the primary predictor.",
    "uncertainty_quality_1_5":
        "1 = no uncertainty language although the record requires it; "
        "3 = generic caution not tied to this decision context; "
        "5 = uncertainty is stated and tied to the specific driver (low confidence, disagreement, abstention, thin history).",
    "decision_usefulness_1_5":
        "1 = an analyst learns nothing beyond the score; "
        "3 = readable but generic; "
        "5 = an analyst could act on it: what the signal is, how reliable, what to look at next.",
    "unsupported_claim_count_manual":
        "Integer count of substantive statements with no supporting evidence record.",
    "causal_overclaim_manual_0_1":
        "1 if the note asserts causal transmission/contagion/spillover as fact, else 0. "
        "Echoing a prohibition ('this is not causal contagion') is 0.",
}


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def compact_context(rec):
    ctx = rec.get("decision_context", {}) or {}
    keep = ["date", "target", "decision_score", "risk_level",
            "reliability_adjusted_confidence", "abstention_flag",
            "disagreement_flag", "reliability_status"]
    return "; ".join(f"{k}={ctx[k]}" for k in keep if k in ctx)


_ap = argparse.ArgumentParser()
_ap.add_argument("--rater", default="rater1",
                 help="label used in the output file names, e.g. rater1 / rater2")
_ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                 help="item order is randomised per rater so that order effects "
                      "are not shared between raters")
_args = _ap.parse_args()

records = read_jsonl(GEN)
rng = np.random.default_rng(_args.seed)
order = rng.permutation(len(records))

rows, key = [], []
for new_pos, i in enumerate(order, start=1):
    rec = records[i]
    item_id = f"ITEM-{new_pos:03d}"
    allowed = rec.get("allowed_evidence_ids", [])
    rows.append({
        "item_id": item_id,
        "decision_date": rec.get("date"),
        "decision_context": compact_context(rec),
        "n_allowed_evidence_ids": len(allowed),
        "allowed_evidence_ids": "; ".join(allowed),
        "generated_note": rec.get("response_text", ""),
        **{c: "" for c in RATING_COLUMNS},
    })
    key.append({
        "item_id": item_id,
        "prompt_id": rec.get("prompt_id"),
        "decision_id": rec.get("decision_id"),
        "condition": rec.get("condition"),
        "original_row": int(i),
    })

sheet = pd.DataFrame(rows)
keydf = pd.DataFrame(key)

out_sheet = EVAL_DIR / f"blind_rating_sheet_{_args.rater}.xlsx"
with pd.ExcelWriter(out_sheet, engine="openpyxl") as xw:
    sheet.to_excel(xw, sheet_name="rate_these", index=False)
    pd.DataFrame(
        [{"criterion": k, "anchors": v} for k, v in RUBRIC.items()]
    ).to_excel(xw, sheet_name="rubric", index=False)
    pd.DataFrame([{
        "instruction": (
            "Rate every item on its own merits. Do not try to infer which prompt "
            "condition produced it. Items are in randomised order and the condition "
            "labels are withheld. Use the full 1-5 range: if two items differ in "
            "quality, they should not receive the same score. Do not open "
            f"blind_rating_key_{_args.rater}.csv until all 40 items are rated. "
            "Do not discuss any item with the other rater before both passes are finished."
        )
    }]).to_excel(xw, sheet_name="instructions", index=False)

key_path = EVAL_DIR / f"blind_rating_key_{_args.rater}.csv"
keydf.to_csv(key_path, index=False)

print(f"Wrote {out_sheet}  ({len(sheet)} items, {len(sheet.columns)} columns)")
print(f"Wrote {key_path}  (keep closed until rating is finished)")
print("\nCondition distribution hidden in the sheet:")
print(keydf["condition"].value_counts().to_string())
