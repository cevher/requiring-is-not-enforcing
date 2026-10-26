"""
14_build_stress_onset_labels.py

Creates a stress-onset label for early warning.

Definition:
A snapshot is eligible for onset evaluation if current_external_stress == 0.
Among eligible snapshots:
- next_20d_stress_onset = 1 if next_20d_external_stress == 1
- next_20d_stress_onset = 0 if next_20d_external_stress == 0

Snapshots already under current stress are marked as not eligible because they do
not represent early warning before a stress episode.

Inputs:
- data_processed/labels/snapshot_forward_labels.csv

Outputs:
- data_processed/labels/snapshot_forward_labels_with_onset.csv
- data_processed/labels/onset_label_summary.csv
"""

import numpy as np
import pandas as pd

from config import LABEL_DIR
from common import ensure_dirs

ensure_dirs(LABEL_DIR)

path = LABEL_DIR / "snapshot_forward_labels.csv"
df = pd.read_csv(path, parse_dates=["date"])

required = {"current_external_stress", "next_20d_external_stress"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in {path}: {missing}")

df["onset_eval_eligible"] = (df["current_external_stress"] == 0).astype(int)

df["next_20d_stress_onset"] = np.where(
    df["onset_eval_eligible"] == 1,
    df["next_20d_external_stress"],
    np.nan,
)

out = LABEL_DIR / "snapshot_forward_labels_with_onset.csv"
df.to_csv(out, index=False)

summary = []
for col in ["next_20d_external_stress", "next_20d_stress_onset"]:
    valid = df[col].dropna()
    summary.append({
        "label": col,
        "n": int(valid.shape[0]),
        "positive_count": int(valid.sum()),
        "positive_share": float(valid.mean()) if len(valid) else np.nan,
    })

summary.append({
    "label": "onset_eval_eligible",
    "n": int(df.shape[0]),
    "positive_count": int(df["onset_eval_eligible"].sum()),
    "positive_share": float(df["onset_eval_eligible"].mean()),
})

summary_df = pd.DataFrame(summary)
summary_df.to_csv(LABEL_DIR / "onset_label_summary.csv", index=False)

print(f"Saved: {out}")
print(f"Saved: {LABEL_DIR / 'onset_label_summary.csv'}")
print(summary_df)
