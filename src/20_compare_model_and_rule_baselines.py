"""
20_compare_model_and_rule_baselines.py

Combines expanding-window model results and rule baselines into one comparison table.

Inputs:
- outputs/reports/expanding_window_results.csv
- outputs/reports/rule_baseline_results.csv

Outputs:
- outputs/reports/final_baseline_comparison.csv
"""

import pandas as pd

from config import REPORT_DIR
from common import ensure_dirs

ensure_dirs(REPORT_DIR)

model_path = REPORT_DIR / "expanding_window_results.csv"
rule_path = REPORT_DIR / "rule_baseline_results.csv"

if not model_path.exists():
    raise FileNotFoundError(f"Missing {model_path}. Run 15_expanding_window_backtest.py first.")
if not rule_path.exists():
    raise FileNotFoundError(f"Missing {rule_path}. Run 17_evaluate_rule_baselines.py first.")

models = pd.read_csv(model_path)
rules = pd.read_csv(rule_path)

models = models.copy()
models["system_type"] = "ml_model"
models["method_name"] = (
    models["feature_set"].astype(str) + " | " +
    models["model"].astype(str) + " | " +
    models["calibration"].astype(str)
)

rules = rules.copy()
rules["system_type"] = "rule_baseline"
rules["method_name"] = rules["baseline"].astype(str)

common_cols = [
    "target", "system_type", "method_name",
    "n_oos", "positive_share_oos",
    "brier", "roc_auc", "average_precision", "log_loss",
    "ece_10_bins", "accuracy", "balanced_accuracy"
]

combined = pd.concat([
    models[[c for c in common_cols if c in models.columns]],
    rules[[c for c in common_cols if c in rules.columns]],
], ignore_index=True)

combined = combined.sort_values(["target", "brier", "ece_10_bins"], ascending=True)

out = REPORT_DIR / "final_baseline_comparison.csv"
combined.to_csv(out, index=False)

print(f"Saved: {out}")
print(combined.head(30))
