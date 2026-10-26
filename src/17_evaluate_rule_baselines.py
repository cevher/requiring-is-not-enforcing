"""
17_evaluate_rule_baselines.py

Adds simple rule/naive baselines for the systemic-risk early warning task.

Why:
A paper with calibrated risk scoring must compare against simple, interpretable baselines:
- constant prevalence baseline
- persistence baseline
- VIX threshold rule
- NFCI threshold rule
- combined VIX/NFCI rule

Inputs:
- data_processed/labels/snapshot_forward_labels_with_onset.csv
- data_processed/labels/stress_labels.csv

Outputs:
- outputs/reports/rule_baseline_results.csv
- outputs/reports/rule_baseline_predictions.csv

Notes:
- Uses expanding-year out-of-sample evaluation for consistency with the model backtest.
- Rule baselines are intentionally simple. They are sanity checks, not final models.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    balanced_accuracy_score,
)

from config import LABEL_DIR, REPORT_DIR
from common import ensure_dirs

ensure_dirs(REPORT_DIR)

FIRST_TEST_YEAR = 2011
TARGETS = ["next_20d_external_stress", "next_20d_stress_onset"]


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob > lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)

    return float(ece)


def safe_metric(fn, y, p):
    try:
        return float(fn(y, p))
    except Exception:
        return np.nan


def safe_log_loss(y, p):
    try:
        return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    except Exception:
        return np.nan


def evaluate(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = (p >= 0.5).astype(int)

    return {
        "brier": safe_metric(brier_score_loss, y, p),
        "roc_auc": safe_metric(roc_auc_score, y, p),
        "average_precision": safe_metric(average_precision_score, y, p),
        "log_loss": safe_log_loss(y, p),
        "ece_10_bins": expected_calibration_error(y, p),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def minmax_score(series, train_min, train_max):
    denom = train_max - train_min
    if not np.isfinite(denom) or denom == 0:
        return np.repeat(0.5, len(series))
    return np.clip((series - train_min) / denom, 0, 1)


label_path = LABEL_DIR / "snapshot_forward_labels_with_onset.csv"
if not label_path.exists():
    raise FileNotFoundError(
        f"Missing {label_path}. Run 14_build_stress_onset_labels.py first."
    )

labels = pd.read_csv(label_path, parse_dates=["date"]).sort_values("date")
daily = pd.read_csv(LABEL_DIR / "stress_labels.csv", parse_dates=["date"]).sort_values("date")

macro_cols = [c for c in ["vix", "nfci"] if c in daily.columns]
macro = daily[["date"] + macro_cols].copy()

df = pd.merge_asof(
    labels.sort_values("date"),
    macro.sort_values("date"),
    on="date",
    direction="nearest",
    tolerance=pd.Timedelta("5D")
)

df["year"] = df["date"].dt.year

pred_records = []

for target in TARGETS:
    if target not in df.columns:
        continue

    rows = df.dropna(subset=[target]).copy()
    rows[target] = rows[target].astype(int)

    for test_year in sorted(rows["year"].unique()):
        if test_year < FIRST_TEST_YEAR:
            continue

        train = rows[rows["year"] < test_year].copy()
        test = rows[rows["year"] == test_year].copy()

        if len(train) < 36 or len(test) == 0:
            continue
        if train[target].nunique() < 2 or test[target].nunique() < 2:
            continue

        y_train = train[target].values
        prevalence = float(np.mean(y_train))

        # Baseline 1: constant historical prevalence
        probs = np.repeat(prevalence, len(test))
        for d, yt, yp in zip(test["date"], test[target], probs):
            pred_records.append({
                "target": target,
                "test_year": test_year,
                "date": d,
                "baseline": "constant_prevalence",
                "y_true": int(yt),
                "y_prob": float(yp),
                "y_pred": int(yp >= 0.5),
            })

        # Baseline 2: current external stress persistence
        if "current_external_stress" in test.columns:
            probs = test["current_external_stress"].fillna(prevalence).astype(float).values
            for d, yt, yp in zip(test["date"], test[target], probs):
                pred_records.append({
                    "target": target,
                    "test_year": test_year,
                    "date": d,
                    "baseline": "current_stress_persistence",
                    "y_true": int(yt),
                    "y_prob": float(yp),
                    "y_pred": int(yp >= 0.5),
                })

        # Baseline 3: VIX score/rule, threshold estimated from past train only
        if "vix" in train.columns and train["vix"].notna().sum() > 20 and "vix" in test.columns:
            vix_thr = train["vix"].quantile(0.80)
            vix_min = train["vix"].quantile(0.05)
            vix_max = train["vix"].quantile(0.95)

            # Probability-like score
            probs = minmax_score(test["vix"].fillna(train["vix"].median()), vix_min, vix_max)
            for d, yt, yp in zip(test["date"], test[target], probs):
                pred_records.append({
                    "target": target,
                    "test_year": test_year,
                    "date": d,
                    "baseline": "vix_scaled_score",
                    "y_true": int(yt),
                    "y_prob": float(yp),
                    "y_pred": int(yp >= 0.5),
                })

            # Hard rule
            probs = (test["vix"].fillna(train["vix"].median()) >= vix_thr).astype(float).values
            for d, yt, yp in zip(test["date"], test[target], probs):
                pred_records.append({
                    "target": target,
                    "test_year": test_year,
                    "date": d,
                    "baseline": "vix_80pct_rule",
                    "y_true": int(yt),
                    "y_prob": float(yp),
                    "y_pred": int(yp >= 0.5),
                })

        # Baseline 4: NFCI score/rule
        if "nfci" in train.columns and train["nfci"].notna().sum() > 20 and "nfci" in test.columns:
            nfci_thr = 0.0
            nfci_min = train["nfci"].quantile(0.05)
            nfci_max = train["nfci"].quantile(0.95)

            probs = minmax_score(test["nfci"].fillna(train["nfci"].median()), nfci_min, nfci_max)
            for d, yt, yp in zip(test["date"], test[target], probs):
                pred_records.append({
                    "target": target,
                    "test_year": test_year,
                    "date": d,
                    "baseline": "nfci_scaled_score",
                    "y_true": int(yt),
                    "y_prob": float(yp),
                    "y_pred": int(yp >= 0.5),
                })

            probs = (test["nfci"].fillna(train["nfci"].median()) > nfci_thr).astype(float).values
            for d, yt, yp in zip(test["date"], test[target], probs):
                pred_records.append({
                    "target": target,
                    "test_year": test_year,
                    "date": d,
                    "baseline": "nfci_positive_rule",
                    "y_true": int(yt),
                    "y_prob": float(yp),
                    "y_pred": int(yp >= 0.5),
                })

        # Baseline 5: combined VIX/NFCI if both exist
        if {"vix", "nfci"}.issubset(train.columns) and {"vix", "nfci"}.issubset(test.columns):
            if train["vix"].notna().sum() > 20 and train["nfci"].notna().sum() > 20:
                vix_thr = train["vix"].quantile(0.80)
                vix_flag = (test["vix"].fillna(train["vix"].median()) >= vix_thr).astype(int)
                nfci_flag = (test["nfci"].fillna(train["nfci"].median()) > 0).astype(int)
                probs = ((vix_flag + nfci_flag) / 2.0).astype(float).values
                for d, yt, yp in zip(test["date"], test[target], probs):
                    pred_records.append({
                        "target": target,
                        "test_year": test_year,
                        "date": d,
                        "baseline": "vix_nfci_combined_rule",
                        "y_true": int(yt),
                        "y_prob": float(yp),
                        "y_pred": int(yp >= 0.5),
                    })

pred_df = pd.DataFrame(pred_records)

result_records = []
if not pred_df.empty:
    for key, g in pred_df.groupby(["target", "baseline"]):
        y = g["y_true"].values
        p = g["y_prob"].values
        if len(np.unique(y)) < 2:
            continue
        row = {
            "target": key[0],
            "baseline": key[1],
            "n_oos": len(g),
            "positive_share_oos": float(np.mean(y)),
        }
        row.update(evaluate(y, p))
        result_records.append(row)

results = pd.DataFrame(result_records)
if not results.empty:
    results = results.sort_values(["target", "brier", "ece_10_bins"], ascending=True)

pred_path = REPORT_DIR / "rule_baseline_predictions.csv"
res_path = REPORT_DIR / "rule_baseline_results.csv"

pred_df.to_csv(pred_path, index=False)
results.to_csv(res_path, index=False)

print(f"Saved: {pred_path}")
print(f"Saved: {res_path}")
print(results)
