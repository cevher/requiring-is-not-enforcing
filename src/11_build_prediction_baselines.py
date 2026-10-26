"""
11_build_prediction_baselines.py

Builds simple temporal baseline models for future stress prediction and evaluates calibration.

Purpose:
Before adding LLM layers, test whether structure-derived financial evidence has predictive
or diagnostic value for next-period stress.

Models:
- Macro-only Logistic Regression
- Network-only Logistic Regression
- Network + Macro Logistic Regression
- Network + Macro Random Forest

Calibration:
- Simple Platt calibration fitted on a chronological calibration split.
- Reports Brier score, ROC-AUC, average precision, log loss, and ECE.

Inputs:
- data_processed/networks/network_metrics.csv
- data_processed/networks/residual_network_metrics.csv, if available
- data_processed/networks/tail_network_metrics.csv, if available
- data_processed/labels/snapshot_forward_labels.csv
- data_processed/labels/stress_labels.csv

Outputs:
- outputs/reports/baseline_model_results.csv
- outputs/reports/baseline_predictions.csv
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.exceptions import UndefinedMetricWarning

from config import NETWORK_DIR, LABEL_DIR, REPORT_DIR
from common import ensure_dirs

ensure_dirs(REPORT_DIR)

TARGET = "next_20d_external_stress"

def collapse_metrics(path, prefix_map=None):
    df = pd.read_csv(path, parse_dates=["date"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = [c for c in numeric_cols if c.endswith("_nodes") or c.endswith("_edges")]
    numeric_cols = [c for c in numeric_cols if c not in drop_cols]

    grouped = df.groupby("date")[numeric_cols].agg(["mean", "max", "std"])
    grouped.columns = ["{}_{}".format(c, stat) for c, stat in grouped.columns]
    grouped = grouped.reset_index()

    if prefix_map:
        grouped = grouped.rename(columns={old: new for old, new in prefix_map.items() if old in grouped.columns})

    return grouped

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

def platt_calibrate(p_cal, y_cal, p_test):
    """
    Fits logistic calibration on one-dimensional base probabilities.
    """
    eps = 1e-6
    p_cal = np.clip(np.asarray(p_cal), eps, 1 - eps)
    p_test = np.clip(np.asarray(p_test), eps, 1 - eps)

    # logit transform gives smoother calibration
    x_cal = np.log(p_cal / (1 - p_cal)).reshape(-1, 1)
    x_test = np.log(p_test / (1 - p_test)).reshape(-1, 1)

    if len(np.unique(y_cal)) < 2:
        # Cannot calibrate if calibration set has one class only.
        return p_test

    cal_model = LogisticRegression()
    cal_model.fit(x_cal, y_cal)
    return cal_model.predict_proba(x_test)[:, 1]

# Load primary network metrics
raw = collapse_metrics(NETWORK_DIR / "network_metrics.csv")

# Optional residual metrics
resid_path = NETWORK_DIR / "residual_network_metrics.csv"
if resid_path.exists():
    resid = collapse_metrics(resid_path)
    raw = raw.merge(resid, on="date", how="left", suffixes=("", "_resid"))

# Optional tail metrics
tail_path = NETWORK_DIR / "tail_network_metrics.csv"
if tail_path.exists():
    tail = collapse_metrics(tail_path)
    raw = raw.merge(tail, on="date", how="left", suffixes=("", "_tail"))

# Labels
fwd = pd.read_csv(LABEL_DIR / "snapshot_forward_labels.csv", parse_dates=["date"])
daily = pd.read_csv(LABEL_DIR / "stress_labels.csv", parse_dates=["date"])

# Macro features at snapshot dates
macro_cols = [c for c in ["vix", "treasury_10y", "treasury_2y", "term_spread_10y_2y", "high_yield_oas", "nfci", "ofr_fsi"] if c in daily.columns]
macro = daily[["date"] + macro_cols].sort_values("date")

df = raw.sort_values("date")
df = pd.merge_asof(df, fwd.sort_values("date"), on="date", direction="nearest", tolerance=pd.Timedelta("5D"))
df = pd.merge_asof(df.sort_values("date"), macro.sort_values("date"), on="date", direction="nearest", tolerance=pd.Timedelta("5D"))

df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(int)

# Feature sets
all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
label_like = [c for c in all_numeric if "stress" in c or c.endswith("_label") or c == TARGET]
macro_features = [c for c in macro_cols if c in df.columns]
network_features = [c for c in all_numeric if c not in label_like + macro_features]
network_features = [c for c in network_features if not c.startswith("vix_threshold")]

feature_sets = {
    "macro_only": macro_features,
    "network_only": network_features,
    "network_plus_macro": network_features + macro_features,
}

# Chronological split: 60% train, 20% calibration, 20% test
df = df.sort_values("date").reset_index(drop=True)
n = len(df)
train_end = int(n * 0.60)
cal_end = int(n * 0.80)

train = df.iloc[:train_end].copy()
cal = df.iloc[train_end:cal_end].copy()
test = df.iloc[cal_end:].copy()

models = {
    "logistic": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced_subsample",
    ),
}

results = []
pred_records = []

for fs_name, features in feature_sets.items():
    features = [f for f in features if f in df.columns]
    if not features:
        continue

    X_train = train[features]
    y_train = train[TARGET].values
    X_cal = cal[features]
    y_cal = cal[TARGET].values
    X_test = test[features]
    y_test = test[TARGET].values

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print(f"Skipping {fs_name}: insufficient class variation in train/test.")
        continue

    for model_name, base_model in models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False) if model_name == "logistic" else SimpleImputer(strategy="median")),
        ])

        # Use a custom preprocessing flow to avoid scaling RandomForest twice
        if model_name == "logistic":
            estimator = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clone(base_model)),
            ])
        else:
            estimator = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", clone(base_model)),
            ])

        estimator.fit(X_train, y_train)

        p_cal = estimator.predict_proba(X_cal)[:, 1] if len(cal) > 0 else estimator.predict_proba(X_train)[:, 1]
        y_cal_used = y_cal if len(cal) > 0 else y_train

        p_test_raw = estimator.predict_proba(X_test)[:, 1]
        p_test_cal = platt_calibrate(p_cal, y_cal_used, p_test_raw)

        for prob_type, probs in [("raw", p_test_raw), ("platt_calibrated", p_test_cal)]:
            preds = (probs >= 0.5).astype(int)

            result = {
                "feature_set": fs_name,
                "model": model_name,
                "probability_type": prob_type,
                "n_train": len(train),
                "n_calibration": len(cal),
                "n_test": len(test),
                "positive_share_train": float(np.mean(y_train)),
                "positive_share_test": float(np.mean(y_test)),
                "brier": safe_metric(brier_score_loss, y_test, probs),
                "roc_auc": safe_metric(roc_auc_score, y_test, probs),
                "average_precision": safe_metric(average_precision_score, y_test, probs),
                "log_loss": safe_metric(log_loss, y_test, probs),
                "ece_10_bins": expected_calibration_error(y_test, probs, n_bins=10),
                "accuracy": float(accuracy_score(y_test, preds)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
                "n_features": len(features),
            }
            results.append(result)

            for date, yt, pr, pd_ in zip(test["date"], y_test, probs, preds):
                pred_records.append({
                    "date": date,
                    "feature_set": fs_name,
                    "model": model_name,
                    "probability_type": prob_type,
                    "target": TARGET,
                    "y_true": yt,
                    "y_prob": pr,
                    "y_pred": pd_,
                })

results_df = pd.DataFrame(results).sort_values(["brier", "ece_10_bins"], ascending=True)
preds_df = pd.DataFrame(pred_records)

results_df.to_csv(REPORT_DIR / "baseline_model_results.csv", index=False)
preds_df.to_csv(REPORT_DIR / "baseline_predictions.csv", index=False)

print(f"Saved: {REPORT_DIR / 'baseline_model_results.csv'}")
print(f"Saved: {REPORT_DIR / 'baseline_predictions.csv'}")
print(results_df.head(20))
