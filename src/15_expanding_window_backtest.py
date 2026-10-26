"""
15_expanding_window_backtest.py

Runs expanding-window temporal validation with calibration.

This replaces the weak single 60/20/20 split.

Targets:
- next_20d_external_stress
- next_20d_stress_onset, if available

Feature sets:
- macro_only
- raw_threshold
- raw_topk
- residual_threshold
- residual_topk
- tail_threshold
- tail_topk
- macro_plus_raw_threshold
- macro_plus_raw_topk
- macro_plus_all_networks

Calibration:
- raw probability
- Platt calibration
- isotonic calibration

Outputs:
- outputs/reports/expanding_window_results.csv
- outputs/reports/expanding_window_predictions.csv
- outputs/reports/expanding_window_fold_summary.csv

Important:
This is still a modeling scaffold. For the final paper, inspect folds and
exclude unstable targets or models if positive classes are too sparse.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    balanced_accuracy_score,
)

from config import NETWORK_DIR, LABEL_DIR, REPORT_DIR
from common import ensure_dirs

ensure_dirs(REPORT_DIR)

SEPARATED_DIR = NETWORK_DIR / "separated"

MIN_TRAIN_OBS = 60
CALIBRATION_MONTHS = 24
FIRST_TEST_YEAR = 2011


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


def safe_metric(metric_fn, y_true, y_score):
    try:
        return float(metric_fn(y_true, y_score))
    except Exception:
        return np.nan


def safe_log_loss(y_true, y_prob):
    try:
        return float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6)))
    except Exception:
        return np.nan


def collapse_network_metrics(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty:
        return pd.DataFrame(columns=["date"])

    base_cols = ["weighted_degree", "betweenness", "eigenvector", "clustering", "network_density", "network_edges", "network_nodes"]
    base_cols = [c for c in base_cols if c in df.columns]

    grouped = df.groupby("date")[base_cols].agg(["mean", "max", "std"])
    grouped.columns = [f"{prefix}_{col}_{stat}" for col, stat in grouped.columns]
    grouped = grouped.reset_index()

    # Sector concentration and financial-sector metrics if available.
    if "sector" in df.columns and "weighted_degree" in df.columns:
        sector = (
            df.groupby(["date", "sector"], as_index=False)
            .agg(sector_weighted_degree_mean=("weighted_degree", "mean"),
                 sector_weighted_degree_max=("weighted_degree", "max"),
                 n_firms=("ticker", "nunique"))
        )

        fin = sector[sector["sector"].astype(str).str.lower().eq("financials")].copy()
        if not fin.empty:
            fin = fin[["date", "sector_weighted_degree_mean", "sector_weighted_degree_max"]]
            fin = fin.rename(columns={
                "sector_weighted_degree_mean": f"{prefix}_financials_degree_mean",
                "sector_weighted_degree_max": f"{prefix}_financials_degree_max",
            })
            grouped = grouped.merge(fin, on="date", how="left")

        # Herfindahl-style concentration across sectors
        tmp = sector.copy()
        total = tmp.groupby("date")["sector_weighted_degree_mean"].transform("sum")
        tmp["share"] = np.where(total > 0, tmp["sector_weighted_degree_mean"] / total, np.nan)
        hhi = tmp.groupby("date")["share"].apply(lambda s: np.nansum(s ** 2)).reset_index()
        hhi = hhi.rename(columns={"share": f"{prefix}_sector_degree_hhi"})
        grouped = grouped.merge(hhi, on="date", how="left")

    return grouped


def add_changes(df: pd.DataFrame, date_col="date") -> pd.DataFrame:
    df = df.sort_values(date_col).copy()
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c != "year"]
    for c in numeric:
        if any(key in c for key in ["density", "degree", "eigenvector", "hhi"]):
            df[f"{c}_delta1"] = df[c].diff()
            df[f"{c}_delta3"] = df[c].diff(3)
    return df


def load_dataset():
    labels_onset = LABEL_DIR / "snapshot_forward_labels_with_onset.csv"
    labels_plain = LABEL_DIR / "snapshot_forward_labels.csv"
    label_path = labels_onset if labels_onset.exists() else labels_plain

    labels = pd.read_csv(label_path, parse_dates=["date"]).sort_values("date")

    daily = pd.read_csv(LABEL_DIR / "stress_labels.csv", parse_dates=["date"]).sort_values("date")
    macro_cols = [
        c for c in [
            "vix", "nfci", "treasury_10y", "treasury_2y",
            "term_spread_10y_2y", "high_yield_oas", "ofr_fsi",
            "universe_market_return"
        ]
        if c in daily.columns
    ]
    macro = daily[["date"] + macro_cols].copy()

    df = labels.copy()
    df = pd.merge_asof(
        df.sort_values("date"),
        macro.sort_values("date"),
        on="date",
        direction="nearest",
        tolerance=pd.Timedelta("5D")
    )

    network_names = [
        "raw_threshold",
        "raw_topk",
        "residual_threshold",
        "residual_topk",
        "tail_threshold",
        "tail_topk",
    ]

    for name in network_names:
        path = SEPARATED_DIR / f"{name}_metrics.csv"
        if path.exists():
            nm = collapse_network_metrics(path, prefix=name)
            df = df.merge(nm, on="date", how="left")
        else:
            print(f"Warning: missing network metrics file: {path}")

    df = add_changes(df)
    df["year"] = df["date"].dt.year

    return df


def feature_groups(df: pd.DataFrame):
    macro_features = [
        c for c in [
            "vix", "nfci", "treasury_10y", "treasury_2y",
            "term_spread_10y_2y", "high_yield_oas", "ofr_fsi",
            "universe_market_return"
        ]
        if c in df.columns
    ]

    groups = {"macro_only": macro_features}

    prefixes = [
        "raw_threshold",
        "raw_topk",
        "residual_threshold",
        "residual_topk",
        "tail_threshold",
        "tail_topk",
    ]

    for p in prefixes:
        feats = [c for c in df.columns if c.startswith(p + "_")]
        groups[p] = feats
        groups[f"macro_plus_{p}"] = macro_features + feats

    all_network = []
    for p in prefixes:
        all_network.extend([c for c in df.columns if c.startswith(p + "_")])

    groups["all_networks"] = all_network
    groups["macro_plus_all_networks"] = macro_features + all_network

    # Remove empty and duplicated feature lists
    clean = {}
    for k, v in groups.items():
        v = list(dict.fromkeys([x for x in v if x in df.columns]))
        if v:
            clean[k] = v
    return clean


def fit_model(model_name: str):
    if model_name == "logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ])

    if model_name == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                max_depth=4,
                min_samples_leaf=5,
                random_state=42,
                class_weight="balanced_subsample",
            )),
        ])

    raise ValueError(model_name)


def calibrate_probs(method, p_cal, y_cal, p_test):
    p_cal = np.clip(np.asarray(p_cal), 1e-6, 1 - 1e-6)
    p_test = np.clip(np.asarray(p_test), 1e-6, 1 - 1e-6)

    if method == "raw":
        return p_test

    if len(np.unique(y_cal)) < 2:
        return p_test

    if method == "platt":
        x_cal = np.log(p_cal / (1 - p_cal)).reshape(-1, 1)
        x_test = np.log(p_test / (1 - p_test)).reshape(-1, 1)
        cal = LogisticRegression()
        cal.fit(x_cal, y_cal)
        return cal.predict_proba(x_test)[:, 1]

    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(p_cal, y_cal)
        return cal.predict(p_test)

    raise ValueError(method)


def evaluate_predictions(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "brier": safe_metric(brier_score_loss, y_true, y_prob),
        "roc_auc": safe_metric(roc_auc_score, y_true, y_prob),
        "average_precision": safe_metric(average_precision_score, y_true, y_prob),
        "log_loss": safe_log_loss(y_true, y_prob),
        "ece_10_bins": expected_calibration_error(y_true, y_prob, n_bins=10),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def run_backtest_for_target(df, target, groups):
    rows = df.dropna(subset=[target]).copy()
    rows[target] = rows[target].astype(int)

    years = sorted(rows["year"].dropna().unique())
    test_years = [y for y in years if y >= FIRST_TEST_YEAR]

    models = ["logistic", "random_forest"]
    calibrators = ["raw", "platt", "isotonic"]

    predictions = []
    fold_summary = []

    for test_year in test_years:
        train_all = rows[rows["year"] < test_year].copy()
        test = rows[rows["year"] == test_year].copy()

        if len(train_all) < MIN_TRAIN_OBS or len(test) == 0:
            continue

        cal_start_date = train_all["date"].max() - pd.DateOffset(months=CALIBRATION_MONTHS)
        train = train_all[train_all["date"] < cal_start_date].copy()
        cal = train_all[train_all["date"] >= cal_start_date].copy()

        if len(train) < 36 or len(cal) < 8:
            continue

        if train[target].nunique() < 2 or test[target].nunique() < 2:
            # Still record fold condition; skip metrics-heavy training for unstable fold.
            fold_summary.append({
                "target": target,
                "test_year": test_year,
                "status": "skipped_class_imbalance",
                "n_train": len(train),
                "n_calibration": len(cal),
                "n_test": len(test),
                "train_positive_share": float(train[target].mean()),
                "cal_positive_share": float(cal[target].mean()),
                "test_positive_share": float(test[target].mean()),
            })
            continue

        fold_summary.append({
            "target": target,
            "test_year": test_year,
            "status": "used",
            "n_train": len(train),
            "n_calibration": len(cal),
            "n_test": len(test),
            "train_positive_share": float(train[target].mean()),
            "cal_positive_share": float(cal[target].mean()),
            "test_positive_share": float(test[target].mean()),
        })

        for group_name, features in groups.items():
            features = [f for f in features if f in rows.columns]
            if not features:
                continue

            X_train = train[features]
            y_train = train[target].values
            X_cal = cal[features]
            y_cal = cal[target].values
            X_test = test[features]
            y_test = test[target].values

            if len(np.unique(y_train)) < 2:
                continue

            for model_name in models:
                model = fit_model(model_name)
                model.fit(X_train, y_train)

                p_cal = model.predict_proba(X_cal)[:, 1]
                p_test_base = model.predict_proba(X_test)[:, 1]

                for cal_method in calibrators:
                    p_test = calibrate_probs(cal_method, p_cal, y_cal, p_test_base)

                    for date, yt, yp in zip(test["date"], y_test, p_test):
                        predictions.append({
                            "target": target,
                            "test_year": test_year,
                            "date": date,
                            "feature_set": group_name,
                            "model": model_name,
                            "calibration": cal_method,
                            "y_true": int(yt),
                            "y_prob": float(yp),
                            "y_pred": int(yp >= 0.5),
                        })

    return pd.DataFrame(predictions), pd.DataFrame(fold_summary)


def summarize_predictions(pred_df):
    if pred_df.empty:
        return pd.DataFrame()

    results = []
    keys = ["target", "feature_set", "model", "calibration"]

    for key, g in pred_df.groupby(keys):
        y = g["y_true"].values
        p = g["y_prob"].values

        if len(np.unique(y)) < 2:
            continue

        metrics = evaluate_predictions(y, p)
        row = dict(zip(keys, key))
        row.update(metrics)
        row["n_oos"] = len(g)
        row["positive_share_oos"] = float(np.mean(y))
        results.append(row)

    return pd.DataFrame(results).sort_values(["target", "brier", "ece_10_bins"], ascending=True)


if __name__ == "__main__":
    df = load_dataset()
    groups = feature_groups(df)

    targets = ["next_20d_external_stress"]
    if "next_20d_stress_onset" in df.columns:
        targets.append("next_20d_stress_onset")

    all_preds = []
    all_folds = []

    for target in targets:
        print(f"Running expanding-window backtest for target: {target}")
        pred, folds = run_backtest_for_target(df, target, groups)
        all_preds.append(pred)
        all_folds.append(folds)

    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    fold_df = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()
    result_df = summarize_predictions(pred_df)

    pred_path = REPORT_DIR / "expanding_window_predictions.csv"
    fold_path = REPORT_DIR / "expanding_window_fold_summary.csv"
    result_path = REPORT_DIR / "expanding_window_results.csv"

    pred_df.to_csv(pred_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    result_df.to_csv(result_path, index=False)

    print(f"Saved: {pred_path}")
    print(f"Saved: {fold_path}")
    print(f"Saved: {result_path}")
    print(result_df.head(30))
