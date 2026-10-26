"""
33_leakfree_label_variants.py

Builds leak-free and VIX-free variants of the forward stress label and re-runs
the main rule/model baselines on each, so that two separate questions can be
answered explicitly:

(Q1) Look-ahead in the label.
     In 04/08 the VIX stress threshold is the 0.80 quantile of the FULL
     2005-2025 sample. The target for a decision at date t therefore depends on
     data observed after t. Variant L1 recomputes that threshold with an
     expanding window using only observations strictly before t.

(Q2) Tautology.
     The published label is (VIX high) OR (NFCI > 0); OFR FSI is absent from
     the macro file, so it contributes nothing. VIX alone accounts for the
     large majority of positive days, and the best baseline is a monotone
     transform of VIX. Variant L2 removes VIX from the LABEL entirely
     (NFCI > 0 only), which is the test the feature-side ablation in the
     manuscript does not perform.

Label variants
  L0_published   VIX >= full-sample q80  OR NFCI > 0      (as published)
  L1_leakfree    VIX >= expanding q80    OR NFCI > 0
  L2_vixfree     NFCI > 0                                 (no VIX in label)

Evaluation: expanding-window, one test year at a time, first test year 2011,
identical protocol for every variant.

Outputs:
  data_processed/labels/label_variants_daily.csv
  data_processed/labels/label_variants_snapshots.csv
  outputs/reports/label_variant_baseline_results.csv
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    brier_score_loss, roc_auc_score, average_precision_score,
    accuracy_score, balanced_accuracy_score,
)

from config import LABEL_DIR, NETWORK_DIR, REPORT_DIR
from common import ensure_dirs

ensure_dirs(LABEL_DIR, REPORT_DIR)

FIRST_TEST_YEAR = 2011
FORWARD_DAYS = 20
BURN_IN = 500          # trading days before an expanding threshold is trusted
VIX_Q = 0.80


def ece(y, p, n_bins=10):
    y = np.asarray(y); p = np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p <= hi) if i == 0 else (p > lo) & (p <= hi)
        if m.sum() == 0:
            continue
        out += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(out)


def metrics(y, p):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
    pred = (p >= 0.5).astype(int)
    def safe(fn):
        try:
            return float(fn(y, p))
        except Exception:
            return np.nan
    return {
        "brier": safe(brier_score_loss),
        "roc_auc": safe(roc_auc_score),
        "average_precision": safe(average_precision_score),
        "ece": ece(y, p),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "n": int(len(y)),
        "positive_share": float(y.mean()),
    }


# ---------------------------------------------------------------- daily layer
daily = pd.read_csv(LABEL_DIR / "stress_labels.csv", parse_dates=["date"]).sort_values("date")
daily = daily.set_index("date")

vix = daily["vix"]
nfci_stress = (daily["nfci"] > 0).astype(int)

# L0: published, full-sample threshold (look-ahead)
thr_full = vix.quantile(VIX_Q)
vix_stress_L0 = (vix >= thr_full).astype(int)

# L1: expanding threshold, strictly past information only
thr_exp = vix.expanding(min_periods=BURN_IN).quantile(VIX_Q).shift(1)
vix_stress_L1 = (vix >= thr_exp).astype(float)

# L3/L4: targets built from realized equity-market outcomes instead of from the
# same volatility index that the best baseline transforms. Thresholds are
# expanding and strictly backward-looking, so these targets are leak-free and
# VIX-independent by construction. VIX is then a legitimate predictor rather
# than a component of the label.
mret = daily["universe_market_return"]
fwd_cum = mret[::-1].rolling(FORWARD_DAYS, min_periods=FORWARD_DAYS).sum()[::-1].shift(-1)
fwd_vol = mret[::-1].rolling(FORWARD_DAYS, min_periods=FORWARD_DAYS).std()[::-1].shift(-1)

dd_thr = fwd_cum.expanding(min_periods=BURN_IN).quantile(0.10).shift(1)
vol_thr = fwd_vol.expanding(min_periods=BURN_IN).quantile(0.80).shift(1)

variants = {
    "L0_published": ((vix_stress_L0 == 1) | (nfci_stress == 1)).astype(float),
    "L1_leakfree": ((vix_stress_L1 == 1) | (nfci_stress == 1)).astype(float).where(thr_exp.notna()),
    "L2_vixfree": nfci_stress.astype(float),
    "L3_fwd_drawdown": (fwd_cum <= dd_thr).astype(float).where(dd_thr.notna() & fwd_cum.notna()),
    "L4_fwd_realized_vol": (fwd_vol >= vol_thr).astype(float).where(vol_thr.notna() & fwd_vol.notna()),
}

lab = pd.DataFrame(variants)
lab["vix"] = vix
lab["nfci"] = daily["nfci"]
lab["threshold_full_sample"] = thr_full
lab["threshold_expanding"] = thr_exp
lab.reset_index().to_csv(LABEL_DIR / "label_variants_daily.csv", index=False)

print("Daily stress share by variant:")
print(lab[list(variants)].mean().round(4).to_string())
print(f"\nFull-sample VIX threshold: {thr_full:.3f}")
print("Expanding threshold, first/last valid: "
      f"{thr_exp.dropna().iloc[0]:.3f} / {thr_exp.dropna().iloc[-1]:.3f}")
print(f"Days where L0 and L1 disagree: {int((lab['L0_published'] != lab['L1_leakfree']).sum())}")

# ------------------------------------------------------------ snapshot layer
metrics_df = pd.read_csv(NETWORK_DIR / "network_metrics.csv", parse_dates=["date"])
snapshot_dates = pd.to_datetime(sorted(metrics_df["date"].dropna().unique()))

idx = lab.index
rows = []
for d in snapshot_dates:
    prev = idx[idx <= d]
    if len(prev) == 0:
        continue
    d0 = prev.max()
    pos = idx.get_loc(d0)
    fwd = lab.iloc[pos + 1: pos + 1 + FORWARD_DAYS]
    if len(fwd) < FORWARD_DAYS:
        continue
    row = {"date": d0, "snapshot_date": d}
    for v in variants:
        cur = lab[v].iloc[pos]
        row[f"current_{v}"] = cur
        if v in ("L3_fwd_drawdown", "L4_fwd_realized_vol"):
            # already defined over the forward window at date d0
            row[f"next20_{v}"] = float(cur) if pd.notna(cur) else np.nan
            row[f"current_{v}"] = np.nan
        else:
            f = fwd[v]
            row[f"next20_{v}"] = float(f.max()) if f.notna().all() else np.nan
    rows.append(row)

snap = pd.DataFrame(rows)
snap.to_csv(LABEL_DIR / "label_variants_snapshots.csv", index=False)
print(f"\nSnapshots: {len(snap)}")
print(snap[[f'next20_{v}' for v in variants]].mean().round(4).to_string())

# ------------------------------------------------------------------ features
# high_yield_oas is 87% missing in data_raw/macro/fred_stress_data.csv
# (712 of 5487 rows) and is dropped rather than silently imputed.
macro_cols = [c for c in ["vix", "nfci", "treasury_10y", "treasury_2y",
                          "term_spread_10y_2y",
                          "universe_market_return"] if c in daily.columns]
macro = daily[macro_cols].reset_index()
df = pd.merge_asof(snap.sort_values("date"), macro.sort_values("date"),
                   on="date", direction="nearest", tolerance=pd.Timedelta("5D"))
df["year"] = df["date"].dt.year

MODEL_SETS = {
    "macro_full": macro_cols,
    "macro_no_vix": [c for c in macro_cols if c != "vix"],
}


def run_variant(target):
    d = df.dropna(subset=[target]).copy()
    years = sorted(d["year"].unique())
    test_years = [y for y in years if y >= FIRST_TEST_YEAR]
    preds = []

    for y in test_years:
        train = d[d["year"] < y]
        test = d[d["year"] == y]
        if len(train) < 24 or test.empty or train[target].nunique() < 2:
            continue

        # rule baselines, thresholds from training period only
        vthr = train["vix"].quantile(0.80)
        vlo, vhi = train["vix"].quantile(0.05), train["vix"].quantile(0.95)
        med = train["vix"].median()
        v = test["vix"].fillna(med)
        scaled = ((v - vlo) / (vhi - vlo)).clip(0, 1) if vhi > vlo else pd.Series(0.5, index=v.index)

        out = {
            "vix_scaled_score": scaled.values,
            "vix_80pct_rule": (v >= vthr).astype(float).values,
            "persistence": test[target.replace("next20_", "current_")].fillna(0).values
                            if test[target.replace("next20_", "current_")].notna().any()
                            else np.full(len(test), np.nan),
            "prevalence": np.repeat(train[target].mean(), len(test)),
        }

        for name, feats in MODEL_SETS.items():
            Xtr, ytr = train[feats], train[target].astype(int)
            Xte = test[feats]
            for mname, clf in [
                ("logistic", LogisticRegression(max_iter=3000, class_weight="balanced")),
                ("random_forest", RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                        class_weight="balanced", random_state=0)),
            ]:
                pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                                 ("sc", StandardScaler()), ("clf", clf)])
                try:
                    pipe.fit(Xtr, ytr)
                    out[f"{name}|{mname}"] = pipe.predict_proba(Xte)[:, 1]
                except Exception:
                    out[f"{name}|{mname}"] = np.full(len(test), np.nan)

        for k, p in out.items():
            preds.append(pd.DataFrame({"date": test["date"].values, "method": k,
                                       "p": p, "y": test[target].astype(int).values}))

    if not preds:
        return pd.DataFrame()
    allp = pd.concat(preds)
    res = []
    for m, g in allp.groupby("method"):
        g = g.dropna(subset=["p"])
        if g.empty:
            continue
        r = metrics(g["y"], g["p"])
        r["method"] = m
        r["target"] = target
        res.append(r)
    return pd.DataFrame(res)


allres = pd.concat([run_variant(f"next20_{v}") for v in variants], ignore_index=True)
allres = allres[["target", "method", "n", "positive_share", "brier", "roc_auc",
                 "average_precision", "ece", "accuracy", "balanced_accuracy"]]
allres = allres.sort_values(["target", "brier"])
allres.to_csv(REPORT_DIR / "label_variant_baseline_results.csv", index=False)

print("\n================ Expanding-window OOS results by label variant ================")
for t in allres["target"].unique():
    print(f"\n--- {t} ---")
    print(allres[allres["target"] == t].round(4).to_string(index=False))
