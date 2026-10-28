"""Wear models, tested leave-one-case-out (LOCO).

Each NASA case is one tool under one set of cutting conditions. To test how a
model does on a tool it has never seen, we train on 15 cases and predict the
16th, then repeat for every case. Splitting by individual run would leak
information, because neighbouring runs of the same tool look almost identical.

Models
------
process_only  Wear from cutting time and settings only, fitted in log space
              (a simple Taylor-style law: wear grows as a power of time,
              scaled by depth of cut, feed and material). No sensors.
ridge         Linear model on all features (standardised).
forest        Random forest on all features.
gbm_q         Gradient boosting, three models for the 10th, 50th and 90th
              percentile of wear. The median is the point prediction; the gap
              between 10th and 90th is the uncertainty used by the risk layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

QUANTILES = (0.1, 0.5, 0.9)
SEED = 0


def _process_design(df: pd.DataFrame) -> np.ndarray:
    t = np.log(df["time"].clip(lower=1e-3).values)
    return np.column_stack([t, np.log(df["DOC"].values), np.log(df["feed"].values),
                            (df["material"].values == 2).astype(float)])


class ProcessOnly:
    """log(VB + eps) = a + b*log(time) + c*log(DOC) + d*log(feed) + e*steel"""
    eps = 0.02

    def fit(self, df, y):
        self.m = LinearRegression().fit(_process_design(df), np.log(y + self.eps))
        return self

    def predict(self, df):
        return np.exp(self.m.predict(_process_design(df))) - self.eps


def _gbm(alpha):
    return GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=300,
                                     max_depth=3, learning_rate=0.05, subsample=0.8,
                                     min_samples_leaf=3, random_state=SEED)


def loco_predict(df: pd.DataFrame, feats: list[str], models=("process_only", "ridge", "forest", "gbm_q")) -> pd.DataFrame:
    """Predict every run of every case using models trained on the other cases."""
    out = []
    train_mask_all = df["VB_label"].notna()
    for case in sorted(df["case"].unique()):
        test = df[df["case"] == case]
        train = df[(df["case"] != case) & train_mask_all]
        X_tr, y_tr, X_te = train[feats].values, train["VB_label"].values, test[feats].values
        pred = test[["case", "run", "time", "VB", "VB_label", "VB_measured", "DOC", "feed", "material"]].copy()

        if "process_only" in models:
            pred["pred_process_only"] = ProcessOnly().fit(train, y_tr).predict(test)
        if "ridge" in models:
            pred["pred_ridge"] = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X_tr, y_tr).predict(X_te)
        if "forest" in models:
            rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
            pred["pred_forest"] = rf.fit(X_tr, y_tr).predict(X_te)
        if "gbm_q" in models:
            qs = np.column_stack([_gbm(a).fit(X_tr, y_tr).predict(X_te) for a in QUANTILES])
            qs.sort(axis=1)  # stop quantiles crossing
            pred["q10"], pred["q50"], pred["q90"] = qs.T
            pred["pred_gbm_q"] = pred["q50"]
        out.append(pred)
    res = pd.concat(out).reset_index(drop=True)
    for c in res.columns:
        if c.startswith("pred_") or c in ("q10", "q50", "q90"):
            res[c] = res[c].clip(lower=0)  # wear cannot be negative
    return res


def score(pred: pd.DataFrame) -> pd.DataFrame:
    """Error on runs where wear was really measured (never on interpolated values)."""
    m = pred[pred["VB_measured"]]
    rows = []
    for c in [c for c in pred.columns if c.startswith("pred_")]:
        err = m[c] - m["VB"]
        rows.append({"model": c.replace("pred_", ""), "MAE_mm": err.abs().mean(),
                     "RMSE_mm": np.sqrt((err**2).mean()), "n_test_points": len(m)})
    out = pd.DataFrame(rows).sort_values("MAE_mm").reset_index(drop=True)
    if {"q10", "q90"} <= set(pred.columns):
        cover = ((m["VB"] >= m["q10"]) & (m["VB"] <= m["q90"])).mean()
        out.attrs["interval_coverage_80"] = float(cover)
    return out
