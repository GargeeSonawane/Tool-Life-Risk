"""Run the whole project: data -> features -> models -> risk -> policies -> figures -> summary.

Usage:
    python run_all.py --data data/mill.mat
    python run_all.py --data data/mill.mat --limit 0.3 --p-star 0.2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src import plots
from src.data import SIGNALS, add_wear_labels, drop_bad_runs, load_mill
from src.features import build_feature_table, feature_columns
from src.models import _gbm, loco_predict, score
from src.risk import (add_risk, expected_cost, fixed_K_loco, fixed_policy, risk_policy, summarise)

SENSOR_NAMES = {"smcAC": "spindle current AC", "smcDC": "spindle current DC", "vib_table": "table vibration",
                "vib_spindle": "spindle vibration", "AE_table": "table acoustic emission",
                "AE_spindle": "spindle acoustic emission"}


def missing_sensor_study(df: pd.DataFrame) -> pd.DataFrame:
    """Median GBM, leave-one-case-out, with different sensor sets."""
    variants = {"all sensors": []}
    for s in SIGNALS:
        variants[f"without {SENSOR_NAMES[s]}"] = [s]
    variants["spindle current only"] = [s for s in SIGNALS if not s.startswith("smc")]
    variants["no sensors (time + settings)"] = list(SIGNALS)

    rows = []
    for name, drop in variants.items():
        feats = feature_columns(df, drop_signals=drop)
        errs = []
        for case in sorted(df["case"].unique()):
            tr = df[(df["case"] != case) & df["VB_label"].notna()]
            te = df[(df["case"] == case) & df["VB_measured"]]
            if te.empty:
                continue
            p = _gbm(0.5).fit(tr[feats].values, tr["VB_label"].values).predict(te[feats].values)
            errs.extend(np.abs(np.clip(p, 0, None) - te["VB"].values))
        rows.append({"features": name, "MAE_mm": float(np.mean(errs)), "n_features": len(feats)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mill.mat")
    ap.add_argument("--limit", type=float, default=0.3, help="wear limit VB in mm (ISO 3685 uses 0.3 mm)")
    ap.add_argument("--p-star", type=float, default=0.2, help="replace when P(next run passes limit) >= p*")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cost-failure", type=float, default=20.0, help="illustrative cost of one failure")
    ap.add_argument("--cost-wasted-run", type=float, default=1.0, help="illustrative cost of one unused safe run")
    args = ap.parse_args()

    out = Path(args.out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    print("1/6 loading data")
    meta, signals = load_mill(args.data)
    n_raw = len(meta)
    meta = drop_bad_runs(meta, signals)
    meta = add_wear_labels(meta)
    print(f"    {n_raw} runs, kept {len(meta)}, {meta['case'].nunique()} cases, "
          f"{int(meta['VB_measured'].sum())} with measured wear")

    print("2/6 features")
    df = build_feature_table(meta, signals)
    feats = feature_columns(df)
    df.to_csv(out / "features.csv", index=False)

    print("3/6 leave-one-case-out models")
    pred = loco_predict(df, feats)
    scores = score(pred)
    cover = scores.attrs.get("interval_coverage_80", np.nan)
    scores.to_csv(out / "model_scores.csv", index=False)
    print(scores.to_string(index=False))
    print(f"    80% interval coverage on unseen tools: {cover:.0%}")

    print("4/6 risk layer and policies")
    risked = add_risk(pred, args.limit)
    risked.to_csv(out / "predictions.csv", index=False)

    chosen_pol = risk_policy(risked, args.limit, args.p_star)
    fixed_loco_pol = fixed_K_loco(risked, args.limit)
    chosen, fixed_loco = summarise(chosen_pol), summarise(fixed_loco_pol)
    chosen_pol.to_csv(out / "policy_risk.csv", index=False)
    fixed_loco_pol.to_csv(out / "policy_fixed.csv", index=False)

    grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    model_curve = pd.DataFrame([{"p_star": p, **summarise(risk_policy(risked, args.limit, p))} for p in grid])
    max_safe = int(max(chosen_pol["safe_runs"].max(), 1)) if not chosen_pol.empty else 1
    fixed_curve = pd.DataFrame([{"K": k, **summarise(fixed_policy(risked, args.limit, k))}
                                for k in range(1, max_safe + 2)])
    model_curve.to_csv(out / "curve_risk.csv", index=False)
    fixed_curve.to_csv(out / "curve_fixed.csv", index=False)

    cost_model = expected_cost(chosen_pol, args.cost_failure, args.cost_wasted_run)
    cost_fixed = expected_cost(fixed_loco_pol, args.cost_failure, args.cost_wasted_run)

    print("5/6 missing-sensor study")
    miss = missing_sensor_study(df)
    miss.to_csv(out / "missing_sensor.csv", index=False)
    print(miss.to_string(index=False))

    print("6/6 figures and summary")
    plots.wear_by_case(risked, args.limit, figs / "wear_by_case.png")
    if not chosen_pol.empty:
        plots.tradeoff(model_curve, fixed_curve, chosen, fixed_loco, figs / "tradeoff.png")
    plots.missing_sensor(miss, figs / "missing_sensor.png")
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=0, n_jobs=-1)
    tr = df[df["VB_label"].notna()]
    rf.fit(tr[feats].values, tr["VB_label"].values)
    plots.importance(feats, rf.feature_importances_, figs / "feature_importance.png")

    censored = meta["case"].nunique() - len(chosen_pol)
    best = scores.iloc[0]
    proc = scores[scores["model"] == "process_only"]["MAE_mm"]
    lines = [
        "# Results summary (generated by run_all.py)",
        "",
        f"Data file: `{args.data}`  ·  wear limit VB = {args.limit} mm  ·  p* = {args.p_star}",
        f"Runs kept: {len(meta)} of {n_raw}  ·  cases: {meta['case'].nunique()}  ·  "
        f"runs with measured wear: {int(meta['VB_measured'].sum())}",
        "",
        "## 1. Wear prediction on unseen tools (leave-one-case-out)",
        "",
        scores.to_markdown(index=False, floatfmt=".3f"),
        "",
        f"Best model: **{best['model']}**, mean absolute error {best['MAE_mm']:.3f} mm"
        + (f" (time-and-settings baseline: {float(proc.iloc[0]):.3f} mm)." if len(proc) else "."),
        f"The 10–90% band contained the measured wear {cover:.0%} of the time (ideal: 80%).",
        "",
        "## 2. Replacement policy on unseen tools",
        "",
        f"Tools that reach the limit within the recorded runs: {len(chosen_pol)} "
        f"({censored} cases never reached it and are left out of this test).",
        "",
        "| Policy | Tools | Failures | Safe life used |",
        "|---|---|---|---|",
        f"| Risk-based, p* = {args.p_star} | {chosen['tools']} | {chosen['failures']} | {chosen['mean_life_used']:.0%} |",
        f"| Fixed K (shortest safe life of the other tools) | {fixed_loco['tools']} | {fixed_loco['failures']} | "
        f"{fixed_loco['mean_life_used']:.0%} |",
        "",
        f"Illustrative cost per tool (failure = {args.cost_failure:g}, unused safe run = {args.cost_wasted_run:g}): "
        f"risk-based {cost_model:.2f}, fixed {cost_fixed:.2f}. These cost weights are assumptions, not data.",
        "",
        "## 3. Missing sensors",
        "",
        miss.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Figures are in `results/figures/`.",
    ]
    (out / "summary.md").write_text("\n".join(lines))
    (out / "summary.json").write_text(json.dumps({"scores": scores.to_dict("records"), "coverage_80": cover,
                                                 "risk_policy": chosen, "fixed_policy": fixed_loco,
                                                 "cost_risk": cost_model, "cost_fixed": cost_fixed}, indent=2,
                                                default=float))
    print(f"done → {out / 'summary.md'}")


if __name__ == "__main__":
    main()
