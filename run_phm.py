"""Second study: PHM Society 2010 milling data (force, vibration, acoustic emission).

Usage:
    python run_phm.py --data data/phm2010
    python run_phm.py --data data/phm2010 --limit 0.15 --p-star 0.2

Put the unzipped c1, c4 and c6 folders (with their c1_wear.csv, c4_wear.csv, c6_wear.csv)
anywhere inside the --data folder; the script finds them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src import plots
from src.models import _gbm
from src.phm import GROUPS, build_features, feature_columns, loco_predict, nested_risk_policy, score
from src.risk import (add_risk, expected_cost, fixed_K_loco, fixed_policy, risk_policy, summarise, true_safe_runs)

SIGMA_FLOOR = 0.005  # mm; PHM wear values are smaller than NASA's, so a smaller floor


def missing_sensor_study(df: pd.DataFrame) -> pd.DataFrame:
    variants = {"all sensors": []}
    for g in GROUPS:
        variants[f"without {g}"] = [g]
    variants["force only"] = ["vibration", "acoustic emission"]
    rows = []
    for name, drop in variants.items():
        feats = feature_columns(df, drop_groups=drop)
        errs = []
        for c in sorted(df["case"].unique()):
            tr, te = df[df["case"] != c], df[df["case"] == c]
            p = _gbm(0.5).fit(tr[feats].values, tr["VB_label"].values).predict(te[feats].values)
            errs.extend(np.abs(np.clip(p, 0, None) - te["VB"].values))
        rows.append({"features": name, "MAE_mm": float(np.mean(errs)), "n_features": len(feats)})
    # no sensors at all: wear from cut number only, same model type
    errs = []
    for c in sorted(df["case"].unique()):
        tr, te = df[df["case"] != c], df[df["case"] == c]
        p = _gbm(0.5).fit(tr[["run"]].values, tr["VB_label"].values).predict(te[["run"]].values)
        errs.extend(np.abs(np.clip(p, 0, None) - te["VB"].values))
    rows.append({"features": "no sensors (cut number only)", "MAE_mm": float(np.mean(errs)), "n_features": 1})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phm2010")
    ap.add_argument("--limit", type=float, default=0.15, help="wear limit (worst flute) in mm")
    ap.add_argument("--p-star", type=float, default=0.2)
    ap.add_argument("--out", default="results_phm")
    ap.add_argument("--cost-failure", type=float, default=20.0)
    ap.add_argument("--cost-wasted-run", type=float, default=1.0)
    ap.add_argument("--rebuild", action="store_true", help="re-read the raw files instead of using the cache")
    args = ap.parse_args()

    out = Path(args.out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    cache = out / "features.csv"
    if args.rebuild and cache.exists():
        cache.unlink()

    print("1/6 reading cuts and building features (first run takes several minutes)")
    df = build_features(args.data, cache=cache)
    print(f"    {len(df)} cuts from cutters {sorted(int(c) for c in df['case'].unique())}")
    for c, g in df.groupby("case"):
        print(f"    c{c}: wear {g['VB'].min():.3f} → {g['VB'].max():.3f} mm")
    never = [c for c, g in df.groupby("case") if g["VB"].max() < args.limit]
    if never:
        print(f"    NOTE: cutters {never} never reach the {args.limit} mm limit; try a lower --limit")

    feats = feature_columns(df)

    print("2/6 leave-one-cutter-out models (with calibrated uncertainty)")
    pred = loco_predict(df, feats)
    scores, extra = score(pred)
    print(scores.to_string(index=False))
    print(f"    10–90% band coverage: before calibration {extra['coverage']['raw']:.0%}, "
          f"after {extra['coverage']['calibrated']:.0%} (target 80%)")
    scores.to_csv(out / "model_scores.csv", index=False)

    print("3/6 risk layer and policies")
    risked = add_risk(pred, args.limit, sigma_floor=SIGMA_FLOOR)
    risked.to_csv(out / "predictions.csv", index=False)
    chosen_pol = risk_policy(risked, args.limit, args.p_star)
    fixed_pol = fixed_K_loco(risked, args.limit)
    chosen, fixed = summarise(chosen_pol), summarise(fixed_pol)
    chosen_pol.to_csv(out / "policy_risk.csv", index=False)
    fixed_pol.to_csv(out / "policy_fixed.csv", index=False)
    grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    nested_pol = nested_risk_policy(df, feats, risked, args.limit, grid, SIGMA_FLOOR)
    nested = summarise(nested_pol)
    nested_pol.to_csv(out / "policy_risk_nested.csv", index=False)
    print(f"    risk policy, p* fixed at {args.p_star}: {chosen['failures']} failures, "
          f"{chosen['mean_life_used']:.0%} life used")
    if not nested_pol.empty:
        print(f"    risk policy, p* chosen from training cutters "
              f"({', '.join(f'c{int(c)}: {p:g}' for c, p in zip(nested_pol['case'], nested_pol['p_star_chosen']))}): "
              f"{nested['failures']} failures, {nested['mean_life_used']:.0%} life used")
    print(f"    fixed rule: {fixed['failures']} failures, {fixed['mean_life_used']:.0%} life used")
    model_curve = pd.DataFrame([{"p_star": p, **summarise(risk_policy(risked, args.limit, p))} for p in grid])
    max_safe = int(chosen_pol["safe_runs"].max()) if not chosen_pol.empty else 1
    step = max(1, max_safe // 40)
    fixed_curve = pd.DataFrame([{"K": k, **summarise(fixed_policy(risked, args.limit, k))}
                                for k in range(1, max_safe + 2, step)])
    model_curve.to_csv(out / "curve_risk.csv", index=False)
    fixed_curve.to_csv(out / "curve_fixed.csv", index=False)
    cost_model = expected_cost(chosen_pol, args.cost_failure, args.cost_wasted_run)
    cost_fixed = expected_cost(fixed_pol, args.cost_failure, args.cost_wasted_run)

    print("4/6 missing-sensor study")
    miss = missing_sensor_study(df)
    miss.to_csv(out / "missing_sensor.csv", index=False)
    print(miss.to_string(index=False))

    print("5/6 figures")
    plots.wear_by_cutter(risked, args.limit, figs / "wear_by_cutter.png")
    if not chosen_pol.empty:
        plots.tradeoff(model_curve, fixed_curve, chosen, fixed, figs / "tradeoff.png")
    plots.missing_sensor(miss, figs / "missing_sensor.png")
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=0, n_jobs=-1)
    rf.fit(df[feats].values, df["VB_label"].values)
    plots.importance(feats, rf.feature_importances_, figs / "feature_importance.png")

    print("6/6 summary")
    per_cut = extra["per_cutter"].assign(case=lambda d: "c" + d["case"].astype(str))
    safe = {f"c{c}": true_safe_runs(g, args.limit) for c, g in risked.groupby("case")}
    lines = [
        "# PHM 2010 results summary (generated by run_phm.py)",
        "",
        f"Data folder: `{args.data}`  ·  wear limit (worst flute) = {args.limit} mm  ·  p* = {args.p_star}",
        f"Cuts: {len(df)}  ·  cutters: {', '.join('c' + str(c) for c in sorted(df['case'].unique()))}  ·  "
        f"safe cuts before the limit: {safe}",
        "",
        "## 1. Wear prediction on an unseen cutter (train on two, test on the third)",
        "",
        scores.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Error per held-out cutter (quantile gradient boosting):",
        "",
        per_cut.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 2. Uncertainty",
        "",
        f"10–90% band coverage before calibration: {extra['coverage']['raw']:.0%}; "
        f"after calibration: {extra['coverage']['calibrated']:.0%} (target 80%).",
        f"Band widening learned from the training cutters only: "
        f"{', '.join(f'c{c}: ×{s:.2f}' for c, s in pred.groupby('case')['band_scale'].first().items())}.",
        "",
        "## 3. Replacement policy",
        "",
        "| Policy | Tools | Failures | Safe life used |",
        "|---|---|---|---|",
        f"| Risk-based, p* = {args.p_star} | {chosen['tools']} | {chosen['failures']} | {chosen['mean_life_used']:.0%} |",
        (f"| Risk-based, p* chosen from the training cutters only "
         f"({', '.join(f'c{int(c)}: {p:g}' for c, p in zip(nested_pol['case'], nested_pol['p_star_chosen']))}) | "
         f"{nested['tools']} | {nested['failures']} | {nested['mean_life_used']:.0%} |") if not nested_pol.empty else "",
        f"| Fixed K (shortest safe life of the other cutters) | {fixed['tools']} | {fixed['failures']} | "
        f"{fixed['mean_life_used']:.0%} |",
        "",
        f"Illustrative cost per tool (failure = {args.cost_failure:g}, unused safe cut = {args.cost_wasted_run:g}): "
        f"risk-based {cost_model:.2f}, fixed {cost_fixed:.2f}. Assumed weights, not data.",
        "",
        "## 4. Missing sensors",
        "",
        miss.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Only three labelled cutters: treat every number here as indicative, not conclusive.",
        "Figures are in `results_phm/figures/`.",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({
        "scores": scores.to_dict("records"), "coverage": extra["coverage"],
        "risk_policy": chosen, "risk_policy_nested": nested, "fixed_policy": fixed, "cost_risk": cost_model, "cost_fixed": cost_fixed,
    }, indent=2, default=float), encoding="utf-8")
    print(f"done → {out / 'summary.md'}")


if __name__ == "__main__":
    main()
