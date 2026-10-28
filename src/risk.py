"""Risk layer: from predicted wear (with uncertainty) to keep / inspect / replace.

1. Uncertainty. The 10th-90th percentile band from the quantile models is
   turned into a normal distribution around the median:
       sigma = (q90 - q10) / (2 * 1.2816)
   with a small floor so the model is never 'certain'.

2. Look one run ahead. The decision is made *after* run r and asks: if the tool
   is allowed to do run r+1, how likely is it to pass the wear limit? Next-run
   wear = current median + recent wear rate (never negative).

3. Decision.
       p < KEEP_BELOW            -> keep cutting
       KEEP_BELOW <= p < p_star  -> inspect (measure the tool)
       p >= p_star               -> replace

4. Policy test on held-out tools. A tool is 'stopped' at the first run where
   the policy says replace. 'Safe runs' for a tool are the runs before it truly
   reaches the limit. We report:
       failure     the tool reached the limit during a run the policy allowed
       life used   stopped runs / safe runs (1.0 = used every safe run)
   and compare with a fixed rule 'replace after K runs', the conservative
   practice described in the GKN project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

Z90 = 1.2816
SIGMA_FLOOR = 0.02  # mm
KEEP_BELOW = 0.05


def add_risk(pred: pd.DataFrame, limit: float) -> pd.DataFrame:
    out = []
    for _, g in pred.groupby("case", sort=True):
        g = g.sort_values("run").copy()
        sigma = np.maximum((g["q90"] - g["q10"]) / (2 * Z90), SIGMA_FLOOR)
        rate = g["q50"].diff().clip(lower=0).rolling(2, min_periods=1).mean().fillna(0)
        nxt = g["q50"] + rate
        g["sigma"] = sigma
        g["p_now"] = 1 - norm.cdf((limit - g["q50"]) / sigma)
        g["p_next"] = 1 - norm.cdf((limit - nxt) / sigma)
        out.append(g)
    return pd.concat(out).reset_index(drop=True)


def decide(p: float, p_star: float) -> str:
    if p >= p_star:
        return "replace"
    if p >= KEEP_BELOW:
        return "inspect"
    return "keep"


def true_safe_runs(g: pd.DataFrame, limit: float):
    """Number of runs before the tool truly reaches the limit, or None if it never does."""
    g = g.sort_values("run").reset_index(drop=True)
    hit = np.where(g["VB_label"].values >= limit)[0]
    if len(hit) == 0:
        return None
    return int(hit[0])  # rows before the first row at/over the limit = safe runs


def _evaluate(stop_run_index: int, safe: int) -> tuple[bool, float]:
    """stop_run_index = how many runs the tool did (1-based count)."""
    failure = stop_run_index > safe
    life = min(stop_run_index, safe) / safe if safe > 0 else 1.0
    return failure, life


def risk_policy(risked: pd.DataFrame, limit: float, p_star: float) -> pd.DataFrame:
    rows = []
    for case, g in risked.groupby("case", sort=True):
        g = g.sort_values("run").reset_index(drop=True)
        safe = true_safe_runs(g, limit)
        if safe is None or safe == 0:
            continue
        flag = np.where(g["p_next"].values >= p_star)[0]
        done = int(flag[0]) + 1 if len(flag) else len(g)  # runs performed
        failure, life = _evaluate(done, safe)
        rows.append({"case": case, "safe_runs": safe, "runs_done": done, "failure": failure, "life_used": life})
    return pd.DataFrame(rows)


def fixed_policy(risked: pd.DataFrame, limit: float, K: int) -> pd.DataFrame:
    rows = []
    for case, g in risked.groupby("case", sort=True):
        safe = true_safe_runs(g, limit)
        if safe is None or safe == 0:
            continue
        done = min(K, len(g))
        failure, life = _evaluate(done, safe)
        rows.append({"case": case, "safe_runs": safe, "runs_done": done, "failure": failure, "life_used": life})
    return pd.DataFrame(rows)


def fixed_K_loco(risked: pd.DataFrame, limit: float) -> pd.DataFrame:
    """Fair fixed rule: for each held-out tool, K is chosen from the *other* tools
    as the largest K that caused no failure there (i.e. the shortest safe life)."""
    safes = {c: true_safe_runs(g, limit) for c, g in risked.groupby("case")}
    safes = {c: s for c, s in safes.items() if s}
    rows = []
    for case, safe in safes.items():
        K = min(s for c, s in safes.items() if c != case)
        failure, life = _evaluate(K, safe)
        rows.append({"case": case, "safe_runs": safe, "runs_done": K, "failure": failure, "life_used": life})
    return pd.DataFrame(rows)


def summarise(policy: pd.DataFrame) -> dict:
    if policy.empty:
        return {"tools": 0, "failures": 0, "failure_rate": np.nan, "mean_life_used": np.nan}
    ok = policy[~policy["failure"]]
    return {"tools": len(policy), "failures": int(policy["failure"].sum()),
            "failure_rate": float(policy["failure"].mean()),
            "mean_life_used": float(policy["life_used"].mean()),
            "mean_life_used_no_failure": float(ok["life_used"].mean()) if len(ok) else np.nan}


def expected_cost(policy: pd.DataFrame, cost_failure: float, cost_per_wasted_run: float) -> float:
    """Illustrative cost per tool: scrap cost if it fails + value of unused safe runs."""
    if policy.empty:
        return np.nan
    wasted = (policy["safe_runs"] - policy["runs_done"]).clip(lower=0)
    return float((policy["failure"] * cost_failure + wasted * cost_per_wasted_run).mean())
