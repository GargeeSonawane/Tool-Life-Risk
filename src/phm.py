"""PHM Society 2010 milling data: loading, features, models and a calibrated band.

Data (per cutter c1, c4, c6):
    c_<id>_<cut>.csv   one file per cut, 7 columns, no header:
                       force X, Y, Z (N) · vibration X, Y, Z (g) · acoustic emission RMS (V)
    c<id>_wear.csv     flank wear of each of the 3 flutes after every cut, in 10^-3 mm

All cuts use the same settings (10,400 rpm, 1,555 mm/min feed, 0.125 mm radial and
0.2 mm axial depth), so cutting time alone cannot separate the tools: the sensors
have to carry the prediction. That is the point of this second study.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .models import _gbm

CHANNELS = ["Fx", "Fy", "Fz", "Vx", "Vy", "Vz", "AE"]
GROUPS = {"force": ["Fx", "Fy", "Fz"], "vibration": ["Vx", "Vy", "Vz"], "acoustic emission": ["AE"]}
CUTTERS = (1, 4, 6)
N_BANDS = 4
TRIM = 0.15
MAX_SAMPLES = 60_000  # take the middle of each cut; enough for stable statistics and fast to run
COVERAGE = 0.80
Z90 = 1.2816


# ---------- loading ----------

def _find(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.rglob(pattern) if p.is_file())


def find_cutter_files(root: str | Path, cutter: int) -> tuple[list[tuple[int, Path]], Path]:
    root = Path(root)
    cut_files = []
    for p in _find(root, f"c_{cutter}_*.csv"):
        m = re.search(rf"c_{cutter}_(\d+)\.csv$", p.name)
        if m:
            cut_files.append((int(m.group(1)), p))
    wear = _find(root, f"c{cutter}_wear.csv")
    if not cut_files:
        raise FileNotFoundError(f"No cut files like c_{cutter}_001.csv found under {root}")
    if not wear:
        raise FileNotFoundError(f"No wear file c{cutter}_wear.csv found under {root}")
    return sorted(cut_files), wear[0]


def load_wear(path: Path) -> pd.DataFrame:
    """Wear per cut = the worst (maximum) of the three flutes, converted to mm."""
    w = pd.read_csv(path)
    w.columns = [str(c).strip().lower() for c in w.columns]
    cut_col = next((c for c in w.columns if "cut" in c), w.columns[0])
    flutes = [c for c in w.columns if c != cut_col]
    out = pd.DataFrame({"cut": w[cut_col].astype(int),
                        "VB": w[flutes].max(axis=1).astype(float) / 1000.0})
    return out


def _read_signal(path: Path) -> np.ndarray:
    x = pd.read_csv(path, header=None, engine="c").to_numpy(dtype=float)
    if x.shape[1] < 7:
        raise ValueError(f"{path.name}: expected 7 columns, found {x.shape[1]}")
    x = x[:, :7]
    n = len(x)
    a, b = int(n * TRIM), int(n * (1 - TRIM))
    x = x[a:b] if b - a > 100 else x
    if len(x) > MAX_SAMPLES:  # keep the central part of the steady cut
        mid = len(x) // 2
        x = x[mid - MAX_SAMPLES // 2: mid + MAX_SAMPLES // 2]
    return x


def _channel_features(x: np.ndarray, name: str) -> dict[str, float]:
    xc = x - x.mean()
    rms_raw = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    f = {
        f"{name}_mean": float(x.mean()),
        f"{name}_rms": rms_raw,
        f"{name}_std": float(x.std()),
        f"{name}_peak": peak,
        f"{name}_p2p": float(np.ptp(x)),
        f"{name}_crest": peak / rms_raw if rms_raw > 0 else 0.0,
        f"{name}_kurtosis": float(kurtosis(x)),
        f"{name}_skew": float(skew(x)),
    }
    spec = np.abs(np.fft.rfft(xc)) ** 2
    total = spec.sum() or 1.0
    for i, band in enumerate(np.array_split(spec, N_BANDS), start=1):
        f[f"{name}_band{i}"] = float(band.sum() / total)
    return f


def build_features(root: str | Path, cache: str | Path | None = None) -> pd.DataFrame:
    """One row per cut for cutters c1, c4, c6. Cached to CSV because reading ~950 files is slow."""
    if cache and Path(cache).exists():
        print(f"    using cached features: {cache}")
        return pd.read_csv(cache)
    rows = []
    for c in CUTTERS:
        files, wear_path = find_cutter_files(root, c)
        wear = load_wear(wear_path).set_index("cut")["VB"]
        print(f"    cutter c{c}: {len(files)} cuts")
        for k, (cut, path) in enumerate(files, start=1):
            x = _read_signal(path)
            row = {"case": c, "run": cut}
            for j, ch in enumerate(CHANNELS):
                row.update(_channel_features(x[:, j], ch))
            row["VB"] = float(wear.get(cut, np.nan))
            rows.append(row)
            if k % 50 == 0:
                print(f"      {k}/{len(files)}")
    df = pd.DataFrame(rows).sort_values(["case", "run"]).reset_index(drop=True)
    df = df[df["VB"].notna()].reset_index(drop=True)
    df["VB_label"] = df["VB"]
    df["VB_measured"] = True
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    return df


def feature_columns(df: pd.DataFrame, drop_groups: list[str] | None = None) -> list[str]:
    drop = [ch for g in (drop_groups or []) for ch in GROUPS[g]]
    keep = [ch for ch in CHANNELS if ch not in drop]
    return [c for c in df.columns if c.split("_")[0] in keep and c not in ("VB_label", "VB_measured")]


# ---------- models ----------

class CutBaseline:
    """log(VB) = a + b*log(cut). Same settings for every cut, so this is the 'no sensors' baseline."""

    def fit(self, df, y):
        self.m = LinearRegression().fit(np.log(df[["run"]].values), np.log(np.clip(y, 1e-4, None)))
        return self

    def predict(self, df):
        return np.exp(self.m.predict(np.log(df[["run"]].values)))


def _quantiles(X_tr, y_tr, X_te):
    qs = np.column_stack([_gbm(a).fit(X_tr, y_tr).predict(X_te) for a in (0.1, 0.5, 0.9)])
    qs.sort(axis=1)
    return qs


def _calibration_scale(train: pd.DataFrame, feats: list[str]) -> float:
    """Split-conformal style widening, learned only from the training cutters.

    Inner leave-one-cutter-out on the training cutters: how much wider must the
    10-90% band be so that it really contains 80% of the held-out measurements?
    """
    ratios = []
    for c in sorted(train["case"].unique()):
        tr, te = train[train["case"] != c], train[train["case"] == c]
        if tr.empty:
            continue
        q = _quantiles(tr[feats].values, tr["VB_label"].values, te[feats].values)
        half = np.maximum((q[:, 2] - q[:, 0]) / 2, 1e-4)
        ratios.extend(np.abs(te["VB"].values - q[:, 1]) / half)
    if not ratios:
        return 1.0
    return float(max(1.0, np.quantile(ratios, COVERAGE)))


def loco_predict(df: pd.DataFrame, feats: list[str], calibrate: bool = True) -> pd.DataFrame:
    """Leave one cutter out: train on two cutters, test on the third."""
    out = []
    for c in sorted(df["case"].unique()):
        train, test = df[df["case"] != c], df[df["case"] == c]
        X_tr, y_tr, X_te = train[feats].values, train["VB_label"].values, test[feats].values
        pred = test[["case", "run", "VB", "VB_label", "VB_measured"]].copy()
        pred["pred_cut_baseline"] = CutBaseline().fit(train, y_tr).predict(test)
        pred["pred_ridge"] = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X_tr, y_tr).predict(X_te)
        rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=0, n_jobs=-1)
        pred["pred_forest"] = rf.fit(X_tr, y_tr).predict(X_te)
        q = _quantiles(X_tr, y_tr, X_te)
        pred["q50"] = q[:, 1]
        pred["pred_gbm_q"] = pred["q50"]
        # the risk layer treats the band as symmetric around the median, so 'raw' is that symmetric band
        half = (q[:, 2] - q[:, 0]) / 2
        pred["q10_raw"], pred["q90_raw"] = pred["q50"] - half, pred["q50"] + half
        scale = _calibration_scale(train, feats) if calibrate else 1.0  # only ever widens (conservative)
        pred["q10"] = pred["q50"] - half * scale
        pred["q90"] = pred["q50"] + half * scale
        pred["band_scale"] = scale
        out.append(pred)
    res = pd.concat(out).reset_index(drop=True)
    for col in [c for c in res.columns if c.startswith("pred_") or c in ("q10", "q50", "q90", "q10_raw", "q90_raw")]:
        res[col] = res[col].clip(lower=0)
    return res


def score(pred: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for c in [c for c in pred.columns if c.startswith("pred_")]:
        err = pred[c] - pred["VB"]
        rows.append({"model": c.replace("pred_", ""), "MAE_mm": err.abs().mean(), "RMSE_mm": np.sqrt((err ** 2).mean())})
    table = pd.DataFrame(rows).sort_values("MAE_mm").reset_index(drop=True)
    inside = lambda lo, hi: float(((pred["VB"] >= pred[lo]) & (pred["VB"] <= pred[hi])).mean())
    cover = {"raw": inside("q10_raw", "q90_raw"), "calibrated": inside("q10", "q90")}
    per_cutter = (pred.assign(err=(pred["pred_gbm_q"] - pred["VB"]).abs())
                  .groupby("case")["err"].mean().rename("MAE_mm_gbm").reset_index())
    return table, {"coverage": cover, "per_cutter": per_cutter}


# ---------- choosing the risk threshold fairly ----------

def choose_p_star(train: pd.DataFrame, feats: list[str], scale: float, limit: float,
                  grid: list[float], sigma_floor: float) -> float:
    """Pick the replacement threshold p* using ONLY the training cutters.

    Inner leave-one-cutter-out: train on one training cutter, predict the other,
    widen the band by the same scale used for the outer fold, run the policy at
    every p* in the grid, and keep the largest p* that caused no failure.
    If every p* fails, fall back to the most cautious one.
    """
    from .risk import add_risk, risk_policy

    inner = []
    for c in sorted(train["case"].unique()):
        tr, te = train[train["case"] != c], train[train["case"] == c]
        if tr.empty:
            continue
        q = _quantiles(tr[feats].values, tr["VB_label"].values, te[feats].values)
        half = (q[:, 2] - q[:, 0]) / 2
        p = te[["case", "run", "VB", "VB_label", "VB_measured"]].copy()
        p["q50"] = np.clip(q[:, 1], 0, None)
        p["q10"] = np.clip(p["q50"] - half * scale, 0, None)
        p["q90"] = p["q50"] + half * scale
        inner.append(p)
    if not inner:
        return min(grid)
    risked = add_risk(pd.concat(inner), limit, sigma_floor=sigma_floor)
    safe = [p for p in sorted(grid) if not risk_policy(risked, limit, p)["failure"].any()]
    return max(safe) if safe else min(grid)


def nested_risk_policy(df: pd.DataFrame, feats: list[str], risked: pd.DataFrame, limit: float,
                       grid: list[float], sigma_floor: float) -> pd.DataFrame:
    """Risk policy where each held-out cutter gets a p* chosen from the other cutters only."""
    from .risk import risk_policy

    rows = []
    for c in sorted(risked["case"].unique()):
        train = df[df["case"] != c]
        scale = float(risked.loc[risked["case"] == c, "band_scale"].iloc[0])
        p_star = choose_p_star(train, feats, scale, limit, grid, sigma_floor)
        pol = risk_policy(risked[risked["case"] == c], limit, p_star)
        if not pol.empty:
            pol = pol.assign(p_star_chosen=p_star)
            rows.append(pol)
    return pd.concat(rows).reset_index(drop=True) if rows else pd.DataFrame()
