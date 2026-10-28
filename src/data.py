"""Load the NASA Ames milling dataset (mill.mat) into Python.

The file holds a 1x167 MATLAB struct array called ``mill``. Each element is one
experimental run and has these fields:

    case        experiment case number (1-16)
    run         run counter within the case
    VB          flank wear in mm, measured after some runs only (NaN otherwise)
    time        cumulative cutting time in the case
    DOC         depth of cut (mm)
    feed        feed (mm/rev)
    material    1 = cast iron, 2 = steel
    smcAC       spindle motor current, AC
    smcDC       spindle motor current, DC
    vib_table   vibration at the table
    vib_spindle vibration at the spindle
    AE_table    acoustic emission at the table
    AE_spindle  acoustic emission at the spindle
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

SIGNALS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
META = ["case", "run", "VB", "time", "DOC", "feed", "material"]


def _scalar(x) -> float:
    arr = np.asarray(x, dtype=float).ravel()
    return float(arr[0]) if arr.size else float("nan")


def load_mill(path: str | Path) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    """Return (meta, signals).

    meta    : one row per run, columns = META plus ``idx`` (row index into signals)
    signals : idx -> {signal name -> 1D numpy array}
    """
    raw = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "mill" not in raw:
        raise KeyError(f"'mill' struct not found in {path}; keys are {list(raw)}")
    mill = np.atleast_1d(raw["mill"])

    rows, signals = [], {}
    for idx, rec in enumerate(mill):
        row = {"idx": idx}
        for name in META:
            row[name] = _scalar(getattr(rec, name))
        rows.append(row)
        signals[idx] = {s: np.asarray(getattr(rec, s), dtype=float).ravel() for s in SIGNALS}

    meta = pd.DataFrame(rows)
    meta["case"] = meta["case"].astype(int)
    meta["run"] = meta["run"].astype(int)
    meta = meta.sort_values(["case", "run"]).reset_index(drop=True)
    return meta, signals


def add_wear_labels(meta: pd.DataFrame) -> pd.DataFrame:
    """Add training labels for wear.

    VB is only measured after some runs. For *training* we linearly interpolate
    VB between measured runs inside the same case (never across cases and never
    beyond the first/last measurement). For *testing* we only score runs where
    VB was actually measured, so the reported error is never helped by our own
    interpolation.
    """
    out = []
    for _, g in meta.groupby("case", sort=True):
        g = g.sort_values("run").copy()
        g["VB_measured"] = g["VB"].notna()
        # interpolate against the run number, only between two real measurements
        g["VB_label"] = g.set_index("run")["VB"].interpolate(method="index", limit_area="inside").values
        out.append(g)
    return pd.concat(out).reset_index(drop=True)


def drop_bad_runs(meta: pd.DataFrame, signals: dict, min_len: int = 1000) -> pd.DataFrame:
    """Drop runs whose signals are missing, too short, or flat (sensor fault)."""
    keep = []
    for _, r in meta.iterrows():
        sig = signals[int(r["idx"])]
        ok = all(len(sig[s]) >= min_len and np.isfinite(sig[s]).all() and np.std(sig[s]) > 0 for s in SIGNALS)
        keep.append(ok)
    return meta[np.array(keep)].reset_index(drop=True)
