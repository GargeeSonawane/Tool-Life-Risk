"""Turn each run's raw sensor signals into a row of numbers (features).

A worn tool cuts less cleanly: it needs more spindle power, vibrates more and
emits more acoustic energy. Simple statistics of each signal capture that.
For every signal we compute:

    rms       overall energy level
    std       spread around the mean
    peak      largest absolute value
    p2p       peak-to-peak range
    crest     peak / rms  (spikiness)
    kurtosis  how heavy the tails are (impacts, chipping)
    skew      asymmetry
    band_1..4 share of energy in four equal frequency bands (FFT)

The first and last part of each run (tool entering and leaving the part) is
trimmed so the features describe steady cutting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from .data import SIGNALS

N_BANDS = 4
TRIM = 0.15  # drop the first and last 15% of each signal


def _signal_features(x: np.ndarray, prefix: str) -> dict[str, float]:
    n = len(x)
    a, b = int(n * TRIM), int(n * (1 - TRIM))
    x = x[a:b] if b - a > 100 else x
    x = x - np.mean(x) if prefix.startswith("vib") or prefix.startswith("AE") else x

    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    feats = {
        f"{prefix}_rms": rms,
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_peak": peak,
        f"{prefix}_p2p": float(np.ptp(x)),
        f"{prefix}_crest": peak / rms if rms > 0 else 0.0,
        f"{prefix}_kurtosis": float(kurtosis(x)),
        f"{prefix}_skew": float(skew(x)),
    }
    spec = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    total = spec.sum() or 1.0
    for i, band in enumerate(np.array_split(spec, N_BANDS), start=1):
        feats[f"{prefix}_band{i}"] = float(band.sum() / total)
    return feats


def build_feature_table(meta: pd.DataFrame, signals: dict) -> pd.DataFrame:
    """One row per run: meta columns + all signal features."""
    rows = []
    for _, r in meta.iterrows():
        sig = signals[int(r["idx"])]
        feats = {}
        for s in SIGNALS:
            feats.update(_signal_features(sig[s], s))
        rows.append(feats)
    return pd.concat([meta.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def feature_columns(df: pd.DataFrame, drop_signals: list[str] | None = None) -> list[str]:
    """Process settings + signal features, optionally without some sensors."""
    drop_signals = drop_signals or []
    process = ["time", "DOC", "feed", "material"]
    sensor = [c for c in df.columns if any(c.startswith(s + "_") for s in SIGNALS)
              and not any(c.startswith(d + "_") for d in drop_signals)]
    return process + sensor
