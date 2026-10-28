"""Make a FAKE mill.mat with the same structure as the NASA file.

For testing the code only. Its numbers mean nothing. Never report results
from this file; use the real NASA mill.mat.
"""
import sys
import numpy as np
from scipy.io import savemat

FIELDS = ["case", "run", "VB", "time", "DOC", "feed", "material",
          "smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]


def main(path="data/synthetic_mill.mat", seed=1):
    rng = np.random.default_rng(seed)
    combos = [(d, f, m) for d in (1.5, 0.75) for f in (0.5, 0.25) for m in (1, 2)]
    recs = []
    for case in range(1, 17):
        doc, feed, mat = combos[(case - 1) % 8]
        k = 0.02 * (doc / 0.75) ** 0.6 * (feed / 0.25) ** 0.5 * (1.6 if mat == 2 else 1.0) * rng.uniform(0.7, 1.3)
        n_runs = rng.integers(7, 22)
        t = 0.0
        for run in range(1, n_runs + 1):
            t += rng.uniform(2, 9)
            vb = k * t ** 0.75 + rng.normal(0, 0.01)
            measured = run == 1 or rng.random() < 0.7
            n = 9000
            x = np.linspace(0, 60 * np.pi, n)
            sev = 1 + 3 * vb
            rec = {"case": case, "run": run, "VB": max(vb, 0) if measured else np.nan, "time": t,
                   "DOC": doc, "feed": feed, "material": mat,
                   "smcAC": sev * np.sin(x) + rng.normal(0, 0.2, n),
                   "smcDC": 2 + 0.8 * sev + rng.normal(0, 0.1, n),
                   "vib_table": rng.normal(0, 0.3 * sev, n),
                   "vib_spindle": rng.normal(0, 0.2 * sev, n) + 0.1 * sev * np.sin(5 * x),
                   "AE_table": np.abs(rng.normal(0, 0.1 * sev, n)),
                   "AE_spindle": np.abs(rng.normal(0, 0.15 * sev, n))}
            recs.append(rec)
    arr = np.zeros((1, len(recs)), dtype=[(f, "O") for f in FIELDS])
    for i, r in enumerate(recs):
        for f in FIELDS:
            arr[0, i][f] = np.atleast_2d(np.asarray(r[f], dtype=float)).T if np.ndim(r[f]) else np.array([[r[f]]])
    savemat(path, {"mill": arr})
    print(f"wrote {path} with {len(recs)} FAKE runs (testing only)")


if __name__ == "__main__":
    main(*sys.argv[1:2])
