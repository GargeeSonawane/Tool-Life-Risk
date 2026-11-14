"""Make FAKE PHM 2010-format files for testing the code only. Numbers mean nothing."""
import sys
from pathlib import Path

import numpy as np


def main(root="data/synthetic_phm", n_cuts=120, n_samples=20000, seed=3):
    rng = np.random.default_rng(seed)
    root = Path(root)
    for c, rate in [(1, 1.0), (4, 1.25), (6, 0.85)]:
        d = root / f"c{c}"
        d.mkdir(parents=True, exist_ok=True)
        cuts = np.arange(1, n_cuts + 1)
        base = 0.03 + 0.10 * rate * (cuts / n_cuts) ** 0.6 + 0.08 * rate * (cuts / n_cuts) ** 6
        flutes = np.column_stack([base * f + rng.normal(0, 0.002, n_cuts) for f in (1.0, 0.95, 0.9)])
        with open(root / f"c{c}_wear.csv", "w") as fh:
            fh.write("cut,flute_1,flute_2,flute_3\n")
            for i, row in zip(cuts, flutes * 1000):
                fh.write(f"{i},{row[0]:.2f},{row[1]:.2f},{row[2]:.2f}\n")
        t = np.linspace(0, 200 * np.pi, n_samples)
        for i in cuts:
            sev = 1 + 6 * base[i - 1]
            sig = np.column_stack([
                sev * 40 * np.sin(t) + rng.normal(0, 5, n_samples),
                sev * 30 * np.cos(t) + rng.normal(0, 5, n_samples),
                sev * 10 + rng.normal(0, 3, n_samples),
                rng.normal(0, 0.2 * sev, n_samples),
                rng.normal(0, 0.15 * sev, n_samples),
                rng.normal(0, 0.1 * sev, n_samples),
                0.05 * sev + np.abs(rng.normal(0, 0.01, n_samples)),
            ])
            np.savetxt(d / f"c_{c}_{i:03d}.csv", sig, delimiter=",", fmt="%.4f")
    print(f"wrote FAKE PHM data to {root} (testing only)")


if __name__ == "__main__":
    main(*sys.argv[1:2])
