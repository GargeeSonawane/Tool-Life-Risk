# Risk-based tool-life prediction for CNC milling

**From sensor signals to keep / inspect / replace decisions.**

Cutting tools are usually replaced after a fixed number of cuts, chosen
conservatively so that no tool ever wears out mid-cut. That wastes a lot of
usable tool life. This project predicts tool wear from machine sensor signals,
estimates the *risk* that the tool will pass its wear limit on the next run,
and turns that risk into a recommendation. It then tests, on tools the model
has never seen, whether this uses more tool life than a fixed rule without
more failures.

Author: Gargee Sonawane · Data: NASA Ames milling data set (Agogino & Goebel, 2007) and PHM Society 2010 Data Challenge

Code written with help from an AI assistant (Claude); experiments run, results checked and interpreted by the author.

---

## Results (NASA milling data, 16 tools, each tested as an unseen tool)

**Headline:** the risk-based policy used **65%** of each tool's safe life, against **45%** for a
conservative fixed replacement rule, but let **2 of 15** tools pass the wear limit against **1 of 15**.
It keeps tools cutting longer, but is not yet safe enough for aerospace use.

### 1. Wear prediction (mean absolute error, lower is better)

| Model | MAE (mm) | RMSE (mm) |
|---|---|---|
| Time-and-settings baseline (wear-growth law, no sensors) | **0.110** | 0.178 |
| Quantile gradient boosting (all sensors) | 0.115 | 0.185 |
| Random forest (all sensors) | 0.133 | 0.204 |
| Ridge regression (all sensors) | 0.139 | 0.204 |

Wear limit: 0.3 mm. Errors are scored only on runs where wear was actually measured (146 points).

### 2. Uncertainty

The 10–90% prediction band contained the measured wear **55%** of the time (target 80%).
The model is overconfident on new tools, mainly because it misses the rapid wear at the end of a tool's life.

### 3. Replacement policy

| Policy | Tools | Failures | Safe life used |
|---|---|---|---|
| Risk-based (replace when risk ≥ 20%) | 15 | 2 | 65% |
| Fixed rule (shortest safe life of the other tools) | 15 | 1 | 45% |

One case never reached the limit and is left out of this test.

### 4. Missing sensors

Removing any single sensor changed the error very little (0.111–0.119 mm vs 0.113 mm with all sensors). This test uses a separately trained median model, so its all-sensor error differs slightly from the 0.115 mm in the first table.
Gradient boosting with **no sensors** (time and settings only) gave the lowest error, **0.092 mm**.

### What this means

On a small lab dataset, sensor features did not beat a time-and-settings baseline, and the
uncertainty was too narrow for safety-critical decisions. The hard problems are catching late-life
rapid wear and calibrating uncertainty, which need richer data and better uncertainty methods.

![Wear predictions](results/figures/wear_by_case.png)
![Policy trade-off](results/figures/tradeoff.png)
![Missing sensors](results/figures/missing_sensor.png)

---

## Second study: PHM 2010 milling data (force, vibration, acoustic emission)

Three tools (c1, c4, c6) cut stainless steel (HRC52) under identical settings, with cutting
forces, vibration and acoustic emission recorded on every cut and flank wear measured after
every cut. Because the settings never change, cutting time alone cannot tell the tools apart:
the sensors have to do the work. Each tool is tested as an unseen tool (train on two, test on
the third). Wear limit: 0.15 mm on the worst flute.

**Headline:** on this data the sensors clearly help, and a simple calibration step fixed most
of the uncertainty problem from the NASA study. Choosing a safe replacement threshold for a new
tool is still the hard part.

| Result | Value |
|---|---|
| Wear error, quantile gradient boosting (sensors) | **0.0156 mm** |
| Wear error, cut-number baseline (no sensors) | 0.0234 mm (sensors about 33% lower) |
| 10–90% band coverage before / after calibration | 43% → **77%** (target 80%) |
| Risk policy, threshold set in advance (p* = 0.2) | **0 of 3** tools failed, 34% of safe life used |
| Risk policy, threshold chosen from the other two tools | 1 of 3 failed, 67% of safe life used |
| Fixed rule (shortest safe life of the other tools) | 1 of 3 failed, 88% of safe life used |

**Calibration.** The band is widened by a factor learned only from the training tools (an
inner leave-one-tool-out check of how often the band missed). The factor ranged from ×2.4 to
×5.8; tool c6 needed by far the widest band, because it behaves differently from c1 and c4.

**What this means.** With honest (wide) uncertainty, the cautious policy was safe but wasted
tool life. A threshold learned from two tools did not transfer to the third. Tool-to-tool
variation, not the wear model itself, is the main obstacle, and it needs data from many more
tools to solve.

**Missing sensors.** Removing force (0.0229 mm) or vibration (0.0211 mm) made predictions
clearly worse; removing acoustic emission changed almost nothing (0.0161 mm vs 0.0160 mm).
Force alone was no better than no sensors. (This test uses a separately trained median model,
so its all-sensor error, 0.0160 mm, differs slightly from the 0.0156 mm above.)

Only three labelled tools: treat these numbers as indicative, not conclusive.

![PHM wear predictions](results_phm/figures/wear_by_cutter.png)
![PHM policy trade-off](results_phm/figures/tradeoff.png)
![PHM missing sensors](results_phm/figures/missing_sensor.png)

**Run it:** download cutters c1, c4 and c6 (for example the Kaggle copy
`rabahba/phm-data-challenge-2010`), put them in `data/phm2010/`, then
`python run_phm.py --data data/phm2010`.

---

## Data (NASA study)

16 milling experiments ("cases"), each one tool run until worn, at different
depths of cut, feeds and two materials (cast iron and steel). After each run
the machine logged six signals: spindle motor current (AC and DC), vibration
at the table and spindle, and acoustic emission at the table and spindle.
Flank wear (VB) was measured after some runs, not all.

**Get it:** download `mill.zip` from
<https://data.nasa.gov/dataset/milling-wear>, unzip it, and put `mill.mat` in
the `data/` folder.

## How to run (NASA study)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_all.py --data data/mill.mat      # about 3–5 minutes
streamlit run app.py                         # interactive demo
```

Options: `--limit 0.3` (wear limit in mm), `--p-star 0.2` (replacement risk threshold).

## Method

| Step | What it does | File |
|---|---|---|
| 1. Load and clean | Read `mill.mat`; drop runs with broken signals | `src/data.py` |
| 2. Labels | Wear is only measured on some runs; for training, interpolate *within* a tool between real measurements. Test scores use **only measured** runs | `src/data.py` |
| 3. Features | For each signal: RMS, std, peak, peak-to-peak, crest factor, kurtosis, skew, and energy in four frequency bands (FFT). Entry and exit of each cut trimmed | `src/features.py` |
| 4. Models | A physics-style baseline (wear as a power law of cutting time, scaled by settings), ridge regression, random forest, and gradient-boosted **quantile** models (10th, 50th, 90th percentile) | `src/models.py` |
| 5. Testing | **Leave-one-case-out**: train on 15 tools, test on the 16th, repeat for all. Splitting by run would leak information between neighbouring runs of the same tool | `src/models.py` |
| 6. Risk | Turn the 10–90% band into a probability that next-run wear passes the limit | `src/risk.py` |
| 7. Decision | keep if risk < 5%; inspect in between; replace if risk ≥ p* | `src/risk.py` |
| 8. Policy test | On each unseen tool, compare tool life used and failures against a fixed rule "replace after K runs", where K is the shortest safe life among the *other* tools (the conservative practice) | `src/risk.py` |
| 9. Missing sensors | Retrain without each sensor, and with spindle current only (a signal every CNC already has) | `run_all.py` |

## Limitations

- Small dataset: 16 tools, lab conditions, cast iron and steel, not titanium.
- Wear is not measured after every run; interpolated labels are used for training only.
- The 10–90% band is not perfectly calibrated with this little data (check the coverage line in `results/summary.md`).
- The decision threshold p* was set in advance, not tuned per tool; the trade-off curve shows all thresholds.
- Cost figures are illustrative assumptions.

## Next steps

- Learn the replacement threshold in a way that transfers between tools (more tools, or tool-specific adaptation).
- Test transfer between cutters and materials.
- Better-calibrated uncertainty: the simple calibration raised PHM coverage to 77%; test it on more tools and on NASA.
- Physics-informed model: use the Taylor-style baseline as a prior and let the data-driven model learn the correction.

## Testing the code without the real data

`python tools/make_synthetic_mill.py` writes a **fake** file in the same format
(`data/synthetic_mill.mat`) so you can check the code runs. Its numbers mean
nothing; never report results from it.

## References

A. Agogino and K. Goebel (2007). BEST Lab, UC Berkeley. "Milling Data Set",
NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

PHM Society 2010 Data Challenge, high-speed CNC milling cutter data (c1, c4, c6).
