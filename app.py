"""Decision-support demo. Run after run_all.py:

    streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.plots import BLUE, INK, MUTED, ORANGE
from src.risk import KEEP_BELOW, add_risk, decide

RESULTS = Path("results/predictions.csv")

st.set_page_config(page_title="Tool-life risk demo", layout="wide")
st.title("Tool-life risk: keep, inspect or replace?")
st.caption("Predictions are for tools the model never saw in training (leave-one-case-out).")

if not RESULTS.exists():
    st.error("Run `python run_all.py --data data/mill.mat` first to create results/predictions.csv.")
    st.stop()

df = pd.read_csv(RESULTS)
with st.sidebar:
    case = st.selectbox("Tool (NASA case)", sorted(df["case"].unique()))
    limit = st.number_input("Wear limit VB (mm)", value=0.30, step=0.05, format="%.2f")
    p_star = st.slider("Replace when risk of passing the limit on the next run ≥", 0.05, 0.9, 0.20, 0.05)
    st.markdown(f"Keep if risk < {KEEP_BELOW:.0%}, inspect in between.")

g = df[df["case"] == case].sort_values("run").reset_index(drop=True)
# recompute risk for the chosen limit
g = add_risk(g.drop(columns=[c for c in ("sigma", "p_now", "p_next") if c in g]), limit)
run = st.slider("Step through the runs", int(g["run"].min()), int(g["run"].max()), int(g["run"].min()))
cur = g[g["run"] <= run]
now = cur.iloc[-1]
action = decide(now["p_next"], p_star)

c1, c2, c3 = st.columns(3)
c1.metric("Predicted wear now (mm)", f"{now['q50']:.3f}", help="median prediction")
c2.metric("Risk of passing limit on next run", f"{now['p_next']:.0%}")
c3.metric("Recommendation", action.upper())

fig, ax = plt.subplots(figsize=(9, 4))
ax.fill_between(cur["run"], cur["q10"], cur["q90"], color=BLUE, alpha=0.18, linewidth=0, label="10–90% band")
ax.plot(cur["run"], cur["q50"], color=BLUE, label="predicted wear (median)")
m = cur[cur["VB_measured"]]
ax.scatter(m["run"], m["VB"], color=INK, s=24, zorder=3, label="measured wear")
ax.axhline(limit, color=MUTED, linestyle="--", linewidth=1, label="wear limit")
first_replace = g[g["p_next"] >= p_star]
if not first_replace.empty and first_replace["run"].iloc[0] <= run:
    ax.axvline(first_replace["run"].iloc[0], color=ORANGE, linewidth=2, label="policy says replace")
ax.set_xlim(g["run"].min() - 0.5, g["run"].max() + 0.5)
ax.set_ylim(0, max(g["q90"].max(), g["VB"].max(skipna=True), limit) * 1.1)
ax.set_xlabel("Run")
ax.set_ylabel("Flank wear VB (mm)")
ax.legend(frameon=False, loc="upper left")
st.pyplot(fig)

table = cur[["run", "time", "VB", "q50", "p_next"]].copy()
table["decision"] = [decide(p, p_star) for p in table["p_next"]]
table = table.rename(columns={"VB": "measured VB", "q50": "predicted VB", "p_next": "risk next run"})
st.dataframe(table.style.format({"measured VB": "{:.3f}", "predicted VB": "{:.3f}", "risk next run": "{:.0%}",
                                 "time": "{:.1f}"}, na_rep="—"), width="stretch")
