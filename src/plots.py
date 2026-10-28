"""Figures for the README. Two colours only: blue = model, orange = fixed rule."""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK,
    "lines.linewidth": 2, "axes.axisbelow": True,
})


def wear_by_case(risked: pd.DataFrame, limit: float, path):
    cases = sorted(risked["case"].unique())
    cols = 4
    rows = math.ceil(len(cases) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows), sharey=True)
    for ax, case in zip(np.ravel(axes), cases):
        g = risked[risked["case"] == case].sort_values("run")
        ax.fill_between(g["run"], g["q10"], g["q90"], color=BLUE, alpha=0.18, linewidth=0)
        ax.plot(g["run"], g["q50"], color=BLUE)
        m = g[g["VB_measured"]]
        ax.scatter(m["run"], m["VB"], s=22, color=INK, zorder=3)
        ax.axhline(limit, color=MUTED, linestyle="--", linewidth=1)
        mat = "cast iron" if g["material"].iloc[0] == 1 else "steel"
        ax.set_title(f"Case {case} · {mat}")
    for ax in np.ravel(axes)[len(cases):]:
        ax.axis("off")
    fig.supxlabel("Run")
    fig.supylabel("Flank wear VB (mm)")
    fig.suptitle("Predicted wear on unseen tools (line = median, band = 10–90%, dots = measured, dashed = limit)",
                 color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tradeoff(model_curve: pd.DataFrame, fixed_curve: pd.DataFrame, chosen: dict, fixed_loco: dict, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(model_curve["failure_rate"] * 100, model_curve["mean_life_used"] * 100,
            color=BLUE, marker="o", markersize=5, label="Risk-based policy (sweep of p*)")
    ax.plot(fixed_curve["failure_rate"] * 100, fixed_curve["mean_life_used"] * 100,
            color=ORANGE, marker="s", markersize=5, label="Fixed rule: replace after K runs")
    ax.scatter([chosen["failure_rate"] * 100], [chosen["mean_life_used"] * 100], s=110,
               facecolor=SURFACE, edgecolor=BLUE, linewidth=2, zorder=4)
    ax.annotate(f"chosen p* (life {chosen['mean_life_used']*100:.0f}%)",
                (chosen["failure_rate"] * 100, chosen["mean_life_used"] * 100),
                xytext=(10, 10), textcoords="offset points", color=INK)
    ax.scatter([fixed_loco["failure_rate"] * 100], [fixed_loco["mean_life_used"] * 100], s=110,
               facecolor=SURFACE, edgecolor=ORANGE, linewidth=2, zorder=4)
    ax.annotate(f"conservative fixed K (life {fixed_loco['mean_life_used']*100:.0f}%)",
                (fixed_loco["failure_rate"] * 100, fixed_loco["mean_life_used"] * 100),
                xytext=(8, 6), textcoords="offset points", color=INK)
    ax.set_xlabel("Tools that reached the wear limit before replacement (%)")
    ax.set_ylabel("Safe tool life used (%)")
    ax.set_title("More tool life at the same risk: up and to the left is better")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def missing_sensor(table: pd.DataFrame, path):
    t = table.sort_values("MAE_mm")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(t) + 1.2))
    ax.barh(t["features"], t["MAE_mm"], color=BLUE, height=0.6)
    for y, v in enumerate(t["MAE_mm"]):
        ax.text(v, y, f"  {v:.3f}", va="center", color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute error on unseen tools (mm) — lower is better")
    ax.set_title("What happens when a sensor is missing")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def importance(names, values, path, top=15):
    order = np.argsort(values)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7, 0.4 * top + 1.2))
    ax.barh([names[i] for i in order], [values[i] for i in order], color=BLUE, height=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (random forest)")
    ax.set_title(f"Top {top} features")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
