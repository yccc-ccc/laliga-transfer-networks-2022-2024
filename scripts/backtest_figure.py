# -*- coding: utf-8 -*-
"""Figure 5: hold-out backtest of the cross-window validation protocol.

Reads the CSV outputs of scripts/backtest_links.py and writes
results/figure5_backtest.png and .pdf.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")

fm = pd.read_csv(os.path.join(RES, "backtest_fold_metrics.csv"))
ac = pd.read_csv(os.path.join(RES, "backtest_allclub_scope.csv"))
rt = pd.read_csv(os.path.join(RES, "backtest_rules.csv"))
it = pd.read_csv(os.path.join(RES, "backtest_recurrence_by_intensity.csv"))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))

# A
ax = axes[0, 0]
order = ["Random", "Jaccard", "Stable (0/1)", "Repeated (>=2)", "Degree product",
         "Last window (0/1)", "Decayed count (1.0/0.5)"]
g = fm.groupby("model")["auc"].agg(["mean", "std"]).reindex(order)
yp = np.arange(len(g))[::-1]
ax.barh(yp, g["mean"], xerr=g["std"], height=0.6, color="#4C78A8", ecolor="#333333", capsize=2)
ax.axvline(0.5, color="#888888", lw=0.9, ls="--")
ax.set_yticks(yp)
ax.set_yticklabels(order, fontsize=8)
ax.set_xlim(0.35, 0.75)
ax.set_xlabel("Mean ROC-AUC (4 hold-out folds)")
ax.set_title("A. LaLiga-internal pairs", loc="left", fontsize=10, fontweight="bold")

# B
ax = axes[0, 1]
order_a = ["stable (0/1)", "repeated >=2", "last window (0/1)", "last-window count"]
lab_a = ["Stable (0/1)", "Repeated (>=2)", "Last window (0/1)", "Last-window count"]
s = ac.set_index("model").loc[order_a]
yp = np.arange(len(s))[::-1]
ax.barh(yp, s["auc_mean"], xerr=s["auc_sd"], height=0.55, color="#F58518",
        ecolor="#333333", capsize=2)
ax.axvline(0.5, color="#888888", lw=0.9, ls="--")
ax.set_yticks(yp)
ax.set_yticklabels(lab_a, fontsize=8)
ax.set_xlim(0.45, 0.80)
ax.set_xlabel("Mean ROC-AUC (3 hold-out folds)")
ax.set_title("B. LaLiga clubs vs all clubs", loc="left", fontsize=10, fontweight="bold")

# C
ax = axes[1, 0]
vals, ns = it["rate"].values, it["n_pairs"].values
ax.bar(np.arange(len(vals)), vals, width=0.55, color="#54A24B")
base_c = rt.base.mean()
ax.axhline(base_c, color="#D62728", lw=1.0, ls="--")
ax.text(0.02, 0.94, f"dashed line: base rate {base_c:.2f}", transform=ax.transAxes,
        color="#D62728", fontsize=7.5, va="top")
for i, (v, n) in enumerate(zip(vals, ns)):
    ax.text(i, v + 0.015, f"{v:.0%}\n(n={n})", ha="center", fontsize=7.5)
ax.set_xticks(np.arange(len(vals)))
ax.set_xticklabels(["1 tie", "2 ties", ">=3 ties"])
ax.set_ylim(0, 0.78)
ax.set_ylabel("Share recurring next window")
ax.set_title("C. Repetition in the most recent window", loc="left", fontsize=10, fontweight="bold")

# D
ax = axes[1, 1]
grp = rt.groupby("rule")[["share", "precision"]].mean()
keys = ["last window >=1", "last window >=2", "last window >=3", "stable (both windows)"]
labels = ["Last window\n>=1", "Last window\n>=2", "Last window\n>=3", "Stable in\nboth windows"]
prec = [grp.loc[k, "precision"] for k in keys]
share = [grp.loc[k, "share"] for k in keys]
ax.bar(np.arange(4), prec, width=0.55, color=["#4C78A8"] * 3 + ["#B0B0B0"])
ax.axhline(rt.base.mean(), color="#D62728", lw=1.0, ls="--")
ax.text(0.02, 0.94, f"dashed line: base rate {rt.base.mean():.2f}", transform=ax.transAxes,
        color="#D62728", fontsize=7.5, va="top")
for i, (p, sh) in enumerate(zip(prec, share)):
    ax.text(i, p + 0.015, f"{p:.0%}\n({sh:.0%} of pairs)", ha="center", fontsize=7.5)
ax.set_xticks(np.arange(4))
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylim(0, 0.72)
ax.set_ylabel("Share of flagged pairs that recur")
ax.set_title("D. Screening rules, LaLiga-internal", loc="left", fontsize=10, fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(RES, "figure5_backtest.png"), dpi=300)
fig.savefig(os.path.join(RES, "figure5_backtest.pdf"))
print("wrote results/figure5_backtest.png/.pdf")
