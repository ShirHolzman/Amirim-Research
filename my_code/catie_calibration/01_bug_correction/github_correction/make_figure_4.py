"""Regenerate Fig. 4 (model comparison) with the corrected CATIE.

Panel a is E[p], panel b is E[log p], pooled over all 12 schedules
(3,332 participants x 100 trials = 333,200 trials), exactly as in the paper.

Model 1 (CATIE) is recomputed from scratch with both corrections applied.
Models 6, 7, 9 and 10 are unchanged and are taken from the published values;
no per-trial data is available for them, so they carry no error bars.

The CATIE bar is computed from
    ../figures/bug_comparison_per_trial.csv.gz
produced by ../bug_benchmark.py. Column `pc_fix` is P(choice made) under the
corrected model.

Run from the repository root:
    python my_code/catie_calibration/01_bug_correction/github_correction/make_figure_4.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 150, "font.size": 12, "font.family": "DejaVu Sans"})

PER_TRIAL = "my_code/catie_calibration/01_bug_correction/figures/bug_comparison_per_trial.csv.gz"
OUT = "my_code/catie_calibration/01_bug_correction/github_correction/figure_4_corrected.png"

# Published values for the unchanged models (paper, Fig. 4 / Tables S1-S2).
published = {
    6: dict(Ep=0.588, Elogp=-0.588),
    7: dict(Ep=0.592, Elogp=-0.676),
    9: dict(Ep=0.618, Elogp=-0.778),
    10: dict(Ep=0.603, Elogp=-0.569),
}
# Colours sampled from the published figure so the unchanged bars look identical.
COLORS = {1: "#EAB246", 6: "#0669B1", 7: "#9D75A6", 9: "#3483C2", 10: "#7A4D8B"}

df = pd.read_csv(PER_TRIAL)
n = len(df)
assert n == 333_200, n

p = df["pc_fix"].to_numpy()
logp = np.log(p)
catie = dict(
    Ep=p.mean(),
    Ep_sem=p.std(ddof=1) / np.sqrt(n),
    Elogp=logp.mean(),
    Elogp_sem=logp.std(ddof=1) / np.sqrt(n),
)
assert abs(catie["Ep"] - 0.6295) < 1e-3, catie["Ep"]
assert abs(catie["Elogp"] - (-0.6654)) < 1e-3, catie["Elogp"]

order = [1, 6, 7, 9, 10]
labels = ["1\n(CATIE)", "6", "7", "9", "10"]

Ep = [catie["Ep"]] + [published[m]["Ep"] for m in order[1:]]
Ep_err = [catie["Ep_sem"]] + [0.0] * 4
Elogp = [catie["Elogp"]] + [published[m]["Elogp"] for m in order[1:]]
Elogp_err = [catie["Elogp_sem"]] + [0.0] * 4
colors = [COLORS[m] for m in order]
x = np.arange(len(order))

fig, (axa, axb) = plt.subplots(1, 2, figsize=(10, 4.4))

for ax, vals, errs, ylim, ylab, panel in (
    (axa, Ep, Ep_err, (0.5, 0.65), "E[$p$]", "a"),
    (axb, Elogp, Elogp_err, (-0.85, 0.0), "E[log($p$)]", "b"),
):
    for xi, v, e, c in zip(x, vals, errs, colors):
        ax.bar(xi, v, width=0.7, color=c, edgecolor="black", linewidth=0.6,
               yerr=e if e > 0 else None, capsize=3,
               error_kw=dict(elinewidth=1, ecolor="black"))
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylab)
    ax.set_xlabel("Model ID")
    ax.set_xticks(x, labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.13, 1.04, panel, transform=ax.transAxes, fontsize=20, fontweight="bold")

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", OUT)
print("CATIE (corrected): E[p] = %.4f (SEM %.4f), E[log p] = %.4f (SEM %.4f)"
      % (catie["Ep"], catie["Ep_sem"], catie["Elogp"], catie["Elogp_sem"]))
