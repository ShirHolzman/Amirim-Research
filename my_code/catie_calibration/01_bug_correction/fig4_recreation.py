"""
Recreation of the paper's Fig. 4 (model comparison, E[p] and E[log p]),
with an added column for the bug-fixed CATIE.

Published bar heights (models 1, 6, 7, 9, 10) are read from
Data_resources/extracted_data/Figure_4_data.md (OCR'd from the paper; no
per-model raw data available, so no error bars for 6/7/9/10). The published
and bug-fixed CATIE bars/error bars are computed directly from this repo's
own re-run of the corrected model on the full 333,200-trial pooled set
(bug_comparison_per_trial.csv.gz from ../01_bug_correction/bug_benchmark.py),
using the paper's per-trial k-mixture weighting (see ../matlab/README.md
section 4 and this phase's README "Resolved discrepancy" note).

Colors for models 1/6/7/9/10 are sampled directly from the published figure
crop (Data_resources/extracted_figures/fig4_model_comparison_Ep_Elogp_crop.png)
so the reproduction is visually identical for the original bars.
"""
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 150, "font.size": 12, "font.family": "DejaVu Sans"})

# --- published bars (Fig. 4a/4b, from the paper's table) ---
published = {
    1: dict(Ep=0.619, Elogp=-0.678),
    6: dict(Ep=0.588, Elogp=-0.588),
    7: dict(Ep=0.592, Elogp=-0.676),
    9: dict(Ep=0.618, Elogp=-0.778),
    10: dict(Ep=0.603, Elogp=-0.569),
}

# colors sampled from the published crop (RGB -> hex)
COLORS = {
    1: "#EAB246",
    6: "#0669B1",
    7: "#9D75A6",
    9: "#3483C2",
    10: "#7A4D8B",
}
FIXED_COLOR = "#2CA02C"  # not used in the original palette -- flags "not in the paper"

# --- our own re-run: published & bug-fixed CATIE, full pooled set (333,200 trials) ---
df = pd.read_csv(
    "my_code/catie_calibration/01_bug_correction/figures/bug_comparison_per_trial.csv.gz"
)
n = len(df)
assert n == 333_200, n


def summarize(col):
    p = df[col].to_numpy()
    logp = np.log(p)
    return dict(
        Ep=p.mean(),
        Ep_sem=p.std(ddof=1) / np.sqrt(n),
        Elogp=logp.mean(),
        Elogp_sem=logp.std(ddof=1) / np.sqrt(n),
    )


catie_pub = summarize("pc_pub")
catie_fix = summarize("pc_fix")

# sanity check against the values quoted in README.md / output.txt
assert abs(catie_pub["Ep"] - 0.6192) < 1e-3
assert abs(catie_fix["Ep"] - 0.6295) < 1e-3
assert abs(catie_pub["Elogp"] - (-0.6776)) < 1e-3
assert abs(catie_fix["Elogp"] - (-0.6654)) < 1e-3

# --- assemble bar order: 1, 1-fixed, 6, 7, 9, 10 ---
order = [1, "1fix", 6, 7, 9, 10]
labels = ["1\n(CATIE)", "1\n(fixed)", "6", "7", "9", "10"]
colors = [COLORS[1], FIXED_COLOR, COLORS[6], COLORS[7], COLORS[9], COLORS[10]]
hatches = [None, "//", None, None, None, None]

Ep_vals, Ep_err = [], []
Elogp_vals, Elogp_err = [], []
for m in order:
    if m == 1:
        Ep_vals.append(catie_pub["Ep"]); Ep_err.append(catie_pub["Ep_sem"])
        Elogp_vals.append(catie_pub["Elogp"]); Elogp_err.append(catie_pub["Elogp_sem"])
    elif m == "1fix":
        Ep_vals.append(catie_fix["Ep"]); Ep_err.append(catie_fix["Ep_sem"])
        Elogp_vals.append(catie_fix["Elogp"]); Elogp_err.append(catie_fix["Elogp_sem"])
    else:
        Ep_vals.append(published[m]["Ep"]); Ep_err.append(0.0)
        Elogp_vals.append(published[m]["Elogp"]); Elogp_err.append(0.0)

x = np.arange(len(order))

fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.6))

# panel a: E[p]
for xi, v, e, c, h in zip(x, Ep_vals, Ep_err, colors, hatches):
    axa.bar(xi, v, width=0.7, color=c, hatch=h, edgecolor="black", linewidth=0.6,
            yerr=e if e > 0 else None, capsize=3, error_kw=dict(elinewidth=1, ecolor="black"))
axa.set_ylim(0.5, 0.64)
axa.set_ylabel("E[$p$]")
axa.set_xlabel("Model ID")
axa.set_xticks(x, labels)
axa.text(-0.13, 1.05, "a", transform=axa.transAxes, fontsize=20, fontweight="bold")

# panel b: E[log p]
for xi, v, e, c, h in zip(x, Elogp_vals, Elogp_err, colors, hatches):
    axb.bar(xi, v, width=0.7, color=c, hatch=h, edgecolor="black", linewidth=0.6,
            yerr=e if e > 0 else None, capsize=3, error_kw=dict(elinewidth=1, ecolor="black"))
axb.set_ylim(-0.85, 0)
axb.set_ylabel("E[log($p$)]")
axb.set_xlabel("Model ID")
axb.set_xticks(x, labels)
axb.text(-0.13, 1.05, "b", transform=axb.transAxes, fontsize=20, fontweight="bold")

for ax in (axa, axb):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Model comparison (paper's Fig. 4) with the bug-fixed CATIE added\n"
    "green hatched bar = corrected heuristic-branch indexing, per-trial k-mixture, "
    "333,200 pooled trials; models 6/7/9/10 as published (no per-trial data available "
    "for error bars)",
    fontsize=9.5, y=1.04,
)

fig.tight_layout()
out = "my_code/catie_calibration/01_bug_correction/figures/fig4_model_comparison_with_fix.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
print("CATIE published:", catie_pub)
print("CATIE fixed:", catie_fix)
