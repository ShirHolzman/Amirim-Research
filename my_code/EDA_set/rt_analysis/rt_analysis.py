"""
rt_analysis.py
==============
RT vs. CATIE choice-probability analysis for the EDA SET.

Threshold parameter
-------------------
Set X (integer, 1–49) to control the category boundaries:
    Surprise  = catie_choice_probability < X/100
    Autopilot = catie_choice_probability > 1 - X/100
    Routine   = everything in between

Example: X=10 → Surprise p < 0.10, Autopilot p > 0.90
         X=5  → Surprise p < 0.05, Autopilot p > 0.95

Each X value writes its output to its own sub-folder:
    figures/X{X}/fig1_bar_chart.png
    figures/X{X}/fig2_kde_overlay.png
    figures/X{X}/fig3_scatter_trend.png
    figures/X{X}/fig4_schedule_bars.png
    figures/X{X}/fig5_trial_trajectory.png
    figures/X{X}/output.txt   ← full console output for this run

Hypothesis
----------
Trials on which CATIE is surprised (p < X/100) should show longer reaction
times because the participant paused to consciously reconsider their pattern.
Trials where CATIE is highly confident (p > 1 - X/100) should show shorter
reaction times because the participant is acting on autopilot.

The analysis uses RT_zscore (reaction time normalised within each participant)
so that individual baseline differences cancel out.

Run
---
    cd my_code/EDA_set/rt_analysis
    python rt_analysis.py
"""

# ── 0. Imports and configuration ─────────────────────────────────────────────

import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Threshold parameter ───────────────────────────────────────────────────────
# Change X to shift all category boundaries simultaneously.
# X=10 → Surprise p < 0.10, Autopilot p > 0.90
# X=5  → Surprise p < 0.05, Autopilot p > 0.95
X = 10

SURPRISE_MAX  = X / 100    
AUTOPILOT_MIN = 1 - X / 100

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = (
    pathlib.Path(__file__).parent.parent
    / "processing"
    / "eda_with_catie_probabilities.csv"
)
# Each X value gets its own sub-folder
OUT_DIR = pathlib.Path(__file__).parent / "figures" / f"X{X}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Tee stdout → console + output.txt ────────────────────────────────────────
class _Tee:
    """Write to multiple streams simultaneously, with safe encoding fallback."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, text):
        for s in self._streams:
            try:
                s.write(text)
            except UnicodeEncodeError:
                enc = getattr(s, "encoding", "utf-8") or "utf-8"
                s.write(text.encode(enc, errors="replace").decode(enc))
            s.flush()
    def flush(self):
        for s in self._streams:
            s.flush()

_log_file = open(OUT_DIR / "output.txt", "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_file)

# ── Ordered category labels (short form used in code; display form in plots) ─
CAT_SURPRISE  = "Surprise"
CAT_ROUTINE   = "Routine"
CAT_AUTOPILOT = "Autopilot"
CAT_ORDER = [CAT_SURPRISE, CAT_ROUTINE, CAT_AUTOPILOT]

CAT_LABELS = {
    CAT_SURPRISE:  f"Surprise\n(p < {SURPRISE_MAX:.2f})",
    CAT_ROUTINE:   f"Routine\n({SURPRISE_MAX:.2f} ≤ p ≤ {AUTOPILOT_MIN:.2f})",
    CAT_AUTOPILOT: f"Autopilot\n(p > {AUTOPILOT_MIN:.2f})",
}

PALETTE = {
    CAT_SURPRISE:  "#ef5f5f",   # red
    CAT_ROUTINE:   "#b2a9a9",   # grey
    CAT_AUTOPILOT: "#93c4e6",   # blue
}

# Display palette keyed on long labels (for seaborn hue)
DISPLAY_PALETTE = {CAT_LABELS[k]: v for k, v in PALETTE.items()}

RANDOM_SEED = 42

# Global matplotlib style
plt.rcParams.update({
    "font.family":   "sans-serif",
    "font.size":     11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi":    120,
})

# ── 1. Load and validate data ─────────────────────────────────────────────────

print("=" * 60)
print("RT vs. CATIE Choice Probability — EDA SET Analysis")
print("=" * 60)
print(f"\nLoading: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Hard assertions — any failure here means the input file is unexpected
assert len(df) == 49_200, (
    f"Expected 49 200 rows, got {len(df)}. "
    "Re-run the MATLAB CATIE wrapper to regenerate the CSV."
)
assert df["catie_choice_probability"].notna().all(), \
    "NaN values found in catie_choice_probability."
assert df["catie_choice_probability"].between(0, 1, inclusive="both").all(), \
    "catie_choice_probability values outside [0, 1] detected."
assert df["RT_zscore"].notna().all(), \
    "NaN values found in RT_zscore."

print(f"Loaded {len(df):,} rows | "
      f"{df['subject_file'].nunique()} subjects | "
      f"{df['trial_number'].nunique()} trials each")
print(f"Schedules: {sorted(df['schedule'].unique())}")

# ── 2. Categorise trials ──────────────────────────────────────────────────────

conditions = [
    df["catie_choice_probability"] < SURPRISE_MAX,
    df["catie_choice_probability"] > AUTOPILOT_MIN,
]
choices = [CAT_SURPRISE, CAT_AUTOPILOT]
df["category"] = np.select(conditions, choices, default=CAT_ROUTINE)
df["category"] = pd.Categorical(df["category"], categories=CAT_ORDER, ordered=True)

# Long label column (for seaborn hue)
df["cat_label"] = df["category"].map(CAT_LABELS)
df["cat_label"] = pd.Categorical(
    df["cat_label"],
    categories=[CAT_LABELS[c] for c in CAT_ORDER],
    ordered=True,
)

counts = df["category"].value_counts().reindex(CAT_ORDER)

# ── 3. Descriptive statistics ─────────────────────────────────────────────────

print("\n── Descriptive Statistics (RT_zscore) ─────────────────────────────")
print(f"{'Category':<15} {'n':>7} {'mean':>8} {'std':>8} {'SEM':>8} "
      f"{'95% CI lower':>13} {'95% CI upper':>13}")

desc = {}
for cat in CAT_ORDER:
    grp = df.loc[df["category"] == cat, "RT_zscore"]
    n = len(grp)
    mean, std = grp.mean(), grp.std(ddof=1)
    sem = std / np.sqrt(n)
    ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
    desc[cat] = {"n": n, "mean": mean, "std": std, "sem": sem,
                 "ci_low": ci_low, "ci_high": ci_high, "data": grp}
    print(f"{cat:<15} {n:>7,} {mean:>8.4f} {std:>8.4f} {sem:>8.4f} "
          f"{ci_low:>13.4f} {ci_high:>13.4f}")

# ── 4. Statistical tests ──────────────────────────────────────────────────────

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples (unequal variance form)."""
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)

COMPARISONS = [
    (CAT_SURPRISE, CAT_ROUTINE),
    (CAT_SURPRISE, CAT_AUTOPILOT),
    (CAT_ROUTINE,  CAT_AUTOPILOT),
]
N_COMPARISONS = len(COMPARISONS)   # Bonferroni denominator

print("\n── Welch's t-tests (Bonferroni corrected, k=3) ─────────────────────")
header = (f"{'Comparison':<30} {'t':>8} {'p (raw)':>12} "
          f"{'p (Bonf.)':>12} {'Cohen d':>9} {'Sig?':>5}")
print(header)
print("-" * len(header))

test_results = {}
# For each 2 different catagories combinations:
for cat_a, cat_b in COMPARISONS:
    # extract each catagory data
    a = desc[cat_a]["data"].values 
    b = desc[cat_b]["data"].values
    if len(a) < 2 or len(b) < 2: 
        #Edge case: for Welch's  T-test we need at least 2 items in each group
        warnings.warn(f"Skipping {cat_a} vs {cat_b}: insufficient data.")
        continue
    # Welch's T-test 
    t_stat, p_raw = stats.ttest_ind(a, b, equal_var=False)
    p_bonf = min(p_raw * N_COMPARISONS, 1.0)
    d = cohens_d(a, b)
    sig = "***" if p_bonf < 0.001 else ("**" if p_bonf < 0.01 else
          ("*"   if p_bonf < 0.05 else "n.s."))
    label = f"{cat_a} vs {cat_b}"
    test_results[label] = {"t": t_stat, "p_raw": p_raw,
                           "p_bonf": p_bonf, "d": d, "sig": sig}
    print(f"{label:<30} {t_stat:>8.3f} {p_raw:>12.2e} "
          f"{p_bonf:>12.2e} {d:>9.4f} {sig:>5}")

# ── 5. Figure 1 — Bar chart with SEM ─────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(7, 5))

x_pos = np.arange(len(CAT_ORDER))
bar_colors = [PALETTE[c] for c in CAT_ORDER]
means = [desc[c]["mean"] for c in CAT_ORDER]
sems  = [desc[c]["sem"]  for c in CAT_ORDER]
ns    = [desc[c]["n"]    for c in CAT_ORDER]

bars = ax1.bar(x_pos, means, yerr=sems, capsize=5,
               color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.8,
               error_kw={"elinewidth": 1.5, "ecolor": "#333333"})

ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

for bar, n in zip(bars, ns):
    h = bar.get_height()
    if h >= 0:
        # Positive bar: label just inside the base, near the 0 line
        y, va = 0.003, "bottom"
    else:
        # Negative bar: label just inside the base, near the 0 line
        y, va = -0.003, "top"
    ax1.text(bar.get_x() + bar.get_width() / 2, y,
             f"n={n:,}", ha="center", va=va, fontsize=9, color="#333333")

ax1.set_xticks(x_pos)
ax1.set_xticklabels([CAT_LABELS[c] for c in CAT_ORDER], fontsize=10)
ax1.set_ylabel("Mean RT z-score")
ax1.set_title("Mean Reaction Time by CATIE Prediction Confidence\n"
              "(error bars = ±1 SEM)")
ax1.spines[["top", "right"]].set_visible(False)
fig1.tight_layout()

# ── 6. Figure 2 — KDE overlay ────────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(8, 5))

for cat in CAT_ORDER:
    grp_data = desc[cat]["data"]
    sns.kdeplot(grp_data, ax=ax2, color=PALETTE[cat], linewidth=2,
                label=f"{CAT_LABELS[cat]}  (n={desc[cat]['n']:,})")
    ax2.axvline(desc[cat]["mean"], color=PALETTE[cat],
                linewidth=1.2, linestyle="--", alpha=0.7)

ax2.set_xlabel("RT z-score")
ax2.set_ylabel("Density")
ax2.set_title("RT z-score Distributions by CATIE Prediction Confidence\n"
              "(dashed lines = group means)")
ax2.legend(loc="upper right", framealpha=0.9)
ax2.spines[["top", "right"]].set_visible(False)
fig2.tight_layout()

# ── 7. Figure 3 — Scatter + trendlines ───────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(8, 5))

# Subsample for dot plot only (avoids over-plotting 49k points)
rng = np.random.default_rng(RANDOM_SEED)
sample_idx = rng.choice(len(df), size=min(5_000, len(df)), replace=False)
sample = df.iloc[sample_idx]

ax3.scatter(
    sample["catie_choice_probability"], sample["RT_zscore"],
    alpha=0.18, s=7, color="#555555", linewidths=0, rasterized=True,
    label="_nolegend_",
)

# Linear regression on the full dataset
x_full = df["catie_choice_probability"].values
y_full = df["RT_zscore"].values
slope, intercept, r_val, p_lin, _ = stats.linregress(x_full, y_full)
x_line = np.array([x_full.min(), x_full.max()])
ax3.plot(x_line, slope * x_line + intercept, color="#ef5f5f", linewidth=2,
         label=f"Linear fit  r = {r_val:.3f},  p = {p_lin:.2e}")

# LOWESS smoothed curve — use the same 5 000-point sample to avoid O(n²) cost
# on 49 200 rows; the resulting curve is visually indistinguishable.
sns.regplot(
    x="catie_choice_probability", y="RT_zscore", data=sample,
    scatter=False, lowess=True, ax=ax3,
    line_kws={"color": "#93c4e6", "linewidth": 2, "linestyle": "--"},
)

# Manually add LOWESS to legend (sns.regplot doesn't label it)
lowess_line = mpatches.Patch(color="#93c4e6",
                              label="LOWESS smooth")
linear_line = mpatches.Patch(color="#ef5f5f",
                              label=f"Linear fit  r = {r_val:.3f},  p = {p_lin:.2e}")
ax3.legend(handles=[linear_line, lowess_line], framealpha=0.9)

ax3.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
ax3.set_xlabel("CATIE choice probability")
ax3.set_ylabel("RT z-score")
ax3.set_title("Reaction Time vs. CATIE Prediction Confidence\n"
              f"(dots = random 5 000-trial sample; regression on all {len(df):,} trials)")
ax3.spines[["top", "right"]].set_visible(False)
fig3.tight_layout()

# ── 8. Figure 4 (Extra 1) — Schedule × category grouped bars ─────────────────
#
# Rationale: if the RT staircase is driven by the CATIE probability and not by
# schedule-specific reward structure, the pattern should replicate in all three
# schedules independently.

fig4, ax4 = plt.subplots(figsize=(9, 5))

schedules = sorted(df["schedule"].unique())
n_sched = len(schedules)
n_cat   = len(CAT_ORDER)
bar_width = 0.22
x_sched = np.arange(n_sched)

for ci, cat in enumerate(CAT_ORDER):
    means_sched, sems_sched = [], []
    for sched in schedules:
        grp = df.loc[(df["category"] == cat) & (df["schedule"] == sched), "RT_zscore"]
        if len(grp) < 2:
            means_sched.append(np.nan)
            sems_sched.append(np.nan)
        else:
            means_sched.append(grp.mean())
            sems_sched.append(grp.std(ddof=1) / np.sqrt(len(grp)))
    offset = (ci - (n_cat - 1) / 2) * bar_width
    ax4.bar(x_sched + offset, means_sched, width=bar_width,
            color=PALETTE[cat], alpha=0.85, label=CAT_LABELS[cat],
            edgecolor="white", linewidth=0.7,
            yerr=sems_sched, capsize=4,
            error_kw={"elinewidth": 1.3, "ecolor": "#333333"})

ax4.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax4.set_xticks(x_sched)
ax4.set_xticklabels([s.replace("_", " ").title() for s in schedules])
ax4.set_ylabel("Mean RT z-score")
ax4.set_title("RT Staircase Replicated Across Reward Schedules\n"
              "(error bars = ±1 SEM)")
ax4.legend(title="Category", framealpha=0.9)
ax4.spines[["top", "right"]].set_visible(False)
fig4.tight_layout()

# ── 9. Figure 5 (Extra 2) — Trial position × category trajectory ─────────────
#
# Rationale: if autopilot behaviour builds up as participants learn the reward
# schedule, the gap between Autopilot and Surprise RT_zscore should widen in
# later quintiles of the session.

N_QUINTILES = 5
bins = pd.cut(df["trial_number"], bins=N_QUINTILES,
              labels=[f"T{i*20}–{(i+1)*20-1}" for i in range(N_QUINTILES)])
df["trial_quintile"] = bins
quintile_labels = df["trial_quintile"].cat.categories.tolist()

fig5, ax5 = plt.subplots(figsize=(9, 5))

for cat in CAT_ORDER:
    q_means, q_sems = [], []
    for ql in quintile_labels:
        grp = df.loc[(df["category"] == cat) & (df["trial_quintile"] == ql),
                     "RT_zscore"]
        if len(grp) < 2:
            q_means.append(np.nan)
            q_sems.append(np.nan)
        else:
            q_means.append(grp.mean())
            q_sems.append(grp.std(ddof=1) / np.sqrt(len(grp)))
    q_means = np.array(q_means, dtype=float)
    q_sems  = np.array(q_sems,  dtype=float)
    x_q = np.arange(len(quintile_labels))
    ax5.plot(x_q, q_means, color=PALETTE[cat], linewidth=2,
             marker="o", markersize=6, label=CAT_LABELS[cat])
    ax5.fill_between(x_q, q_means - q_sems, q_means + q_sems,
                     color=PALETTE[cat], alpha=0.15)

ax5.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax5.set_xticks(np.arange(len(quintile_labels)))
ax5.set_xticklabels(quintile_labels, fontsize=10)
ax5.set_xlabel("Trial quintile (session position)")
ax5.set_ylabel("Mean RT z-score")
ax5.set_title("RT z-score Trajectory Across Session by Category\n"
              "(shaded band = ±1 SEM; shows whether autopilot effect builds over time)")
ax5.legend(title="Category", framealpha=0.9)
ax5.spines[["top", "right"]].set_visible(False)
fig5.tight_layout()

# ── 10. Save all figures ──────────────────────────────────────────────────────

figures = {
    "fig1_bar_chart":       fig1,
    "fig2_kde_overlay":     fig2,
    "fig3_scatter_trend":   fig3,
    "fig4_schedule_bars":   fig4,
    "fig5_trial_trajectory": fig5,
}

print("\n── Saving figures ──────────────────────────────────────────────────")
for name, fig in figures.items():
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  {name}.png")

print(f"\nAll figures saved to: {OUT_DIR}")
print(f"Console output saved to: {OUT_DIR / 'output.txt'}")
print("\nDone.")

# Restore stdout and close the log file before showing plots
sys.stdout = sys.__stdout__
_log_file.close()

plt.show()
