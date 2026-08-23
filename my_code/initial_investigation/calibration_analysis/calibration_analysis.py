"""
calibration_analysis.py — CATIE Probability Calibration Analysis

The CSV stores catie_choice_probability = the probability CATIE assigned
to whichever option the participant actually chose. That quantity already
bakes in the outcome, so it cannot be used directly to test calibration.

Instead we recover CATIE's raw, pre-outcome forecast:
    catie_p_biased = P(participant chooses the biased option)
                    = catie_choice_probability        if is_biased_choice
                    = 1 - catie_choice_probability     otherwise

This is an exact algebraic inversion of the model's own stored output
(no leakage, no engineered feature) and lets us ask the standard
reliability-diagram question: "of all trials where CATIE said p% chance
of the biased choice, did the biased choice happen p% of the time?"
"""

import pathlib
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = (
    pathlib.Path(__file__).parent.parent
    / "processing"
    / "eda_with_catie_probabilities.csv"
)
OUT_DIR = pathlib.Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

N_BINS = 10
HIGH_CONF_THRESHOLD = 0.85

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

# ── Section 1 — Load and validate ─────────────────────────────────────────────

print("=" * 62)
print("CATIE Probability Calibration Analysis")
print("=" * 62)
print(f"\nLoading: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows")

assert len(df) == 49200, f"Expected 49200 rows, got {len(df)}"
assert df["catie_choice_probability"].between(0, 1, inclusive="both").all()

df["is_biased_choice"] = df["is_biased_choice"].astype(str).str.upper() == "TRUE"

# ── Section 2 — Recover the raw, pre-outcome forecast ────────────────────────

df["catie_p_biased"] = np.where(
    df["is_biased_choice"],
    df["catie_choice_probability"],
    1 - df["catie_choice_probability"],
)
df["confidence"] = np.maximum(df["catie_p_biased"], 1 - df["catie_p_biased"])

print("\n-- Recovered catie_p_biased = P(choose biased option), independent of outcome")
print(df["catie_p_biased"].describe().to_string())

# ── Section 3 — Figure 1: Reliability diagram ─────────────────────────────────

bins = np.linspace(0, 1, N_BINS + 1)
df["p_bin"] = pd.cut(df["catie_p_biased"], bins=bins, include_lowest=True)

rel = (
    df.groupby("p_bin", observed=True)
    .agg(predicted=("catie_p_biased", "mean"),
         empirical=("is_biased_choice", "mean"),
         n=("is_biased_choice", "size"))
    .reset_index()
)
rel = rel[rel["n"] >= 30].reset_index(drop=True)
rel["se"] = np.sqrt(rel["empirical"] * (1 - rel["empirical"]) / rel["n"])

print(f"\n-- Reliability Diagram Data ({N_BINS} bins, dropped bins with n<30)")
print(rel[["predicted", "empirical", "n"]].to_string(index=False))

fig1, ax1 = plt.subplots(figsize=(6.5, 6.5))
ax1.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2, label="Perfect calibration")
ax1.errorbar(rel["predicted"], rel["empirical"], yerr=1.96 * rel["se"],
             fmt="o-", color="#d62728", linewidth=2, markersize=7,
             capsize=3, label="CATIE (observed)")
for _, r in rel.iterrows():
    ax1.annotate(f"n={int(r['n']):,}", (r["predicted"], r["empirical"]),
                 textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel("CATIE predicted P(choose biased option)")
ax1.set_ylabel("Empirical P(chose biased option)")
ax1.set_title("Reliability Diagram: Is CATIE's Stated Confidence Warranted?")
ax1.legend(fontsize=9, loc="upper left")
fig1.tight_layout()

# ── Section 4 — Figure 2: Confidence vs trial number, by schedule ────────────

trial_conf = (
    df.groupby(["schedule", "trial_number"], observed=True)["confidence"]
    .mean()
    .reset_index()
)

fig2, ax2 = plt.subplots(figsize=(9, 4.5))
for sched, grp in trial_conf.groupby("schedule"):
    ax2.plot(grp["trial_number"], grp["confidence"], linewidth=1.5, alpha=0.85, label=sched)
ax2.set_xlabel("Trial number")
ax2.set_ylabel("Mean confidence  max(p, 1-p)")
ax2.set_title("CATIE's Confidence Across Trial Position, by Schedule")
ax2.legend(fontsize=7, ncol=2, loc="lower right")
fig2.tight_layout()

# ── Section 5 — Figure 3: High-confidence shrinkage ──────────────────────────

hi = df[df["confidence"] > HIGH_CONF_THRESHOLD].copy()
print(f"\n-- High-Confidence Trials (confidence > {HIGH_CONF_THRESHOLD})")
print(f"   n = {len(hi):,} ({100*len(hi)/len(df):.1f}% of all trials)")

hi_bin_edges = np.linspace(HIGH_CONF_THRESHOLD, 1.0, 4)
hi["conf_bin"] = pd.cut(hi["confidence"], bins=hi_bin_edges, include_lowest=True)
hi["predicted_action_matches"] = (hi["catie_p_biased"] > 0.5) == hi["is_biased_choice"]

hi_summary = (
    hi.groupby("conf_bin", observed=True)
    .agg(predicted=("confidence", "mean"),
         empirical=("predicted_action_matches", "mean"),
         n=("predicted_action_matches", "size"))
    .reset_index()
)
print(hi_summary[["predicted", "empirical", "n"]].to_string(index=False))

fig3, ax3 = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(hi_summary))
width = 0.35
ax3.bar(x - width / 2, hi_summary["predicted"], width,
        label="CATIE's stated confidence", color="#1f77b4")
ax3.bar(x + width / 2, hi_summary["empirical"], width,
        label="Empirical accuracy", color="#d62728")
for i, (_, r) in enumerate(hi_summary.iterrows()):
    ax3.annotate(f"n={int(r['n']):,}",
                 (i, max(r["predicted"], r["empirical"]) + 0.015),
                 ha="center", fontsize=8)
ax3.set_xticks(x)
ax3.set_xticklabels([str(b) for b in hi_summary["conf_bin"]], fontsize=8)
ax3.set_ylim(0, 1.1)
ax3.set_ylabel("Probability")
ax3.set_xlabel("Confidence bin")
ax3.set_title(f"High-Confidence Predictions (confidence > {HIGH_CONF_THRESHOLD}):\n"
              f"Stated Confidence vs. Actual Accuracy")
ax3.legend(fontsize=9)
fig3.tight_layout()

# ── Section 6 — Save all figures ──────────────────────────────────────────────

print("\n-- Saving figures")
figure_map = {
    "fig1_reliability_diagram":          fig1,
    "fig2_confidence_by_trial_schedule": fig2,
    "fig3_high_confidence_shrinkage":    fig3,
}

for name, fig in figure_map.items():
    fig.savefig(OUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"  {name}.png")

plt.close("all")
print(f"\nAll figures saved to: {OUT_DIR}")
print(f"Console output saved to: {OUT_DIR / 'output.txt'}")
print("Done.")

sys.stdout = sys.__stdout__
_log_file.close()
