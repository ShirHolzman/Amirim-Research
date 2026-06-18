"""
surprise_analysis.py — CATIE Failure Point Analysis

Why does CATIE (catie_choice_probability < 0.15) fail to predict
human behavior? This script engineers historical features from observed
data only (partial-feedback paradigm: biased_reward, unbiased_reward,
and unobserved_reward are NEVER used), then identifies what separates
surprise trials from baseline trials.
"""

import pathlib
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = (
    pathlib.Path(__file__).parent.parent
    / "processing"
    / "eda_with_catie_probabilities.csv"
)
OUT_DIR = pathlib.Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

SURPRISE_THRESHOLD = 0.15
RNG_SEED = 42

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
print("CATIE Surprise Point Analysis")
print("=" * 62)
print(f"\nLoading: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows")

assert len(df) == 49200, f"Expected 49200 rows, got {len(df)}"
assert df["catie_choice_probability"].between(0, 1, inclusive="both").all()
assert df["catie_choice_probability"].notna().all()

df["is_biased_choice"] = df["is_biased_choice"].astype(str).str.upper() == "TRUE"
df = df.sort_values(["subject_file", "trial_number"]).reset_index(drop=True)

# ── Section 2 — Engineer historical features ──────────────────────────────────

print("\n-- Engineering historical features...")

def compute_streak(series):
    """Count consecutive identical values ending at each position."""
    arr = series.values
    streak = np.ones(len(arr), dtype=int)
    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1]:
            streak[i] = streak[i - 1] + 1
    return pd.Series(streak, index=series.index)

g = df.groupby("subject_file", sort=False)

# Choice-based lags (biased vs unbiased — cognitive inertia)
for k in [1, 2, 3]:
    df[f"prev_choice_biased_{k}"] = g["is_biased_choice"].shift(k).astype("boolean")

# Observed reward lags only — PARTIAL FEEDBACK: only observed_reward is visible
for k in [1, 2, 3]:
    df[f"prev_reward_{k}"] = g["observed_reward"].shift(k)

# Physical side switch (t-1 vs t-2)
df["prev_side_switch"] = (
    g["side_choice"].transform(lambda s: (s != s.shift(1)).astype(float))
)

# Consecutive same biased/unbiased choice streak
df["streak_same_choice"] = g["is_biased_choice"].transform(compute_streak)

# Consecutive same physical side streak (motor inertia)
df["streak_same_side"] = g["side_choice"].transform(compute_streak)

# Recent reward rate (observed only, last 5 trials)
df["recent_reward_rate"] = (
    g["observed_reward"]
    .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
)

# Recent side-switch rate (last 5 trials)
df["recent_switch_rate"] = (
    df.groupby("subject_file")["prev_side_switch"]
    .transform(lambda s: s.rolling(5, min_periods=1).mean())
)

# Cumulative bias rate (running proportion of biased choices so far)
df["cumulative_bias_rate"] = (
    g["is_biased_choice"]
    .transform(lambda s: s.astype(float).expanding().mean().shift(1))
)

# CATIE confidence at t-1 (was model confident just before this surprise?)
df["catie_p_prev"] = g["catie_choice_probability"].shift(1)

# trial_position already present as trial_number

ALL_FEATURES = [
    "prev_choice_biased_1", "prev_choice_biased_2", "prev_choice_biased_3",
    "prev_reward_1", "prev_reward_2", "prev_reward_3",
    "prev_side_switch",
    "streak_same_choice", "streak_same_side",
    "recent_reward_rate", "recent_switch_rate",
    "cumulative_bias_rate",
    "catie_p_prev",
    "trial_number",
]

print(f"  Features engineered: {len(ALL_FEATURES)}")

# ── Section 3 — Label surprise points ────────────────────────────────────────

df["is_surprise"] = (df["catie_choice_probability"] < SURPRISE_THRESHOLD).astype(int)

n_total    = len(df)
n_surprise = df["is_surprise"].sum()
n_baseline = n_total - n_surprise
pct        = 100 * n_surprise / n_total

print(f"\n-- Surprise Points (p < {SURPRISE_THRESHOLD})")
print(f"   Total trials   : {n_total:,}")
print(f"   Surprise trials: {n_surprise:,}  ({pct:.1f}%)")
print(f"   Baseline trials: {n_baseline:,}  ({100-pct:.1f}%)")

# ── Section 3b — Figure 0: Probability distribution histogram ────────────────

fig0, ax0 = plt.subplots(figsize=(8, 4.5))
ax0.hist(df["catie_choice_probability"], bins=50, color="#4c72b0", edgecolor="white")
ax0.axvline(SURPRISE_THRESHOLD, color="#d62728", linestyle="--", linewidth=1.5)
ymax = ax0.get_ylim()[1]
ax0.axvspan(0, SURPRISE_THRESHOLD, color="#d62728", alpha=0.12)
ax0.text(SURPRISE_THRESHOLD / 2, ymax * 0.92,
         f"Surprise\nn={n_surprise:,} ({pct:.1f}%)",
         ha="center", va="top", color="#d62728", fontsize=10, fontweight="bold")
ax0.set_xlabel("CATIE choice probability (probability assigned to the actual choice)")
ax0.set_ylabel("Trial count")
ax0.set_title(f"Distribution of CATIE's Predicted Probability Across All Trials (n={n_total:,})")
fig0.tight_layout()

# ── Section 4 — Descriptive comparison (Welch t-test per feature) ─────────────

print("\n-- Descriptive Comparison: Surprise vs Baseline")
print(f"   (Bonferroni k={len(ALL_FEATURES)})")
print()

surprise_df  = df[df["is_surprise"] == 1]
baseline_df  = df[df["is_surprise"] == 0]
k_bonferroni = len(ALL_FEATURES)

rows = []
for feat in ALL_FEATURES:
    s_vals = surprise_df[feat].dropna().astype(float)
    b_vals = baseline_df[feat].dropna().astype(float)
    if len(s_vals) < 2 or len(b_vals) < 2:
        continue
    t_stat, p_raw = stats.ttest_ind(s_vals, b_vals, equal_var=False)
    p_corr = min(p_raw * k_bonferroni, 1.0)
    pooled_sd = np.sqrt((s_vals.var(ddof=1) + b_vals.var(ddof=1)) / 2)
    cohens_d  = (s_vals.mean() - b_vals.mean()) / pooled_sd if pooled_sd > 0 else 0.0
    rows.append({
        "feature":   feat,
        "mean_surp": s_vals.mean(),
        "mean_base": b_vals.mean(),
        "t_stat":    t_stat,
        "p_raw":     p_raw,
        "p_corr":    p_corr,
        "cohens_d":  cohens_d,
    })

stats_df = pd.DataFrame(rows).sort_values("cohens_d", key=abs, ascending=False)

header = f"{'Feature':<28} {'Mean(Surp)':>10} {'Mean(Base)':>10} {'Cohen_d':>8} {'p_corr':>10}"
print(header)
print("-" * len(header))
for _, r in stats_df.iterrows():
    sig = "***" if r["p_corr"] < 0.001 else ("**" if r["p_corr"] < 0.01 else ("*" if r["p_corr"] < 0.05 else "  ns"))
    print(f"{r['feature']:<28} {r['mean_surp']:>10.4f} {r['mean_base']:>10.4f} "
          f"{r['cohens_d']:>8.4f} {r['p_corr']:>10.4e}  {sig}")

# ── Section 5 — Figure 1: Feature comparison bar chart ───────────────────────

top8 = stats_df.head(8).copy()
colors = ["#d62728" if d > 0 else "#1f77b4" for d in top8["cohens_d"]]

fig1, ax1 = plt.subplots(figsize=(9, 5))
bars = ax1.barh(top8["feature"], top8["cohens_d"], color=colors, edgecolor="white")
ax1.axvline(0, color="black", linewidth=0.8, linestyle="--")
for bar, (_, row) in zip(bars, top8.iterrows()):
    sig = "***" if row["p_corr"] < 0.001 else ("**" if row["p_corr"] < 0.01 else ("*" if row["p_corr"] < 0.05 else "ns"))
    x = bar.get_width()
    ax1.text(x + (0.005 if x >= 0 else -0.005), bar.get_y() + bar.get_height() / 2,
             sig, va="center", ha="left" if x >= 0 else "right", fontsize=9)
ax1.set_xlabel("Cohen's d  (positive = higher in Surprise trials)")
ax1.set_title(f"Top Features Distinguishing Surprise (p < {SURPRISE_THRESHOLD}) from Baseline\n"
              f"Red = higher in Surprise, Blue = lower in Surprise")
ax1.invert_yaxis()
fig1.tight_layout()

# ── Section 6 — Figure 2: Reward history heatmap ─────────────────────────────

df["rh"] = df["prev_reward_1"].map({1: "Win", 0: "Loss", np.nan: None})
df["rh2"] = df["prev_reward_2"].map({1: "Win", 0: "Loss", np.nan: None})
heat_data = (
    df.dropna(subset=["rh", "rh2"])
    .groupby(["rh", "rh2"])["is_surprise"]
    .agg(["mean", "count"])
    .reset_index()
)
heat_pivot = heat_data.pivot(index="rh", columns="rh2", values="mean")
heat_pivot = heat_pivot.reindex(index=["Win", "Loss"], columns=["Win", "Loss"])

fig2, ax2 = plt.subplots(figsize=(5, 4))
sns.heatmap(
    heat_pivot * 100,
    annot=True, fmt=".1f", cmap="Reds",
    linewidths=0.5, ax=ax2,
    cbar_kws={"label": "Surprise rate (%)"},
)
ax2.set_xlabel("Reward at t-2")
ax2.set_ylabel("Reward at t-1")
ax2.set_title(f"Surprise Rate by Recent Reward Sequence\n(p < {SURPRISE_THRESHOLD})")
fig2.tight_layout()

# ── Section 7 — Figure 3: Streak length vs surprise rate ─────────────────────

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

for ax, col, title in [
    (ax3a, "streak_same_choice", "Choice (Biased/Unbiased) Streak"),
    (ax3b, "streak_same_side",   "Physical Side (L/R) Streak"),
]:
    streak_data = (
        df.groupby(col.split()[0] if " " in col else col)["is_surprise"]
        .agg(["mean", "count"])
        .reset_index()
    )
    # Cap at streak = 10+ for display
    df["_streak_cap"] = df[col].clip(upper=10)
    sd = df.groupby("_streak_cap")["is_surprise"].agg(["mean", "count"]).reset_index()
    sd = sd[sd["count"] >= 50]
    ax.plot(sd["_streak_cap"], sd["mean"] * 100, "o-", linewidth=2, markersize=6)
    ax.axhline(pct, color="gray", linestyle="--", linewidth=1, label=f"Overall rate ({pct:.1f}%)")
    ax.set_xlabel("Streak length (trials)")
    ax.set_ylabel("Surprise rate (%)")
    ax.set_title(f"Surprise Rate vs {title}")
    ax.legend(fontsize=9)

df.drop(columns=["_streak_cap"], inplace=True)
fig3.suptitle(f"Does Repetition Predict Pattern-Breaking? (threshold p < {SURPRISE_THRESHOLD})",
              fontsize=12, y=1.01)
fig3.tight_layout()

# ── Section 8 — Figure 4: Trial position vs surprise rate ────────────────────

df["quintile"] = pd.cut(
    df["trial_number"],
    bins=[0, 20, 40, 60, 80, 100],
    labels=["T0-19", "T20-39", "T40-59", "T60-79", "T80-99"],
    right=False,
)

quintile_data = (
    df.groupby(["quintile", "schedule"], observed=True)["is_surprise"]
    .mean()
    .reset_index()
)

fig4, ax4 = plt.subplots(figsize=(8, 4))
for sched, grp in quintile_data.groupby("schedule"):
    ax4.plot(grp["quintile"].astype(str), grp["is_surprise"] * 100, "o-",
             linewidth=2, markersize=6, label=sched)
ax4.axhline(pct, color="gray", linestyle="--", linewidth=1, label=f"Overall ({pct:.1f}%)")
ax4.set_xlabel("Trial block (quintile)")
ax4.set_ylabel("Surprise rate (%)")
ax4.set_title(f"Surprise Rate Across Trial Position by Schedule\n(p < {SURPRISE_THRESHOLD})")
ax4.legend(fontsize=9)
fig4.tight_layout()

# ── Section 9 — Random Forest feature importance ─────────────────────────────

print("\n-- Random Forest Feature Importance")

# Convert boolean columns to float for sklearn
feat_cols = ALL_FEATURES.copy()
rf_df = df[feat_cols + ["is_surprise"]].copy()
for c in feat_cols:
    rf_df[c] = pd.to_numeric(rf_df[c], errors="coerce")
rf_df = rf_df.dropna()

print(f"   Training on {len(rf_df):,} rows (dropped NaN-lag rows)")

rf = RandomForestClassifier(
    n_estimators=300, max_depth=8,
    class_weight="balanced", random_state=RNG_SEED, n_jobs=-1,
)
rf.fit(rf_df[feat_cols], rf_df["is_surprise"])

importances = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=True)

fig6, ax6 = plt.subplots(figsize=(8, 5))
ax6.barh(importances.index, importances.values, color="#2ca02c", edgecolor="white")
ax6.set_xlabel("Mean decrease in impurity (feature importance)")
ax6.set_title("Random Forest Feature Importances\n(predicting Surprise trial)")
fig6.tight_layout()

# Shallow decision tree for human-readable rules
dt = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=RNG_SEED)
dt.fit(rf_df[feat_cols], rf_df["is_surprise"])

print("\n-- Shallow Decision Tree Rules (depth 4)")
print(export_text(dt, feature_names=feat_cols))

# ── Section 10 — Figure 5: Prior CATIE confidence distribution ───────────────

fig5, ax5 = plt.subplots(figsize=(8, 4))
for label, subset, color in [
    ("Baseline", df[df["is_surprise"] == 0]["catie_p_prev"].dropna(), "#1f77b4"),
    ("Surprise", df[df["is_surprise"] == 1]["catie_p_prev"].dropna(), "#d62728"),
]:
    sns.kdeplot(subset, ax=ax5, label=label, color=color, linewidth=2)
    ax5.axvline(subset.mean(), color=color, linestyle="--", linewidth=1.2,
                label=f"{label} mean = {subset.mean():.3f}")

ax5.set_xlabel("CATIE predicted probability at t-1")
ax5.set_ylabel("Density")
ax5.set_title(f"CATIE Confidence at t-1 Before Surprise vs Baseline Trials\n"
              f"(Surprise = p < {SURPRISE_THRESHOLD})")
ax5.legend(fontsize=9)
fig5.tight_layout()

# ── Section 11 — Save all figures ────────────────────────────────────────────

print("\n-- Saving figures")
figure_map = {
    "fig0_probability_histogram":  fig0,
    "fig1_feature_comparison":     fig1,
    "fig2_reward_history_heatmap": fig2,
    "fig3_streak_vs_surprise":     fig3,
    "fig4_trial_position":         fig4,
    "fig5_prior_catie_confidence": fig5,
    "fig6_rf_feature_importance":  fig6,
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
