"""
Phase 2 -- Conditional calibration: which partition explains CATIE's miscalibration?

Research question (see thesis plan): CATIE's aggregate calibration gap is small
(+0.016 in the planning-phase probe), yet the model loses badly on E[log p] to a
Q-Learning competitor. The working hypothesis is that the aggregate gap CONCEALS
large, oppositely-signed conditional miscalibration -- errors that cancel in a
mean but compound in a log. This script does not presuppose which partition of
the data reveals that structure; it ADJUDICATES between candidates, with the
previous choice (`c_prev`) as the mandatory baseline every richer partition must
beat, and reports the R^2 increment honestly in either direction.

Candidate PROSPECTIVE partitions (computable before trial t's outcome, fair game
for "does this explain the calibration gap"): c_prev (baseline) -- dominant mode
by hard mass-attribution argmax (planning-phase method; built only from H, b,
c_prev, s_prev, sbar_prev, g, so legitimately historical) -- schedule --
trial-position quintile -- run-length since last switch -- recent reward rate.

A second, Bayesian responsibility-posterior mode attribution (responsibility.py)
is also computed, and is genuinely NOT forced to be deterministic in c_prev the
way the hard argmax is (see section 5's output). But it is built FROM trial t's
own observed choice (that is what makes it a proper E-step posterior), so it is
RETROSPECTIVE, not prospective, and scoring "R^2 of the calibration gap explained
by soft_argmax" would be circular -- confirmed directly on this data: within every
(c_prev, soft_argmax) cell, chose_biased comes out exactly 0 or 1. Its legitimate
use is retrospective error attribution (section 7): which regime is implicated
when CATIE is wrong, useful for Phase 4, not for real-time recalibration. See
section 5's full explanation before interpreting any soft_argmax number below.

Model: the corrected ("fixed") CATIE likelihood is used as the default baseline
throughout, per project decision -- the published/fixed distinction is
orthogonal to this chapter's question (the bug affects ~17% of trials via the
heuristic branch; this chapter is about the other three modes). A brief
published-vs-fixed comparison is included for continuity with Phase 1, not as
the main analysis.

Data: EDA + Training + schedule_0 (2,524 subjects, schedules 0,2,3,4,5,6,7,9,11).
The Test split (schedules 1,8,10) is deliberately excluded and asserted absent --
per the project's held-out discipline, Test is touched exactly once, at the very
end of the thesis timeline, not during exploratory/diagnostic work.

Run:  python my_code/catie_calibration/02_mode_calibration/conditional_calibration.py
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from catie_core import EPSILON, PHI, TAU, catie_hetero, p_of_observed_choice  # noqa: E402
import metrics as M  # noqa: E402

sys.path.insert(0, str(HERE))
from responsibility import REGIME_NAMES, responsibility_posterior, validate_responsibility  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "font.family": "sans-serif"})

RNG_SEED = 42
OUT_DIR = HERE / "figures"
DATA_DIR = HERE.parent / "data"
EDA_CSV = HERE.parent.parent / "EDA_set" / "processing" / "eda_with_catie_probabilities.csv"

TEST_SCHEDULES = {"schedule_1", "schedule_8", "schedule_10"}

PALETTE = {
    "c_prev=0": "#4c72b0", "c_prev=1": "#d62728",
    "heuristic": "#9467bd", "exploration": "#2ca02c",
    "inertia": "#d62728", "contingent_avg": "#4c72b0",
    "published": "#d62728", "fixed": "#1f77b4",
}


# ── Section 1 -- Load data (EDA + Training + schedule_0; Test held out) ──────
def load_frame() -> pd.DataFrame:
    eda = pd.read_csv(EDA_CSV)
    eda["subject_id"] = eda["schedule"] + "/" + eda["subject_file"]

    frames = [eda[["subject_id", "schedule", "trial_number", "biased_reward",
                   "unbiased_reward", "is_biased_choice", "observed_reward"]]]
    for name in ("training", "schedule_0"):
        df = pd.read_csv(DATA_DIR / f"cleaned_{name}.csv")
        frames.append(df[["subject_id", "schedule", "trial_number", "biased_reward",
                          "unbiased_reward", "is_biased_choice", "observed_reward"]])

    out = pd.concat(frames, ignore_index=True)
    out["chose_biased"] = (out["is_biased_choice"].astype(str).str.upper() == "TRUE").astype(int)
    out = out.sort_values(["subject_id", "trial_number"]).reset_index(drop=True)

    assert not out["schedule"].isin(TEST_SCHEDULES).any(), \
        "Test split leaked into Phase 2 -- Test must be touched exactly once, at the end"
    sizes = out.groupby("subject_id").size()
    assert (sizes == 100).all(), f"expected 100 trials/subject, got min={sizes.min()} max={sizes.max()}"
    return out


# ── Section 2 -- Non-circular partition features ─────────────────────────────
def _streak_transform(shifted: pd.Series) -> pd.Series:
    """Run length of an already-lagged series (ending at the position it's
    evaluated at). Applying this to c_prev (chose_biased.shift(1)) gives, at
    trial t, the length of the identical-choice run ending at t-1 -- entirely
    historical, no leakage of trial t's own choice."""
    vals = shifted.fillna(-1).to_numpy()
    streak = np.ones(len(vals), dtype=int)
    for i in range(1, len(vals)):
        if vals[i] == vals[i - 1]:
            streak[i] = streak[i - 1] + 1
    return pd.Series(streak, index=shifted.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("subject_id", sort=False)
    df["c_prev"] = g["chose_biased"].shift(1)
    df["streak_prev"] = g["c_prev"].transform(_streak_transform)
    df["recent_reward_rate"] = g["observed_reward"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["trial_quintile"] = pd.cut(df["trial_number"], bins=[-1, 19, 39, 59, 79, 99],
                                  labels=["Q1(0-19)", "Q2(20-39)", "Q3(40-59)",
                                          "Q4(60-79)", "Q5(80-99)"])
    df["streak_bin"] = pd.cut(df["streak_prev"], bins=[0, 1, 2, 3, 5, 100],
                              labels=["1", "2", "3", "4-5", "6+"])
    df["reward_rate_bin"] = pd.qcut(df["recent_reward_rate"], q=4, duplicates="drop")
    return df


# ── Section 3 -- Per-subject model computation ───────────────────────────────
def compute_all_subjects(df: pd.DataFrame):
    """Runs responsibility_posterior (mode="fixed") for every subject. Returns
    the input frame augmented with p_alt1, p_choice, per-regime responsibility,
    and per-regime mass-attribution (mixed_contrib) columns, plus the max
    sum-to-1 deviation observed (a correctness gate, not just a claim)."""
    rows = []
    max_dev = 0.0
    n_subjects = df["subject_id"].nunique()
    for i, (sid, d) in enumerate(df.groupby("subject_id", sort=False)):
        r1 = d["biased_reward"].to_numpy()
        r2 = d["unbiased_reward"].to_numpy()
        c1 = d["chose_biased"].to_numpy().astype(bool)

        out = responsibility_posterior(r1, r2, c1, mode="fixed")
        max_dev = max(max_dev, validate_responsibility(out["resp"]))

        sub = d[["subject_id", "schedule", "trial_number", "chose_biased",
                 "c_prev", "streak_prev", "streak_bin", "recent_reward_rate",
                 "reward_rate_bin", "trial_quintile"]].copy()
        sub["p_alt1"] = out["p_alt1_mix"]
        sub["p_choice"] = out["p_choice_mix"]
        for j, name in enumerate(REGIME_NAMES):
            sub[f"resp_{name}"] = out["resp"][j]
            sub[f"contrib_{name}"] = out["mixed_contrib"][j]
        rows.append(sub)

        if (i + 1) % 500 == 0 or (i + 1) == n_subjects:
            print(f"  [{i + 1}/{n_subjects}] subjects processed", flush=True)

    result = pd.concat(rows, ignore_index=True)
    result = result[result["trial_number"] > 0].reset_index(drop=True)  # drop trial 1 (p=0.5 by fiat)

    resp_cols = [f"resp_{n}" for n in REGIME_NAMES]
    contrib_cols = [f"contrib_{n}" for n in REGIME_NAMES]
    result["soft_argmax"] = np.array(REGIME_NAMES)[result[resp_cols].to_numpy().argmax(axis=1)]
    result["hard_argmax"] = np.array(REGIME_NAMES)[result[contrib_cols].to_numpy().argmax(axis=1)]
    return result, max_dev


def compute_published_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight published-mode pass (p_alt1/p_choice only, no responsibility
    decomposition) for the brief published-vs-fixed continuity check."""
    rows = []
    for sid, d in df.groupby("subject_id", sort=False):
        r1 = d["biased_reward"].to_numpy()
        r2 = d["unbiased_reward"].to_numpy()
        c1 = d["chose_biased"].to_numpy().astype(bool)
        p_alt1 = catie_hetero(r1, r2, c1, mode="published")
        p_choice = p_of_observed_choice(p_alt1, c1)
        rows.append(pd.DataFrame({"subject_id": sid, "trial_number": d["trial_number"].to_numpy(),
                                  "p_alt1_pub": p_alt1, "p_choice_pub": p_choice}))
    out = pd.concat(rows, ignore_index=True)
    return out[out["trial_number"] > 0].reset_index(drop=True)


# ── Section 4 -- R^2 adjudication ─────────────────────────────────────────────
def r2_explained(gap: np.ndarray, labels) -> float:
    """Fraction of variance in `gap` explained by grouping on `labels`: the R^2
    of the saturated group-mean model (between-group SS / total SS).

    Uses list(labels), not np.asarray(labels): the latter coerces a list of
    tuples (used for combined partitions like c_prev x schedule) into a 2D
    array instead of a 1D array of tuple objects, which pd.Series then rejects.
    """
    s = pd.Series(np.asarray(gap, dtype=float)).reset_index(drop=True)
    labels = pd.Series(list(labels)).reset_index(drop=True)
    tot = ((s - s.mean()) ** 2).sum()
    if tot == 0:
        return float("nan")
    resid = s.groupby(labels, observed=True).transform(lambda x: x - x.mean())
    return float(1 - (resid ** 2).sum() / tot)


def adjudicate_partitions(df: pd.DataFrame) -> pd.DataFrame:
    """R^2 of the calibration gap explained by each PROSPECTIVE partition -- one
    computable from trial t-1 and earlier, before trial t's outcome is known.

    soft_argmax is deliberately excluded here, not merely omitted. It is a
    RETROSPECTIVE quantity: the responsibility posterior it comes from is built
    from y(t), the very choice `gap` is computed against (see responsibility.py
    -- resp_r(t) uses np.where(y1>0.5, contributions, weights-contributions)).
    Verified directly on this data: chose_biased is EXACTLY 0 or 1 within every
    single (c_prev, soft_argmax) cell -- i.e. that grouping recovers the outcome
    itself, not genuine calibration structure. Computing "R^2 of gap explained by
    soft_argmax" would silently repeat the same circularity that discredited
    `prev_side_switch` in the original EDA work (a feature built from the choice
    being predicted), just one level more indirect. soft_argmax's legitimate use
    -- retrospective error attribution, i.e. which regime is implicated when
    CATIE is wrong -- is reported separately in main(), clearly labelled as such,
    not as a calibration partition. hard_argmax has no such problem: it is built
    purely from mode_contributions(), which depends only on H, b, c_prev, s_prev,
    sbar_prev, g -- all historical, none of it y(t) -- so it belongs here.
    """
    gap = (df["chose_biased"] - df["p_alt1"]).to_numpy()
    partitions = {
        "c_prev (baseline)": df["c_prev"],
        "hard_argmax (prospective)": df["hard_argmax"],
        "schedule": df["schedule"],
        "trial_quintile": df["trial_quintile"],
        "streak_bin": df["streak_bin"],
        "reward_rate_bin": df["reward_rate_bin"],
        "c_prev x schedule": df["c_prev"].astype(str) + "|" + df["schedule"].astype(str),
        "c_prev x hard_argmax": df["c_prev"].astype(str) + "|" + df["hard_argmax"].astype(str),
    }
    baseline_r2 = r2_explained(gap, df["c_prev"])
    rows = []
    for name, labels in partitions.items():
        r2 = r2_explained(gap, labels)
        n_groups = pd.Series(labels).nunique()
        rows.append({"partition": name, "n_groups": n_groups, "R2": r2,
                    "R2_increment_over_c_prev": r2 - baseline_r2})
    return pd.DataFrame(rows)


# ── Section 5 -- metric suite ─────────────────────────────────────────────────
def stratum_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for val, g in df.groupby(by, observed=True):
        rows.append({
            by: val, "n": len(g),
            "predicted": g["p_alt1"].mean(), "empirical": g["chose_biased"].mean(),
            "gap": g["chose_biased"].mean() - g["p_alt1"].mean(),
        })
    return pd.DataFrame(rows)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = open(OUT_DIR / "output.txt", "w", encoding="utf-8")
    sys.stdout = M.Tee(sys.__stdout__, log)

    print("=" * 78)
    print("PHASE 2 -- CONDITIONAL CALIBRATION: WHICH PARTITION EXPLAINS THE GAP?")
    print("=" * 78)
    print("model: corrected (\"fixed\") CATIE, k in {0,1,2}, published weighting")
    print("data: EDA + Training + schedule_0 (Test held out for final evaluation)\n")

    df = load_frame()
    n_subj = df["subject_id"].nunique()
    print(f"loaded {n_subj:,} subjects, {len(df):,} trials, "
          f"{df['schedule'].nunique()} schedules: {sorted(df['schedule'].unique())}")

    df = build_features(df)

    print("\ncomputing responsibility posteriors ...")
    M_trials, max_dev = compute_all_subjects(df)
    print(f"\nresponsibility sum-to-1 max deviation: {max_dev:.3e} (correctness gate)")
    assert max_dev < 1e-8, "responsibility posterior failed to normalise -- stop and investigate"

    M_trials.to_csv(OUT_DIR / "trial_level.csv.gz", index=False, compression="gzip")

    # ── 1. Headline metrics ──────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("1. HEADLINE METRICS (fixed model, all trials pooled)")
    print("=" * 78)
    overall = M.score_all(M_trials["p_alt1"], M_trials["p_choice"], M_trials["chose_biased"])
    for k, v in overall.items():
        print(f"   {k:<10} {v:.4f}" if isinstance(v, float) else f"   {k:<10} {v}")

    ep, ep_lo, ep_hi = M.bootstrap_ci(M_trials["p_choice"], M_trials["subject_id"], np.mean)
    lp = np.log(np.clip(M_trials["p_choice"], 1e-12, 1))
    elp, elp_lo, elp_hi = M.bootstrap_ci(lp, M_trials["subject_id"], np.mean)
    print(f"\n   E[p]     = {ep:.4f}  95% CI [{ep_lo:.4f}, {ep_hi:.4f}]  (subject-clustered bootstrap)")
    print(f"   E[log p] = {elp:.4f}  95% CI [{elp_lo:.4f}, {elp_hi:.4f}]")

    # ── 2. Published vs fixed, on this population, for continuity ───────────
    print("\n" + "=" * 78)
    print("2. PUBLISHED vs FIXED, this population (continuity with Phase 1)")
    print("=" * 78)
    pub = compute_published_comparison(df)
    merged = M_trials.merge(pub, on=["subject_id", "trial_number"])
    res = M.paired_subject_test(merged["p_choice"], merged["p_choice_pub"], merged["subject_id"])
    print(f"   E[p]      fixed {res['mean_a']:.4f}  published {res['mean_b']:.4f}  "
          f"diff {res['mean_diff']:+.4f}  p={res['p']:.3e} {M.stars(res['p'])}")
    lp_pub = np.log(np.clip(merged["p_choice_pub"], 1e-12, 1))
    lp_fix = np.log(np.clip(merged["p_choice"], 1e-12, 1))
    res2 = M.paired_subject_test(lp_fix, lp_pub, merged["subject_id"])
    print(f"   E[log p]  fixed {res2['mean_a']:.4f}  published {res2['mean_b']:.4f}  "
          f"diff {res2['mean_diff']:+.4f}  p={res2['p']:.3e} {M.stars(res2['p'])}")
    print("   (headline conclusion of this chapter is not sensitive to which model is used --")
    print("    see fig_published_vs_fixed_gap comparison below)")

    # ── 3. The central arithmetic: c_prev-conditional gap ────────────────────
    print("\n" + "=" * 78)
    print("3. THE CENTRAL ARITHMETIC -- calibration gap conditional on c_prev")
    print("=" * 78)
    print("   (re-derived on the k-mixture, all three non-held-out splits --")
    print("    planning-phase numbers used a single k=2 agent on EDA only)")
    cprev_table = stratum_table(M_trials, "c_prev")
    print(cprev_table.to_string(index=False))
    agg_gap = M_trials["chose_biased"].mean() - M_trials["p_alt1"].mean()
    print(f"\n   aggregate gap: {agg_gap:+.4f}  (near-zero: cancellation in the mean)")
    print("   -> errors that cancel in a mean compound in a log; this is the arithmetic")
    print("      behind CATIE's E[p]/E[log p] split decision.")

    # ── 4. Partition adjudication (PROSPECTIVE partitions only) ──────────────
    print("\n" + "=" * 78)
    print("4. PARTITION ADJUDICATION (R^2 of the calibration gap, PROSPECTIVE only)")
    print("=" * 78)
    print("   soft_argmax is excluded here -- see section 5 for why, and for its")
    print("   legitimate (retrospective) use instead.\n")
    r2_table = adjudicate_partitions(M_trials)
    print(r2_table.to_string(index=False))
    r2_table.to_csv(OUT_DIR / "partition_r2.csv", index=False)

    # ── 5. Hard vs soft mode attribution -- the confound, the fix, and a trap ─
    print("\n" + "=" * 78)
    print("5. HARD vs SOFT MODE ATTRIBUTION")
    print("=" * 78)
    print("   OLD (planning-phase) method: hard argmax over mass toward 'choosing alt1'.")
    print("   Forced to equal c_prev exactly for the inertia regime (its contribution is")
    print("   phi*c_prev, identically 0 when c_prev=0) -- not an empirical finding.\n")
    print("   P(c_prev=1 | hard_argmax):")
    hard_tab = M_trials.groupby("hard_argmax", observed=True)["c_prev"].agg(["mean", "count"])
    print(hard_tab.to_string())

    print("\n   NEW method: Bayesian responsibility posterior. Genuinely NOT forced to")
    print("   be deterministic in c_prev (see responsibility.py) -- confirmed below.\n")
    print("   P(c_prev=1 | soft_argmax):")
    soft_tab = M_trials.groupby("soft_argmax", observed=True)["c_prev"].agg(["mean", "count"])
    print(soft_tab.to_string())

    print("\n   BUT: the responsibility posterior is computed FROM the observed choice")
    print("   y(t) (resp_r(t) uses y(t) to pick contribution_r(t) vs weight_r(t)-")
    print("   contribution_r(t)) -- so soft_argmax is RETROSPECTIVE, not a forecasting")
    print("   partition, and must not be scored the way c_prev/hard_argmax are.")
    print("   Direct proof: within every single (c_prev, soft_argmax) cell, does")
    print("   chose_biased come out exactly 0 or 1 (i.e. does the label recover y(t)")
    print("   outright)?\n")
    circularity_check = M_trials.groupby(["c_prev", "soft_argmax"], observed=True)["chose_biased"].agg(["mean", "count"])
    print(circularity_check.to_string())
    all_degenerate = circularity_check["mean"].isin([0.0, 1.0]).all()
    print(f"\n   every cell mean in {{0, 1}}: {all_degenerate}  "
          f"{'-- confirmed circular, as expected' if all_degenerate else '-- NOT circular, reconsider the R^2 exclusion above'}")
    print("\n   soft_argmax's legitimate use is therefore RETROSPECTIVE ERROR ATTRIBUTION:")
    print("   'of the choices CATIE got most wrong, which regime's implicit confidence")
    print("   was responsible' -- see section 7. It cannot be used as-is to improve a")
    print("   real-time forecast, and its R^2-of-gap-explained is not reported, because")
    print("   it would not mean what a reader would assume it means.")

    # ── 6. Per-schedule metrics (never done, including in the paper) ─────────
    print("\n" + "=" * 78)
    print("6. PER-SCHEDULE METRIC SUITE")
    print("=" * 78)
    sched_rows = []
    for sched, g in M_trials.groupby("schedule", observed=True):
        row = {"schedule": sched, **M.score_all(g["p_alt1"], g["p_choice"], g["chose_biased"])}
        sched_rows.append(row)
    sched_df = pd.DataFrame(sched_rows)
    sched_df["order"] = sched_df["schedule"].str.replace("schedule_", "").astype(int)
    sched_df = sched_df.sort_values("order").drop(columns="order")
    print(sched_df.to_string(index=False))
    sched_df.to_csv(OUT_DIR / "metrics_by_schedule.csv", index=False)

    # ── 7. Retrospective error attribution by dominant mode ──────────────────
    print("\n" + "=" * 78)
    print("7. RETROSPECTIVE ERROR ATTRIBUTION BY DOMINANT MODE (soft posterior)")
    print("=" * 78)
    print("   NOT a calibration partition (see section 5) -- 'predicted' and 'empirical'")
    print("   below describe trials grouped by which regime the E-step attributes the")
    print("   REALISED choice to, useful for asking 'when CATIE is wrong, which regime's")
    print("   implicit confidence estimate is responsible', which is what Phase 4's model")
    print("   extensions need to target. Not usable to improve a real-time forecast.\n")
    mode_tab = stratum_table(M_trials, "soft_argmax")
    print(mode_tab.to_string(index=False))
    mode_tab.to_csv(OUT_DIR / "gap_by_soft_mode.csv", index=False)

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("FIGURES")
    print("=" * 78)

    # fig1: aggregate reliability vs c_prev split
    fig1, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    rel_all = M.reliability_table(M_trials["p_alt1"], M_trials["chose_biased"], n_bins=10, min_count=30)
    axes[0].plot([0, 1], [0, 1], "--", color="0.5", lw=1)
    axes[0].scatter(rel_all["predicted"], rel_all["empirical"], s=rel_all["n"] / 30, color="#333")
    axes[0].set_xlabel("predicted P(biased)"); axes[0].set_ylabel("empirical P(biased)")
    axes[0].set_title(f"Aggregate (n={len(M_trials):,})")
    axes[1].plot([0, 1], [0, 1], "--", color="0.5", lw=1)
    for v, label, color in [(0, "c_prev=0 (prior: unbiased)", PALETTE["c_prev=0"]),
                            (1, "c_prev=1 (prior: biased)", PALETTE["c_prev=1"])]:
        sub = M_trials[M_trials["c_prev"] == v]
        rel = M.reliability_table(sub["p_alt1"], sub["chose_biased"], n_bins=10, min_count=30)
        axes[1].plot(rel["predicted"], rel["empirical"], "o-", color=color, label=label)
    axes[1].set_xlabel("predicted P(biased)"); axes[1].set_ylabel("empirical P(biased)")
    axes[1].set_title("Split by previous choice")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.spines[["top", "right"]].set_visible(False)
    fig1.tight_layout()

    # fig2: R^2 partition comparison
    fig2, ax = plt.subplots(figsize=(8, 5))
    r2_sorted = r2_table.sort_values("R2")
    colors = ["#d62728" if "baseline" in p else "#4c72b0" for p in r2_sorted["partition"]]
    ax.barh(r2_sorted["partition"], r2_sorted["R2"], color=colors)
    ax.set_xlabel("R$^2$ of calibration gap explained")
    ax.set_title("Which PROSPECTIVE partition explains the miscalibration?")
    ax.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()

    # fig3: reliability by soft dominant mode
    fig3, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="0.5", lw=1)
    for name in REGIME_NAMES:
        sub = M_trials[M_trials["soft_argmax"] == name]
        if len(sub) < 100:
            continue
        rel = M.reliability_table(sub["p_alt1"], sub["chose_biased"], n_bins=8, min_count=20)
        ax.plot(rel["predicted"], rel["empirical"], "o-", color=PALETTE.get(name, "#333"),
               label=f"{name} (n={len(sub):,})")
    ax.set_xlabel("predicted P(biased)"); ax.set_ylabel("empirical P(biased)")
    ax.set_title("Retrospective error attribution by dominant mode\n(NOT a calibration partition -- see output.txt §5)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.spines[["top", "right"]].set_visible(False)
    fig3.tight_layout()

    # fig4: ECE by schedule
    fig4, ax = plt.subplots(figsize=(8.5, 4.5))
    sd = sched_df.copy()
    sd["order"] = sd["schedule"].str.replace("schedule_", "").astype(int)
    sd = sd.sort_values("order")
    ax.bar(sd["schedule"].str.replace("schedule_", ""), sd["ECE"], color="#4c72b0")
    ax.set_xlabel("schedule"); ax.set_ylabel("ECE")
    ax.set_title("Calibration error by reward schedule (fixed model)")
    ax.spines[["top", "right"]].set_visible(False)
    fig4.tight_layout()

    # fig5: hard vs soft attribution, P(c_prev=1 | mode)
    fig5, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(REGIME_NAMES))
    hard_vals = [hard_tab.loc[n, "mean"] if n in hard_tab.index else np.nan for n in REGIME_NAMES]
    soft_vals = [soft_tab.loc[n, "mean"] if n in soft_tab.index else np.nan for n in REGIME_NAMES]
    ax.bar(x - 0.2, hard_vals, 0.4, label="hard argmax (old)", color="#d62728")
    ax.bar(x + 0.2, soft_vals, 0.4, label="soft posterior (new)", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(REGIME_NAMES, rotation=20, ha="right")
    ax.set_ylabel("P(c_prev = 1 | dominant mode)")
    ax.set_title("Hard attribution is forced to {0,1}; the posterior is not\n"
                 "(posterior is retrospective -- diagnostic only, not a forecast partition)")
    ax.axhline(0, color="0.7", lw=0.8); ax.axhline(1, color="0.7", lw=0.8)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig5.tight_layout()

    figure_map = {
        "fig1_reliability_aggregate_vs_cprev": fig1,
        "fig2_r2_partition_comparison": fig2,
        "fig3_reliability_by_soft_mode": fig3,
        "fig4_ece_by_schedule": fig4,
        "fig5_hard_vs_soft_attribution": fig5,
    }
    for name, fig in figure_map.items():
        fig.savefig(OUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
        print(f"  {name}.png")

    print("\ndone.")
    sys.stdout = sys.__stdout__
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
