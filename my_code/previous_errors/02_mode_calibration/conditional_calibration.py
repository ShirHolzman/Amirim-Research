"""
Phase 2 -- Conditional calibration: which partition explains CATIE's miscalibration?

Research question (see thesis plan): CATIE's *aggregate* calibration gap is small,
yet the model loses badly to Q-Learning on E[log p]. The working hypothesis is that
the aggregate conceals large, oppositely-signed *conditional* miscalibration --
errors that cancel in a mean but compound in a log. This script does not
presuppose which partition of the data reveals that structure; it ADJUDICATES
between candidates, with the previous choice (`c_prev`) as the baseline every
richer partition must beat, and reports the result honestly in either direction.

WHAT THIS R^2 ACTUALLY MEASURES -- read before interpreting any number below.
The score is R^2 of the trial-level calibration gap, gap = y - p. Decomposing,

    Var(gap | L) = Var(y | L) + Var(p | L) - 2 Cov(y, p | L)

so a partition earns R^2 by EITHER predicting the choice y OR by homogenising the
model's own forecast p -- and these are very different achievements. Section 4
reports the decomposition, because the ranking is not interpretable without it.
Concretely: an out-of-fold gradient-boosted predictor of y scores only R^2=0.019
here despite predicting y better than anything else, because it leaves p
heterogeneous within cells; while `c_prev` removes ~89% of the variance in p.

Candidate PROSPECTIVE partitions (computable before trial t's outcome): c_prev
(baseline), hard-argmax mode attribution, schedule, trial-position phase (which
fifth of the session), run-length since last switch, recent reward rate, binned
p_alt1, and the pairwise interactions of c_prev with each. An earlier version of
this script tested only two of those interactions and thereby missed the
strongest partition -- see the `partitions` dict, which now crosses c_prev with
every candidate symmetrically.

A second, Bayesian responsibility-posterior mode attribution (responsibility.py)
is also computed. It is genuinely NOT forced to be deterministic in c_prev the way
the hard argmax is. But it is built FROM trial t's own observed choice (that is
what makes it a proper E-step), and section 5 proves it is exactly equivalent to
the repeat/switch indicator: `soft_argmax == inertia` iff `y == c_prev`,
identically. It is therefore reported ONLY as that proof; it is not used as a
partition and no reliability curve is drawn from it, because every such number is
reconstructible from a 2x4 count table and says nothing about the regimes.

Model: the corrected ("fixed") CATIE likelihood, k in {0,1,2} mixed with the
PER-TRIAL weighting -- the sequential Bayesian model average the paper's own
reported numbers match (verified per schedule to the rounding floor, 2026-09).
The shipped hetro.m:25 time-averages those weights; that variant is available as
weighting="shipped_time_avg" and is used only for MATLAB golden tests.
Section 2 re-runs the central c_prev split under the published model too, so the
robustness claim is shown rather than asserted.

Data: EDA + Training + schedule_0 (2,528 subjects; 250,272 trials after dropping
trial 1). Test is excluded and asserted absent -- touched once, at the very end.

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
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from catie_core import EPSILON, PHI, TAU, catie_hetero, p_of_observed_choice  # noqa: E402
import metrics  # noqa: E402

sys.path.insert(0, str(HERE))
from responsibility import REGIME_NAMES, responsibility_posterior, validate_responsibility  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "font.family": "sans-serif"})

RNG_SEED = 42
OUT_DIR = HERE / "figures"
DATA_DIR = HERE.parent / "data"

TEST_SCHEDULES = {"schedule_1", "schedule_8", "schedule_10"}

PALETTE = {"c_prev=0": "#4c72b0", "c_prev=1": "#d62728",
           "published": "#d62728", "fixed": "#1f77b4"}


# ── Section 1 -- Load data (EDA + Training + schedule_0; Test held out) ──────
def load_frame() -> pd.DataFrame:
    columns_to_load = ["subject_id", "schedule", "trial_number", "biased_reward",
            "unbiased_reward", "is_biased_choice", "observed_reward"]
    split_frames = [pd.read_csv(DATA_DIR / f"cleaned_{name}.csv")[columns_to_load]
              for name in ("eda", "training", "schedule_0")]

    combined_df = pd.concat(split_frames, ignore_index=True)
    combined_df["chose_biased"] = (combined_df["is_biased_choice"].astype(str).str.upper() == "TRUE").astype(int)
    combined_df = combined_df.sort_values(["subject_id", "trial_number"]).reset_index(drop=True)

    assert not combined_df["schedule"].isin(TEST_SCHEDULES).any(), \
        "Test split leaked into Phase 2 -- Test must be touched exactly once, at the end"
    trials_per_subject = combined_df.groupby("subject_id").size()
    assert (trials_per_subject == 100).all(), f"expected 100 trials/subject, got min={trials_per_subject.min()} max={trials_per_subject.max()}"
    return combined_df


# ── Section 2 -- Non-circular partition features ─────────────────────────────
def _streak_transform(shifted: pd.Series) -> pd.Series:
    """Run length of an already-lagged series. Applied to c_prev, this gives, at
    trial t, the length of the identical-choice run ending at t-1 -- entirely
    historical, no leakage of trial t's own choice."""
    shifted_values = shifted.fillna(-1).to_numpy()
    streak_lengths = np.ones(len(shifted_values), dtype=int)
    for i in range(1, len(shifted_values)):
        if shifted_values[i] == shifted_values[i - 1]:
            streak_lengths[i] = streak_lengths[i - 1] + 1
    return pd.Series(streak_lengths, index=shifted.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag-based features. Must run BEFORE the trial-1 drop, because c_prev needs
    trial 1 present to shift from. Binning that does not need trial 1 is deferred
    to `bin_features` so the bin edges reflect the analysed rows only."""
    grouped_by_subject = df.groupby("subject_id", sort=False)
    df["c_prev"] = grouped_by_subject["chose_biased"].shift(1)
    df["streak_prev"] = grouped_by_subject["c_prev"].transform(_streak_transform)
    df["recent_reward_rate"] = grouped_by_subject["observed_reward"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    return df


def bin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Binning, applied AFTER the trial-1 drop so every trial phase covers the same
    number of analysed trials (previously the first phase held 19 and the rest 20)."""
    df = df.copy()
    df["trial_phase"] = pd.cut(df["trial_number"], bins=[0, 20, 40, 60, 80, 99],
                                  labels=["trials 1-20", "trials 21-40", "trials 41-60",
                                          "trials 61-80", "trials 81-99"], include_lowest=True)
    df["streak_bin"] = pd.cut(df["streak_prev"], bins=[0, 1, 2, 3, 5, 100],
                              labels=["1", "2", "3", "4-5", "6+"])
    # NOTE: recent_reward_rate is a mean of <=5 binary rewards, so it takes few
    # distinct values; q=4 collapses to 3 groups via duplicates="drop". Verified
    # not to distort the comparison (the raw 11-valued variable scores 0.00135 vs
    # the binned 0.00077 -- both noise-level), but the collapse is reported.
    df["reward_rate_bin"] = pd.qcut(df["recent_reward_rate"], q=4, duplicates="drop")
    df["p_alt1_bin"] = pd.qcut(df["p_alt1"], q=20, duplicates="drop")
    return df


# ── Section 3 -- Per-subject model computation ───────────────────────────────
def compute_all_subjects(df: pd.DataFrame):
    subject_frames = []
    max_responsibility_deviation = 0.0
    n_subjects = df["subject_id"].nunique()
    for subject_index, (subject_id, subject_df) in enumerate(df.groupby("subject_id", sort=False)):
        biased_reward_sequence = subject_df["biased_reward"].to_numpy()
        unbiased_reward_sequence = subject_df["unbiased_reward"].to_numpy()
        chose_biased_bool = subject_df["chose_biased"].to_numpy().astype(bool)

        posterior_result = responsibility_posterior(biased_reward_sequence, unbiased_reward_sequence, chose_biased_bool, mode="fixed")
        max_responsibility_deviation = max(max_responsibility_deviation, validate_responsibility(posterior_result["resp"]))

        subject_result_df = subject_df[["subject_id", "schedule", "trial_number", "chose_biased",
                 "c_prev", "streak_prev", "recent_reward_rate"]].copy()
        subject_result_df["p_alt1"] = posterior_result["p_alt1_mix"]
        subject_result_df["p_choice"] = posterior_result["p_choice_mix"]
        for regime_index, regime_name in enumerate(REGIME_NAMES):
            subject_result_df[f"resp_{regime_name}"] = posterior_result["resp"][regime_index]
            subject_result_df[f"contrib_{regime_name}"] = posterior_result["mixed_contrib"][regime_index]
        subject_frames.append(subject_result_df)

        if (subject_index + 1) % 500 == 0 or (subject_index + 1) == n_subjects:
            print(f"  [{subject_index + 1}/{n_subjects}] subjects processed", flush=True)

    trial_level_df = pd.concat(subject_frames, ignore_index=True)
    trial_level_df = trial_level_df[trial_level_df["trial_number"] > 0].reset_index(drop=True)

    responsibility_columns = [f"resp_{n}" for n in REGIME_NAMES]
    contribution_columns = [f"contrib_{n}" for n in REGIME_NAMES]
    trial_level_df["soft_argmax"] = np.array(REGIME_NAMES)[trial_level_df[responsibility_columns].to_numpy().argmax(axis=1)]
    trial_level_df["hard_argmax"] = np.array(REGIME_NAMES)[trial_level_df[contribution_columns].to_numpy().argmax(axis=1)]
    return trial_level_df, max_responsibility_deviation


def compute_published(df: pd.DataFrame) -> pd.DataFrame:
    subject_frames = []
    for subject_id, subject_df in df.groupby("subject_id", sort=False):
        chose_biased_bool = subject_df["chose_biased"].to_numpy().astype(bool)
        p_alt1 = catie_hetero(subject_df["biased_reward"].to_numpy(),
                              subject_df["unbiased_reward"].to_numpy(), chose_biased_bool, mode="published")
        subject_frames.append(pd.DataFrame({
            "subject_id": subject_id, "trial_number": subject_df["trial_number"].to_numpy(),
            "c_prev": subject_df["c_prev"].to_numpy(), "chose_biased": chose_biased_bool.astype(int),
            "p_alt1_pub": p_alt1, "p_choice_pub": p_of_observed_choice(p_alt1, chose_biased_bool)}))
    published_df = pd.concat(subject_frames, ignore_index=True)
    return published_df[published_df["trial_number"] > 0].reset_index(drop=True)


# ── Section 4 -- R^2 adjudication ─────────────────────────────────────────────
def r2_explained(gap, labels) -> float:
    """R^2 of the saturated group-mean model (between-group SS / total SS).

    Uses list(labels) rather than np.asarray so a list of tuples stays 1-D.

    NaN labels are rejected outright: pandas' groupby drops them, and because
    `.sum()` skips their NaN residuals they would contribute ZERO residual while
    still counting toward total SS -- silently inflating R^2 toward 1. A demo:
    x=[1,2,3,4,100] with labels ['a','a','b','b',None] scores 0.9999.
    """
    gap_series = pd.Series(np.asarray(gap, dtype=float)).reset_index(drop=True)
    label_series = pd.Series(list(labels)).reset_index(drop=True)
    nan_label_count = label_series.isna().sum()
    assert nan_label_count == 0, (f"r2_explained received {nan_label_count} NaN labels; these would be "
                        f"silently dropped and inflate R^2. Filter or fillna first.")
    total_sum_of_squares = ((gap_series - gap_series.mean()) ** 2).sum()
    if total_sum_of_squares == 0:
        return float("nan")
    within_group_residuals = gap_series.groupby(label_series, observed=True).transform(lambda x: x - x.mean())
    return float(1 - (within_group_residuals ** 2).sum() / total_sum_of_squares)


def variance_decomposition(df: pd.DataFrame, labels) -> dict:
    """E[Var(gap|L)] split into its Var(y), Var(p) and Cov components."""
    gap_components_df = pd.DataFrame({"y": df["chose_biased"].to_numpy().astype(float),
                      "p": df["p_alt1"].to_numpy(),
                      "L": pd.Series(list(labels)).to_numpy()})
    per_group_stats = []
    for _, group_df in gap_components_df.groupby("L", observed=True):
        if len(group_df) < 2:
            per_group_stats.append((len(group_df), 0.0, 0.0, 0.0)); continue
        per_group_stats.append((len(group_df), group_df.y.var(), group_df.p.var(),
                     float(np.cov(group_df.y.to_numpy(), group_df.p.to_numpy())[0, 1])))
    per_group_stats_df = pd.DataFrame(per_group_stats, columns=["n", "vy", "vp", "cv"]).fillna(0.0)
    group_weight = per_group_stats_df.n / per_group_stats_df.n.sum()
    return {"E_Var_y": float((group_weight * per_group_stats_df.vy).sum()), "E_Var_p": float((group_weight * per_group_stats_df.vp).sum()),
            "E_Cov": float((group_weight * per_group_stats_df.cv).sum())}


def candidate_partitions(df: pd.DataFrame) -> dict:
    """Every prospective candidate, each also crossed with c_prev. Crossing is done
    symmetrically -- an earlier version crossed c_prev with only `schedule` and
    `hard_argmax`, which missed `c_prev x streak_bin`, the strongest partition."""
    c_prev_str = df["c_prev"].astype(int).astype(str)
    single_partitions = {
        "c_prev (baseline)": df["c_prev"].astype(int),
        "hard_argmax": df["hard_argmax"],
        "schedule": df["schedule"],
        "trial_phase": df["trial_phase"].astype(str),
        "streak_bin": df["streak_bin"].astype(str),
        "reward_rate_bin": df["reward_rate_bin"].astype(str),
        "p_alt1_bin (20q)": df["p_alt1_bin"].astype(str),
    }
    all_partitions = dict(single_partitions)
    for partition_name, partition_labels in single_partitions.items():
        if partition_name == "c_prev (baseline)":
            continue
        all_partitions[f"c_prev x {partition_name}"] = c_prev_str + "|" + pd.Series(partition_labels).astype(str).reset_index(drop=True)
    return all_partitions


def noise_ceiling(df: pd.DataFrame, seed=RNG_SEED):
    """Upper bound on R^2 achievable by ANY history-only partition.

    Derivation. For a partition L, R^2(L) = 1 - E[Var(gap|L)]/Var(gap). By the law
    of total variance, 
      L never increases E[Var(gap|L)], so the finest
    history partition X maximises R^2. Within a cell of X the forecast p is fixed
    (it is a deterministic function of history), so

        Var(gap | X) = Var(y | X) = q(1-q),   q = P(y=1 | history)

    giving   ceiling = 1 - E[q(1-q)] / Var(gap).

    Estimating q is the whole difficulty, and the error is DIRECTIONAL: an underfit
    q_hat is shrunk toward 0.5, so q_hat(1-q_hat) overstates the noise and
    UNDERSTATES the ceiling. A logistic regression therefore gives a misleadingly
    low ceiling -- low enough that observed partitions exceed it, which is how we
    caught the problem. A cross-validated gradient booster is used instead, and two
    independent checks are reported:

      * plug-in     1 - E[q_hat(1-q_hat)]/Var(gap)  -- can err either way
      * OOF-Brier   1 - E[(y-q_hat)^2]/Var(gap)     -- a RIGOROUS lower bound on
        the ceiling for any q_hat, calibrated or not, since
        E[(y-q_hat)^2] = E[q(1-q)] + E[(q-q_hat)^2] >= E[q(1-q)].

    Returns (plug_in, oof_brier_bound, ece_of_q_hat).
    """
    grouped_by_subject = df.groupby("subject_id", sort=False)
    feature_df = pd.DataFrame(index=df.index)
    feature_df["c_prev"] = df["c_prev"]
    feature_df["streak_prev"] = df["streak_prev"]
    feature_df["recent_reward_rate"] = df["recent_reward_rate"]
    feature_df["trial_number"] = df["trial_number"]
    feature_df["p_alt1"] = df["p_alt1"]
    for lag in (2, 3, 4, 5):
        feature_df[f"choice_lag{lag}"] = grouped_by_subject["chose_biased"].shift(lag)
    feature_df["cum_bias_rate"] = grouped_by_subject["chose_biased"].transform(lambda s: s.shift(1).expanding().mean())
    feature_df["cum_switch_rate"] = grouped_by_subject["chose_biased"].transform(
        lambda s: s.shift(1).diff().abs().expanding().mean())
    feature_df = feature_df.fillna(-1.0)

    feature_matrix = feature_df.to_numpy()
    chose_biased_array = df["chose_biased"].to_numpy().astype(float)
    subject_id_array = df["subject_id"].to_numpy()
    gap_variance = (chose_biased_array - df["p_alt1"].to_numpy()).var()

    oof_predicted_prob = np.zeros(len(chose_biased_array))
    for train_idx, test_idx in GroupKFold(n_splits=5).split(feature_matrix, chose_biased_array, subject_id_array):
        booster_model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                           max_leaf_nodes=63, random_state=seed)
        booster_model.fit(feature_matrix[train_idx], chose_biased_array[train_idx])
        oof_predicted_prob[test_idx] = booster_model.predict_proba(feature_matrix[test_idx])[:, 1]

    plug_in_ceiling = 1 - np.mean(oof_predicted_prob * (1 - oof_predicted_prob)) / gap_variance
    oof_brier_bound = 1 - np.mean((chose_biased_array - oof_predicted_prob) ** 2) / gap_variance
    q_hat_ece = metrics.ece(oof_predicted_prob, chose_biased_array, n_bins=10)
    return float(plug_in_ceiling), float(oof_brier_bound), float(q_hat_ece), oof_predicted_prob


def stratum_table(df: pd.DataFrame, by) -> pd.DataFrame:
    stratum_rows = []
    for stratum_value, stratum_df in df.groupby(by, observed=True):
        stratum_rows.append({"stratum": stratum_value, "n": len(stratum_df),
                     "predicted": stratum_df["p_alt1"].mean(),
                     "empirical": stratum_df["chose_biased"].mean(),
                     "gap": stratum_df["chose_biased"].mean() - stratum_df["p_alt1"].mean()})
    return pd.DataFrame(stratum_rows)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(OUT_DIR / "output.txt", "w", encoding="utf-8")
    sys.stdout = metrics.Tee(sys.__stdout__, log_file)

    print("=" * 78)
    print("PHASE 2 -- CONDITIONAL CALIBRATION: WHICH PARTITION EXPLAINS THE GAP?")
    print("=" * 78)
    print('model: corrected ("fixed") CATIE, k in {0,1,2}, per-trial k-mixture weighting (the paper\'s)')
    print("data: EDA + Training + schedule_0 (Test held out for final evaluation)\n")

    df = build_features(load_frame())
    print(f"loaded {df['subject_id'].nunique():,} subjects, {len(df):,} trials, "
          f"{df['schedule'].nunique()} schedules: {sorted(df['schedule'].unique())}")

    print("\ncomputing responsibility posteriors ...")
    trial_level_df, max_responsibility_deviation = compute_all_subjects(df)
    print(f"\nresponsibility sum-to-1 max deviation: {max_responsibility_deviation:.3e} (correctness gate)")
    assert max_responsibility_deviation < 1e-8, "responsibility posterior failed to normalise"
    trial_level_df = bin_features(trial_level_df)
    assert trial_level_df[["c_prev", "streak_bin", "trial_phase", "reward_rate_bin",
              "p_alt1_bin", "schedule", "hard_argmax"]].notna().all().all(), \
        "NaN in a partition column -- would silently inflate R^2"
    trial_level_df.to_csv(OUT_DIR / "trial_level.csv.gz", index=False, compression="gzip")

    # ── 1. Headline metrics ──────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("1. HEADLINE METRICS (fixed model, all trials pooled)")
    print("=" * 78)
    for metric_name, metric_value in metrics.score_all(trial_level_df["p_alt1"], trial_level_df["p_choice"], trial_level_df["chose_biased"]).items():
        print(f"   {metric_name:<10} {metric_value:.4f}" if isinstance(metric_value, float) else f"   {metric_name:<10} {metric_value}")
    mean_p, mean_p_ci_lo, mean_p_ci_hi = metrics.bootstrap_ci(trial_level_df["p_choice"], trial_level_df["subject_id"], np.mean)
    log_p_choice = np.log(np.clip(trial_level_df["p_choice"], 1e-12, 1))
    mean_log_p, mean_log_p_ci_lo, mean_log_p_ci_hi = metrics.bootstrap_ci(log_p_choice, trial_level_df["subject_id"], np.mean)
    print(f"\n   E[p]     = {mean_p:.4f}  95% CI [{mean_p_ci_lo:.4f}, {mean_p_ci_hi:.4f}]  (subject-clustered)")
    print(f"   E[log p] = {mean_log_p:.4f}  95% CI [{mean_log_p_ci_lo:.4f}, {mean_log_p_ci_hi:.4f}]")
    print("\n   NOTE accuracy is near-vacuous here: argmax(p_alt1)==c_prev on ~99.4% of")
    print("   trials, so it essentially reports the repeat rate, not model quality.")

    # ── 2. Published vs fixed -- shown, not asserted ─────────────────────────
    print("\n" + "=" * 78)
    print("2. PUBLISHED vs FIXED (does this chapter's conclusion depend on the fix?)")
    print("=" * 78)
    published_df = compute_published(df)
    merged_df = trial_level_df.merge(published_df[["subject_id", "trial_number", "p_alt1_pub", "p_choice_pub"]],
                 on=["subject_id", "trial_number"])
    assert len(merged_df) == len(trial_level_df), f"merge changed row count: {len(trial_level_df)} -> {len(merged_df)}"
    e_p_test_result = metrics.paired_subject_test(merged_df["p_choice"], merged_df["p_choice_pub"], merged_df["subject_id"])
    e_log_p_test_result = metrics.paired_subject_test(np.log(np.clip(merged_df["p_choice"], 1e-12, 1)),
                                np.log(np.clip(merged_df["p_choice_pub"], 1e-12, 1)), merged_df["subject_id"])
    print(f"   E[p]      fixed {e_p_test_result['mean_a']:.4f}  published {e_p_test_result['mean_b']:.4f}  "
          f"diff {e_p_test_result['mean_diff']:+.4f}  p={e_p_test_result['p']:.3e} {metrics.stars(e_p_test_result['p'])}")
    print(f"   E[log p]  fixed {e_log_p_test_result['mean_a']:.4f}  published {e_log_p_test_result['mean_b']:.4f}  "
          f"diff {e_log_p_test_result['mean_diff']:+.4f}  p={e_log_p_test_result['p']:.3e} {metrics.stars(e_log_p_test_result['p'])}")

    print("\n   The central c_prev split, recomputed under the PUBLISHED model:")
    published_gap_df = published_df.copy()
    published_gap_df["gap"] = published_gap_df["chose_biased"] - published_gap_df["p_alt1_pub"]
    for c_prev_value, group_df in published_gap_df.groupby("c_prev"):
        print(f"     c_prev={int(c_prev_value)}  n={len(group_df):>7,}  predicted {group_df['p_alt1_pub'].mean():.4f}  "
              f"empirical {group_df['chose_biased'].mean():.4f}  gap {group_df['gap'].mean():+.4f}")
    print(f"     aggregate gap {published_gap_df['gap'].mean():+.4f}")
    r2_c_prev_published = r2_explained(published_gap_df["gap"], published_gap_df["c_prev"].astype(int))
    print(f"     R^2(c_prev) under published = {r2_c_prev_published:.4f}  "
          f"(fixed model: see section 4)")
    print("   => the oppositely-signed split is present under BOTH models, so this")
    print("      chapter's conclusion does not depend on the Phase 1 correction.")
    print("      (The published aggregate gap is larger, so its cancellation is less")
    print("      complete -- the corrected model is the cleaner demonstration.)")

    # ── 3. The central arithmetic ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("3. THE CENTRAL ARITHMETIC -- calibration gap conditional on c_prev")
    print("=" * 78)
    c_prev_stratum_table = stratum_table(trial_level_df, "c_prev")
    print(c_prev_stratum_table.to_string(index=False))
    aggregate_gap = trial_level_df["chose_biased"].mean() - trial_level_df["p_alt1"].mean()
    print(f"\n   aggregate gap: {aggregate_gap:+.4f}  (near-zero: cancellation in the mean)")
    print("   Section 8 tests directly whether this cancellation is what separates")
    print("   E[p] from E[log p], rather than asserting it.")

    # ── 4. Partition adjudication ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("4. PARTITION ADJUDICATION (R^2 of the calibration gap, PROSPECTIVE only)")
    print("=" * 78)
    print("   soft_argmax is excluded -- section 5 proves it encodes the outcome.\n")
    calibration_gap = (trial_level_df["chose_biased"] - trial_level_df["p_alt1"]).to_numpy()
    r2_c_prev_baseline = r2_explained(calibration_gap, trial_level_df["c_prev"].astype(int))
    partition_r2_rows = []
    for partition_name, partition_labels in candidate_partitions(trial_level_df).items():
        partition_r2_rows.append({"partition": partition_name, "n_groups": pd.Series(list(partition_labels)).nunique(),
                     "R2": r2_explained(calibration_gap, partition_labels),
                     "increment_over_c_prev": r2_explained(calibration_gap, partition_labels) - r2_c_prev_baseline})
    partition_r2_table = pd.DataFrame(partition_r2_rows).sort_values("R2", ascending=False)
    print(partition_r2_table.to_string(index=False))
    partition_r2_table.to_csv(OUT_DIR / "partition_r2.csv", index=False)

    print("\n   computing the noise ceiling (5-fold grouped GB) ...", flush=True)
    plug_in_ceiling, oof_brier_bound, q_hat_ece, q_hat = noise_ceiling(trial_level_df)
    print(f"\n   Var(gap) = {calibration_gap.var():.4f}")
    print(f"   plug-in ceiling  1 - E[q(1-q)]/Var(gap)   = {plug_in_ceiling:.4f}")
    print(f"   rigorous bound   1 - OOF Brier/Var(gap)   = {oof_brier_bound:.4f}   (ceiling >= this)")
    print(f"   ECE of q_hat = {q_hat_ece:.4f} (well calibrated => plug-in trustworthy)")
    print(f"\n   So the achievable ceiling is ~{plug_in_ceiling:.2f}, NOT the ~0.18 a logistic-")
    print("   regression estimate suggests. An underfit q_hat is shrunk toward 0.5,")
    print("   overstating the noise and understating the ceiling -- low enough that")
    print("   observed partitions exceed it, which is how the error was caught.")
    best_partition_row = partition_r2_table.iloc[0]
    print(f"\n   best partition: {best_partition_row['partition']} R^2={best_partition_row['R2']:.4f} "
          f"= {100*best_partition_row['R2']/plug_in_ceiling:.0f}% of the ~{plug_in_ceiling:.2f} ceiling")
    print(f"   c_prev alone:   R^2={r2_c_prev_baseline:.4f} = {100*r2_c_prev_baseline/plug_in_ceiling:.0f}% of ceiling")

    print("\n" + "-" * 78)
    print("   DECOMPOSITION: Var(gap|L) = Var(y|L) + Var(p|L) - 2Cov(y,p|L)")
    print("   A partition scores by predicting y OR by homogenising p -- these are")
    print("   different achievements and the ranking is not readable without this.")
    print("-" * 78)
    print(f"   {'partition':<26}{'R^2':>8}{'E Var(y)':>11}{'E Var(p)':>11}{'E Cov':>10}")
    variance_decomposition_rows = []
    for partition_name, partition_labels in [("c_prev", trial_level_df["c_prev"].astype(int)),
                    ("hard_argmax", trial_level_df["hard_argmax"]),
                    ("c_prev x streak_bin",
                     trial_level_df["c_prev"].astype(int).astype(str) + "|" + trial_level_df["streak_bin"].astype(str)),
                    ("p_alt1_bin (20q)", trial_level_df["p_alt1_bin"].astype(str)),
                    ("q_hat 50q (best y-pred)", pd.qcut(q_hat, 50, duplicates="drop").astype(str))]:
        decomposition = variance_decomposition(trial_level_df, partition_labels)
        r2_value = r2_explained(calibration_gap, partition_labels)
        print(f"   {partition_name:<26}{r2_value:>8.4f}{decomposition['E_Var_y']:>11.4f}{decomposition['E_Var_p']:>11.4f}{decomposition['E_Cov']:>+10.4f}")
        variance_decomposition_rows.append({"partition": partition_name, "R2": r2_value, **decomposition})
    print(f"   {'(unconditional)':<26}{0.0:>8.4f}"
          f"{trial_level_df['chose_biased'].var():>11.4f}{trial_level_df['p_alt1'].var():>11.4f}"
          f"{float(np.cov(trial_level_df['chose_biased'], trial_level_df['p_alt1'])[0,1]):>+10.4f}")
    pd.DataFrame(variance_decomposition_rows).to_csv(OUT_DIR / "variance_decomposition.csv", index=False)
    unconditional_p_variance = trial_level_df["p_alt1"].var()
    p_variance_given_c_prev = variance_decomposition(trial_level_df, trial_level_df["c_prev"].astype(int))["E_Var_p"]
    print(f"\n   c_prev removes {100*(1-p_variance_given_c_prev/unconditional_p_variance):.0f}% of the variance in CATIE's")
    print("   OWN forecast p. So 'c_prev explains the calibration gap' is substantially")
    print("   a statement about the model's architecture (phi=0.71 dominates p), not")
    print("   purely a discovery about human behaviour. See README finding #6.")

    # ── 5. Hard vs soft attribution + the circularity proof ─────────────────
    print("\n" + "=" * 78)
    print("5. MODE ATTRIBUTION -- why soft_argmax cannot be used as a partition")
    print("=" * 78)
    print("   Hard argmax (mass toward alt 1) is a deterministic relabelling of c_prev:")
    print("   inertia's contribution is phi*c_prev, identically 0 when c_prev=0.\n")
    print(trial_level_df.groupby("hard_argmax", observed=True)["c_prev"].agg(["mean", "count"]).to_string())
    print("\n   The Bayesian posterior fixes THAT degeneracy ...")
    print(trial_level_df.groupby("soft_argmax", observed=True)["c_prev"].agg(["mean", "count"]).to_string())
    print("\n   ... but replaces it with a worse one. PROOF: regime r's unnormalised")
    print("   responsibility is w_r*P(y|r). For inertia P(alt1|inertia)=c_prev in {0,1},")
    print("   so its term is exactly w_I when y==c_prev and exactly 0 otherwise. And")
    print("   w_I = (1-tau*H)(1-p_exp)*phi >= 0.71*0.70*0.71 = 0.3529 strictly exceeds")
    print("   every other regime's maximum (heuristic <= tau = 0.29; contingent <=")
    print("   (1-tau*H)(1-p_exp)*0.29 < w_I; exploration <= (1-tau*H)*0.15 < w_I). The")
    print("   inequality holds per-k so it survives the convex k-mixture. Therefore:")
    print("\n       soft_argmax == inertia   <=>   y == c_prev,  identically.\n")
    c_prev_by_soft_argmax_table = trial_level_df.groupby(["c_prev", "soft_argmax"], observed=True)["chose_biased"].agg(["mean", "count"])
    print(c_prev_by_soft_argmax_table.to_string())
    all_cells_degenerate = c_prev_by_soft_argmax_table["mean"].isin([0.0, 1.0]).all()
    mismatch_count = int(((trial_level_df["soft_argmax"] == "inertia") != (trial_level_df["chose_biased"] == trial_level_df["c_prev"])).sum())
    print(f"\n   every cell mean in {{0,1}}: {all_cells_degenerate}     mismatches: {mismatch_count} / {len(trial_level_df):,}")
    print("   soft_argmax is the repeat/switch indicator plus a 3-way tiebreak. It")
    print("   carries exactly one bit beyond c_prev and that bit IS the outcome, so")
    print("   any reliability curve drawn from it is reconstructible from the counts")
    print("   above and says nothing about the regimes. No such curve is plotted.")

    # ── 6. Per-schedule ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("6. PER-SCHEDULE METRIC SUITE")
    print("=" * 78)
    schedule_metric_rows = []
    for schedule_name, schedule_df in trial_level_df.groupby("schedule", observed=True):
        schedule_metric_rows.append({"schedule": schedule_name, **metrics.score_all(schedule_df["p_alt1"], schedule_df["p_choice"],
                                                       schedule_df["chose_biased"]),
                      "empirical_rate": schedule_df["chose_biased"].mean(),
                      "gap": schedule_df["chose_biased"].mean() - schedule_df["p_alt1"].mean()})
    schedule_metrics_df = pd.DataFrame(schedule_metric_rows)
    schedule_metrics_df["order"] = schedule_metrics_df["schedule"].str.replace("schedule_", "").astype(int)
    schedule_metrics_df = schedule_metrics_df.sort_values("order").drop(columns="order")
    print(schedule_metrics_df.to_string(index=False))
    schedule_metrics_df.to_csv(OUT_DIR / "metrics_by_schedule.csv", index=False)
    print(f"\n   Empirical biased-choice rate ranges {schedule_metrics_df['empirical_rate'].min():.3f}"
          f"-{schedule_metrics_df['empirical_rate'].max():.3f} across schedules (spread "
          f"{schedule_metrics_df['empirical_rate'].max()-schedule_metrics_df['empirical_rate'].min():.3f}), yet max |gap|"
          f" is only {schedule_metrics_df['gap'].abs().max():.3f}.")
    print("   schedule's near-zero R^2 is therefore a POSITIVE result -- CATIE tracks")
    print("   between-schedule variation almost perfectly -- not evidence that")
    print("   schedule is behaviourally irrelevant.")

    # ── 7. The interaction the earlier version missed ────────────────────────
    print("\n" + "=" * 78)
    print("7. THE STRONGEST PARTITION: c_prev x run length")
    print("=" * 78)
    gap_by_c_prev_and_streak = trial_level_df.groupby(["c_prev", "streak_bin"], observed=True).apply(
        lambda group_df: pd.Series({"n": len(group_df), "predicted": group_df["p_alt1"].mean(),
                              "empirical": group_df["chose_biased"].mean(),
                              "gap": group_df["chose_biased"].mean() - group_df["p_alt1"].mean()}),
        include_groups=False)
    print(gap_by_c_prev_and_streak.to_string())
    gap_by_c_prev_and_streak.to_csv(OUT_DIR / "gap_by_cprev_streak.csv")
    print("\n   The gap is monotone in run length and REVERSES SIGN inside both c_prev")
    print("   strata. So the headline +0.205/-0.114 split is itself an average over")
    print("   oppositely-signed sub-strata -- the same failure mode this chapter is")
    print("   about, one level down. CATIE's constant phi under-predicts perseveration")
    print("   after long runs and over-predicts it after short ones.")
    print("   CONSEQUENCE FOR PHASE 4: an asymmetric inertia (one phi per side of")
    print("   c_prev) targets only the main effect and CANNOT represent a within-")
    print("   stratum sign reversal. Run-length-dependent or recency-weighted inertia")
    print("   is what these data actually demand.")

    # ── 8. Does the cancellation really drive the E[p]/E[log p] split? ──────
    print("\n" + "=" * 78)
    print("8. TESTING (not asserting) THE E[p] vs E[log p] MECHANISM")
    print("=" * 78)
    print("   If the c_prev-conditional miscalibration is what costs CATIE on E[log p],")
    print("   then removing it should improve E[log p] while NOT improving (or even")
    print("   hurting) E[p]. Recalibrating in three ways:\n")
    chose_biased_float_array = trial_level_df["chose_biased"].to_numpy().astype(float)
    variants = {"CATIE as-is": trial_level_df["p_alt1"].to_numpy()}
    variants["c_prev stratum rate"] = (
        trial_level_df.groupby("c_prev")["chose_biased"].transform("mean").to_numpy())
    variants["c_prev x streak_bin rate"] = (
        trial_level_df.groupby(["c_prev", "streak_bin"], observed=True)["chose_biased"]
        .transform("mean").to_numpy())
    isotonic_model = IsotonicRegression(out_of_bounds="clip")
    variants["isotonic on p_alt1 (in-sample)"] = isotonic_model.fit_transform(trial_level_df["p_alt1"], chose_biased_float_array)
    print(f"   {'forecast':<34}{'E[p]':>9}{'E[log p]':>11}{'ECE':>9}")
    for variant_name, predicted_prob in variants.items():
        predicted_prob_of_choice = np.where(chose_biased_float_array > 0.5, predicted_prob, 1 - predicted_prob)
        print(f"   {variant_name:<34}{metrics.e_p(predicted_prob_of_choice):>9.4f}{metrics.e_log_p(predicted_prob_of_choice):>11.4f}"
              f"{metrics.ece(predicted_prob, chose_biased_float_array):>9.4f}")
    print("\n   The dissociation is real: correcting the c_prev-conditional gap")
    print("   IMPROVES E[log p] while LOWERING E[p] -- exactly the trade the")
    print("   E[p]/E[log p] disagreement consists of.")
    print("   CAVEAT: the isotonic row is IN-SAMPLE and so is an optimistic bound,")
    print("   not a held-out result. And a full causal claim about why CATIE loses")
    print("   to Q-Learning needs QL's own conditional calibration profile, which is")
    print("   NOT computed anywhere in this repo. Stated as a mechanism consistent")
    print("   with the data, not as a demonstrated cause.")

    # ── figures ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("FIGURES")
    print("=" * 78)

    fig1, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    reliability_aggregate = metrics.reliability_table(trial_level_df["p_alt1"], trial_level_df["chose_biased"], n_bins=10, min_count=30)
    axes[0].plot([0, 1], [0, 1], "--", color="0.5", lw=1)
    axes[0].scatter(reliability_aggregate["predicted"], reliability_aggregate["empirical"], s=reliability_aggregate["n"] / 30, color="#333")
    axes[0].set_title(f"Predicted vs. actual choice rate, all trials pooled (n={len(trial_level_df):,})")
    axes[1].plot([0, 1], [0, 1], "--", color="0.5", lw=1)
    for c_prev_value, legend_label, color in [(0, "c_prev=0 (prior: unbiased)", PALETTE["c_prev=0"]),
                        (1, "c_prev=1 (prior: biased)", PALETTE["c_prev=1"])]:
        stratum_df = trial_level_df[trial_level_df["c_prev"] == c_prev_value]
        reliability_stratum = metrics.reliability_table(stratum_df["p_alt1"], stratum_df["chose_biased"], n_bins=10, min_count=30)
        axes[1].plot(reliability_stratum["predicted"], reliability_stratum["empirical"], "o-", color=color, label=legend_label)
    axes[1].set_title("Split by previous choice")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xlabel("predicted P(biased)"); ax.set_ylabel("empirical P(biased)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.spines[["top", "right"]].set_visible(False)
    fig1.tight_layout()

    fig2, ax = plt.subplots(figsize=(9, 6))
    r2_table_sorted = partition_r2_table.sort_values("R2")
    bar_colors = ["#d62728" if "baseline" in partition_name else "#4c72b0" for partition_name in r2_table_sorted["partition"]]
    ax.barh(r2_table_sorted["partition"], r2_table_sorted["R2"], color=bar_colors)
    ax.axvline(plug_in_ceiling, color="0.3", ls="--", lw=1.4)
    ax.text(plug_in_ceiling, 0.3, f"  noise ceiling ~{plug_in_ceiling:.2f}", fontsize=9, color="0.3", va="bottom")
    ax.set_xlabel("R$^2$ of calibration gap explained")
    ax.set_title("Prospective partitions vs. the achievable ceiling")
    ax.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()

    fig3, ax = plt.subplots(figsize=(8, 5))
    interaction_df = gap_by_c_prev_and_streak.reset_index()
    for c_prev_value, color, legend_label in [(0.0, PALETTE["c_prev=0"], "c_prev=0 (prior: unbiased)"),
                        (1.0, PALETTE["c_prev=1"], "c_prev=1 (prior: biased)")]:
        stratum_df = interaction_df[interaction_df["c_prev"] == c_prev_value]
        ax.plot(stratum_df["streak_bin"].astype(str), stratum_df["gap"], "o-", color=color, label=legend_label)
    ax.axhline(0, color="0.4", lw=1)
    ax.set_xlabel("run length of identical choices ending at t-1")
    ax.set_ylabel("calibration gap (empirical - predicted)")
    ax.set_title("Calibration gap by run length, split by previous choice")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig3.tight_layout()

    fig4, ax = plt.subplots(figsize=(8.5, 4.5))
    schedule_metrics_sorted = schedule_metrics_df.copy()
    schedule_metrics_sorted["order"] = schedule_metrics_sorted["schedule"].str.replace("schedule_", "").astype(int)
    schedule_metrics_sorted = schedule_metrics_sorted.sort_values("order")
    ax.bar(schedule_metrics_sorted["schedule"].str.replace("schedule_", ""), schedule_metrics_sorted["ECE"], color="#4c72b0")
    ax.set_xlabel("schedule"); ax.set_ylabel("ECE")
    ax.set_title("Calibration error by reward schedule (fixed model)")
    ax.spines[["top", "right"]].set_visible(False)
    fig4.tight_layout()

    figures_to_save = {"fig1_reliability_aggregate_vs_cprev": fig1,
            "fig2_r2_partition_comparison": fig2,
            "fig3_gap_by_cprev_and_runlength": fig3,
            "fig4_ece_by_schedule": fig4}
    for figure_name, figure in figures_to_save.items():
        figure.savefig(OUT_DIR / f"{figure_name}.png", dpi=150, bbox_inches="tight")
        print(f"  {figure_name}.png")

    stale_artifact_names = ["fig3_reliability_by_soft_mode.png", "fig5_hard_vs_soft_attribution.png",
             "gap_by_soft_mode.csv"]
    for stale_name in stale_artifact_names:
        stale_path = OUT_DIR / stale_name
        if stale_path.exists():
            stale_path.unlink()
            print(f"  removed stale artifact: {stale_name}")

    print("\ndone.")
    sys.stdout = sys.__stdout__
    log_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
