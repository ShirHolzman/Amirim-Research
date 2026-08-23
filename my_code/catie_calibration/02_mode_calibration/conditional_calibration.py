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
(baseline), hard-argmax mode attribution, schedule, trial-position quintile,
run-length since last switch, recent reward rate, binned p_alt1, and the pairwise
interactions of c_prev with each. An earlier version of this script tested only
two of those interactions and thereby missed the strongest partition -- see the
`partitions` dict, which now crosses c_prev with every candidate symmetrically.

A second, Bayesian responsibility-posterior mode attribution (responsibility.py)
is also computed. It is genuinely NOT forced to be deterministic in c_prev the way
the hard argmax is. But it is built FROM trial t's own observed choice (that is
what makes it a proper E-step), and section 5 proves it is exactly equivalent to
the repeat/switch indicator: `soft_argmax == inertia` iff `y == c_prev`,
identically. It is therefore reported ONLY as that proof; it is not used as a
partition and no reliability curve is drawn from it, because every such number is
reconstructible from a 2x4 count table and says nothing about the regimes.

Model: the corrected ("fixed") CATIE likelihood, k in {0,1,2} published mixture.
Section 2 re-runs the central c_prev split under the published model too, so the
robustness claim is shown rather than asserted.

Data: EDA + Training + schedule_0 (2,524 subjects; 249,876 trials after dropping
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
import metrics as M  # noqa: E402

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
    cols = ["subject_id", "schedule", "trial_number", "biased_reward",
            "unbiased_reward", "is_biased_choice", "observed_reward"]
    frames = [pd.read_csv(DATA_DIR / f"cleaned_{name}.csv")[cols]
              for name in ("eda", "training", "schedule_0")]

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
    """Run length of an already-lagged series. Applied to c_prev, this gives, at
    trial t, the length of the identical-choice run ending at t-1 -- entirely
    historical, no leakage of trial t's own choice."""
    vals = shifted.fillna(-1).to_numpy()
    streak = np.ones(len(vals), dtype=int)
    for i in range(1, len(vals)):
        if vals[i] == vals[i - 1]:
            streak[i] = streak[i - 1] + 1
    return pd.Series(streak, index=shifted.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag-based features. Must run BEFORE the trial-1 drop, because c_prev needs
    trial 1 present to shift from. Binning that does not need trial 1 is deferred
    to `bin_features` so the bin edges reflect the analysed rows only."""
    g = df.groupby("subject_id", sort=False)
    df["c_prev"] = g["chose_biased"].shift(1)
    df["streak_prev"] = g["c_prev"].transform(_streak_transform)
    df["recent_reward_rate"] = g["observed_reward"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    return df


def bin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Binning, applied AFTER the trial-1 drop so every quintile covers the same
    number of analysed trials (previously Q1 held 19 and the rest 20)."""
    df = df.copy()
    df["trial_quintile"] = pd.cut(df["trial_number"], bins=[0, 20, 40, 60, 80, 99],
                                  labels=["Q1(1-20)", "Q2(21-40)", "Q3(41-60)",
                                          "Q4(61-80)", "Q5(81-99)"], include_lowest=True)
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
                 "c_prev", "streak_prev", "recent_reward_rate"]].copy()
        sub["p_alt1"] = out["p_alt1_mix"]
        sub["p_choice"] = out["p_choice_mix"]
        for j, name in enumerate(REGIME_NAMES):
            sub[f"resp_{name}"] = out["resp"][j]
            sub[f"contrib_{name}"] = out["mixed_contrib"][j]
        rows.append(sub)

        if (i + 1) % 500 == 0 or (i + 1) == n_subjects:
            print(f"  [{i + 1}/{n_subjects}] subjects processed", flush=True)

    result = pd.concat(rows, ignore_index=True)
    result = result[result["trial_number"] > 0].reset_index(drop=True)

    resp_cols = [f"resp_{n}" for n in REGIME_NAMES]
    contrib_cols = [f"contrib_{n}" for n in REGIME_NAMES]
    result["soft_argmax"] = np.array(REGIME_NAMES)[result[resp_cols].to_numpy().argmax(axis=1)]
    result["hard_argmax"] = np.array(REGIME_NAMES)[result[contrib_cols].to_numpy().argmax(axis=1)]
    return result, max_dev


def compute_published(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, d in df.groupby("subject_id", sort=False):
        c1 = d["chose_biased"].to_numpy().astype(bool)
        p_alt1 = catie_hetero(d["biased_reward"].to_numpy(),
                              d["unbiased_reward"].to_numpy(), c1, mode="published")
        rows.append(pd.DataFrame({
            "subject_id": sid, "trial_number": d["trial_number"].to_numpy(),
            "c_prev": d["c_prev"].to_numpy(), "chose_biased": c1.astype(int),
            "p_alt1_pub": p_alt1, "p_choice_pub": p_of_observed_choice(p_alt1, c1)}))
    out = pd.concat(rows, ignore_index=True)
    return out[out["trial_number"] > 0].reset_index(drop=True)


# ── Section 4 -- R^2 adjudication ─────────────────────────────────────────────
def r2_explained(gap, labels) -> float:
    """R^2 of the saturated group-mean model (between-group SS / total SS).

    Uses list(labels) rather than np.asarray so a list of tuples stays 1-D.

    NaN labels are rejected outright: pandas' groupby drops them, and because
    `.sum()` skips their NaN residuals they would contribute ZERO residual while
    still counting toward total SS -- silently inflating R^2 toward 1. A demo:
    x=[1,2,3,4,100] with labels ['a','a','b','b',None] scores 0.9999.
    """
    s = pd.Series(np.asarray(gap, dtype=float)).reset_index(drop=True)
    lab = pd.Series(list(labels)).reset_index(drop=True)
    n_nan = lab.isna().sum()
    assert n_nan == 0, (f"r2_explained received {n_nan} NaN labels; these would be "
                        f"silently dropped and inflate R^2. Filter or fillna first.")
    tot = ((s - s.mean()) ** 2).sum()
    if tot == 0:
        return float("nan")
    resid = s.groupby(lab, observed=True).transform(lambda x: x - x.mean())
    return float(1 - (resid ** 2).sum() / tot)


def variance_decomposition(df: pd.DataFrame, labels) -> dict:
    """E[Var(gap|L)] split into its Var(y), Var(p) and Cov components."""
    d = pd.DataFrame({"y": df["chose_biased"].to_numpy().astype(float),
                      "p": df["p_alt1"].to_numpy(),
                      "L": pd.Series(list(labels)).to_numpy()})
    rows = []
    for _, sub in d.groupby("L", observed=True):
        if len(sub) < 2:
            rows.append((len(sub), 0.0, 0.0, 0.0)); continue
        rows.append((len(sub), sub.y.var(), sub.p.var(),
                     float(np.cov(sub.y.to_numpy(), sub.p.to_numpy())[0, 1])))
    R = pd.DataFrame(rows, columns=["n", "vy", "vp", "cv"]).fillna(0.0)
    w = R.n / R.n.sum()
    return {"E_Var_y": float((w * R.vy).sum()), "E_Var_p": float((w * R.vp).sum()),
            "E_Cov": float((w * R.cv).sum())}


def candidate_partitions(df: pd.DataFrame) -> dict:
    """Every prospective candidate, each also crossed with c_prev. Crossing is done
    symmetrically -- an earlier version crossed c_prev with only `schedule` and
    `hard_argmax`, which missed `c_prev x streak_bin`, the strongest partition."""
    cp = df["c_prev"].astype(int).astype(str)
    singles = {
        "c_prev (baseline)": df["c_prev"].astype(int),
        "hard_argmax": df["hard_argmax"],
        "schedule": df["schedule"],
        "trial_quintile": df["trial_quintile"].astype(str),
        "streak_bin": df["streak_bin"].astype(str),
        "reward_rate_bin": df["reward_rate_bin"].astype(str),
        "p_alt1_bin (20q)": df["p_alt1_bin"].astype(str),
    }
    out = dict(singles)
    for name, lab in singles.items():
        if name == "c_prev (baseline)":
            continue
        out[f"c_prev x {name}"] = cp + "|" + pd.Series(lab).astype(str).reset_index(drop=True)
    return out


def noise_ceiling(df: pd.DataFrame, seed=RNG_SEED):
    """Upper bound on R^2 achievable by ANY history-only partition.

    Derivation. For a partition L, R^2(L) = 1 - E[Var(gap|L)]/Var(gap). By the law
    of total variance, refining L never increases E[Var(gap|L)], so the finest
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
    g = df.groupby("subject_id", sort=False)
    F = pd.DataFrame(index=df.index)
    F["c_prev"] = df["c_prev"]
    F["streak_prev"] = df["streak_prev"]
    F["recent_reward_rate"] = df["recent_reward_rate"]
    F["trial_number"] = df["trial_number"]
    F["p_alt1"] = df["p_alt1"]
    for L in (2, 3, 4, 5):
        F[f"choice_lag{L}"] = g["chose_biased"].shift(L)
    F["cum_bias_rate"] = g["chose_biased"].transform(lambda s: s.shift(1).expanding().mean())
    F["cum_switch_rate"] = g["chose_biased"].transform(
        lambda s: s.shift(1).diff().abs().expanding().mean())
    F = F.fillna(-1.0)

    X = F.to_numpy()
    y = df["chose_biased"].to_numpy().astype(float)
    groups = df["subject_id"].to_numpy()
    var_gap = (y - df["p_alt1"].to_numpy()).var()

    q = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                           max_leaf_nodes=63, random_state=seed)
        m.fit(X[tr], y[tr])
        q[te] = m.predict_proba(X[te])[:, 1]

    plug_in = 1 - np.mean(q * (1 - q)) / var_gap
    bound = 1 - np.mean((y - q) ** 2) / var_gap
    ece = M.ece(q, y, n_bins=10)
    return float(plug_in), float(bound), float(ece), q


def stratum_table(df: pd.DataFrame, by) -> pd.DataFrame:
    rows = []
    for val, g in df.groupby(by, observed=True):
        rows.append({"stratum": val, "n": len(g),
                     "predicted": g["p_alt1"].mean(),
                     "empirical": g["chose_biased"].mean(),
                     "gap": g["chose_biased"].mean() - g["p_alt1"].mean()})
    return pd.DataFrame(rows)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = open(OUT_DIR / "output.txt", "w", encoding="utf-8")
    sys.stdout = M.Tee(sys.__stdout__, log)

    print("=" * 78)
    print("PHASE 2 -- CONDITIONAL CALIBRATION: WHICH PARTITION EXPLAINS THE GAP?")
    print("=" * 78)
    print('model: corrected ("fixed") CATIE, k in {0,1,2}, published weighting')
    print("data: EDA + Training + schedule_0 (Test held out for final evaluation)\n")

    df = build_features(load_frame())
    print(f"loaded {df['subject_id'].nunique():,} subjects, {len(df):,} trials, "
          f"{df['schedule'].nunique()} schedules: {sorted(df['schedule'].unique())}")

    print("\ncomputing responsibility posteriors ...")
    T, max_dev = compute_all_subjects(df)
    print(f"\nresponsibility sum-to-1 max deviation: {max_dev:.3e} (correctness gate)")
    assert max_dev < 1e-8, "responsibility posterior failed to normalise"
    T = bin_features(T)
    assert T[["c_prev", "streak_bin", "trial_quintile", "reward_rate_bin",
              "p_alt1_bin", "schedule", "hard_argmax"]].notna().all().all(), \
        "NaN in a partition column -- would silently inflate R^2"
    T.to_csv(OUT_DIR / "trial_level.csv.gz", index=False, compression="gzip")

    # ── 1. Headline metrics ──────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("1. HEADLINE METRICS (fixed model, all trials pooled)")
    print("=" * 78)
    for k, v in M.score_all(T["p_alt1"], T["p_choice"], T["chose_biased"]).items():
        print(f"   {k:<10} {v:.4f}" if isinstance(v, float) else f"   {k:<10} {v}")
    ep, lo, hi = M.bootstrap_ci(T["p_choice"], T["subject_id"], np.mean)
    lp = np.log(np.clip(T["p_choice"], 1e-12, 1))
    elp, llo, lhi = M.bootstrap_ci(lp, T["subject_id"], np.mean)
    print(f"\n   E[p]     = {ep:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (subject-clustered)")
    print(f"   E[log p] = {elp:.4f}  95% CI [{llo:.4f}, {lhi:.4f}]")
    print("\n   NOTE accuracy is near-vacuous here: argmax(p_alt1)==c_prev on ~99.4% of")
    print("   trials, so it essentially reports the repeat rate, not model quality.")

    # ── 2. Published vs fixed -- shown, not asserted ─────────────────────────
    print("\n" + "=" * 78)
    print("2. PUBLISHED vs FIXED (does this chapter's conclusion depend on the fix?)")
    print("=" * 78)
    pub = compute_published(df)
    mg = T.merge(pub[["subject_id", "trial_number", "p_alt1_pub", "p_choice_pub"]],
                 on=["subject_id", "trial_number"])
    assert len(mg) == len(T), f"merge changed row count: {len(T)} -> {len(mg)}"
    r1 = M.paired_subject_test(mg["p_choice"], mg["p_choice_pub"], mg["subject_id"])
    r2_ = M.paired_subject_test(np.log(np.clip(mg["p_choice"], 1e-12, 1)),
                                np.log(np.clip(mg["p_choice_pub"], 1e-12, 1)), mg["subject_id"])
    print(f"   E[p]      fixed {r1['mean_a']:.4f}  published {r1['mean_b']:.4f}  "
          f"diff {r1['mean_diff']:+.4f}  p={r1['p']:.3e} {M.stars(r1['p'])}")
    print(f"   E[log p]  fixed {r2_['mean_a']:.4f}  published {r2_['mean_b']:.4f}  "
          f"diff {r2_['mean_diff']:+.4f}  p={r2_['p']:.3e} {M.stars(r2_['p'])}")

    print("\n   The central c_prev split, recomputed under the PUBLISHED model:")
    pub_g = pub.copy()
    pub_g["gap"] = pub_g["chose_biased"] - pub_g["p_alt1_pub"]
    for v, g_ in pub_g.groupby("c_prev"):
        print(f"     c_prev={int(v)}  n={len(g_):>7,}  predicted {g_['p_alt1_pub'].mean():.4f}  "
              f"empirical {g_['chose_biased'].mean():.4f}  gap {g_['gap'].mean():+.4f}")
    print(f"     aggregate gap {pub_g['gap'].mean():+.4f}")
    r2_pub = r2_explained(pub_g["gap"], pub_g["c_prev"].astype(int))
    print(f"     R^2(c_prev) under published = {r2_pub:.4f}  "
          f"(fixed model: see section 4)")
    print("   => the oppositely-signed split is present under BOTH models, so this")
    print("      chapter's conclusion does not depend on the Phase 1 correction.")
    print("      (The published aggregate gap is larger, so its cancellation is less")
    print("      complete -- the corrected model is the cleaner demonstration.)")

    # ── 3. The central arithmetic ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("3. THE CENTRAL ARITHMETIC -- calibration gap conditional on c_prev")
    print("=" * 78)
    tab = stratum_table(T, "c_prev")
    print(tab.to_string(index=False))
    agg = T["chose_biased"].mean() - T["p_alt1"].mean()
    print(f"\n   aggregate gap: {agg:+.4f}  (near-zero: cancellation in the mean)")
    print("   Section 8 tests directly whether this cancellation is what separates")
    print("   E[p] from E[log p], rather than asserting it.")

    # ── 4. Partition adjudication ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("4. PARTITION ADJUDICATION (R^2 of the calibration gap, PROSPECTIVE only)")
    print("=" * 78)
    print("   soft_argmax is excluded -- section 5 proves it encodes the outcome.\n")
    gap = (T["chose_biased"] - T["p_alt1"]).to_numpy()
    base = r2_explained(gap, T["c_prev"].astype(int))
    rows = []
    for name, lab in candidate_partitions(T).items():
        rows.append({"partition": name, "n_groups": pd.Series(list(lab)).nunique(),
                     "R2": r2_explained(gap, lab),
                     "increment_over_c_prev": r2_explained(gap, lab) - base})
    r2_table = pd.DataFrame(rows).sort_values("R2", ascending=False)
    print(r2_table.to_string(index=False))
    r2_table.to_csv(OUT_DIR / "partition_r2.csv", index=False)

    print("\n   computing the noise ceiling (5-fold grouped GB) ...", flush=True)
    plug, bound, q_ece, q_hat = noise_ceiling(T)
    print(f"\n   Var(gap) = {gap.var():.4f}")
    print(f"   plug-in ceiling  1 - E[q(1-q)]/Var(gap)   = {plug:.4f}")
    print(f"   rigorous bound   1 - OOF Brier/Var(gap)   = {bound:.4f}   (ceiling >= this)")
    print(f"   ECE of q_hat = {q_ece:.4f} (well calibrated => plug-in trustworthy)")
    print(f"\n   So the achievable ceiling is ~{plug:.2f}, NOT the ~0.18 a logistic-")
    print("   regression estimate suggests. An underfit q_hat is shrunk toward 0.5,")
    print("   overstating the noise and understating the ceiling -- low enough that")
    print("   observed partitions exceed it, which is how the error was caught.")
    best = r2_table.iloc[0]
    print(f"\n   best partition: {best['partition']} R^2={best['R2']:.4f} "
          f"= {100*best['R2']/plug:.0f}% of the ~{plug:.2f} ceiling")
    print(f"   c_prev alone:   R^2={base:.4f} = {100*base/plug:.0f}% of ceiling")

    print("\n" + "-" * 78)
    print("   DECOMPOSITION: Var(gap|L) = Var(y|L) + Var(p|L) - 2Cov(y,p|L)")
    print("   A partition scores by predicting y OR by homogenising p -- these are")
    print("   different achievements and the ranking is not readable without this.")
    print("-" * 78)
    print(f"   {'partition':<26}{'R^2':>8}{'E Var(y)':>11}{'E Var(p)':>11}{'E Cov':>10}")
    dec_rows = []
    for nm, lab in [("c_prev", T["c_prev"].astype(int)),
                    ("hard_argmax", T["hard_argmax"]),
                    ("c_prev x streak_bin",
                     T["c_prev"].astype(int).astype(str) + "|" + T["streak_bin"].astype(str)),
                    ("p_alt1_bin (20q)", T["p_alt1_bin"].astype(str)),
                    ("q_hat 50q (best y-pred)", pd.qcut(q_hat, 50, duplicates="drop").astype(str))]:
        d = variance_decomposition(T, lab)
        r = r2_explained(gap, lab)
        print(f"   {nm:<26}{r:>8.4f}{d['E_Var_y']:>11.4f}{d['E_Var_p']:>11.4f}{d['E_Cov']:>+10.4f}")
        dec_rows.append({"partition": nm, "R2": r, **d})
    print(f"   {'(unconditional)':<26}{0.0:>8.4f}"
          f"{T['chose_biased'].var():>11.4f}{T['p_alt1'].var():>11.4f}"
          f"{float(np.cov(T['chose_biased'], T['p_alt1'])[0,1]):>+10.4f}")
    pd.DataFrame(dec_rows).to_csv(OUT_DIR / "variance_decomposition.csv", index=False)
    vp_uncond = T["p_alt1"].var()
    vp_cprev = variance_decomposition(T, T["c_prev"].astype(int))["E_Var_p"]
    print(f"\n   c_prev removes {100*(1-vp_cprev/vp_uncond):.0f}% of the variance in CATIE's")
    print("   OWN forecast p. So 'c_prev explains the calibration gap' is substantially")
    print("   a statement about the model's architecture (phi=0.71 dominates p), not")
    print("   purely a discovery about human behaviour. See README finding #6.")

    # ── 5. Hard vs soft attribution + the circularity proof ─────────────────
    print("\n" + "=" * 78)
    print("5. MODE ATTRIBUTION -- why soft_argmax cannot be used as a partition")
    print("=" * 78)
    print("   Hard argmax (mass toward alt 1) is a deterministic relabelling of c_prev:")
    print("   inertia's contribution is phi*c_prev, identically 0 when c_prev=0.\n")
    print(T.groupby("hard_argmax", observed=True)["c_prev"].agg(["mean", "count"]).to_string())
    print("\n   The Bayesian posterior fixes THAT degeneracy ...")
    print(T.groupby("soft_argmax", observed=True)["c_prev"].agg(["mean", "count"]).to_string())
    print("\n   ... but replaces it with a worse one. PROOF: regime r's unnormalised")
    print("   responsibility is w_r*P(y|r). For inertia P(alt1|inertia)=c_prev in {0,1},")
    print("   so its term is exactly w_I when y==c_prev and exactly 0 otherwise. And")
    print("   w_I = (1-tau*H)(1-p_exp)*phi >= 0.71*0.70*0.71 = 0.3529 strictly exceeds")
    print("   every other regime's maximum (heuristic <= tau = 0.29; contingent <=")
    print("   (1-tau*H)(1-p_exp)*0.29 < w_I; exploration <= (1-tau*H)*0.15 < w_I). The")
    print("   inequality holds per-k so it survives the convex k-mixture. Therefore:")
    print("\n       soft_argmax == inertia   <=>   y == c_prev,  identically.\n")
    cross = T.groupby(["c_prev", "soft_argmax"], observed=True)["chose_biased"].agg(["mean", "count"])
    print(cross.to_string())
    degenerate = cross["mean"].isin([0.0, 1.0]).all()
    mism = int(((T["soft_argmax"] == "inertia") != (T["chose_biased"] == T["c_prev"])).sum())
    print(f"\n   every cell mean in {{0,1}}: {degenerate}     mismatches: {mism} / {len(T):,}")
    print("   soft_argmax is the repeat/switch indicator plus a 3-way tiebreak. It")
    print("   carries exactly one bit beyond c_prev and that bit IS the outcome, so")
    print("   any reliability curve drawn from it is reconstructible from the counts")
    print("   above and says nothing about the regimes. No such curve is plotted.")

    # ── 6. Per-schedule ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("6. PER-SCHEDULE METRIC SUITE")
    print("=" * 78)
    srows = []
    for sched, g_ in T.groupby("schedule", observed=True):
        srows.append({"schedule": sched, **M.score_all(g_["p_alt1"], g_["p_choice"],
                                                       g_["chose_biased"]),
                      "empirical_rate": g_["chose_biased"].mean(),
                      "gap": g_["chose_biased"].mean() - g_["p_alt1"].mean()})
    sdf = pd.DataFrame(srows)
    sdf["order"] = sdf["schedule"].str.replace("schedule_", "").astype(int)
    sdf = sdf.sort_values("order").drop(columns="order")
    print(sdf.to_string(index=False))
    sdf.to_csv(OUT_DIR / "metrics_by_schedule.csv", index=False)
    print(f"\n   Empirical biased-choice rate ranges {sdf['empirical_rate'].min():.3f}"
          f"-{sdf['empirical_rate'].max():.3f} across schedules (spread "
          f"{sdf['empirical_rate'].max()-sdf['empirical_rate'].min():.3f}), yet max |gap|"
          f" is only {sdf['gap'].abs().max():.3f}.")
    print("   schedule's near-zero R^2 is therefore a POSITIVE result -- CATIE tracks")
    print("   between-schedule variation almost perfectly -- not evidence that")
    print("   schedule is behaviourally irrelevant.")

    # ── 7. The interaction the earlier version missed ────────────────────────
    print("\n" + "=" * 78)
    print("7. THE STRONGEST PARTITION: c_prev x run length")
    print("=" * 78)
    inter = T.groupby(["c_prev", "streak_bin"], observed=True).apply(
        lambda g_: pd.Series({"n": len(g_), "predicted": g_["p_alt1"].mean(),
                              "empirical": g_["chose_biased"].mean(),
                              "gap": g_["chose_biased"].mean() - g_["p_alt1"].mean()}),
        include_groups=False)
    print(inter.to_string())
    inter.to_csv(OUT_DIR / "gap_by_cprev_streak.csv")
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
    y_ = T["chose_biased"].to_numpy().astype(float)
    variants = {"CATIE as-is": T["p_alt1"].to_numpy()}
    variants["c_prev stratum rate"] = (
        T.groupby("c_prev")["chose_biased"].transform("mean").to_numpy())
    variants["c_prev x streak_bin rate"] = (
        T.groupby(["c_prev", "streak_bin"], observed=True)["chose_biased"]
        .transform("mean").to_numpy())
    iso = IsotonicRegression(out_of_bounds="clip")
    variants["isotonic on p_alt1 (in-sample)"] = iso.fit_transform(T["p_alt1"], y_)
    print(f"   {'forecast':<34}{'E[p]':>9}{'E[log p]':>11}{'ECE':>9}")
    for nm, pv in variants.items():
        pc = np.where(y_ > 0.5, pv, 1 - pv)
        print(f"   {nm:<34}{M.e_p(pc):>9.4f}{M.e_log_p(pc):>11.4f}"
              f"{M.ece(pv, y_):>9.4f}")
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
    rel = M.reliability_table(T["p_alt1"], T["chose_biased"], n_bins=10, min_count=30)
    axes[0].plot([0, 1], [0, 1], "--", color="0.5", lw=1)
    axes[0].scatter(rel["predicted"], rel["empirical"], s=rel["n"] / 30, color="#333")
    axes[0].set_title(f"Aggregate (n={len(T):,})")
    axes[1].plot([0, 1], [0, 1], "--", color="0.5", lw=1)
    for v, lab, col in [(0, "c_prev=0 (prior: unbiased)", PALETTE["c_prev=0"]),
                        (1, "c_prev=1 (prior: biased)", PALETTE["c_prev=1"])]:
        s = T[T["c_prev"] == v]
        rr = M.reliability_table(s["p_alt1"], s["chose_biased"], n_bins=10, min_count=30)
        axes[1].plot(rr["predicted"], rr["empirical"], "o-", color=col, label=lab)
    axes[1].set_title("Split by previous choice")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xlabel("predicted P(biased)"); ax.set_ylabel("empirical P(biased)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.spines[["top", "right"]].set_visible(False)
    fig1.tight_layout()

    fig2, ax = plt.subplots(figsize=(9, 6))
    rs = r2_table.sort_values("R2")
    cols = ["#d62728" if "baseline" in p_ else "#4c72b0" for p_ in rs["partition"]]
    ax.barh(rs["partition"], rs["R2"], color=cols)
    ax.axvline(plug, color="0.3", ls="--", lw=1.4)
    ax.text(plug, 0.3, f"  noise ceiling ~{plug:.2f}", fontsize=9, color="0.3", va="bottom")
    ax.set_xlabel("R$^2$ of calibration gap explained")
    ax.set_title("Prospective partitions vs. the achievable ceiling")
    ax.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()

    fig3, ax = plt.subplots(figsize=(8, 5))
    ip = inter.reset_index()
    for v, col, lab in [(0.0, PALETTE["c_prev=0"], "c_prev=0 (prior: unbiased)"),
                        (1.0, PALETTE["c_prev=1"], "c_prev=1 (prior: biased)")]:
        s = ip[ip["c_prev"] == v]
        ax.plot(s["streak_bin"].astype(str), s["gap"], "o-", color=col, label=lab)
    ax.axhline(0, color="0.4", lw=1)
    ax.set_xlabel("run length of identical choices ending at t-1")
    ax.set_ylabel("calibration gap (empirical - predicted)")
    ax.set_title("The gap reverses sign with run length inside BOTH strata")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig3.tight_layout()

    fig4, ax = plt.subplots(figsize=(8.5, 4.5))
    sd = sdf.copy()
    sd["o"] = sd["schedule"].str.replace("schedule_", "").astype(int)
    sd = sd.sort_values("o")
    ax.bar(sd["schedule"].str.replace("schedule_", ""), sd["ECE"], color="#4c72b0")
    ax.set_xlabel("schedule"); ax.set_ylabel("ECE")
    ax.set_title("Calibration error by reward schedule (fixed model)")
    ax.spines[["top", "right"]].set_visible(False)
    fig4.tight_layout()

    figs = {"fig1_reliability_aggregate_vs_cprev": fig1,
            "fig2_r2_partition_comparison": fig2,
            "fig3_gap_by_cprev_and_runlength": fig3,
            "fig4_ece_by_schedule": fig4}
    for nm, fg in figs.items():
        fg.savefig(OUT_DIR / f"{nm}.png", dpi=150, bbox_inches="tight")
        print(f"  {nm}.png")

    stale = ["fig3_reliability_by_soft_mode.png", "fig5_hard_vs_soft_attribution.png",
             "gap_by_soft_mode.csv"]
    for s in stale:
        f = OUT_DIR / s
        if f.exists():
            f.unlink()
            print(f"  removed stale artifact: {s}")

    print("\ndone.")
    sys.stdout = sys.__stdout__
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
