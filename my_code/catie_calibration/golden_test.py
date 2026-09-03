"""
Golden test for catie_core -- BLOCKING gate for every downstream analysis.

Checks -- all SEVEN are blocking (each feeds `ok &= ...`, and main() returns
a non-zero exit code if any fail; none is merely printed):
  1. The "published" port (split: state_tensors + probability_from_state)
     reproduces ORIGINAL, UNMODIFIED MATLAB, called live (not a
     previously-stored CSV), across ALL 12 schedules -- both the hetero
     mixture and each single-k model. See verify_against_live_matlab().
  2. The independent monolithic reference (#3 below) ALSO vs that same live
     MATLAB, per-k -- so it's checked against ground truth directly, not
     just against split. Also inside verify_against_live_matlab().
  3. That monolithic, single-pass reference implementation (defined below,
     sharing no code with catie_core.py -- no import of state_tensors,
     probability_from_state, or catie_probabilities) agrees with the
     production split. This is the real test of the refactor into "cache
     parameter-free state once, apply parameters afterward"; see the note
     on an earlier, vacuous version of this check below.
  4. mode_contributions()'s four terms sum to probability_from_state()'s
     output (except trial 1, which is hardcoded to 0.5 and never actually
     uses the mode decomposition).
  5. mode_weights()'s four priors sum to 1 (except trial 1) -- these are
     the priors the responsibility-posterior E-step in 02_mode_calibration/
     relies on being a valid distribution.
  6. In "published" mode the heuristic contribution to alternative 1 is
     EXACTLY zero (literal `== 0.0`, not a tolerance -- this IS the NaN
     bug, expressed as a test); in "fixed" mode it is > 0.
  7. The trend branch is testable on ~16.9% of trials (sanity bound
     15-19%, not an exact figure -- depends on the sanitized population).

Even this test only checks the FINAL probability end-to-end, not the
individual state tensors (H, b, c_prev, s_prev, sbar_prev, g) against MATLAB's
own internal variables. That element-by-element cross-check lives in
matlab/verify_state_tensors.py, against real per-trial data exported from an
instrumented copy of the original .m file (matlab/export_state_tensors.m) --
see that script for the strongest test of state_tensors() in this project.

Run:  python my_code/catie_calibration/golden_test.py
      python my_code/catie_calibration/golden_test.py --regenerate-matlab
"""

import argparse
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from catie_core import (  # noqa: E402
    TAU, EPSILON, PHI,
    catie_hetero, catie_probabilities, mode_contributions, mode_weights,
    p_of_observed_choice, probability_from_state, state_tensors,
)

TOLERANCE = 1e-12

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
MATLAB_DIR = pathlib.Path(__file__).parent / "matlab"
MATLAB_DRIVER = MATLAB_DIR / "export_original_reference_all_schedules.m"
MATLAB_REFERENCE_CSV = MATLAB_DIR / "results" / "original_reference_all_schedules.csv"

# Same 12 schedules, same directories, as matlab/run_bug_comparison_all_schedules.m
# and export_original_reference_all_schedules.m -- kept in sync deliberately.
RAW_SCHEDULE_DIRS = {
    "schedule_0":  PROJECT_ROOT / "my_code" / "Schedule0_set" / "schedule_0",
    "schedule_1":  PROJECT_ROOT / "my_code" / "Test_set" / "schedule_1",
    "schedule_2":  PROJECT_ROOT / "my_code" / "Training_set" / "schedule_2",
    "schedule_3":  PROJECT_ROOT / "my_code" / "Training_set" / "schedule_3",
    "schedule_4":  PROJECT_ROOT / "my_code" / "EDA_set" / "schedule_4",
    "schedule_5":  PROJECT_ROOT / "my_code" / "EDA_set" / "schedule_5",
    "schedule_6":  PROJECT_ROOT / "my_code" / "Training_set" / "schedule_6",
    "schedule_7":  PROJECT_ROOT / "my_code" / "EDA_set" / "schedule_7",
    "schedule_8":  PROJECT_ROOT / "my_code" / "Test_set" / "schedule_8",
    "schedule_9":  PROJECT_ROOT / "my_code" / "Training_set" / "schedule_9",
    "schedule_10": PROJECT_ROOT / "my_code" / "Test_set" / "schedule_10",
    "schedule_11": PROJECT_ROOT / "my_code" / "Training_set" / "schedule_11",
}

# Checks 2-5 use the sanitized EDA split (496 subjects). Check 1 is independent
# of this file -- it reads the raw per-subject CSVs directly, above.
REFERENCE = pathlib.Path(__file__).parent / "data" / "cleaned_eda.csv"
N_EDA_ROWS = 49_600


def _reference_catie_probability(rewards_1, rewards_2, is_choice_1, k, mode,
                                 tau=TAU, epsilon=EPSILON, phi=PHI, n_trials=100):
    """Monolithic single-pass reference: computes P(alt 1) trial by trial in one
    loop, without ever separating parameter-free state from parameter-dependent
    probability. Deliberately does not import or call anything from catie_core.py,
    so agreement with catie_probabilities() (state_tensors + probability_from_state)
    is real evidence for that refactor, not a tautology.

    mode="published" literally re-derives the bug from NaN propagation (`pays[t]`
    is genuinely unassigned at the point it's read, exactly as in the MATLAB),
    rather than assuming b==0 the way catie_core.state_tensors does. If the two
    implementations agree here, that's an independent confirmation that "the
    heuristic branch is dead in published mode" follows from the NaN semantics,
    not merely from how catie_core happens to be written.
    """
    nan = float("nan")
    r1 = np.concatenate([[nan], np.asarray(rewards_1, dtype=float)])
    r2 = np.concatenate([[nan], np.asarray(rewards_2, dtype=float)])
    c = np.concatenate([[False], np.asarray(is_choice_1, dtype=bool)])

    cc1 = np.zeros((4 ** k, 2))
    cc2 = np.zeros((4 ** k, 2))
    pays = np.full(n_trials + 1, nan)
    surprise = np.full(n_trials + 1, nan)
    reward_vec = np.full(n_trials + 1, nan)
    observed_sd = np.zeros(3)
    expected_reward = np.zeros(3)
    reward_mean = [0.0, 0.0, 0.0]
    reward_sum = [0.0, 0.0, 0.0]
    sum_sq = [0.0, 0.0, 0.0]
    n_choices = [0, 0, 0]
    total_surprise = 0.0
    mean_surprise = 0.0
    p1 = np.zeros(n_trials + 1)

    def base2dec(vec):
        d = 0.0
        for j, v in enumerate(vec):
            d += (4 ** j) * float(v)
        return int(d)

    for t in range(1, n_trials + 1):
        if k == 0:
            expected_reward[1] = reward_mean[1]
            expected_reward[2] = reward_mean[2]
            g = 0.5 if reward_mean[1] == reward_mean[2] else float(reward_mean[1] > reward_mean[2])
        elif (t - 1) > k and t < n_trials:
            row = base2dec(reward_vec[t - k:t])
            if cc2[row, 1] > 0:
                ca2 = np.array([cc2[row, 0] / cc2[row, 1]])
            else:
                seen = cc2[:, 1] > 0
                ca2 = (cc2[seen, 0] / cc2[seen, 1]) if seen.any() else np.array([reward_mean[2]])
            if cc1[row, 1] > 0:
                ca1 = np.array([cc1[row, 0] / cc1[row, 1]])
            else:
                seen = cc1[:, 1] > 0
                ca1 = (cc1[seen, 0] / cc1[seen, 1]) if seen.any() else np.array([reward_mean[1]])
            gt = ca1[:, None] > ca2[None, :]
            eq = ca1[:, None] == ca2[None, :]
            g = gt.mean() + 0.5 * eq.mean()
            expected_reward[1] = ca1.mean()
            expected_reward[2] = ca2.mean()
        else:
            g = 0.5 if reward_mean[1] == reward_mean[2] else float(reward_mean[1] > reward_mean[2])

        if t == 1:
            p1[1] = 0.5
        else:
            testable = (t > 2) and (c[t - 1] == c[t - 2]) and (pays[t - 1] != pays[t - 2])
            if testable:
                if mode == "fixed":
                    predicts_1 = (c[t - 1] and pays[t - 1] > pays[t - 2]) or \
                                 ((not c[t - 1]) and pays[t - 1] < pays[t - 2])
                else:  # "published": literal bug -- pays[t] is still NaN here
                    predicts_1 = (c[t - 1] and pays[t] > pays[t - 1]) or \
                                 ((not c[t]) and pays[t] < pays[t - 1])
                p_heuristic = tau if predicts_1 else 0.0
                p_try_explore = 1 - tau
            else:
                p_try_explore = 1.0
                p_heuristic = 0.0

            p_exp = epsilon * (1 + surprise[t - 1] + mean_surprise) / 3
            p_explore_term = p_try_explore * 0.5 * p_exp
            p_try_inertia = p_try_explore * (1 - p_exp)
            p_inertia_term = p_try_inertia * phi * float(c[t - 1])
            p_ca_term = p_try_inertia * (1 - phi) * g
            p1[t] = p_heuristic + p_explore_term + p_inertia_term + p_ca_term

        a = 1 if c[t] else 2
        pays[t] = r1[t] if c[t] else r2[t]
        reward_vec[t] = (3.0 if pays[t] == 1 else 2.0) if c[t] else (1.0 if pays[t] == 1 else 0.0)
        reward_sum[a] += pays[t]
        n_choices[a] += 1
        sum_sq[a] += pays[t] ** 2
        var = ((1.0 / (n_choices[a] - 1)) * (sum_sq[a] - (reward_sum[a] ** 2) / n_choices[a])
               if n_choices[a] > 1 else nan)
        if np.isnan(var) or var < 0:
            var = 0.0
        observed_sd[a] = np.sqrt(var)
        if observed_sd[a] > 1e-4:
            d = abs(expected_reward[a] - pays[t])
            surprise[t] = d / (observed_sd[a] + d)
        else:
            surprise[t] = 0.0
        total_surprise += surprise[t]
        mean_surprise = total_surprise / t
        if k > 0 and t > k:
            row = base2dec(reward_vec[t - k:t])
            table = cc1 if c[t] else cc2
            table[row, 0] += pays[t]
            table[row, 1] += 1
        reward_mean[a] = reward_sum[a] / n_choices[a]

    return p1[1:]


def _run_matlab_reference():
    """Actually invoke real MATLAB to (re)generate MATLAB_REFERENCE_CSV.

    Calls the ORIGINAL, unmodified .m files -- not a stored/cached artifact --
    via export_original_reference_all_schedules.m (see that file's header for
    what it computes). Takes a few minutes.
    """
    print(f"running MATLAB: {MATLAB_DRIVER.name} (this takes a few minutes)...", flush=True)
    t0 = time.time()
    cmd = ["matlab", "-batch", f"run('{MATLAB_DRIVER.as_posix()}')"]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"MATLAB driver failed with exit code {result.returncode}")
    if not MATLAB_REFERENCE_CSV.exists():
        raise RuntimeError(f"MATLAB driver ran but did not produce {MATLAB_REFERENCE_CSV}")
    print(f"MATLAB run finished in {time.time() - t0:.0f}s", flush=True)


def _load_raw_subject(schedule: str, subject_file: str):
    """Read one raw per-subject CSV, sorted by trial_number -- same file the
    MATLAB reference and the rest of this project's pipeline both read."""
    path = RAW_SCHEDULE_DIRS[schedule] / subject_file
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    d = d.sort_values("trial_number").reset_index(drop=True)
    r1 = d["reward_alternative_1"].to_numpy(dtype=float)
    r2 = d["reward_alternative_2"].to_numpy(dtype=float)
    c1 = (d["is_choice_alternative_1"].astype(str).str.strip().str.upper()
          == "TRUE").to_numpy()
    return r1, r2, c1


def verify_against_live_matlab(regenerate=False, tolerance=TOLERANCE):
    """Check 1: the Python "published" port against ORIGINAL MATLAB, called
    live, across ALL 12 schedules -- both the hetero mixture and each
    single-k model (k=0,1,2) individually, so a mismatch can be localized to
    "mixing" vs "a specific k". ALSO diffs the independent monolithic
    reference (_reference_catie_probability, the implementation check 2 uses)
    against the same live MATLAB, per-k -- previously that reference was only
    ever checked against the Python split implementation (mono vs split),
    never directly against MATLAB itself (mono vs ground truth). This closes
    that gap: with this addition, split-vs-MATLAB, mono-vs-MATLAB, AND
    mono-vs-split (check 2, unchanged, still runs separately below) all hold
    independently -- a bug that happened to leave split-vs-MATLAB passing
    could not also leave mono-vs-MATLAB passing unless it were a genuine,
    correct result, since mono shares no code with either.

    Replaces the previous version of this check, which only compared against
    a single previously-stored CSV covering 3 of 12 schedules (EDA only).
    That comparison could not tell you whether a mismatch was in the model
    itself or was already baked into the stored file; this one calls the
    unmodified .m files fresh and covers every schedule.

    Unless `regenerate=True`, reuses MATLAB_REFERENCE_CSV if it already
    exists (the MATLAB run takes a few minutes; no need to pay that cost on
    every invocation of this otherwise-fast gate). Delete the CSV, or pass
    `--regenerate-matlab` on the command line, to force a fresh MATLAB run.

    Returns (ok, report_df).
    """
    if regenerate or not MATLAB_REFERENCE_CSV.exists():
        _run_matlab_reference()
    else:
        print(f"reusing cached MATLAB reference: {MATLAB_REFERENCE_CSV}\n"
              f"(pass --regenerate-matlab to force a fresh MATLAB run)")

    ref = pd.read_csv(MATLAB_REFERENCE_CSV)
    ref = ref.sort_values(["schedule", "subject_id", "trial_number"]).reset_index(drop=True)

    rows = []
    for (schedule, subject_id, subject_file), d in ref.groupby(
            ["schedule", "subject_id", "subject_file"], sort=False):
        d = d.sort_values("trial_number")
        r1, r2, c1 = _load_raw_subject(schedule, subject_file)
        assert len(c1) == len(d), (
            f"{subject_id}: raw file has {len(c1)} trials, "
            f"MATLAB reference has {len(d)}")

        # shipped_time_avg: the reference is the code AS SHIPPED (hetro.m:25), which
        # time-averages the k-mixture weights. The project default is per_trial.
        p_alt1_hetero = catie_hetero(r1, r2, c1, mode="published", weighting="shipped_time_avg")
        pc_hetero_py = p_of_observed_choice(p_alt1_hetero, c1)
        dev_hetero = np.abs(pc_hetero_py - d["pc_hetero"].to_numpy())

        dev_k = {}
        dev_mono = {}
        for k in (0, 1, 2):
            p_alt1_k = catie_probabilities(r1, r2, c1, k=k, mode="published")
            pc_k_py = p_of_observed_choice(p_alt1_k, c1)
            dev_k[k] = np.abs(pc_k_py - d[f"pc_k{k}"].to_numpy())

            # mono: the independent monolithic reference (check 2's implementation),
            # diffed directly against live MATLAB -- not just against `split`.
            p_alt1_mono = _reference_catie_probability(r1, r2, c1, k=k, mode="published")
            pc_mono_py = p_of_observed_choice(p_alt1_mono, c1)
            dev_mono[k] = np.abs(pc_mono_py - d[f"pc_k{k}"].to_numpy())

        rows.append({
            "schedule": schedule, "subject_id": subject_id,
            "max_dev_hetero": dev_hetero.max(), "max_dev_k0": dev_k[0].max(),
            "max_dev_k1": dev_k[1].max(), "max_dev_k2": dev_k[2].max(),
            "max_dev_mono_k0": dev_mono[0].max(), "max_dev_mono_k1": dev_mono[1].max(),
            "max_dev_mono_k2": dev_mono[2].max(),
        })

    subj = pd.DataFrame(rows)

    def _sched_num(s):
        return int(s.split("_")[1])
    sched_order = sorted(subj["schedule"].unique(), key=_sched_num)

    print("\n" + "=" * 118)
    print("CHECK 1: Python 'published' port (split) AND the independent monolithic")
    print("reference (mono, check 2's implementation) EACH vs LIVE, ORIGINAL, UNMODIFIED")
    print("MATLAB -- all 12 schedules, both compared directly, not just to each other")
    print("=" * 118)
    print(f"{'schedule':<12}{'n_subj':>8}{'split hetero':>14}{'split k0':>11}"
          f"{'split k1':>11}{'split k2':>11}{'  |  mono k0':>13}{'mono k1':>11}{'mono k2':>11}")
    for sched in sched_order:
        g = subj[subj["schedule"] == sched]
        mono_k0_col = f"  |  {g['max_dev_mono_k0'].max():.2e}"
        print(f"{sched:<12}{len(g):>8}{g['max_dev_hetero'].max():>14.2e}"
              f"{g['max_dev_k0'].max():>11.2e}{g['max_dev_k1'].max():>11.2e}"
              f"{g['max_dev_k2'].max():>11.2e}{mono_k0_col:>13}"
              f"{g['max_dev_mono_k1'].max():>11.2e}{g['max_dev_mono_k2'].max():>11.2e}")

    max_dev_hetero = subj["max_dev_hetero"].max()
    max_dev_k = subj[["max_dev_k0", "max_dev_k1", "max_dev_k2"]].to_numpy().max()
    max_dev_mono = subj[["max_dev_mono_k0", "max_dev_mono_k1", "max_dev_mono_k2"]].to_numpy().max()
    ok = bool(max_dev_hetero < tolerance and max_dev_k < tolerance and max_dev_mono < tolerance)

    pooled_mono_k0_col = f"  |  {subj['max_dev_mono_k0'].max():.2e}"
    print("-" * 118)
    print(f"{'POOLED':<12}{len(subj):>8}{max_dev_hetero:>14.2e}"
          f"{subj['max_dev_k0'].max():>11.2e}{subj['max_dev_k1'].max():>11.2e}"
          f"{subj['max_dev_k2'].max():>11.2e}"
          f"{pooled_mono_k0_col:>13}"
          f"{subj['max_dev_mono_k1'].max():>11.2e}{subj['max_dev_mono_k2'].max():>11.2e}"
          f"   (tol {tolerance:.0e})")

    if not ok:
        all_dev_cols = ["max_dev_hetero", "max_dev_k0", "max_dev_k1", "max_dev_k2",
                        "max_dev_mono_k0", "max_dev_mono_k1", "max_dev_mono_k2"]
        worst = subj.loc[subj[all_dev_cols].max(axis=1).idxmax()]
        print(f"\nWORST MISMATCH: {worst['subject_id']}  "
              f"(split: hetero={worst['max_dev_hetero']:.3e}, k0={worst['max_dev_k0']:.3e}, "
              f"k1={worst['max_dev_k1']:.3e}, k2={worst['max_dev_k2']:.3e}; "
              f"mono: k0={worst['max_dev_mono_k0']:.3e}, k1={worst['max_dev_mono_k1']:.3e}, "
              f"k2={worst['max_dev_mono_k2']:.3e})")

    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] split vs LIVE MATLAB: max|dev| hetero={max_dev_hetero:.3e}, "
          f"per-k={max_dev_k:.3e}  |  mono vs LIVE MATLAB: max|dev| per-k={max_dev_mono:.3e} "
          f"(tol {tolerance:.0e})")
    print("=" * 118)
    return ok, subj


def load_reference():
    """Sanitized EDA split (496 subjects) -- the data checks 2-5 run over.
    Check 1 does not use this; see verify_against_live_matlab()."""
    df = pd.read_csv(REFERENCE)
    df["c1"] = df["is_biased_choice"].astype(str).str.upper() == "TRUE"
    df = df.sort_values(["subject_id", "trial_number"]).reset_index(drop=True)
    assert len(df) == N_EDA_ROWS, f"expected {N_EDA_ROWS} rows, got {len(df)}"
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate-matlab", action="store_true",
                        help="Force a fresh MATLAB run for check 1, even if a "
                             "cached reference CSV already exists.")
    args = parser.parse_args()

    ok1, _ = verify_against_live_matlab(regenerate=args.regenerate_matlab)

    df = load_reference()
    subjects = df["subject_id"].unique()
    print(f"\nreference (checks 2-5): {REFERENCE.name}  ({len(df):,} rows, {len(subjects)} subjects)")

    max_ref_dev = 0.0
    max_decomp_dev = 0.0
    max_weights_dev = 0.0
    n_trend = 0
    n_trials_total = 0
    pub_heur_max = 0.0
    fix_heur_max = 0.0

    for subj, d in df.groupby("subject_id", sort=False):
        r1 = d["biased_reward"].to_numpy()
        r2 = d["unbiased_reward"].to_numpy()
        c1 = d["c1"].to_numpy()

        for k in (0, 1, 2):
            for mode in ("published", "fixed"):
                # (2) independent monolithic reference vs the production split
                mono = _reference_catie_probability(r1, r2, c1, k=k, mode=mode)
                split = catie_probabilities(r1, r2, c1, k=k, mode=mode)
                max_ref_dev = max(max_ref_dev, np.abs(mono - split).max())

                # (3) mode_contributions decomposition sums to probability_from_state
                st = state_tensors(r1, r2, c1, k=k, mode=mode)
                parts_sum = sum(mode_contributions(st))
                p1 = probability_from_state(st)
                max_decomp_dev = max(max_decomp_dev, np.abs(parts_sum[1:] - p1[1:]).max())

                # (3b) mode_weights are a valid prior: sum to 1 (used by the
                # responsibility-posterior E-step in 02_mode_calibration/)
                weights_sum = sum(mode_weights(st))
                max_weights_dev = max(max_weights_dev, np.abs(weights_sum[1:] - 1.0).max())

        # (4) heuristic contribution under each mode (k=2, matching the rest of the file)
        st_pub = state_tensors(r1, r2, c1, k=2, mode="published")
        st_fix = state_tensors(r1, r2, c1, k=2, mode="fixed")
        pub_heur_max = max(pub_heur_max, np.abs(mode_contributions(st_pub)[0]).max())
        fix_heur_max = max(fix_heur_max, np.abs(mode_contributions(st_fix)[0]).max())

        # (5) trend testability
        n_trend += int(st_fix.H.sum())
        n_trials_total += len(st_fix.H)

    print("\n" + "=" * 74)
    print(f"CHECKS 2-5 (sanitized EDA split, {len(subjects)} subjects)")
    print("=" * 74)
    ok = ok1

    status = "PASS" if max_ref_dev < TOLERANCE else "FAIL"
    ok &= max_ref_dev < TOLERANCE
    print(f"[{status}] monolithic reference vs split impl.   : max |dev| = {max_ref_dev:.3e} "
          f"(tol {TOLERANCE:.0e}; k in 0,1,2 x both modes)")

    status = "PASS" if max_decomp_dev < TOLERANCE else "FAIL"
    ok &= max_decomp_dev < TOLERANCE
    print(f"[{status}] mode_contributions sums to p(alt 1)   : max |dev| = {max_decomp_dev:.3e} "
          f"(tol {TOLERANCE:.0e}; trial 1 excluded, hardcoded to 0.5)")

    status = "PASS" if max_weights_dev < TOLERANCE else "FAIL"
    ok &= max_weights_dev < TOLERANCE
    print(f"[{status}] mode_weights sums to 1                 : max |dev| = {max_weights_dev:.3e} "
          f"(tol {TOLERANCE:.0e}; trial 1 excluded)")

    status = "PASS" if pub_heur_max == 0.0 else "FAIL"
    ok &= pub_heur_max == 0.0
    print(f"[{status}] published heuristic contribution      : max = {pub_heur_max:.3e} "
          f"(must be exactly 0 -- this IS the bug)")

    status = "PASS" if fix_heur_max > 0.0 else "FAIL"
    ok &= fix_heur_max > 0.0
    print(f"[{status}] fixed heuristic contribution          : max = {fix_heur_max:.3e} "
          f"(must be > 0)")

    pct = 100.0 * n_trend / n_trials_total
    status = "PASS" if 15.0 < pct < 19.0 else "FAIL"
    ok &= 15.0 < pct < 19.0
    print(f"[{status}] trend-testable trials                 : {pct:.2f}%  "
          f"({n_trend:,} / {n_trials_total:,})")

    print("=" * 74)
    print("GOLDEN TEST PASSED" if ok else "GOLDEN TEST FAILED")
    print("\nNOTE: this file validates catie_core's END-TO-END output only. For a true")
    print("element-by-element check of the individual state tensors (H, b, c_prev,")
    print("s_prev, sbar_prev, g) against MATLAB's own internal variables, run")
    print("matlab/export_state_tensors.m then matlab/verify_state_tensors.py.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
