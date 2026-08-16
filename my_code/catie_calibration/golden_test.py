"""
Golden test for catie_core -- BLOCKING gate for every downstream analysis.

Checks:
  1. The "published" port reproduces the stored MATLAB output
     (my_code/EDA_set/processing/eda_with_catie_probabilities.csv) to < 1e-12.
  2. A genuinely independent, monolithic, single-pass reference implementation
     (defined below, sharing no code with catie_core.py) agrees with the
     production split -- state_tensors() followed by probability_from_state().
     This is the real test of the refactor into "cache parameter-free state
     once, apply parameters afterward"; see the note on an earlier, vacuous
     version of this check below.
  3. mode_contributions()'s four terms sum to probability_from_state()'s
     output (except trial 1, which is hardcoded to 0.5 and never actually
     uses the mode decomposition).
  4. In "published" mode the heuristic contribution to alternative 1 is
     identically zero; in "fixed" mode it is not.
  5. The trend branch is testable on ~16.9% of trials.

On an earlier version of this check, "check 2" compared
    probability_from_state(st_pub, ...)
to
    probability_from_state(st_pub, ...)
-- the same function called twice on the same object. That is guaranteed to
return zero deviation regardless of whether either function is correct; it
tested determinism, not correctness. It has been replaced by a comparison
against `_reference_catie_probability` below, a hand-written, single-loop
port that never splits state from parameters and does not call anything in
catie_core.py, so a bug in the refactor (as opposed to a bug shared by both
because they were transcribed from the same MATLAB line) has a real chance of
being caught.

Even this test only checks the FINAL probability end-to-end, not the
individual state tensors (H, b, c_prev, s_prev, sbar_prev, g) against MATLAB's
own internal variables. That element-by-element cross-check lives in
matlab/verify_state_tensors.py, against real per-trial data exported from an
instrumented copy of the original .m file (matlab/export_state_tensors.m) --
see that script for the strongest test of state_tensors() in this project.

Run:  python my_code/catie_calibration/golden_test.py
"""

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from catie_core import (  # noqa: E402
    TAU, EPSILON, PHI,
    catie_hetero, catie_probabilities, mode_contributions, mode_weights,
    p_of_observed_choice, probability_from_state, state_tensors,
)

TOLERANCE = 1e-12
REFERENCE = (pathlib.Path(__file__).parent.parent
             / "EDA_set" / "processing" / "eda_with_catie_probabilities.csv")


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


def load_reference():
    df = pd.read_csv(REFERENCE)
    df["c1"] = df["is_biased_choice"].astype(str).str.upper() == "TRUE"
    df = df.sort_values(["subject_file", "trial_number"]).reset_index(drop=True)
    assert len(df) == 49_200, f"expected 49 200 rows, got {len(df)}"
    assert df["catie_choice_probability"].notna().all()
    return df


def main():
    df = load_reference()
    subjects = df["subject_file"].unique()
    print(f"reference: {REFERENCE.name}  ({len(df):,} rows, {len(subjects)} subjects)")

    max_dev = 0.0
    max_ref_dev = 0.0
    max_decomp_dev = 0.0
    max_weights_dev = 0.0
    n_trend = 0
    n_trials_total = 0
    pub_heur_max = 0.0
    fix_heur_max = 0.0

    for subj, d in df.groupby("subject_file", sort=False):
        r1 = d["biased_reward"].to_numpy()
        r2 = d["unbiased_reward"].to_numpy()
        c1 = d["c1"].to_numpy()
        ref = d["catie_choice_probability"].to_numpy()

        # (1) published port vs stored MATLAB output, in P(observed choice) space
        p_alt1 = catie_hetero(r1, r2, c1, mode="published")
        p_choice = p_of_observed_choice(p_alt1, c1)
        max_dev = max(max_dev, np.abs(p_choice - ref).max())

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
    ok = True

    status = "PASS" if max_dev < TOLERANCE else "FAIL"
    ok &= max_dev < TOLERANCE
    print(f"[{status}] published port vs stored MATLAB      : max |dev| = {max_dev:.3e} "
          f"(tol {TOLERANCE:.0e})")

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
