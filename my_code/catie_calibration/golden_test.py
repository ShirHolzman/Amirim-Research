"""
Golden test for catie_core -- BLOCKING gate for every downstream analysis.

Checks:
  1. The "published" port reproduces the stored MATLAB output
     (my_code/EDA_set/processing/eda_with_catie_probabilities.csv) to < 1e-12.
  2. The cached-tensor path reproduces the direct-port path bitwise.
  3. In "published" mode the heuristic contribution to alternative 1 is identically
     zero; in "fixed" mode it is not.
  4. The trend branch is testable on ~16.9% of trials.

Run:  python my_code/catie_calibration/golden_test.py
"""

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from catie_core import (  # noqa: E402
    TAU, EPSILON, PHI,
    catie_hetero, mode_contributions, p_of_observed_choice,
    probability_from_state, state_tensors,
)

TOLERANCE = 1e-12
REFERENCE = (pathlib.Path(__file__).parent.parent
             / "EDA_set" / "processing" / "eda_with_catie_probabilities.csv")


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
    max_cache_dev = 0.0
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

        # (2) cached-tensor path vs direct port (k=2 agent)
        st_pub = state_tensors(r1, r2, c1, k=2, mode="published")
        st_fix = state_tensors(r1, r2, c1, k=2, mode="fixed")
        direct = probability_from_state(st_pub, TAU, EPSILON, PHI)
        cached = probability_from_state(st_pub, TAU, EPSILON, PHI)
        max_cache_dev = max(max_cache_dev, np.abs(direct - cached).max())

        # (3) heuristic contribution under each mode
        pub_heur_max = max(pub_heur_max, np.abs(mode_contributions(st_pub)[0]).max())
        fix_heur_max = max(fix_heur_max, np.abs(mode_contributions(st_fix)[0]).max())

        # (4) trend testability
        n_trend += int(st_fix.H.sum())
        n_trials_total += len(st_fix.H)

    print("\n" + "=" * 70)
    ok = True

    status = "PASS" if max_dev < TOLERANCE else "FAIL"
    ok &= max_dev < TOLERANCE
    print(f"[{status}] published port vs stored MATLAB : max |dev| = {max_dev:.3e} "
          f"(tol {TOLERANCE:.0e})")

    status = "PASS" if max_cache_dev == 0.0 else "FAIL"
    ok &= max_cache_dev == 0.0
    print(f"[{status}] cached path vs direct port      : max |dev| = {max_cache_dev:.3e} "
          f"(must be exactly 0)")

    status = "PASS" if pub_heur_max == 0.0 else "FAIL"
    ok &= pub_heur_max == 0.0
    print(f"[{status}] published heuristic contribution: max = {pub_heur_max:.3e} "
          f"(must be exactly 0 -- this IS the bug)")

    status = "PASS" if fix_heur_max > 0.0 else "FAIL"
    ok &= fix_heur_max > 0.0
    print(f"[{status}] fixed heuristic contribution    : max = {fix_heur_max:.3e} "
          f"(must be > 0)")

    pct = 100.0 * n_trend / n_trials_total
    status = "PASS" if 15.0 < pct < 19.0 else "FAIL"
    ok &= 15.0 < pct < 19.0
    print(f"[{status}] trend-testable trials           : {pct:.2f}%  "
          f"({n_trend:,} / {n_trials_total:,})")

    print("=" * 70)
    print("GOLDEN TEST PASSED" if ok else "GOLDEN TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
