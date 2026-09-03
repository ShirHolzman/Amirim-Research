"""
Isolated verification that catie_core.mix_agents() reproduces the original
MATLAB's Bayesian-model-averaging (k-mixture) arithmetic exactly.

WHY THIS IS NOT ALREADY COVERED. golden_test.py check 1 compares the END of the
pipeline -- state recursion + per-trial probability + BMA mixing -- against live
MATLAB. It would pass even if the state recursion and the mixing contained
compensating errors. verify_state_tensors.py covers the state recursion in
isolation. This file closes the remaining gap: it takes the per-agent
probability matrix MATLAB ITSELF produced, feeds that identical matrix to
mix_agents(), and compares only the mixing.

THE SUBSTANTIVE THING UNDER TEST -- the two implementations mix in DIFFERENT
PROBABILITY SPACES:

  COMPETITION_CATIE_schedule_choice_probability.m returns P(CHOICE ACTUALLY
  MADE) (its lines 136-139: `if is_choice_1(trial), p = current_p_choice_1;
  else p = 1-current_p_choice_1`). So hetro.m:25 forms its weighted average in
  CHOICE space:
        p_decisions = mean(agents_choice_probabilities' * W, 2)

  catie_core.mix_agents() instead keeps P(alt 1) and mixes in ALT-1 space:
        mixed = (P.T @ W).mean(axis=1)
  converting to choice space only afterwards via p_of_observed_choice().

These agree only because the BMA weights form a convex combination (sum to 1
per column), which makes "mix then flip" == "flip then mix":
        sum_a w_a (1 - P_a) = 1 - sum_a w_a P_a     iff  sum_a w_a = 1
That identity is the translation's load-bearing assumption. This script checks
it numerically rather than trusting the algebra -- and checks the weights really
do sum to 1, which is what licenses it.

MUTATION CONTROLS. A pass is only meaningful if the test could have failed, so
four deliberately-wrong mixing rules are run through the same comparison. Each
is a plausible translation error; all four must show a clearly nonzero deviation.

Prerequisite: run the exporter in MATLAB first --
    matlab -batch "run('my_code/catie_calibration/matlab/export_bma_mixing_inputs.m')"

Run:  python my_code/catie_calibration/matlab/verify_bma_mixing.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from catie_core import mix_agents, p_of_observed_choice  # noqa: E402

MATLAB_CSV = HERE / "results" / "bma_mixing_reference.csv"

# The exporter writes %.17g, which round-trips an IEEE double exactly, so there
# is no text-rounding floor here as there is in verify_state_tensors.py. This is
# a pure floating-point-arithmetic comparison and should land at machine epsilon.
TOLERANCE = 1e-12

K_COLS = ["p_k0", "p_k1", "p_k2"]


def published_mix_in_choice_space(P_choice: np.ndarray) -> np.ndarray:
    """hetro.m:20-25 transcribed LITERALLY, operating in choice space like the
    original -- including its (n_trials x n_trials) intermediate and the
    mean(...,2) that time-averages the weights. Used as a second, independent
    reference so that a shared misreading of the MATLAB would have to occur
    twice, in two different probability spaces, to go undetected."""
    n_agents = P_choice.shape[0]
    L = np.cumprod(np.hstack([np.ones((n_agents, 1)), P_choice]), axis=1)
    W = L / L.sum(axis=0, keepdims=True)
    W = W[:, :-1]
    return (P_choice.T @ W).mean(axis=1)


# ---- mutation controls: each is a plausible mistranslation of hetro.m -------

def _normalised_cumprod(P_choice):
    """hetro.m:20-21 -- the (n_agents, n_trials+1) normalised weight matrix,
    BEFORE hetro.m:22 drops its last column."""
    n_agents = P_choice.shape[0]
    L = np.cumprod(np.hstack([np.ones((n_agents, 1)), P_choice]), axis=1)
    return L / L.sum(axis=0, keepdims=True)


def mut_lookahead(P_alt1, P_choice):
    """Weights not lagged: uses columns 1..end instead of hetro.m:22's
    `(:, 1:end-1)`, so trial t's weight already contains trial t's own outcome.

    NOTE -- this is the SAME error as "forgot the ones(...) prior column in
    hetro.m:20". Prepending the flat prior and then dropping the last column is
    precisely what lags the weight matrix by one trial; the prior column IS the
    lag. Both mistranslations produce a bit-identical weight matrix, verified in
    main(). They are therefore one control, not two.
    """
    W = _normalised_cumprod(P_choice)[:, 1:]
    return (P_alt1.T @ W).mean(axis=1)


def mut_likelihood_in_alt1_space(P_alt1, P_choice):
    """Accumulated the wrong likelihood: ran the cumprod over P(alt 1) instead of
    P(choice made). A highly plausible translation slip, since mix_agents()
    receives P(alt 1) and must convert to choice space itself (catie_core.py:393)
    -- forgetting that one line lands exactly here."""
    W = _normalised_cumprod(P_alt1)[:, :-1]
    return (P_alt1.T @ W).mean(axis=1)


def mut_per_trial(P_alt1, P_choice):
    """Used the per-trial weighting hetro.m's own commented-out line 26 describes,
    instead of the time-averaged one line 25 actually runs. NOTE: per-trial is the
    PROJECT DEFAULT (it is what the paper's numbers match); it is a 'mutation' here
    only relative to the shipped MATLAB this script validates against."""
    W = _normalised_cumprod(P_choice)[:, :-1]
    return (P_alt1 * W).sum(axis=0)


def mut_uniform(P_alt1, P_choice):
    """No Bayesian updating at all -- flat 1/3 weights (the prior, never moved)."""
    return P_alt1.mean(axis=0)


MUTATIONS = [
    ("un-lagged weights (== dropping hetro.m:20's prior column)", mut_lookahead),
    ("cumprod over P(alt 1) instead of P(choice made)", mut_likelihood_in_alt1_space),
    ("per-trial weights (hetro.m:26, the commented-out line)", mut_per_trial),
    ("uniform 1/3 weights (no BMA at all)", mut_uniform),
]


def main() -> int:
    if not MATLAB_CSV.exists():
        print(f"MISSING: {MATLAB_CSV}")
        print("Run:  matlab -batch \"run('my_code/catie_calibration/matlab/"
              "export_bma_mixing_inputs.m')\"")
        return 1

    df = pd.read_csv(MATLAB_CSV)
    print(f"MATLAB reference : {MATLAB_CSV}")
    print(f"rows             : {len(df):,}")

    groups = list(df.groupby(["schedule", "subject"], sort=False))
    print(f"subjects         : {len(groups):,}")
    print(f"schedules        : {df['schedule'].nunique()}")
    print(f"tolerance        : {TOLERANCE:g}\n")

    dev_main = []        # mix_agents (alt-1 space) vs MATLAB
    dev_literal = []     # literal choice-space transcription vs MATLAB
    dev_space = []       # the two Python routes vs each other
    dev_wsum = []        # |sum_a W[a,t] - 1|
    dev_wbarsum = []     # |sum_a w_bar[a] - 1|
    dev_prior_is_lag = []  # "no prior column" vs "no lag" -- same matrix?
    dev_mut = {name: [] for name, _ in MUTATIONS}

    for (_sched, _subj), g in groups:
        g = g.sort_values("trial")
        c = g["is_choice_1"].to_numpy().astype(bool)
        P_choice = g[K_COLS].to_numpy().T          # (3, 100) as MATLAB produced it
        p_hetro = g["p_hetro"].to_numpy()          # MATLAB's BMA output

        # MATLAB gives P(choice made); recover P(alt 1), which is what
        # mix_agents() consumes. Exact: a single 1-x flip, no arithmetic loss.
        P_alt1 = np.where(c[None, :], P_choice, 1.0 - P_choice)

        # --- the actual thing under test -----------------------------------
        mixed_alt1, W = mix_agents(P_alt1, c, weighting="shipped_time_avg",
                                   return_weights=True)
        mixed_choice = p_of_observed_choice(mixed_alt1, c)
        dev_main.append(np.abs(mixed_choice - p_hetro).max())

        # --- independent literal transcription, in MATLAB's own space -------
        dev_literal.append(np.abs(published_mix_in_choice_space(P_choice) - p_hetro).max())
        dev_space.append(np.abs(published_mix_in_choice_space(P_choice) - mixed_choice).max())

        # --- the convexity precondition that licenses the space swap --------
        dev_wsum.append(np.abs(W.sum(axis=0) - 1.0).max())
        dev_wbarsum.append(abs(W.mean(axis=1).sum() - 1.0))

        # Two seemingly-distinct mistranslations are actually one: prepending the
        # flat prior then dropping the last column IS the one-trial lag. Asserted,
        # not assumed, so the mutation list can't silently double-count a control.
        L_noprior = np.cumprod(P_choice, axis=1)
        W_noprior = L_noprior / L_noprior.sum(axis=0, keepdims=True)
        dev_prior_is_lag.append(
            np.abs(W_noprior - _normalised_cumprod(P_choice)[:, 1:]).max())

        # --- mutation controls ---------------------------------------------
        for name, fn in MUTATIONS:
            got = p_of_observed_choice(fn(P_alt1, P_choice), c)
            dev_mut[name].append(np.abs(got - p_hetro).max())

    def report(label, devs, expect_zero=True):
        m = float(np.max(devs))
        mark = "OK  " if (m <= TOLERANCE) == expect_zero else "FAIL"
        print(f"  [{mark}] {label:<58} max dev {m:.3e}")
        return (m <= TOLERANCE) == expect_zero

    print("=" * 78)
    print("1. mix_agents() vs the ORIGINAL hetro.m, on identical inputs")
    print("=" * 78)
    ok = True
    ok &= report("mix_agents(shipped_time_avg) -> choice space  vs  MATLAB", dev_main)
    ok &= report("literal choice-space transcription     vs  MATLAB", dev_literal)
    ok &= report("alt-1-space route  vs  choice-space route", dev_space)

    print("\n" + "=" * 78)
    print("2. Convexity precondition (what licenses mixing in a different space)")
    print("=" * 78)
    ok &= report("per-trial weight columns sum to 1", dev_wsum)
    ok &= report("time-averaged w_bar sums to 1", dev_wbarsum)

    print("\n" + "=" * 78)
    print("3. Mutation controls -- each MUST differ, or the test has no power")
    print("=" * 78)
    for name, _ in MUTATIONS:
        ok &= report(name, dev_mut[name], expect_zero=False)
    print("\n   (identity check, so no control is double-counted:)")
    ok &= report("'no prior column' and 'no lag' are the same weight matrix",
                 dev_prior_is_lag)

    print("\n" + "=" * 78)
    if ok:
        print("PASS -- the BMA translation is exact, and the test can detect it "
              "being wrong.")
    else:
        print("FAIL -- see the lines marked FAIL above.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
