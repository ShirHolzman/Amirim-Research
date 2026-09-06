"""
Bayesian responsibility posterior over CATIE's four regimes.

CATIE's per-trial choice probability is an exact mixture over four regimes
(heuristic, exploration, inertia, contingent-average), each with a prior WEIGHT of
being the active regime this trial (mode_weights) and a conditional probability of
choosing alternative 1 given that regime is active:

    P(alt1 | heuristic)   = b          (0 or 1)
    P(alt1 | exploration) = 0.5
    P(alt1 | inertia)     = c_prev     (0 or 1)
    P(alt1 | contingent)  = g

    mode_contributions(state)[i] == mode_weights(state)[i] * P(alt1 | regime i)

Given the CHOICE ACTUALLY MADE, Bayes' rule gives a posterior "responsibility" for
each regime:

    resp_r(t) = weight_r(t) * P(y(t) | r) / P(y(t))
    P(y(t) | r) = P(alt1|r)      if y(t) == 1 (chose alt 1)
                = 1 - P(alt1|r)  if y(t) == 0 (chose alt 2)

Using mode_contributions directly: contribution_r(t) = weight_r(t) * P(alt1|r), so

    weight_r(t) * P(y(t)|r) = contribution_r(t)               if y(t) == 1
                            = weight_r(t) - contribution_r(t)  if y(t) == 0

and P(y(t)) is exactly p_choice(t), the already-computed probability of the
observed choice -- so responsibilities normalise to 1 by construction (verified at
runtime below, not just claimed).

WHY THIS EXISTS: an earlier probe (see project memory / thesis plan) attributed
"dominant mode" by a HARD ARGMAX over mode_contributions -- i.e. which regime
contributed the most mass toward "choosing alt 1". That is a mass-attribution
toward one specific outcome, not a posterior over what generated the ACTUAL
choice, and it is mathematically forced to equal c_prev: inertia's contribution
toward alt1 is identically phi*c_prev, which is exactly 0 whenever c_prev==0,
so inertia can never win that argmax on a c_prev==0 trial regardless of what
was actually chosen. Measured: P(c_prev=1 | argmax==inertia) = 1.0000 exactly.

The responsibility posterior computed here does not have that flaw: on a
c_prev==0 trial where the participant ALSO chose alt 2 (i.e. repeated their
prior choice, consistent with inertia), P(alt1|inertia) = c_prev = 0, so
P(y=0|inertia) = 1, giving inertia a large -- not zero -- responsibility.
Whether this in fact explains more variance than c_prev alone is an empirical
question, answered in conditional_calibration.py; this module only computes the
posterior, honestly, without presupposing the answer.

Heterogeneous (k-mixture) extension: identical logic applied per k-agent, then
combined using the EXACT SAME per-trial weight matrix W that catie_core.mix_agents
uses internally to mix P(alt 1) (obtained via mix_agents(..., return_weights=True),
not recomputed here), so responsibility and probability mixing cannot silently
diverge from each other.

MIXING WEIGHTS. The default weighting="per_trial" uses the causal sequential
k-posterior W[:, t] (choices before t only), so p_alt1 and everything derived from
it is a genuine function of history -- "prospective" in the strict sense. This is
also the weighting the paper's reported numbers match. The alternative
weighting="shipped_time_avg" reproduces hetro.m:25 as shipped, where the weight is
w_bar = W.mean(axis=1), the time-average over all trials, which depends on y(t)
and on every later choice; it is kept only for reproducing the shipped code.
Historical note: Phase 2 was first run under the time-averaged weights. Comparing
the two full pipeline runs, switching to per-trial shifted the headline c_prev gaps
from +0.205/-0.114/+0.003 to +0.208/-0.113/+0.004 and R^2(c_prev) from 0.104 to
0.106; the strongest partition (c_prev x streak_bin) moved 0.180 -> 0.178 against a
noise ceiling of 0.268 -> 0.265. A trial-level probe measured mean |delta p_alt1| at
0.0099 (max 0.206) with the hard_argmax label changing on 1.04% of trials. Every
conclusion is unchanged; only third decimals moved. (The two runs also differ by 4
subjects, 2,524 -> 2,528, from the organized-data-release migration, so these
deltas are not attributable to the weighting alone.)
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from catie_core import (  # noqa: E402
    EPSILON, PHI, TAU,
    mix_agents, mode_contributions, mode_weights, p_of_observed_choice,
    probability_from_state, state_tensors,
)

REGIME_NAMES = ("heuristic", "exploration", "inertia", "contingent_avg")


def responsibility_posterior(rewards_1, rewards_2, is_choice_1, mode="fixed",
                             ks=(0, 1, 2), weighting="per_trial",
                             tau=TAU, epsilon=EPSILON, phi=PHI, n_trials=100):
    """Posterior P(regime | observed choice) for one subject, marginalised over
    the k-mixture.

    Returns a dict with:
      "resp"          (4, n_trials) posterior responsibility, rows in REGIME_NAMES
                      order, columns sum to 1 except column 0 (trial 1), which is
                      NaN -- trial 1 is hardcoded to p=0.5 by probability_from_state
                      and was never generated by any of the four regimes. Always
                      exclude it.
      "mixed_contrib"  (4, n_trials) mass-attribution toward "choosing alt 1",
                      mixed across k with the SAME weights as "resp". This is the
                      k-mixture generalisation of mode_contributions() and is what
                      the OLD (planning-phase) hard-argmax attribution should use
                      for a fair comparison -- that probe used a single k=2 agent
                      only. Provided here so both attribution rules are compared
                      on an identical mixture, differing only in the rule itself.
      "p_alt1_mix"    (n_trials,) mixed forecast P(choose alt 1)
      "p_choice_mix"  (n_trials,) mixed P(choice actually made)
    """
    chose_alt1_bool = np.asarray(is_choice_1, dtype=bool)
    chose_alt1_indicator = chose_alt1_bool.astype(float)  # 1.0 if alt1 chosen

    n_agents = len(ks)
    p_alt1_by_agent = np.zeros((n_agents, n_trials))
    responsibility_by_regime_and_agent = np.zeros((4, n_agents, n_trials))
    contribution_by_regime_and_agent = np.zeros((4, n_agents, n_trials))

    for agent_index, k in enumerate(ks):
        state = state_tensors(rewards_1, rewards_2, is_choice_1, k=k, mode=mode, n_trials=n_trials)
        p_alt1 = probability_from_state(state, tau=tau, epsilon=epsilon, phi=phi)
        p_alt1_by_agent[agent_index, :] = p_alt1

        regime_contributions = np.vstack(mode_contributions(state, tau=tau, epsilon=epsilon, phi=phi))  # (4, n)
        regime_weights = np.vstack(mode_weights(state, tau=tau, epsilon=epsilon, phi=phi))               # (4, n)
        contribution_by_regime_and_agent[:, agent_index, :] = regime_contributions
        # weight_r * P(y|r): contribution_r if alt1 was chosen, else weight_r - contribution_r
        responsibility_by_regime_and_agent[:, agent_index, :] = np.where(
            chose_alt1_indicator[None, :] > 0.5, regime_contributions, regime_weights - regime_contributions)

    p_alt1_mix, mixing_weights = mix_agents(p_alt1_by_agent, is_choice_1, weighting=weighting, return_weights=True)
    p_choice_mix = p_of_observed_choice(p_alt1_mix, is_choice_1)

    if weighting == "per_trial":
        mixed_responsibility = np.einsum("rat,at->rt", responsibility_by_regime_and_agent, mixing_weights)
        mixed_contribution = np.einsum("rat,at->rt", contribution_by_regime_and_agent, mixing_weights)
    elif weighting == "shipped_time_avg":
        time_averaged_agent_weights = mixing_weights.mean(axis=1)  # (n_agents,), sums to 1
        mixed_responsibility = np.tensordot(
            responsibility_by_regime_and_agent, time_averaged_agent_weights, axes=([1], [0]))  # (4, n_trials)
        mixed_contribution = np.tensordot(
            contribution_by_regime_and_agent, time_averaged_agent_weights, axes=([1], [0]))  # (4, n_trials)
    else:
        raise ValueError(f"unknown weighting {weighting!r}; use 'per_trial' or 'shipped_time_avg'")

    with np.errstate(invalid="ignore", divide="ignore"):
        responsibility = mixed_responsibility / p_choice_mix[None, :]
    responsibility[:, 0] = np.nan  # trial 1: not generated by any regime

    return {"resp": responsibility, "mixed_contrib": mixed_contribution,
            "p_alt1_mix": p_alt1_mix, "p_choice_mix": p_choice_mix}


def validate_responsibility(resp, tol=1e-9):
    """Runtime check: responsibilities sum to 1 for every trial except trial 1.
    Returns the max |deviation|; raises if it exceeds tol. Call this once per
    subject during the main analysis run, not just trust the derivation above.
    """
    responsibility_sums = np.nansum(resp, axis=0)
    responsibility_sums[0] = 1.0  # trial 1 is NaN by design; don't penalise it
    max_deviation = float(np.abs(responsibility_sums - 1.0).max())
    if max_deviation > tol:
        raise AssertionError(f"responsibility posterior does not sum to 1: max dev {max_deviation:.3e}")
    return max_deviation
