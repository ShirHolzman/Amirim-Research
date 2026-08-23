"""
catie_core — exact Python port of the competition's CATIE choice-probability model.

Reference implementation being ported:
    Data_resources/competition_analysis-main/CATIE/
        COMPETITION_CATIE_schedule_choice_probability.m        (per-k likelihood)
        COMPETITION_CATIE_schedule_choice_probability_hetro.m  (k in {0,1,2} mixture)
        CATIE_implementation_helpers/base2dec.m
        CATIE_implementation_helpers/getExploreProb.m

Two behaviours are exposed via `mode`:

    "published"  "bug-for-bug" faithful to the MATLAB above. Reproduces
                 my_code/initial_investigation/processing/eda_with_catie_probabilities.csv
                 to ~2e-15 (see golden_test.py).

    "fixed"      corrects the trend/heuristic branch. In the MATLAB, lines 108-109
                 read `pays(trial)`, which is still NaN (preallocated line 16,
                 assigned line 147). MATLAB evaluates `NaN > x` and `NaN < x` as
                 false, so the branch never fires -- yet `p_try_explore` is still
                 reduced to (1 - tau) on line 114. The result is that on every
                 trend-testable trial, tau of the probability mass is silently
                 handed to alternative 2. The intended semantics are unambiguous in
                 the generative simulator, CATIE_schedule_1.m:152-157:
                     same choice on t-1 and t-2, payoffs differ ->
                     repeat that choice if the payoff rose, switch if it fell.

KEY STRUCTURAL PROPERTY
-----------------------
The likelihood is conditioned on the participant's observed choices, so CATIE never
samples. Every internal state variable (reward means, observed SDs, surprise,
contingency tables, g, H, c_prev) is therefore a function of the DATA ONLY -- none
depends on tau, epsilon or phi. Given the cached state, the choice probability is a
closed form:

    P(alt 1) = tau*H*b + (1 - tau*H) * [ p_exp/2 + (1 - p_exp) * (phi*c_prev + (1-phi)*g) ]
    p_exp    = epsilon * (1 + s_prev + sbar_prev) / 3

`state_tensors` computes the data-only part once; `probability_from_state` applies
parameters. The direct-port path calls exactly these two functions in sequence, so
the cached path cannot silently diverge from it.

Note also that "published" vs "fixed" differ ONLY in the value of `b`: published is
equivalent to b == 0 everywhere. H is identical in both.
"""

from __future__ import annotations

import numpy as np

# Published parameters (Plonsky & Erev 2017, fitted on Erev et al. 2010; reported in
# Dan, Plonsky & Loewenstein 2025 p.8 as tau=0.29, eps=0.30, phi=0.71, K=2).
# See CATIE_single_schedule_score.m:4
TAU = 0.29
EPSILON = 0.30
PHI = 0.71
K_DEFAULT = 2
N_TRIALS = 100

MODES = ("published", "fixed")


# ── State: the parameter-free recursion ──────────────────────────────────────────
class CatieState:
    """Per-trial, parameter-independent quantities driving the choice probability.

    Attributes are length-n arrays, index 0 == trial 1 (MATLAB's `trial` 1..nTrials).

    H          1.0 if the trend/heuristic branch is testable on this trial
    b          heuristic verdict: 1.0 if the trend rule selects alternative 1.
               Identically 0.0 in "published" mode (that is precisely the bug).
    c_prev     1.0 if the previous choice was alternative 1
    s_prev     surprise on the previous trial
    sbar_prev  running mean surprise through the previous trial
    g          contingency-mode probability of choosing alternative 1
    """

    __slots__ = ("H", "b", "c_prev", "s_prev", "sbar_prev", "g", "ca_1", "ca_2")

    def __init__(self, H, b, c_prev, s_prev, sbar_prev, g, ca_1=None, ca_2=None):
        self.H = H
        self.b = b
        self.c_prev = c_prev
        self.s_prev = s_prev
        self.sbar_prev = sbar_prev
        self.g = g
        self.ca_1 = ca_1  # ragged; only populated when collect_ca=True (Phase 4 M5)
        self.ca_2 = ca_2

    def as_dict(self):
        return {
            "H": self.H, "b": self.b, "c_prev": self.c_prev,
            "s_prev": self.s_prev, "sbar_prev": self.sbar_prev, "g": self.g,
        }

    def as_published(self):
        """View of this state under the published (buggy) heuristic branch.

        The published and fixed models differ ONLY in `b` -- everything else in the
        recursion is identical, because `b` never feeds back into the state. So one
        state pass computed with mode="fixed" serves both models, halving the cost of
        any published-vs-fixed comparison.
        """
        return CatieState(self.H, np.zeros_like(self.b), self.c_prev,
                          self.s_prev, self.sbar_prev, self.g, self.ca_1, self.ca_2)


def _base2dec(base, vec):
    """MATLAB base2dec.m:5-9 -- the OLDEST element of the window carries weight base^0."""
    d = 0.0
    for j, v in enumerate(vec):
        d += (base ** j) * float(v)
    return int(d)


def state_tensors(rewards_1, rewards_2, is_choice_1, k=K_DEFAULT, mode="fixed",
                  n_trials=N_TRIALS, collect_ca=False):
    """Run the parameter-free CATIE recursion and return a CatieState.

    rewards_1/2  : length-n binary reward schedules for alternatives 1 and 2
    is_choice_1  : length-n booleans, True where the participant chose alternative 1
    k            : contingency depth (CAB-k)
    mode         : "published" forces b == 0 (reproduces the MATLAB bug); "fixed"
                   computes the heuristic verdict per CATIE_schedule_1.m:152-157
    collect_ca   : also retain the per-trial contingent-average value sets, needed
                   only by the soft-CA extension (Phase 4, M5). Costs memory.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    n = n_trials
    nan = float("nan")

    # 1-indexed views so the port reads like the MATLAB it mirrors.
    r1 = np.concatenate([[nan], np.asarray(rewards_1, dtype=float)])
    r2 = np.concatenate([[nan], np.asarray(rewards_2, dtype=float)])
    c = np.concatenate([[False], np.asarray(is_choice_1, dtype=bool)])

    cc1 = np.zeros((4 ** k, 2))          # contingency table: [sum of payoffs, visits]
    cc2 = np.zeros((4 ** k, 2))
    pays = np.full(n + 1, nan)
    surprise = np.full(n + 1, nan)
    reward_vec = np.full(n + 1, nan)     # outcome symbol in {0,1,2,3}
    observed_sd = np.zeros(3)            # 1-indexed by alternative
    expected_reward = np.zeros(3)        # MATLAB initialises to 0 and, on the
                                         # insufficient-history branch, leaves it stale
    reward_mean = [0.0, 0.0, 0.0]
    reward_sum = [0.0, 0.0, 0.0]
    sum_sq = [0.0, 0.0, 0.0]
    n_choices = [0, 0, 0]
    total_surprise = 0.0
    mean_surprise = 0.0

    H = np.zeros(n + 1)
    b = np.zeros(n + 1)
    c_prev = np.zeros(n + 1)
    s_prev = np.zeros(n + 1)
    sbar_prev = np.zeros(n + 1)
    g_arr = np.zeros(n + 1)
    ca_1_all = [None] * (n + 1) if collect_ca else None
    ca_2_all = [None] * (n + 1) if collect_ca else None

    for t in range(1, n + 1):
        # ── contingency-mode probability, before this trial's reward is observed ──
        ca_1 = ca_2 = None
        if k == 0:
            expected_reward[1] = reward_mean[1]
            expected_reward[2] = reward_mean[2]
            g = 0.5 if reward_mean[1] == reward_mean[2] else float(reward_mean[1] > reward_mean[2])
        elif (t - 1) > k and t < n:
            # MATLAB gate is `(trial-1) > k && trial < nTrials`; the final trial
            # therefore always falls back to the grand-mean comparison.
            row = _base2dec(4, reward_vec[t - k:t])
            if cc2[row, 1] > 0:
                ca_2 = np.array([cc2[row, 0] / cc2[row, 1]])
            else:
                seen = cc2[:, 1] > 0
                # "Confusion": uniform over DISTINCT encountered rows, not weighted
                # by visit counts (CATIE_schedule_1.m:119 uses datasample over rows).
                ca_2 = (cc2[seen, 0] / cc2[seen, 1]) if seen.any() else np.array([reward_mean[2]])
            if cc1[row, 1] > 0:
                ca_1 = np.array([cc1[row, 0] / cc1[row, 1]])
            else:
                seen = cc1[:, 1] > 0
                ca_1 = (cc1[seen, 0] / cc1[seen, 1]) if seen.any() else np.array([reward_mean[1]])
            # Exact marginalisation over the independent confusion draws
            # (MATLAB meshgrid, lines 84-87).
            gt = ca_1[:, None] > ca_2[None, :]
            eq = ca_1[:, None] == ca_2[None, :]
            g = gt.mean() + 0.5 * eq.mean()
            expected_reward[1] = ca_1.mean()
            expected_reward[2] = ca_2.mean()
        else:
            g = 0.5 if reward_mean[1] == reward_mean[2] else float(reward_mean[1] > reward_mean[2])

        # ── record the parameter-free decision inputs ────────────────────────────
        if t > 1:
            testable = (t > 2) and (c[t - 1] == c[t - 2]) and (pays[t - 1] != pays[t - 2])
            H[t] = float(testable)
            if testable and mode == "fixed":
                # CATIE_schedule_1.m:152-157 -- payoff rose => repeat, fell => switch.
                b[t] = float(
                    (c[t - 1] and pays[t - 1] > pays[t - 2])
                    or ((not c[t - 1]) and pays[t - 1] < pays[t - 2])
                )
            c_prev[t] = float(c[t - 1])
            s_prev[t] = surprise[t - 1]
            sbar_prev[t] = mean_surprise
            g_arr[t] = g
        if collect_ca:
            ca_1_all[t] = ca_1
            ca_2_all[t] = ca_2

        # ── observe this trial's outcome and update internals ────────────────────
        a = 1 if c[t] else 2
        pays[t] = r1[t] if c[t] else r2[t]
        if c[t]:
            reward_vec[t] = 3.0 if pays[t] == 1 else 2.0
        else:
            reward_vec[t] = 1.0 if pays[t] == 1 else 0.0

        reward_sum[a] += pays[t]
        n_choices[a] += 1
        sum_sq[a] += pays[t] ** 2
        if n_choices[a] > 1:
            var = (1.0 / (n_choices[a] - 1)) * (sum_sq[a] - (reward_sum[a] ** 2) / n_choices[a])
        else:
            var = nan  # MATLAB: Inf*0 -> NaN on the first visit; caught by the guard
        if np.isnan(var) or var < 0:
            var = 0.0
        observed_sd[a] = np.sqrt(var)

        # Surprise uses the SD updated THROUGH trial t, but `expected_reward` as it
        # stood at the START of t. That ordering is load-bearing -- see MATLAB 182-189.
        if observed_sd[a] > 1e-4:
            diff = abs(expected_reward[a] - pays[t])
            surprise[t] = diff / (observed_sd[a] + diff)
        else:
            surprise[t] = 0.0
        total_surprise += surprise[t]
        mean_surprise = total_surprise / t

        if k > 0 and t > k:
            row = _base2dec(4, reward_vec[t - k:t])
            table = cc1 if c[t] else cc2
            table[row, 0] += pays[t]
            table[row, 1] += 1

        reward_mean[a] = reward_sum[a] / n_choices[a]

    return CatieState(
        H[1:], b[1:], c_prev[1:], s_prev[1:], sbar_prev[1:], g_arr[1:],
        ca_1_all[1:] if collect_ca else None,
        ca_2_all[1:] if collect_ca else None,
    )


# ── Parameters applied to cached state ───────────────────────────────────────────
def probability_from_state(state, tau=TAU, epsilon=EPSILON, phi=PHI, phi_2=None,
                           lapse=0.0):
    """Closed-form P(choose alternative 1) per trial, given a CatieState.

    phi_2  : optional asymmetric inertia (Phase 4, M3). When given, `phi` applies
             after an alternative-1 choice and `phi_2` after an alternative-2 choice.
    lapse  : optional lapse rate (Phase 4, M4), p <- lapse/2 + (1-lapse)*p. Decouples
             the probability floor from epsilon, which currently controls both.

    Trial 1 is fixed at 0.5 (MATLAB line 103).
    """
    p_exp = epsilon * (1.0 + state.s_prev + state.sbar_prev) / 3.0
    th = tau * state.H                                   # decide: heuristic weight

    if phi_2 is None:
        phi_eff = phi
    else:
        phi_eff = np.where(state.c_prev > 0.5, phi, phi_2)

    inner = phi_eff * state.c_prev + (1.0 - phi_eff) * state.g  # inertia vs CA value
    p1 = th * state.b + (1.0 - th) * (0.5 * p_exp + (1.0 - p_exp) * inner)
    p1 = np.asarray(p1, dtype=float).copy()
    p1[0] = 0.5

    if lapse:
        p1 = lapse / 2.0 + (1.0 - lapse) * p1
    return p1


def mode_weights(state, tau=TAU, epsilon=EPSILON, phi=PHI):
    """Prior probability of each of CATIE's four regimes being the one that
    generates this trial's choice, BEFORE conditioning on what the choice
    actually was. Sums to 1 for every trial except trial 1 (probability_from_state
    hardcodes p=0.5 there, outside the mode cascade entirely).

    Structurally parallel to mode_contributions(): for each regime i,
        mode_contributions(state)[i] == mode_weights(state)[i] * P(alt 1 | regime i)
    with P(alt1 | heuristic)=b, P(alt1 | exploration)=0.5, P(alt1 | inertia)=c_prev,
    P(alt1 | contingent)=g. This decomposition is what makes a genuine Bayesian
    responsibility posterior possible (see 02_mode_calibration/ for the E-step that
    uses it) -- mode_contributions alone only supports a mass-attribution toward
    "chose alt 1", which is a deterministic function of c_prev for the inertia term
    (identically 0 whenever c_prev==0) and therefore cannot be used to ask "which
    regime actually explains the choice that was made" without that confound.

    Return value, read as a question per regime: e.g. w_inertia answers "how
    likely, according to CATIE's own theory, is it that the person was in
    inertia-mode when they made this choice" -- as a PRIOR, before the actual
    choice is used as evidence. (Combine with mode_contributions for the
    posterior version of that question.)
    """
    p_exp = epsilon * (1.0 + state.s_prev + state.sbar_prev) / 3.0
    th = tau * state.H
    rest = 1.0 - th
    w_heuristic = th                                   # decide: trend fires?
    w_exploration = rest * p_exp                        # decide: explore fires?
    w_inertia = rest * (1.0 - p_exp) * phi               # decide: inertia fires?
    w_contingent = rest * (1.0 - p_exp) * (1.0 - phi)    # decide: else, use CA
    return w_heuristic, w_exploration, w_inertia, w_contingent


def mode_contributions(state, tau=TAU, epsilon=EPSILON, phi=PHI):
    """Exact additive decomposition of P(alt 1) into its four modal terms.

    Returns (heuristic, exploration, inertia, contingent_average); these sum to
    `probability_from_state(...)` except at trial 0, which is fixed at 0.5.

    Caution (established during planning): a hard argmax over these four terms is a
    DETERMINISTIC relabelling of `c_prev`, because the inertia term is phi*c_prev and
    is identically zero when c_prev == 0. Measured P(c_prev=1 | inertia) = 1.0000 and
    P(c_prev=1 | any other) = 0.0000. Treat argmax attribution accordingly; prefer a
    responsibility posterior for any claim about mechanism.
    """
    p_exp = epsilon * (1.0 + state.s_prev + state.sbar_prev) / 3.0
    th = tau * state.H
    rest = 1.0 - th
    heuristic = th * state.b                             # value: b, trend verdict
    exploration = rest * 0.5 * p_exp                      # value: 0.5, random side
    inertia = rest * (1.0 - p_exp) * phi * state.c_prev    # value: c_prev, repeat
    contingent = rest * (1.0 - p_exp) * (1.0 - phi) * state.g  # value: g, CA rule
    return heuristic, exploration, inertia, contingent


# ── Direct-port entry points ─────────────────────────────────────────────────────
def catie_probabilities(rewards_1, rewards_2, is_choice_1, k=K_DEFAULT,
                        mode="fixed", tau=TAU, epsilon=EPSILON, phi=PHI,
                        n_trials=N_TRIALS):
    """P(alt 1) per trial for a single k-agent. Mirrors the per-k MATLAB function."""
    st = state_tensors(rewards_1, rewards_2, is_choice_1, k=k, mode=mode, n_trials=n_trials)
    return probability_from_state(st, tau=tau, epsilon=epsilon, phi=phi)


def catie_hetero(rewards_1, rewards_2, is_choice_1, mode="fixed", ks=(0, 1, 2),
                 weighting="published", tau=TAU, epsilon=EPSILON, phi=PHI,
                 n_trials=N_TRIALS, return_agents=False):
    """Heterogeneous CATIE: mixture over contingency depths k.

    weighting="published" reproduces COMPETITION_..._hetro.m:25, which computes
        mean(agents_choice_probabilities' * normalised_likelihoods, 2)
    -- a 100x100 outer product collapsed by a row mean. That yields
        p(t) = sum_k p_k(t) * mean_s w_k(s)
    i.e. TIME-AVERAGED posterior weights applied uniformly to every trial, not the
    per-trial sequential weighting the file's own comment (line 26) describes. It is
    still a valid convex combination, and it is what produced the published numbers,
    so it is the default.

    weighting="per_trial" uses the intended line 26. Measured on the EDA set it is
    slightly WORSE (E[log p] -0.6992 -> -0.7042), so it is a footnote, not a fix.
    """
    P = np.vstack([
        catie_probabilities(rewards_1, rewards_2, is_choice_1, k=k, mode=mode,
                            tau=tau, epsilon=epsilon, phi=phi, n_trials=n_trials)
        for k in ks
    ])
    p_mix = mix_agents(P, is_choice_1, weighting=weighting)
    return (p_mix, P) if return_agents else p_mix


def mix_agents(P, is_choice_1, weighting="published", return_weights=False):
    """Combine per-k P(alt 1) matrices into the heterogeneous mixture.

    P : (n_agents, n_trials) array of P(alt 1).

    The MATLAB accumulates likelihoods of the CHOICES ACTUALLY MADE, so P must be
    converted to choice-probability space before the cumprod.

    return_weights=True additionally returns W, the (n_agents, n_trials) per-trial
    normalised posterior weight matrix (columns sum to 1) used to build the mixture.
    Exposed so that other per-agent quantities (e.g. the mode-responsibility
    posterior in 02_mode_calibration/) can be mixed with the EXACT same weights
    this function uses internally, rather than recomputing this cumprod and
    risking silent drift between two copies of the same logic.
    """
    c = np.asarray(is_choice_1, dtype=bool)
    P_choice = np.where(c[None, :], P, 1.0 - P)
    n_agents = P.shape[0]
    L = np.cumprod(np.hstack([np.ones((n_agents, 1)), P_choice]), axis=1)
    W = L / L.sum(axis=0, keepdims=True)
    W = W[:, :-1]
    if weighting == "published":
        mixed = (P.T @ W).mean(axis=1)
    elif weighting == "per_trial":
        mixed = (P * W).sum(axis=0)
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    return (mixed, W) if return_weights else mixed


def p_of_observed_choice(p_alt1, is_choice_1):
    """Convert P(alt 1) to P(the choice actually made) -- the likelihood quantity.

    This is what `catie_choice_probability` in the stored CSV holds, and what E[p]
    and E[log p] are computed over. Distinct from the P(alt 1) forecast used for
    calibration curves; conflating them is an easy and costly mistake.
    """
    c = np.asarray(is_choice_1, dtype=bool)
    return np.where(c, p_alt1, 1.0 - p_alt1)
