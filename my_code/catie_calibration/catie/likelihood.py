"""
Vectorised CATIE likelihood over cached state tensors, for parameter fitting.

Everything here operates on whole (n_subjects x n_trials) matrices loaded from
build_cache.py, so a likelihood evaluation at a given (tau, eps, phi) costs a
handful of numpy ops and no Python loop.

THE ONE SUBTLETY IN THE k-MIXTURE. For a single k the choice probability is a
closed form in (tau, eps, phi) given cached state. For the heterogeneous mixture it
is NOT simply a convex combination with fixed weights: the mixture weights are
built from the cumulative product of the per-agent choice probabilities, which
themselves depend on the parameters. So the weights move when the parameters move
and must be recomputed inside every likelihood evaluation. They are -- see
`mix_per_trial` / `mix_shipped_time_avg`. Treating them as fixed would give a subtly
wrong gradient and a subtly wrong optimum.

WHICH MIXING. `weighting="per_trial"` (default) is the sequential Bayesian model
average the paper's numbers were produced with (verified against Fig S5 per schedule
to the rounding floor). `"shipped_time_avg"` reproduces hetro.m:25 as shipped, which
time-averages the weights; it exists only so the shipped code can be reproduced and
must not be quoted as the published model. Same convention as catie_core.mix_agents.

Parameterisation for the optimiser: tau, eps, phi are all probabilities in (0,1),
so they are optimised in logit space. This removes the need for bounded solvers,
keeps the search unconstrained, and avoids the optimiser parking exactly on a
boundary where the Hessian is undefined.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from catie_core import EPSILON, PHI, TAU  # noqa: E402  (published values)

CACHE_DIR = pathlib.Path(__file__).parent.parent / "cache"
TENSORS = ("H", "b", "c_prev", "s_prev", "sbar_prev", "g")
EPS_CLIP = 1e-12


# ── cache loading ────────────────────────────────────────────────────────────
class StateCache:
    """Stacked parameter-free state for one split.

    st[k][name] -> (n_subjects, n_trials) array.  y -> (n_subjects, n_trials).
    """

    def __init__(self, name: str, ks=(0, 1, 2)):
        z = np.load(CACHE_DIR / f"state_{name}.npz", allow_pickle=False)
        self.name = name
        self.ks = tuple(ks)
        self.subject_id = z["subject_id"]
        self.schedule = z["schedule"]
        self.y = z["y"].astype(np.int8)
        self.st = {k: {t: z[f"k{k}_{t}"] for t in TENSORS} for k in self.ks}
        self.n_subjects, self.n_trials = self.y.shape

    def subset(self, mask) -> "StateCache":
        """Row-subset (e.g. to one schedule, or a bootstrap resample of subjects)."""
        out = object.__new__(StateCache)
        out.name = self.name
        out.ks = self.ks
        out.subject_id = self.subject_id[mask]
        out.schedule = self.schedule[mask]
        out.y = self.y[mask]
        out.st = {k: {t: v[mask] for t, v in d.items()} for k, d in self.st.items()}
        out.n_subjects, out.n_trials = out.y.shape
        return out


# ── the model, vectorised ────────────────────────────────────────────────────
def p_alt1_single_k(state, tau=TAU, eps=EPSILON, phi=PHI, phi_long=None,
                    streak=None, streak_thresh=None):
    """P(choose alternative 1) for one k-agent, over all subjects and trials.

    phi_long / streak / streak_thresh implement the run-length-dependent inertia
    motivated by Phase 2 (the calibration gap reverses sign with run length, which a
    single constant phi -- or even one phi per side of c_prev -- cannot represent).
    When given, phi applies to runs shorter than `streak_thresh` and phi_long to runs
    at or beyond it. When phi_long is None the model reduces exactly to standard
    CATIE, which the golden test checks.
    """
    H, b = state["H"], state["b"]
    c_prev, g = state["c_prev"], state["g"]
    p_exp = eps * (1.0 + state["s_prev"] + state["sbar_prev"]) / 3.0
    th = tau * H

    if phi_long is None:
        phi_eff = phi
    else:
        phi_eff = np.where(streak >= streak_thresh, phi_long, phi)

    inner = phi_eff * c_prev + (1.0 - phi_eff) * g
    p1 = th * b + (1.0 - th) * (0.5 * p_exp + (1.0 - p_exp) * inner)
    p1 = np.array(p1, dtype=float, copy=True)
    p1[:, 0] = 0.5   # trial 1 is fixed at 0.5 by the reference implementation
    return p1


WEIGHTINGS = ("per_trial", "shipped_time_avg")


def _posterior_weights(P, y):
    """Lagged, normalised BMA weights W (n_agents, n_subjects, n_trials), columns sum
    to 1; W[..., t] uses the choices made on trials < t (hetro.m:20-22).
    P : (n_agents, n_subjects, n_trials) of P(alt 1)."""
    yb = y.astype(bool)[None, :, :]
    P_choice = np.where(yb, P, 1.0 - P)                      # (a, s, t)
    n_a, n_s, n_t = P_choice.shape
    ones = np.ones((n_a, n_s, 1))
    L = np.cumprod(np.concatenate([ones, P_choice], axis=2), axis=2)   # (a, s, t+1)
    W = L / L.sum(axis=0, keepdims=True)
    return W[:, :, :-1]


def mix_per_trial(P, y):
    """Sequential Bayesian model average, p(t) = sum_k P_k(t) W_k(t) -- hetro.m's
    commented line 26, and what the paper's reported numbers match. Weights depend on
    the parameters through P, so they are recomputed on every call, not cached."""
    return (P * _posterior_weights(P, y)).sum(axis=0)


def mix_shipped_time_avg(P, y):
    """hetro.m:25 as shipped: mean(agents' * normalised_likelihoods, 2), i.e.
        p(t) = sum_k P_k(t) * mean_s W_k(s)
    -- time-averaged weights applied uniformly to every trial. Reproduces the shipped
    code, NOT the paper's numbers (see module docstring). Kept for reproduction only."""
    w_bar = _posterior_weights(P, y).mean(axis=2)            # (a, s)
    return np.einsum("ast,as->st", P, w_bar)


def mix_agents_3d(P, y, weighting="per_trial"):
    if weighting == "per_trial":
        return mix_per_trial(P, y)
    if weighting == "shipped_time_avg":
        return mix_shipped_time_avg(P, y)
    raise ValueError(f"unknown weighting {weighting!r}; use one of {WEIGHTINGS}")


def p_choice_matrix(cache: StateCache, tau=TAU, eps=EPSILON, phi=PHI,
                    ks=None, phi_long=None, streak_thresh=None, streak=None,
                    lapse=0.0, published_b=False, weighting="per_trial"):
    """P(the choice actually made), (n_subjects, n_trials).

    ks=None uses the cache's k set as a heterogeneous mixture; a single int uses
    that k alone, which isolates one k-agent for sensitivity analysis. The paper's
    model is the {0,1,2} mixture: its "K = 2" is the upper bound of a uniform draw
    (CATIE_single_schedule_score.m:5, k = randi([0,2]); hetro.m:6, K = 0:2).
    weighting : "per_trial" (default, the paper's) or "shipped_time_avg" (the code
                as shipped) -- see module docstring.
    """
    if ks is None:
        ks = cache.ks
    single = isinstance(ks, (int, np.integer))
    klist = [int(ks)] if single else list(ks)

    Ps = []
    for k in klist:
        st = cache.st[k]
        if published_b:
            st = dict(st); st["b"] = np.zeros_like(st["b"])
        Ps.append(p_alt1_single_k(st, tau, eps, phi, phi_long, streak, streak_thresh))

    if single:
        p1 = Ps[0]
    else:
        p1 = mix_agents_3d(np.stack(Ps, axis=0), cache.y, weighting)

    if lapse:
        p1 = lapse / 2.0 + (1.0 - lapse) * p1
    return np.where(cache.y.astype(bool), p1, 1.0 - p1)


def mean_log_p(cache: StateCache, drop_first=True, **kw) -> float:
    """E[log p] -- the quantity the paper's Table S2 reports and this phase fits."""
    pc = p_choice_matrix(cache, **kw)
    if drop_first:
        pc = pc[:, 1:]
    return float(np.mean(np.log(np.clip(pc, EPS_CLIP, 1.0))))


def mean_p(cache: StateCache, drop_first=True, **kw) -> float:
    pc = p_choice_matrix(cache, **kw)
    if drop_first:
        pc = pc[:, 1:]
    return float(np.mean(pc))


# ── optimisation helpers ─────────────────────────────────────────────────────
def _logit(x):
    x = np.clip(x, 1e-9, 1 - 1e-9)
    return np.log(x / (1 - x))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def negloglik_factory(cache: StateCache, ks=None, free=("tau", "eps", "phi"),
                      fixed=None, **extra):
    """Return f(z) = -E[log p] with z the logit-space vector of `free` params.

    Logit space keeps the search unconstrained and stops the optimiser parking on a
    boundary where the Hessian is undefined.
    """
    fixed = dict(fixed or {})
    order = list(free)

    def f(z):
        vals = dict(fixed)
        for nm, zi in zip(order, np.atleast_1d(z)):
            vals[nm] = float(_sigmoid(zi))
        return -mean_log_p(cache, ks=ks, **vals, **extra)

    return f, order


def unpack(z, order):
    return {nm: float(_sigmoid(zi)) for nm, zi in zip(order, np.atleast_1d(z))}


# ── per-subject scores, for subject-clustered (sandwich) standard errors ─────
def subject_nll(cache: StateCache, drop_first=True, **kw) -> np.ndarray:
    """Each subject's SUMMED -log p over their scored trials, shape (n_subjects,)."""
    pc = p_choice_matrix(cache, **kw)
    if drop_first:
        pc = pc[:, 1:]
    return -np.log(np.clip(pc, EPS_CLIP, 1.0)).sum(axis=1)


def subject_scores(cache: StateCache, z, order, ks=None, fixed=None, h=1e-4,
                   **extra) -> np.ndarray:
    """Gradient w.r.t. z (logit space) of every subject's summed nll: (n_subjects, len(z)).

    Central finite differences on the per-subject nll vector; the objective is smooth
    so h = 1e-4 in logit space is ample. Same (order, fixed, ks, extra) convention as
    `negloglik_factory`, so scores line up with that objective's z. Summing the rows
    gives the gradient of the SUMMED nll, which is ~0 at the MLE; the outer-product
    sum  M = S^T S  is the "meat" of the subject-clustered sandwich covariance.
    """
    fixed = dict(fixed or {})
    z = np.atleast_1d(np.asarray(z, dtype=float))

    def nll_vec(zz):
        vals = dict(fixed)
        for nm, zi in zip(order, zz):
            vals[nm] = float(_sigmoid(zi))
        return subject_nll(cache, ks=ks, **vals, **extra)

    S = np.empty((cache.n_subjects, len(z)))
    for j in range(len(z)):
        e = np.zeros_like(z)
        e[j] = h
        S[:, j] = (nll_vec(z + e) - nll_vec(z - e)) / (2.0 * h)
    return S
