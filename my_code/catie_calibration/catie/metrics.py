"""
Scoring and calibration metrics for trial-level choice predictions.

Two distinct quantities are scored throughout this project and must not be conflated:

    p_choice  P(the choice the participant actually made). E[p] and E[log p] are
              computed over this. It is what the stored `catie_choice_probability`
              column holds and what the paper's Tables S1/S2 report.

    p_alt1    P(choose alternative 1 / the biased option) -- a forecast of a fixed
              event. Calibration (reliability, ECE, Brier) must be computed over
              THIS, because calibration is only meaningful for a forecast of a fixed
              target. Computing a reliability curve over p_choice is degenerate: the
              outcome is 1 by construction.

CLUSTERING
----------
Trials within a participant are strongly dependent (CATIE is an inertia-dominated
sequential model; the raw autocorrelation of choices is high). Every confidence
interval and significance test here therefore resamples SUBJECTS, not trials. The
project's earlier scripts treated trials as independent, which inflates significance
by roughly sqrt(trials per subject) -- about 10x here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS_CLIP = 1e-12
DEFAULT_SEED = 42


# ── Point scores ─────────────────────────────────────────────────────────────────
def e_p(p_choice) -> float:
    """Mean probability assigned to the choice actually made."""
    return float(np.mean(p_choice))


def e_log_p(p_choice) -> float:
    """Mean log-probability of the choice actually made (per-trial log-likelihood)."""
    return float(np.mean(np.log(np.clip(p_choice, EPS_CLIP, 1.0))))


def brier(p_alt1, outcome) -> float:
    """Brier score of the alternative-1 forecast. Lower is better."""
    return float(np.mean((np.asarray(p_alt1, float) - np.asarray(outcome, float)) ** 2))


def accuracy(p_alt1, outcome) -> float:
    """Argmax accuracy, ties counted as 0.5."""
    p = np.asarray(p_alt1, float)
    y = np.asarray(outcome, float)
    return float(np.mean(np.where(p == 0.5, 0.5, (p > 0.5) == (y > 0.5))))


# ── Calibration ──────────────────────────────────────────────────────────────────
def _bin_edges(p, n_bins=10, strategy="uniform"):
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
        return edges
    raise ValueError(f"unknown strategy {strategy!r}")


def _bin_index(p, edges):
    return np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)


def reliability_table(p_alt1, outcome, n_bins=10, strategy="uniform",
                      min_count=1, edges=None) -> pd.DataFrame:
    """Binned reliability table.

    strategy : "uniform"  equal-width bins over [0, 1]
               "quantile" equal-mass bins (robust when p is clumped, which it is
                          here -- CATIE's output is heavily concentrated by the
                          [0.05, 0.95] clamp that epsilon induces)
    edges    : pre-computed bin edges. When given, `strategy` and `n_bins` are not
               consulted and the edges are used as they stand. This exists for
               resampling: under strategy="quantile" the edges are a FUNCTION OF
               THE DATA, so recomputing them inside every bootstrap replicate
               would give a confidence interval for a moving target. Callers that
               resample (`reliability_table_ci`, `ece_ci`) compute the edges once
               on the full sample and pass them in. Default None = unchanged
               behaviour: derive the edges from the data handed in.
    """
    p = np.asarray(p_alt1, float)
    y = np.asarray(outcome, float)

    edges = _bin_edges(p, n_bins, strategy) if edges is None else np.asarray(edges, float)
    idx = _bin_index(p, edges)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n < min_count:
            continue
        rows.append({
            "bin": b,
            "lo": edges[b],
            "hi": edges[b + 1],
            "n": n,
            "predicted": float(p[m].mean()),
            "empirical": float(y[m].mean()),
            "gap": float(y[m].mean() - p[m].mean()),
        })
    return pd.DataFrame(rows)


def ece(p_alt1, outcome, n_bins=10, strategy="uniform", edges=None) -> float:
    """Expected calibration error: n-weighted mean |empirical - predicted|.

    `edges` is passed straight through to `reliability_table`; see its docstring.
    Omitting it (the default) leaves the behaviour unchanged.
    """
    t = reliability_table(p_alt1, outcome, n_bins=n_bins, strategy=strategy, edges=edges)
    if t.empty:
        return float("nan")
    return float((t["n"] / t["n"].sum() * t["gap"].abs()).sum())


def mce(p_alt1, outcome, n_bins=10, strategy="uniform", min_count=30) -> float:
    """Maximum calibration error over bins holding at least `min_count` trials."""
    t = reliability_table(p_alt1, outcome, n_bins=n_bins, strategy=strategy,
                          min_count=min_count)
    if t.empty:
        return float("nan")
    return float(t["gap"].abs().max())


def score_all(p_alt1, p_choice, outcome, n_bins=10) -> dict:
    """The full metric suite as a flat dict."""
    return {
        "E[p]": e_p(p_choice),
        "E[log p]": e_log_p(p_choice),
        "ECE": ece(p_alt1, outcome, n_bins=n_bins, strategy="uniform"),
        "ECE_q": ece(p_alt1, outcome, n_bins=n_bins, strategy="quantile"),
        "MCE": mce(p_alt1, outcome, n_bins=n_bins),
        "Brier": brier(p_alt1, outcome),
        "accuracy": accuracy(p_alt1, outcome),
        "n": int(len(p_choice)),
    }


# ── Subject-clustered inference ──────────────────────────────────────────────────
def _subject_means(values, subject_ids):
    """Per-subject mean of a trial-level quantity, plus the subject index."""
    s = pd.Series(np.asarray(values, float))
    grouped = s.groupby(np.asarray(subject_ids))
    return grouped.mean().to_numpy(), grouped.size().index.to_numpy()


def bootstrap_ci(values, subject_ids, statistic=np.mean, n_boot=10_000,
                 alpha=0.05, seed=DEFAULT_SEED):
    """Cluster bootstrap CI: resample SUBJECTS with replacement.

    Returns (point_estimate, lo, hi). The point estimate is the statistic applied to
    the per-subject means, so subjects are weighted equally regardless of trial count
    (all have 100 here, so this matches the trial-level mean).
    """
    per_subj, _ = _subject_means(values, subject_ids)
    rng = np.random.default_rng(seed)
    n = len(per_subj)
    draws = rng.integers(0, n, size=(n_boot, n))
    boot = statistic(per_subj[draws], axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(statistic(per_subj)), float(lo), float(hi)


def cluster_bootstrap(stat_fn, subject_ids, *arrays, n_boot=2000, alpha=0.05,
                      seed=DEFAULT_SEED):
    """Subject-cluster bootstrap of an arbitrary (possibly vector-valued) statistic.

    `arrays` are trial-level and aligned with `subject_ids`. Each resample draws
    SUBJECTS with replacement and hands `stat_fn` the concatenation of the chosen
    subjects' trials (in subject-sorted order), so per-bin empirical rates, ECE,
    E[p], E[log p] or any other function of the trial-level data can be given a
    clustered CI. `stat_fn(*arrays)` must return a scalar or a 1-D array (e.g. one
    value per reliability bin); NaNs (empty bins) are ignored by the percentiles.

    Returns (point, lo, hi): the statistic on the original data and the
    alpha/2, 1-alpha/2 percentiles over resamples, elementwise. Scalars come back
    as floats, vectors as arrays.

    With one trial per subject this is an ordinary trial bootstrap; with the
    statistic = mean of per-subject means and equal trial counts it reproduces
    `bootstrap_ci` exactly (same seed, same draw shape).
    """
    ids = np.asarray(subject_ids)
    arrs = [np.asarray(a) for a in arrays]
    if not arrs:
        raise ValueError("cluster_bootstrap needs at least one trial-level array")
    n_trials = len(ids)
    for a in arrs:
        if len(a) != n_trials:
            raise ValueError("every array must be aligned with subject_ids")

    # sort trials by subject so each subject is one contiguous slice
    order = np.argsort(ids, kind="stable")
    _, starts, counts = np.unique(ids[order], return_index=True, return_counts=True)
    sorted_arrs = [a[order] for a in arrs]
    n_subj = len(starts)

    point = np.asarray(stat_fn(*arrs), dtype=float)
    scalar = point.ndim == 0

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_subj, size=(n_boot, n_subj))
    boot = np.empty((n_boot,) + point.shape)
    for b in range(n_boot):
        pick = draws[b]
        c = counts[pick]
        csum = np.cumsum(c)
        # idx = concatenation of the chosen subjects' slices, built without a loop:
        # element j of block k is starts[pick[k]] + (j - block offset)
        idx = np.arange(csum[-1]) + np.repeat(starts[pick] - (csum - c), c)
        boot[b] = np.asarray(stat_fn(*[a[idx] for a in sorted_arrs]), dtype=float)
    lo, hi = np.nanpercentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    if scalar:
        return float(point), float(lo), float(hi)
    return point, lo, hi


def reliability_table_ci(p_alt1, outcome, subject_ids, n_bins=10, strategy="uniform",
                         min_count=1, n_boot=2000, seed=DEFAULT_SEED) -> pd.DataFrame:
    """`reliability_table` plus subject-cluster-bootstrap columns `emp_lo`, `emp_hi`
    (95% CI of each bin's empirical rate). Bin edges are fixed from the original
    data (for "quantile" too), so the CI is for the rate inside a fixed bin."""
    p = np.asarray(p_alt1, float)
    y = np.asarray(outcome, float)
    edges = _bin_edges(p, n_bins, strategy)   # computed ONCE, on the full sample
    table = reliability_table(p, y, n_bins=n_bins, strategy=strategy,
                              min_count=min_count, edges=edges)
    if table.empty:
        return table
    bins = table["bin"].to_numpy()

    n_all = len(edges) - 1

    def per_bin_rate(pp, yy):
        idx = _bin_index(pp, edges)
        cnt = np.bincount(idx, minlength=n_all)[bins]
        tot = np.bincount(idx, weights=yy, minlength=n_all)[bins]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

    _, lo, hi = cluster_bootstrap(per_bin_rate, subject_ids, p, y, n_boot=n_boot, seed=seed)
    table = table.copy()
    table["emp_lo"] = lo
    table["emp_hi"] = hi
    return table


def ece_ci(p_alt1, outcome, subject_ids, n_bins=10, strategy="uniform", n_boot=2000,
           seed=DEFAULT_SEED):
    """ECE with a subject-cluster-bootstrap 95% CI: (point, lo, hi).

    Bin edges are computed ONCE on the full sample and held fixed across every
    replicate, exactly as `reliability_table_ci` does. This matters only for
    strategy="quantile", where the edges are a function of the data: letting each
    replicate re-derive its own edges gives a CI for a bin definition that moves
    with the resample rather than for the ECE of a fixed binning.
    """
    p = np.asarray(p_alt1, float)
    y = np.asarray(outcome, float)
    edges = _bin_edges(p, n_bins, strategy)
    return cluster_bootstrap(
        lambda pp, yy: ece(pp, yy, n_bins=n_bins, strategy=strategy, edges=edges),
        subject_ids, p, y, n_boot=n_boot, seed=seed)


def paired_subject_test(values_a, values_b, subject_ids, n_boot=10_000,
                        seed=DEFAULT_SEED):
    """Paired comparison of two scorings of the SAME trials, clustered by subject.

    Aggregates each scoring to per-subject means, then tests the paired difference
    with both a paired t-test and a cluster bootstrap. Returns a dict.
    """
    from scipy import stats

    a, ids = _subject_means(values_a, subject_ids)
    b, _ = _subject_means(values_b, subject_ids)
    diff = a - b

    t_stat, p_val = stats.ttest_rel(a, b)
    rng = np.random.default_rng(seed)
    n = len(diff)
    draws = rng.integers(0, n, size=(n_boot, n))
    boot = diff[draws].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])

    sd = diff.std(ddof=1)
    return {
        "n_subjects": int(n),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "t": float(t_stat),
        "p": float(p_val),
        "cohens_dz": float(diff.mean() / sd) if sd > 0 else float("nan"),
    }


def stars(p: float) -> str:
    """Significance marker, matching the convention used elsewhere in this repo."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ── Output helper shared by the analysis scripts ─────────────────────────────────
class Tee:
    """Write to several streams at once, tolerating consoles with narrow encodings.

    Mirrors the `_Tee` used by the existing analysis scripts under my_code/initial_investigation/
    so that every analysis directory keeps producing a figures/output.txt transcript.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text):
        for s in self._streams:
            try:
                s.write(text)
            except UnicodeEncodeError:
                enc = getattr(s, "encoding", "utf-8") or "utf-8"
                s.write(text.encode(enc, errors="replace").decode(enc))
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()
