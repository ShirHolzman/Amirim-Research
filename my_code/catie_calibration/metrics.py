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
def reliability_table(p_alt1, outcome, n_bins=10, strategy="uniform",
                      min_count=1) -> pd.DataFrame:
    """Binned reliability table.

    strategy : "uniform"  equal-width bins over [0, 1]
               "quantile" equal-mass bins (robust when p is clumped, which it is
                          here -- CATIE's output is heavily concentrated by the
                          [0.05, 0.95] clamp that epsilon induces)
    """
    p = np.asarray(p_alt1, float)
    y = np.asarray(outcome, float)

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
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


def ece(p_alt1, outcome, n_bins=10, strategy="uniform") -> float:
    """Expected calibration error: n-weighted mean |empirical - predicted|."""
    t = reliability_table(p_alt1, outcome, n_bins=n_bins, strategy=strategy)
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

    Mirrors the `_Tee` used by the existing analysis scripts under my_code/EDA_set/
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
