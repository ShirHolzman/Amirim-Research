"""
Unit tests for metrics.py.

The project has no pytest dependency (see the venv at the repo root), so every
test here is a plain function that raises AssertionError on failure, and the
module doubles as its own runner:

    python my_code/catie_calibration/tests/test_metrics.py

The functions are named `test_*` and take no fixtures, so `pytest` will collect
and run them unchanged if it is ever added to the environment.

WHAT IS BEING PROTECTED
-----------------------
1. `ece_ci` under strategy="quantile". Quantile bin edges are a FUNCTION OF THE
   DATA. `ece_ci` bootstraps by resampling subjects; if the edges are recomputed
   inside each replicate, every replicate scores a different binning and the
   resulting interval is a CI for a moving target rather than for the ECE of the
   binning that the point estimate used. `reliability_table_ci` already pinned
   its edges; `ece_ci` did not, because it delegated to `ece(...)` which called
   `_bin_edges` on the resampled p. These tests fix the edges once and prove it.
2. `cluster_bootstrap`'s two documented reductions to `bootstrap_ci` (the Day 0
   sanity checks, which lived in a scratch script until now).
3. `reliability_table_ci`'s edges, for both strategies.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import metrics as M  # noqa: E402


# ── synthetic data ───────────────────────────────────────────────────────────
def _clumped_data(n_subjects=40, n_trials=25, seed=0):
    """Subjects of two types, forecasts clumped near 0.15 and near 0.85.

    Deliberately built so quantile edges MOVE under resampling: the two clumps
    are far apart and the mix of types in a bootstrap resample of only 40
    subjects swings enough to shift the interior quantiles by a visible amount.
    That is what makes the moving-edge bug detectable rather than a rounding
    difference.
    """
    rng = np.random.default_rng(seed)
    ids, p, y = [], [], []
    for s in range(n_subjects):
        centre = 0.15 if s % 2 == 0 else 0.85
        ps = np.clip(rng.normal(centre, 0.07, n_trials), 0.01, 0.99)
        ids.append(np.full(n_trials, s))
        p.append(ps)
        y.append(rng.random(n_trials) < ps)
    return (np.concatenate(ids), np.concatenate(p),
            np.concatenate(y).astype(float))


def _ece_ci_moving_edges(p, y, ids, n_bins=10, strategy="quantile", n_boot=400,
                         seed=M.DEFAULT_SEED):
    """The OLD `ece_ci`: edges re-derived inside every replicate.

    Kept here, and only here, as the thing the fix is tested against.
    """
    return M.cluster_bootstrap(
        lambda pp, yy: M.ece(pp, yy, n_bins=n_bins, strategy=strategy),
        ids, p, y, n_boot=n_boot, seed=seed)


# ── 1. ece_ci: the point value, and edges held fixed ─────────────────────────
def test_ece_ci_point_matches_ece():
    """The CI's point estimate is the plain ECE, on both strategies."""
    ids, p, y = _clumped_data()
    for strategy in ("uniform", "quantile"):
        point, lo, hi = M.ece_ci(p, y, ids, n_bins=10, strategy=strategy, n_boot=200)
        direct = M.ece(p, y, n_bins=10, strategy=strategy)
        assert point == direct, f"{strategy}: {point!r} != {direct!r}"
        assert lo <= point <= hi, f"{strategy}: point {point} outside [{lo}, {hi}]"


def test_ece_ci_computes_edges_once():
    """`_bin_edges` is called exactly once per `ece_ci`, not once per replicate."""
    ids, p, y = _clumped_data()
    calls = []
    original = M._bin_edges

    def counting(pp, n_bins=10, strategy="uniform"):
        calls.append(strategy)
        return original(pp, n_bins, strategy)

    M._bin_edges = counting
    try:
        M.ece_ci(p, y, ids, n_bins=10, strategy="quantile", n_boot=200)
    finally:
        M._bin_edges = original
    assert len(calls) == 1, (
        f"_bin_edges called {len(calls)} times; edges must be derived once on the "
        f"full sample and reused by every bootstrap replicate")


def test_ece_ci_edges_constant_across_replicates():
    """Every replicate scores the SAME binning."""
    ids, p, y = _clumped_data()
    seen = []
    original = M.reliability_table

    def recording(p_alt1, outcome, n_bins=10, strategy="uniform", min_count=1,
                  edges=None):
        seen.append(None if edges is None else np.asarray(edges, float).copy())
        return original(p_alt1, outcome, n_bins=n_bins, strategy=strategy,
                        min_count=min_count, edges=edges)

    M.reliability_table = recording
    try:
        M.ece_ci(p, y, ids, n_bins=10, strategy="quantile", n_boot=200)
    finally:
        M.reliability_table = original

    assert len(seen) > 100, "expected one reliability_table call per replicate"
    assert all(e is not None for e in seen), (
        "at least one replicate was scored with edges=None, i.e. it re-derived its "
        "own quantile edges from the resampled p")
    first = seen[0]
    for k, e in enumerate(seen[1:], 1):
        assert e.shape == first.shape and np.array_equal(e, first), (
            f"replicate {k} used different bin edges:\n  {e}\nvs\n  {first}")


def test_ece_ci_quantile_differs_from_moving_edges():
    """The fix changes the answer -- i.e. the old behaviour would fail the above.

    This is the guard against a fix that is a no-op on this data: if the fixed-edge
    and moving-edge intervals were identical, the three tests above would pass
    vacuously.
    """
    ids, p, y = _clumped_data()
    fixed = M.ece_ci(p, y, ids, n_bins=10, strategy="quantile", n_boot=400)
    moving = _ece_ci_moving_edges(p, y, ids, n_bins=10, strategy="quantile",
                                  n_boot=400)
    assert fixed[0] == moving[0], "the point estimate is edges-on-full-data either way"
    spread = max(abs(fixed[1] - moving[1]), abs(fixed[2] - moving[2]))
    assert spread > 1e-6, (
        f"fixed-edge CI {fixed[1:]} is indistinguishable from the moving-edge CI "
        f"{moving[1:]}; this dataset does not exercise the bug")


def test_ece_ci_invariant_to_row_order():
    """Shuffling the trial rows must not move the CI.

    `cluster_bootstrap` sorts by subject before resampling, and the edges now come
    from the full sample, so both the binning and the replicate draws are properties
    of the data rather than of the order it arrived in.
    """
    ids, p, y = _clumped_data()
    perm = np.random.default_rng(7).permutation(len(ids))
    a = M.ece_ci(p, y, ids, n_bins=10, strategy="quantile", n_boot=200)
    b = M.ece_ci(p[perm], y[perm], ids[perm], n_bins=10, strategy="quantile",
                 n_boot=200)
    assert np.allclose(a, b, rtol=0, atol=1e-12), f"{a} != {b}"


def test_ece_edges_argument_default_is_unchanged():
    """Passing the edges `ece` would have derived itself changes nothing."""
    ids, p, y = _clumped_data()
    del ids
    for strategy in ("uniform", "quantile"):
        edges = M._bin_edges(p, 10, strategy)
        assert M.ece(p, y, n_bins=10, strategy=strategy) == \
               M.ece(p, y, n_bins=10, strategy=strategy, edges=edges)


# ── 2. cluster_bootstrap reductions to bootstrap_ci (Day 0 sanity checks) ────
def test_cluster_bootstrap_one_trial_per_subject_matches_bootstrap_ci():
    """One trial per subject: an ordinary trial bootstrap, bit-for-bit."""
    rng = np.random.default_rng(3)
    n = 200
    ids = np.arange(n)
    v = rng.normal(size=n)
    a = M.cluster_bootstrap(np.mean, ids, v, n_boot=500, seed=M.DEFAULT_SEED)
    b = M.bootstrap_ci(v, ids, statistic=np.mean, n_boot=500, seed=M.DEFAULT_SEED)
    assert a == b, f"cluster_bootstrap {a} != bootstrap_ci {b}"


def test_cluster_bootstrap_mean_of_subject_means_matches_bootstrap_ci():
    """Equal trial counts: the flat mean IS the mean of subject means.

    Not bit-for-bit, unlike the one-trial-per-subject case above: the two routes
    add the same numbers in a different order (one flat pass over n_subj*n_trials
    values, versus a mean of per-subject means), so they agree to floating-point
    summation error, ~1e-17 here. The draws themselves are identical.
    """
    rng = np.random.default_rng(11)
    n_subj, n_trials = 60, 12
    ids = np.repeat(np.arange(n_subj), n_trials)
    v = rng.normal(size=n_subj * n_trials)
    a = M.cluster_bootstrap(np.mean, ids, v, n_boot=500, seed=M.DEFAULT_SEED)
    b = M.bootstrap_ci(v, ids, statistic=np.mean, n_boot=500, seed=M.DEFAULT_SEED)
    assert np.allclose(a, b, rtol=0, atol=1e-12), \
        f"cluster_bootstrap {a} != bootstrap_ci {b}"


def test_cluster_bootstrap_resamples_subjects_not_trials():
    """A subject-level signal must survive resampling as a whole-subject unit.

    With every trial of a subject identical, the clustered CI is the CI of the
    subject-level values; a trial bootstrap would give a far narrower one.
    """
    rng = np.random.default_rng(5)
    n_subj, n_trials = 30, 50
    per_subject = rng.normal(0.0, 1.0, n_subj)
    ids = np.repeat(np.arange(n_subj), n_trials)
    v = np.repeat(per_subject, n_trials)
    _, lo, hi = M.cluster_bootstrap(np.mean, ids, v, n_boot=1000)
    _, lo_s, hi_s = M.bootstrap_ci(per_subject, np.arange(n_subj), n_boot=1000)
    assert np.isclose(hi - lo, hi_s - lo_s, rtol=1e-9)
    trial_se = v.std(ddof=1) / np.sqrt(len(v))
    assert (hi - lo) > 6 * trial_se, (
        "clustered CI is no wider than a trial-level one -- clustering is not "
        "being applied")


# ── 3. reliability_table_ci edges ────────────────────────────────────────────
def test_reliability_table_ci_edges_are_fixed():
    """Both strategies: one edge computation, and the table's lo/hi bracket the rate."""
    ids, p, y = _clumped_data()
    for strategy in ("uniform", "quantile"):
        calls = []
        original = M._bin_edges

        def counting(pp, n_bins=10, strategy=strategy):
            calls.append(strategy)
            return original(pp, n_bins, strategy)

        M._bin_edges = counting
        try:
            t = M.reliability_table_ci(p, y, ids, n_bins=10, strategy=strategy,
                                       n_boot=200)
        finally:
            M._bin_edges = original
        assert len(calls) == 1, (
            f"{strategy}: _bin_edges called {len(calls)} times, expected once")
        assert not t.empty
        plain = M.reliability_table(p, y, n_bins=10, strategy=strategy)
        assert np.array_equal(t["bin"].to_numpy(), plain["bin"].to_numpy())
        assert np.allclose(t["empirical"], plain["empirical"])
        assert ((t["emp_lo"] <= t["empirical"] + 1e-12) &
                (t["empirical"] <= t["emp_hi"] + 1e-12)).all(), (
            f"{strategy}: a bin's empirical rate falls outside its own CI")


# ── runner ───────────────────────────────────────────────────────────────────
def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:                      # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}\n        {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
