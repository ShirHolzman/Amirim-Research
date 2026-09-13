"""
Verification for 02_reliability/reliability.py -- see sub_plans/02_reliability.md
"Verification" section for what each check is for and why.

Ten checks, in the same order as the sub-plan, with two adjustments made while
writing this file (both reported in the sub-plan's Outcome section, not silently
applied):

  * Check 2's second half ("...and equal the values likelihood_test.py reproduces
    from the paper") does not apply as written: likelihood_test.py only validates
    the PUBLISHED (buggy) model against the paper -- the paper never reports a
    number for the corrected model, and likelihood_test's own pooled corrected
    figure uses trial 1 KEPT, while this stage drops it. There is no external
    reference to compare a drop_first=True corrected E[p]/E[log p] against. What
    remains, and is exact, is checking this stage's arithmetic against
    catie.likelihood.mean_p/mean_log_p directly (to float precision) -- that is
    what test_anchor_to_validated_base does.
  * Check 9 (determinism) as written means "rerun the whole 16-minute script and
    diff the CSVs". That cost is not worth paying on every test invocation. The
    only randomness anywhere in the pipeline is inside cluster_bootstrap's RNG,
    which is freshly seeded per call -- so the real content of the check is
    "does the SAME bootstrap call give the SAME answer twice", which is cheap.
    test_bootstrap_is_deterministic checks that directly, on the actual training
    data. (The full-script rerun was also done once, by hand, when the CSVs were
    first produced -- see the Outcome section.)

Run:  python -m pytest tests/02_reliability -q
      python tests/02_reliability/reliability_test.py
"""

from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
CATIE_CAL = HERE.parents[1]                       # catie_calibration/
sys.path.insert(0, str(CATIE_CAL))
sys.path.insert(0, str(CATIE_CAL / "02_reliability"))

from catie.likelihood import StateCache, p_choice_matrix, mean_p, mean_log_p  # noqa: E402
from catie import metrics as M  # noqa: E402
import reliability as R  # noqa: E402

STAGE_DIR = CATIE_CAL / "02_reliability"
BINS_CSV = STAGE_DIR / "reliability.csv"
SUMMARY_CSV = STAGE_DIR / "reliability_summary.csv"
README = STAGE_DIR / "README.md"

N_TRAINING_SCORED = 146_817
N_DOUBLED_SCORED = 293_634
N_EDA_SCORED = 49_104
N_SPLIT = {"training": N_TRAINING_SCORED, "eda": N_EDA_SCORED}


# ── 1. p_alt1 round-trip ──────────────────────────────────────────────────────
def test_p_alt1_round_trip():
    cache = StateCache("training")
    pc = p_choice_matrix(cache)
    p1 = np.where(cache.y.astype(bool), pc, 1.0 - pc)
    reconstructed = np.where(cache.y.astype(bool), p1, 1.0 - p1)
    diff = np.abs(reconstructed - pc).max()
    assert diff < 1e-12, f"p_alt1 round-trip max abs diff {diff} >= 1e-12"


# ── 2. anchor to the validated base ───────────────────────────────────────────
def test_anchor_to_validated_base():
    for split in ("training", "eda"):
        flat = R.load_flat(split)
        cache = StateCache(split)
        want_p = mean_p(cache, drop_first=True)
        want_lp = mean_log_p(cache, drop_first=True)
        got_p = M.e_p(flat["pc"])
        got_lp = M.e_log_p(flat["pc"])
        assert abs(got_p - want_p) < 1e-12, f"{split}: E[p] {got_p} != {want_p}"
        assert abs(got_lp - want_lp) < 1e-12, f"{split}: E[log p] {got_lp} != {want_lp}"

        row = pd.read_csv(SUMMARY_CSV).query(
            "diagram == 'baseline_biased'" if split == "training" else "diagram == 'doubled_eda'"
        )
        if split == "eda":
            continue  # doubled_eda's e_p/e_log_p is over eda['pc'], checked below instead
        assert abs(row["e_p"].iloc[0] - want_p) < 1e-9
        assert abs(row["e_log_p"].iloc[0] - want_lp) < 1e-9


def test_eda_summary_matches_direct_computation():
    flat = R.load_flat("eda")
    row = pd.read_csv(SUMMARY_CSV).query("diagram == 'doubled_eda'").iloc[0]
    assert abs(row["e_p"] - M.e_p(flat["pc"])) < 1e-9
    assert abs(row["e_log_p"] - M.e_log_p(flat["pc"])) < 1e-9


# ── 3. doubling identities ─────────────────────────────────────────────────────
def test_doubling_identities():
    flat = R.load_flat("training")
    p_d, y_d, _ = R.double(flat["p1"], flat["y"], flat["subj"])
    assert abs(p_d.mean() - 0.5) < 1e-12, f"doubled mean predicted {p_d.mean()} != 0.5"
    assert abs(y_d.mean() - 0.5) < 1e-12, f"doubled mean empirical {y_d.mean()} != 0.5"

    t = M.reliability_table(p_d, y_d, n_bins=10, strategy="uniform")
    t_flipped = M.reliability_table(1.0 - p_d, 1.0 - y_d, n_bins=10, strategy="uniform")
    # bin b of the flipped table should equal bin (n_bins-1-b) of the original, with
    # predicted/empirical mirrored around 0.5 and gap sign-flipped.
    n_bins = len(t)
    for b in range(n_bins):
        mirror = t_flipped.iloc[n_bins - 1 - b]
        orig = t.iloc[b]
        assert orig["n"] == mirror["n"], f"bin {b}: n mismatch under label flip"
        assert abs((1 - orig["predicted"]) - mirror["predicted"]) < 1e-9
        assert abs((1 - orig["empirical"]) - mirror["empirical"]) < 1e-9
        assert abs(orig["gap"] + mirror["gap"]) < 1e-9


# ── 4. binning conservation ────────────────────────────────────────────────────
def test_binning_conservation():
    bins = pd.read_csv(BINS_CSV)
    for (diagram, split, stratum, n_bins, strategy), g in bins.groupby(
        ["diagram", "split", "stratum", "n_bins", "strategy"]
    ):
        n_total = int(g["n"].sum())
        if diagram == "baseline_biased":
            expect = N_SPLIT[split]
        elif "run" not in diagram:
            expect = N_SPLIT[split] * 2
        else:
            continue  # run-length strata: total checked separately (test 7)
        assert n_total == expect, (
            f"{diagram}/{split}/{stratum}/{n_bins}/{strategy}: sum(n)={n_total} != {expect}"
        )
        w_pred = np.average(g["predicted"], weights=g["n"])
        w_emp = np.average(g["empirical"], weights=g["n"])
        if "doubled" in diagram:
            assert abs(w_pred - 0.5) < 1e-9 and abs(w_emp - 0.5) < 1e-9
    # run-length strata partition the training scored trials exactly
    run_bins = bins[bins["diagram"].str.startswith("doubled_run")]
    per_stratum = run_bins.groupby("stratum")["n"].sum() // 2  # undo doubling
    assert int(per_stratum.sum()) == N_TRAINING_SCORED, (
        f"run-length strata sum to {int(per_stratum.sum())}, expected {N_TRAINING_SCORED}"
    )


# ── 5. run-length brute force ─────────────────────────────────────────────────
def _brute_force_run_length(y_subj: np.ndarray) -> np.ndarray:
    """Naive double loop, one subject's row of y (length n_trials)."""
    n_trials = len(y_subj)
    run = np.zeros(n_trials, dtype=np.int64)
    if n_trials > 1:
        run[1] = 1
    for t in range(2, n_trials):
        run[t] = run[t - 1] + 1 if y_subj[t - 1] == y_subj[t - 2] else 1
    return run


def test_run_length_brute_force():
    cache = StateCache("training")
    rng = np.random.default_rng(42)
    picks = rng.choice(cache.n_subjects, size=50, replace=False)
    run_vectorised = R.compute_run_length(cache.y)
    for i in picks:
        expected = _brute_force_run_length(cache.y[i])
        assert np.array_equal(run_vectorised[i], expected), f"subject index {i} mismatch"


# ── 6. run-length has no leakage ──────────────────────────────────────────────
def test_run_length_no_leakage():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(20, 100)).astype(np.int8)
    run_before = R.compute_run_length(y)
    for t in (2, 10, 50, 97):
        y2 = y.copy()
        y2[:, t] = 1 - y2[:, t]
        run_after = R.compute_run_length(y2)
        assert np.array_equal(run_after[:, t], run_before[:, t]), (
            f"flipping y[:, {t}] changed run[:, {t}] -- leakage into the trial being stratified"
        )
        assert not np.array_equal(run_after[:, t + 1], run_before[:, t + 1]), (
            f"flipping y[:, {t}] left run[:, {t+1}] unchanged -- history-only property violated"
        )


# ── 7. strata partition ───────────────────────────────────────────────────────
def test_strata_partition():
    tr = R.load_flat("training")
    masks = []
    for lo, hi, label in R.RUN_STRATA:
        mask = (tr["run"] >= lo) & ((tr["run"] <= hi) if hi is not None else True)
        masks.append(mask)
    stacked = np.vstack(masks)
    assert (stacked.sum(axis=0) == 1).all(), "run-length strata are not a partition"
    counts = stacked.sum(axis=1)
    assert (counts > 0).all(), "an empty run-length stratum"
    assert counts.sum() == N_TRAINING_SCORED
    min_per_bin = counts.min() / 10
    if min_per_bin < 200:
        print(f"   ! smallest run-length stratum has only {min_per_bin:.0f} rows/bin "
              f"at 10 bins (< 200)")


# ── 8. CI sanity ───────────────────────────────────────────────────────────────
def test_ci_sanity():
    from scipy.stats import spearmanr

    bins = pd.read_csv(BINS_CSV)
    tol = 1e-9
    bad = bins[(bins["emp_lo"] > bins["empirical"] + tol) |
               (bins["empirical"] > bins["emp_hi"] + tol)]
    assert bad.empty, f"{len(bad)} bin(s) with empirical rate outside its own CI"

    ten_bin = bins[bins["n_bins"] == 10]
    neg = 0
    total = 0
    for _, g in ten_bin.groupby(["diagram", "split", "stratum"]):
        if len(g) < 4:
            continue
        width = g["emp_hi"] - g["emp_lo"]
        rho, _ = spearmanr(g["n"], width)
        total += 1
        neg += rho < 0
    assert total > 0 and neg / total >= 0.8, (
        f"CI width is not predominantly narrower with more n: {neg}/{total} curves negative"
    )


# ── 9. bootstrap determinism (see module docstring for why this replaces a full rerun) ─
def test_bootstrap_is_deterministic():
    tr = R.load_flat("training")
    a = M.reliability_table_ci(tr["p1"], tr["y"], tr["subj"], n_bins=10, n_boot=300, seed=42)
    b = M.reliability_table_ci(tr["p1"], tr["y"], tr["subj"], n_bins=10, n_boot=300, seed=42)
    pd.testing.assert_frame_equal(a, b)
    ea = M.ece_ci(tr["p1"], tr["y"], tr["subj"], n_bins=10, n_boot=300, seed=42)
    eb = M.ece_ci(tr["p1"], tr["y"], tr["subj"], n_bins=10, n_boot=300, seed=42)
    assert ea == eb


# ── 10. README fidelity ────────────────────────────────────────────────────────
def test_readme_numbers_match_csv():
    if not README.exists():
        print("   ! README.md not written yet -- skipping (run again after step 6)")
        return
    text = README.read_text(encoding="utf-8")
    numbers = [float(m.replace("−", "-")) for m in
               re.findall(r"[-−]?\d+\.\d{3,}", text)]
    bins = pd.read_csv(BINS_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    pool = np.concatenate([
        bins[["lo", "hi", "predicted", "empirical", "gap", "emp_lo", "emp_hi"]].to_numpy().ravel(),
        summary[["ece", "ece_lo", "ece_hi", "mce", "brier", "e_p", "e_log_p"]].to_numpy().ravel(),
    ])
    unmatched = [n for n in numbers if not np.any(np.abs(pool - n) < 5e-4)]
    assert not unmatched, f"README numbers with no matching CSV value: {unmatched}"


# ── runner ─────────────────────────────────────────────────────────────────────
def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}\n        {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
