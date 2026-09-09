"""
Precompute CATIE's parameter-free state tensors for every subject and cache them.

WHY THIS IS POSSIBLE (the enabling structure of the whole project): the CATIE
likelihood is conditioned on the participant's observed choices, so the model never
samples. Every internal state variable -- reward means, observed SDs, surprise,
contingency tables, g, H, c_prev -- is a function of the DATA ONLY, independent of
tau, epsilon and phi. Only K alters the recursion, so K is handled by enumeration.

Given the cache, evaluating the likelihood at any (tau, eps, phi) is pure vectorised
arithmetic over (n_subjects x n_trials) matrices -- no Python loop, no re-running the
recursion, no autodiff. That turns parameter fitting from an overnight job into
milliseconds per evaluation, which is what makes dense grids, multi-start
optimisation, profile likelihoods and bootstrap all affordable in Phase 3.

Caches per split and per k:
    H, b, c_prev, s_prev, sbar_prev, g   each (n_subjects, 100), float64
    y  (n_subjects, 100) int8   -- the observed choice (1 = biased alternative)
    subject_id, schedule                 -- alignment metadata

Test (schedules 1, 8, 10) is deliberately NOT cached: it is touched exactly once, at
the very end of the project, and having it absent from the cache makes accidental
use structurally impossible rather than merely discouraged.

Run:  cd my_code/catie_calibration && python -m catie.cache
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import pandas as pd

from .core import state_tensors

HERE = pathlib.Path(__file__).parent          # catie/
CACHE_DIR = HERE.parent / "cache"
DATA_DIR = HERE.parent / "data"

K_VALUES = (0, 1, 2, 3)
N_TRIALS = 100
TENSORS = ("H", "b", "c_prev", "s_prev", "sbar_prev", "g")

# Test is excluded on purpose -- see module docstring.
SPLITS = ("training", "eda", "schedule_0")


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"cleaned_{name}.csv")
    df["chose_biased"] = (df["is_biased_choice"].astype(str).str.upper() == "TRUE").astype(int)
    df = df.sort_values(["subject_id", "trial_number"]).reset_index(drop=True)
    assert (df.groupby("subject_id").size() == N_TRIALS).all(), \
        f"{name}: every subject must have exactly {N_TRIALS} trials"
    assert not df["schedule"].isin({"schedule_1", "schedule_8", "schedule_10"}).any(), \
        f"{name}: Test schedule present -- Test must never be cached"
    return df


def build_split(name: str) -> None:
    df = load_split(name)
    subjects = df["subject_id"].drop_duplicates().to_numpy()
    n = len(subjects)
    print(f"[{name}] {n:,} subjects x {len(K_VALUES)} k-values", flush=True)

    sched = df.groupby("subject_id", sort=False)["schedule"].first().reindex(subjects).to_numpy()
    y = np.zeros((n, N_TRIALS), dtype=np.int8)
    store = {k: {t: np.zeros((n, N_TRIALS)) for t in TENSORS} for k in K_VALUES}

    t0 = time.time()
    for i, (sid, d) in enumerate(df.groupby("subject_id", sort=False)):
        r1 = d["biased_reward"].to_numpy()
        r2 = d["unbiased_reward"].to_numpy()
        c1 = d["chose_biased"].to_numpy().astype(bool)
        y[i] = c1.astype(np.int8)
        for k in K_VALUES:
            # mode="fixed": the corrected heuristic branch is the project baseline.
            # The published variant is recoverable from the same cache by zeroing b
            # (they differ ONLY in b -- see catie.core.CatieState.as_published).
            st = state_tensors(r1, r2, c1, k=k, mode="fixed")
            for t in TENSORS:
                store[k][t][i] = getattr(st, t)
        if (i + 1) % 250 == 0 or (i + 1) == n:
            el = time.time() - t0
            print(f"   [{i+1}/{n}] {el:.0f}s elapsed, ~{el/(i+1)*(n-i-1):.0f}s left", flush=True)

    assert (subjects == pd.unique(df["subject_id"])).all(), "subject ordering drifted"

    out = {"subject_id": subjects.astype(str), "schedule": sched.astype(str), "y": y}
    for k in K_VALUES:
        for t in TENSORS:
            out[f"k{k}_{t}"] = store[k][t]
    dest = CACHE_DIR / f"state_{name}.npz"
    np.savez_compressed(dest, **out)
    print(f"   wrote {dest.name} ({dest.stat().st_size/1e6:.1f} MB)\n", flush=True)


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"cache -> {CACHE_DIR}")
    print(f"k values: {K_VALUES}   (Test split deliberately not cached)\n")
    for name in SPLITS:
        build_split(name)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
