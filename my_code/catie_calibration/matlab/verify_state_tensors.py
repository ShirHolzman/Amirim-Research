"""
Element-by-element cross-check of catie_core.state_tensors() against MATLAB's own
internal per-trial variables, extracted directly from an instrumented copy of the
original .m likelihood function.

Every prior check in this project (golden_test.py, 01_bug_correction/) compared
only the FINAL choice probability, end-to-end. This compares the intermediate
state tensors themselves -- H, b, c_prev, s_prev, sbar_prev, g -- trial by trial,
subject by subject, for k in {0,1,2}, against real MATLAB internals across the
full ~3,300-subject population. It is the strongest test of state_tensors() in
this project: a bug that happened to leave the final probability unchanged (an
implausible but not structurally impossible compensating-error scenario) would
still be caught here, because it compares intermediate values that the end-to-end
checks never look at.

Source of ground truth:
    matlab/CATIE_FIXED/COMPETITION_CATIE_schedule_choice_probability_INSTRUMENTED.m
    matlab/export_state_tensors.m

Prerequisite: run export_state_tensors.m in MATLAB first --
    matlab -batch "run('my_code/catie_calibration/matlab/export_state_tensors.m')"

Run:  python my_code/catie_calibration/matlab/verify_state_tensors.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from catie_core import state_tensors  # noqa: E402

MATLAB_CSV = HERE / "results" / "state_tensors_matlab.csv"
DATA_DIR = HERE.parent / "data"
EDA_CSV = HERE.parent.parent / "EDA_set" / "processing" / "eda_with_catie_probabilities.csv"

# MATLAB's CSV round-trips through text, so allow a little more slack than the
# 1e-12 used for pure in-process Python comparisons. Any real logic error would
# show up as a deviation of order 0.01-0.3 (a whole mode's worth of probability
# mass, or a flipped 0/1 flag), so this tolerance still separates "text rounding"
# from "actually wrong" by ten orders of magnitude.
TOLERANCE = 1e-9

TENSOR_NAMES = ("H", "b", "c_prev", "s_prev", "sbar_prev", "g")


def load_all_subjects() -> pd.DataFrame:
    """Same population and subject_id convention as 01_bug_correction/bug_benchmark.py
    and sanitize_splits.py: EDA (from the pre-existing reference CSV) plus the three
    splits sanitize_splits.py produces.
    """
    frames = []

    eda = pd.read_csv(EDA_CSV)
    eda["subject_id"] = eda["schedule"] + "/" + eda["subject_file"]
    frames.append(eda[["subject_id", "trial_number", "biased_reward",
                       "unbiased_reward", "is_biased_choice"]])

    for name in ("training", "test", "schedule_0"):
        path = DATA_DIR / f"cleaned_{name}.csv"
        if not path.exists():
            print(f"  !! {path} missing -- run sanitize_splits.py first")
            continue
        df = pd.read_csv(path)
        frames.append(df[["subject_id", "trial_number", "biased_reward",
                          "unbiased_reward", "is_biased_choice"]])

    out = pd.concat(frames, ignore_index=True)
    out["c1"] = out["is_biased_choice"].astype(str).str.upper() == "TRUE"
    out = out.sort_values(["subject_id", "trial_number"]).reset_index(drop=True)
    return out


def main() -> int:
    if not MATLAB_CSV.exists():
        print(f"missing {MATLAB_CSV}")
        print("run: matlab -batch \"run('my_code/catie_calibration/matlab/export_state_tensors.m')\"")
        return 1

    print("loading MATLAB export ...")
    mat = pd.read_csv(MATLAB_CSV)
    print(f"  {len(mat):,} rows, {mat['subject_id'].nunique():,} subjects, "
          f"k in {sorted(mat['k'].unique())}")

    print("loading sanitized subject data ...")
    subj = load_all_subjects()
    by_r1 = {sid: g["biased_reward"].to_numpy() for sid, g in subj.groupby("subject_id", sort=False)}
    by_r2 = {sid: g["unbiased_reward"].to_numpy() for sid, g in subj.groupby("subject_id", sort=False)}
    by_c1 = {sid: g["c1"].to_numpy() for sid, g in subj.groupby("subject_id", sort=False)}

    mat_subjects = set(mat["subject_id"].unique())
    py_subjects = set(by_r1.keys())
    missing = mat_subjects - py_subjects
    if missing:
        print(f"  !! {len(missing)} MATLAB subjects have no matching sanitized data "
              f"(skipped): {sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}")

    max_dev = {t: 0.0 for t in TENSOR_NAMES}
    worst = {t: None for t in TENSOR_NAMES}
    n_compared = 0
    n_trials_compared = 0

    for k in (0, 1, 2):
        mk = mat[mat["k"] == k]
        for sid, g in mk.groupby("subject_id", sort=False):
            if sid not in by_r1:
                continue
            g = g.sort_values("trial")
            if len(g) != 100:
                print(f"  !! {sid} (k={k}): expected 100 trials, got {len(g)} -- skipped")
                continue

            st = state_tensors(by_r1[sid], by_r2[sid], by_c1[sid], k=k, mode="fixed")

            for t in ("H", "b", "c_prev", "s_prev", "sbar_prev"):
                dev = float(np.abs(getattr(st, t) - g[t].to_numpy()).max())
                if dev > max_dev[t]:
                    max_dev[t] = dev
                    worst[t] = (sid, k)

            # g: MATLAB's trial-1 value is real (computed unconditionally every
            # trial); Python's index 0 is intentionally left at its initialised 0
            # (trial 1 never reads g -- probability_from_state hardcodes p=0.5
            # there). Compare indices 1..99 (MATLAB trials 2..100) only.
            dev = float(np.abs(st.g[1:] - g["g"].to_numpy()[1:]).max())
            if dev > max_dev["g"]:
                max_dev["g"] = dev
                worst["g"] = (sid, k)

            n_compared += 1
            n_trials_compared += 100

    print(f"\ncompared {n_compared:,} (subject, k) pairs, {n_trials_compared:,} trials "
          f"({n_compared // 3:,} subjects x 3 k-values)")
    print("=" * 70)
    ok = True
    for t in TENSOR_NAMES:
        status = "PASS" if max_dev[t] < TOLERANCE else "FAIL"
        ok &= max_dev[t] < TOLERANCE
        note = f"  (worst: subject={worst[t][0]!r}, k={worst[t][1]})" if max_dev[t] > 0 else ""
        print(f"[{status}] {t:<10} max |dev| = {max_dev[t]:.3e}{note}")
    print("=" * 70)
    print("STATE TENSOR CROSS-CHECK PASSED" if ok else "STATE TENSOR CROSS-CHECK FAILED")
    if ok:
        print("\nEvery one of H, b, c_prev, s_prev, sbar_prev, g in catie_core.py's")
        print("state_tensors() matches MATLAB's own internal loop variables, trial by")
        print("trial, across the full sanitized population and all three k values.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
