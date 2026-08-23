"""
Sanitize the competition's per-subject CSVs into one tidy CSV per data split.

Exclusion rule: drop the files the curators tagged "..._INVALID_BIAS.csv". On all
12 schedules this reproduces the per-schedule N reported in Supplementary Data 1
exactly (3,332 valid of 3,386 raw).

Source columns are renamed to this project's canonical names, and observed_reward
is derived (it is not stored, but equals the reward on the chosen side):

    trial_number             ->  trial_number
    is_choice_alternative_1  ->  is_biased_choice
    reward_alternative_1     ->  biased_reward
    reward_alternative_2     ->  unbiased_reward
                             ->  observed_reward   (derived)

Never open the outputs in Excel -- it silently rewrites values (a previous
cleaned CSV lost its `time` column that way, "05-25-2020 15:11:02.738300" ->
"11:02.7", and had its booleans re-cased).

Run:  python my_code/catie_calibration/sanitize_splits.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

MY_CODE = pathlib.Path(__file__).parent.parent
OUT_DIR = pathlib.Path(__file__).parent / "data"

N_TRIALS = 100
MIN_CHOICES_PER_SIDE = 5

SOURCE_COLUMNS = ["trial_number", "is_choice_alternative_1",
                  "reward_alternative_1", "reward_alternative_2"]

RENAME = {
    "is_choice_alternative_1": "is_biased_choice",
    "reward_alternative_1": "biased_reward",
    "reward_alternative_2": "unbiased_reward",
}

OUTPUT_COLUMNS = ["subject_id", "schedule", "split", "subject_file", "trial_number",
                  "is_biased_choice", "biased_reward", "unbiased_reward",
                  "observed_reward"]

# split name -> directory holding its schedule_N subfolders
SPLITS = {
    "training": MY_CODE / "Training_set",
    "test": MY_CODE / "Test_set",
    "eda": MY_CODE / "EDA_set",
    "schedule_0": MY_CODE / "Schedule0_set",
}


def sanitize_split(name: str, split_dir: pathlib.Path) -> pd.DataFrame | None:
    if not split_dir.exists():
        print(f"  !! {split_dir} does not exist -- skipping")
        return None

    frames = []
    kept = 0
    dropped = {"invalid_bias": 0, "bad_trials": 0, "missing_cols": 0, "read_error": 0}

    for sched_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for csv in sorted(sched_dir.glob("*.csv")):
            if "invalid_bias" in csv.name.lower():
                dropped["invalid_bias"] += 1
                continue
            try:
                df = pd.read_csv(csv)
            except Exception as exc:  # noqa: BLE001 -- report and continue
                print(f"  !! read error {csv.name}: {exc}")
                dropped["read_error"] += 1
                continue

            df.columns = [c.strip() for c in df.columns]
            if any(col not in df.columns for col in SOURCE_COLUMNS):
                dropped["missing_cols"] += 1
                continue

            # The CATIE likelihood hardcodes nTrials = 100 and mis-indexes otherwise.
            if len(df) != N_TRIALS or sorted(df["trial_number"]) != list(range(N_TRIALS)):
                dropped["bad_trials"] += 1
                continue

            df = df[SOURCE_COLUMNS].rename(columns=RENAME)
            df["is_biased_choice"] = (
                df["is_biased_choice"].astype(str).str.strip().str.lower() == "true")
            df["observed_reward"] = np.where(
                df["is_biased_choice"], df["biased_reward"], df["unbiased_reward"])

            df["schedule"] = sched_dir.name
            df["split"] = name
            df["subject_file"] = csv.name
            df["subject_id"] = f"{sched_dir.name}/{csv.name}"
            frames.append(df)
            kept += 1

    if not frames:
        print(f"  !! no valid subjects found in {split_dir}")
        return None

    out = pd.concat(frames, ignore_index=True)[OUTPUT_COLUMNS]
    out = out.sort_values(["schedule", "subject_id", "trial_number"]).reset_index(drop=True)

    assert out["subject_id"].nunique() == kept, (
        f"subject_id collision: {kept} kept but {out['subject_id'].nunique()} unique ids")
    sizes = out.groupby("subject_id").size()
    assert (sizes == N_TRIALS).all(), (
        f"expected {N_TRIALS} trials per subject, got min={sizes.min()} max={sizes.max()}")

    # The curators' INVALID_BIAS tag should already cover every subject the old
    # inferred rule would have dropped. Assert rather than filter: if this ever
    # fires, the two criteria have diverged and that is a finding, not noise.
    per_subject_min = out.groupby("subject_id")["is_biased_choice"].agg(
        lambda s: min(int(s.sum()), int((~s).sum())))
    offenders = per_subject_min[per_subject_min < MIN_CHOICES_PER_SIDE]
    assert offenders.empty, (
        f"{len(offenders)} subject(s) below {MIN_CHOICES_PER_SIDE} choices per side "
        f"survived the INVALID_BIAS filter: {list(offenders.index[:5])}")

    print(f"  kept {kept} subjects ({len(out):,} rows), dropped {sum(dropped.values())}")
    for reason, n in dropped.items():
        if n:
            print(f"     - {reason}: {n}")
    by_sched = out.groupby("schedule")["subject_id"].nunique()
    print("     subjects per schedule: " +
          ", ".join(f"{s}={n}" for s, n in by_sched.items()))
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"output -> {OUT_DIR}\n")

    summary = []
    for name, split_dir in SPLITS.items():
        print(f"[{name}]  {split_dir}")
        out = sanitize_split(name, split_dir)
        if out is None:
            continue
        dest = OUT_DIR / f"cleaned_{name}.csv"
        out.to_csv(dest, index=False, encoding="utf-8")
        print(f"  wrote {dest.name}\n")
        summary.append((name, out["subject_id"].nunique(), len(out)))

    print("=" * 62)
    print(f"{'split':<14}{'subjects':>10}{'rows':>12}")
    for name, n_subj, n_rows in summary:
        print(f"{name:<14}{n_subj:>10}{n_rows:>12,}")
    print(f"{'TOTAL':<14}{sum(s for _, s, _ in summary):>10}"
          f"{sum(r for _, _, r in summary):>12,}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
