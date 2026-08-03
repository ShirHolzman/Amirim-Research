"""
Sanitize the raw per-subject CSVs for every data split into one tidy CSV per split.

Generalises my_code/EDA_set/early_proccessing/sanitize.py, which hardcodes the EDA
path (line 7) and is therefore not reusable for Training_set / Test_set / schedule_0.

Differences from the original, all deliberate:

  * Takes a split directory; handles both the nested layout (Training_set/schedule_N/)
    and the flat layout (schedule_0/).
  * Does NOT derive RT_net or RT_zscore. Reaction time is a closed direction in this
    project -- the reported U-shape at extreme p is an artifact of z-scoring subjects
    with near-zero personal SD against the 1.5 s hardware floor. The raw `RT` column
    is carried through untouched for provenance; nothing downstream reads it.
  * Enforces exactly 100 trials per subject. The CATIE likelihood hardcodes
    nTrials = 100 and silently mis-indexes otherwise.
  * Writes UTF-8 with pandas only. Never open these files in Excel: the existing
    cleaned_eda_data.csv had its `time` column destroyed that way
    ("05-25-2020 15:11:02.738300" -> "11:02.7") and its booleans re-cased.

Exclusion rule is unchanged from the original: a subject is dropped if they chose one
side fewer than 5 times (matching the competition's own exclusion, 54/3386 = 1.6%).

Run:  python my_code/catie_calibration/sanitize_splits.py
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

MY_CODE = pathlib.Path(__file__).parent.parent
OUT_DIR = pathlib.Path(__file__).parent / "data"

N_TRIALS = 100
MIN_CHOICES_PER_SIDE = 5

REQUIRED = [
    "trial_number", "is_biased_choice", "side_choice",
    "observed_reward", "biased_reward", "unbiased_reward",
]

# split name -> (directory, layout)
SPLITS = {
    "training": (MY_CODE / "Training_set", "nested"),
    "test": (MY_CODE / "Test_set", "nested"),
    "schedule_0": (MY_CODE / "schedule_0", "flat"),
}


def subject_files(split_dir: pathlib.Path, layout: str):
    """Yield (schedule_name, csv_path) pairs."""
    if layout == "nested":
        for sched_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for csv in sorted(sched_dir.glob("*.csv")):
                yield sched_dir.name, csv
    else:
        for csv in sorted(split_dir.glob("*.csv")):
            yield split_dir.name, csv


def sanitize_split(name: str, split_dir: pathlib.Path, layout: str) -> pd.DataFrame | None:
    if not split_dir.exists():
        print(f"  !! {split_dir} does not exist -- skipping")
        return None

    frames = []
    kept = 0
    dropped = {"invalid_bias": 0, "few_choices": 0, "bad_trials": 0,
               "missing_cols": 0, "read_error": 0}

    for sched, csv in subject_files(split_dir, layout):
        if "invalid_bias" in csv.name.lower():
            dropped["invalid_bias"] += 1
            continue
        try:
            df = pd.read_csv(csv)
        except Exception as exc:  # noqa: BLE001 -- report and continue, as the original did
            print(f"  !! read error {csv.name}: {exc}")
            dropped["read_error"] += 1
            continue

        if any(col not in df.columns for col in REQUIRED):
            dropped["missing_cols"] += 1
            continue

        counts = df["side_choice"].value_counts()
        if len(counts) < 2 or counts.min() < MIN_CHOICES_PER_SIDE:
            dropped["few_choices"] += 1
            continue

        if len(df) != N_TRIALS or sorted(df["trial_number"]) != list(range(N_TRIALS)):
            dropped["bad_trials"] += 1
            continue

        df = df.copy()
        df["subject_file"] = csv.name
        df["schedule"] = sched
        df["split"] = name
        # `subject_file` is NOT unique across schedules -- filenames are derived from
        # a timestamp and do collide (e.g. 1609192765_75.csv appears in both
        # schedule_6 and schedule_11 as two different participants). Grouping on
        # subject_file alone silently merges them into a 200-trial subject and
        # corrupts every per-subject statistic. Always group on `subject_id`.
        df["subject_id"] = f"{sched}/{csv.name}"
        frames.append(df)
        kept += 1

    if not frames:
        print(f"  !! no valid subjects found in {split_dir}")
        return None

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["schedule", "subject_id", "trial_number"]).reset_index(drop=True)

    assert out["subject_id"].nunique() == kept, (
        f"subject_id collision: {kept} subjects kept but "
        f"{out['subject_id'].nunique()} unique ids")
    sizes = out.groupby("subject_id").size()
    assert (sizes == N_TRIALS).all(), (
        f"expected {N_TRIALS} trials per subject, got "
        f"min={sizes.min()} max={sizes.max()}")

    n_collide = (out.groupby("subject_file")["schedule"].nunique() > 1).sum()
    if n_collide:
        print(f"  note: {n_collide} filename(s) reused across schedules "
              f"-- disambiguated by subject_id")

    total_dropped = sum(dropped.values())
    print(f"  kept {kept} subjects ({len(out):,} rows), dropped {total_dropped}")
    for reason, n in dropped.items():
        if n:
            print(f"     - {reason}: {n}")
    by_sched = out.groupby("schedule")["subject_file"].nunique()
    print("     subjects per schedule: " +
          ", ".join(f"{s}={n}" for s, n in by_sched.items()))
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"output -> {OUT_DIR}\n")

    summary = []
    for name, (split_dir, layout) in SPLITS.items():
        print(f"[{name}]  {split_dir}")
        out = sanitize_split(name, split_dir, layout)
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
    print("=" * 62)
    print("NOTE: EDA split already exists as "
          "my_code/EDA_set/processing/eda_with_catie_probabilities.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
