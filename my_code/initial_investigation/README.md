# Initial investigation (archived)

The first round of work on this project, kept for provenance. **Not part of the active
pipeline** — nothing under `my_code/catie_calibration/` reads from this folder.

Formerly `my_code/EDA_set/`; renamed 2026-08-22 when a new `my_code/EDA_set/` was created
to hold *only* the EDA split's reward schedules.

## Contents

| folder | what it was |
|---|---|
| `early_proccessing/` | first sanitizer (`sanitize.py`), hardcoded to the EDA path; superseded by `catie_calibration/sanitize_splits.py` |
| `processing/` | `compute_eda_catie_probabilities.m` and its output `eda_with_catie_probabilities.csv` — the original stored MATLAB CATIE probabilities for the EDA set |
| `rt_analysis/` | reaction-time analysis |
| `surprise_analysis/` | surprise/routine contrast |
| `calibration_analysis/` | first calibration look (reliability diagrams, confidence-by-trial) |
| `schedule_4/5/7/` | the raw per-subject CSVs these analyses ran on |

## Two caveats if you read the numbers here

**They predate the organized-data migration.** These analyses ran on the 492-subject EDA
set built from the raw data dump. The active pipeline now uses the competition's organized
release, where `schedule_7` has 119 valid subjects rather than 115, making the EDA split
496. Any subject count or EDA-conditioned statistic in this folder is therefore slightly
stale — see `../completed_issues/SCHEDULE_N_RECONCILIATION.md`.

**Two directions here were investigated and closed.** They are recorded in
`catie_calibration/REVIEW_PLAN.md` as permanently closed, and should not be reopened:

- **Reaction time** — the U-shape at extreme p is an artifact of z-scoring subjects with
  near-zero personal SD against a 1.5 s hardware floor. At the robust threshold (X=10%)
  the Surprise–Routine contrast is n.s. (d=0.02, p≈0.82).
- **Current-trial-choice features** — circular by construction:
  `P(surprise | no switch) = 0.000` exactly, and `BLUE_RIGHT_LEFT_RED` is constant within
  every subject, so `side_choice ≡ is_biased_choice` up to relabeling.

`eda_with_catie_probabilities.csv` is likewise retired as a data source: `golden_test.py`
now validates against a live, all-12-schedule MATLAB reference instead of this single
stored 3-schedule file. Note this CSV also has known Excel damage (its `time` column was
rewritten, `05-25-2020 15:11:02.738300` → `11:02.7`, and booleans re-cased) — harmless for
the columns the old analyses used, but a reason not to revive it.
