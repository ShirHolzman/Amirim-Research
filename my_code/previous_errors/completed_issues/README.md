# Completed issues

Investigations and migrations that are **finished and closed**. Kept for provenance —
they record how a problem was found, what was decided, and what the outcome was — but
they describe work already done and are not active guidance.

| document | issue | outcome |
|---|---|---|
| `SCHEDULE_N_RECONCILIATION.md` | `schedule_7` held 115 valid subjects against the paper's reported N = 119 | **Resolved.** The competition's organized data release has the 4 missing participants; all 12 schedules now match the paper exactly (3,332 total). |
| `REORGANIZATION_PLAN.md` | Migrate onto the organized data release, rename the first-investigations folder, simplify `sanitize_splits.py` | **Executed 2026-08-22.** Golden test passes; every verification item met. Includes the execution record and two deviations from plan. |

Unqualified file paths inside these documents are relative to
`my_code/catie_calibration/`, where both files used to live.

## One finding here that is still open

The migration **ruled out** the most plausible explanation for the systematic
**~+0.005 E[log p]** residual against the paper: the missing `schedule_7` subjects. The
population now matches the paper exactly on all 12 schedules and the residual is
unchanged (training mean +0.0051, EDA mean +0.0046, same sign on every schedule, while
E[p] matches to ≤0.0018). That defect remains the project's most concrete open item and is
tracked in `../catie_calibration/REVIEW_PLAN.md`.
