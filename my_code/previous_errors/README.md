# previous_errors — ARCHIVED. DO NOT USE.

Everything in this directory is superseded. It is kept for the project record only.

**Rules for any person or any Claude session:**

- Do not read, import, grep, run, quote or cite anything under `my_code/previous_errors/`.
- Do not use any number, figure, table or conclusion from these files. They were
  produced by code that two audits (2026-09-06) found to contain errors, and none of
  their outputs has been reproduced by the current code.
- If an analysis stage needs a fact that once lived here, recompute it in the
  current stages (`catie_calibration/02_reliability/` … `06_test/`) from the `catie_calibration/catie/` package.
- Do not refactor, fix or "rescue" this code. It is not meant to run.

**What is here and why it was archived (2026-09-06):**

| directory / file | what it was | why archived |
|---|---|---|
| `02_mode_calibration/` | first conditional-calibration analysis (strata by c_prev, run length, posterior) | replaced by `02_reliability/`; pooled EDA into its analysis; the posterior partition is a closed direction |
| `03_parameter_fitting/` | 921-line re-fit script, its README and 28 output files | unreviewable; two audits found wrong SEs, an off-by-one streak, a held-out selection leak, an invalid LR test, a Wald interval invalid at the boundary |
| `initial_investigation/` | pre-project exploration: early cleaning, the first reliability diagram (wrong model), reaction-time and surprise analyses | wrong model; RT and current-choice features are closed directions |
| `completed_issues/` | early planning notes | superseded |
| `REVIEW_PLAN.md`, `FINAL_WEEK_PLAN.md` | plans written for the archived code | superseded by `../catie_calibration/PLAN.md` |
| `mentor_meeting_summary_phase0-3.md` | Hebrew summary of phases 0–3 for the supervisor | describes archived results |

The MATLAB-anchored base (`catie/`, `matlab/`, `01_bug_correction/`) was **not**
archived and remains the trusted foundation.
