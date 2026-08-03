# MATLAB validation of the bug correction

Independent, MATLAB-native cross-check of the Python reproduction in
`../01_bug_correction/`. No Python is involved anywhere in this folder — the
comparison runs the **unmodified original** `.m` files from
`Data_resources/competition_analysis-main/CATIE/` against a **minimally patched
copy** kept here, both executed by real MATLAB (R2026a).

## Files

| File | Role |
|---|---|
| `CATIE_FIXED/COMPETITION_CATIE_schedule_choice_probability_FIXED.m` | Verbatim copy of the original per-k likelihood function, with only the heuristic-mode block corrected (see its header comment for the exact diff) |
| `CATIE_FIXED/COMPETITION_CATIE_schedule_choice_probability_hetro_FIXED.m` | Verbatim copy of the original mixture wrapper, calling the `_FIXED` per-k function instead of the original |
| `run_bug_comparison_all_schedules.m` | Driver: reads every raw subject CSV across all 12 schedules, runs both the original and fixed hetro functions, reports E[p]/E[log p] |
| `results/matlab_per_subject_metrics.csv` | Per-subject E[p], E[log p] under both models (3,328 rows) |
| `results/matlab_per_schedule_metrics.csv` | Per-schedule aggregates |

The original `base2dec.m` and `getExploreProb.m` helpers are used unmodified by
both the original and fixed functions — nothing in the helpers changed.

## Result

Pooled, all 12 schedules, 3,328 subjects (identical to the Python run in
`../01_bug_correction/`):

| | Published | Fixed | Δ |
|---|---|---|---|
| E[p] | 0.6191 | 0.6293 | +0.0103 |
| E[log p] | −0.6732 | −0.6610 | +0.0122 |

Paired t-test (subject-clustered, n=3328): E[p] t(3327)=39.09, p=2.45×10⁻²⁷⁵;
E[log p] t(3327)=27.74, p=1.71×10⁻¹⁵². Every per-schedule value matches the
Python reproduction to the digits reported. **This is agreement between two
independent implementations (a from-scratch Python port and a minimally-edited
copy of the original MATLAB), not a rerun of the same code** — it rules out an
implementation-specific artifact in either one.

## A bug caught by this script's own sanity check

The driver includes a self-check: for schedules 4/5/7, the original model's
output on the raw per-subject CSVs should reproduce
`my_code/EDA_set/processing/eda_with_catie_probabilities.csv` (a completely
independent, previously-generated file) to floating-point precision, since both
were produced by the same unmodified `.m` files.

The first run failed this check by 0.9 — nowhere close to floating-point noise.
Before trusting any headline number, that gap was run down rather than
dismissed. It was **not** a data problem: a full row-by-row diff of all 492 raw
EDA subject files against the reference CSV found zero mismatches in
`biased_reward`, `unbiased_reward`, or `is_biased_choice`. The bug was in this
driver script itself:
`COMPETITION_CATIE_schedule_choice_probability_hetro` already returns P(the
choice actually made) — its inner per-k function applies the
`is_choice_1(trial) ? p : 1-p` conversion internally. The first version of this
driver applied that conversion a second time, silently flipping ~40% of trials
(every trial where the participant chose the unbiased alternative). After
removing the redundant conversion, the sanity check passed at 5.55×10⁻¹⁶ and
every downstream number matched the Python analysis exactly.

Kept here as a reminder to verify a pipeline against an independent reference
before reporting its output, not just to check that it runs without error.

## Statistics without the toolbox

This MATLAB installation does not have the Statistics and Machine Learning
Toolbox, so `ttest`/`tinv` are unavailable. `paired_ttest_manual` (bottom of
`run_bug_comparison_all_schedules.m`) implements the paired t-test from the
closed-form two-sided Student-t tail identity
`P(|T_df| > |t|) = betainc(df/(df+t^2), df/2, 1/2)`, using only `betainc` /
`betaincinv`, which are base MATLAB. Validated against textbook values
(t=2, df=20 → p≈0.0591) before trusting it on the real data.

## How to run

```
matlab -batch "run('my_code/catie_calibration/matlab/run_bug_comparison_all_schedules.m')"
```

Takes a few minutes for ~3,300 subjects × 2 models × 3 k-agents.
