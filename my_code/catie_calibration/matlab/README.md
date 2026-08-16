# MATLAB validation

Two independent things are validated here, in two subsections below:
1. **The bug correction** (`run_bug_comparison_all_schedules.m`) -- does fixing the
   three index errors actually improve E[p]/E[log p], measured in real MATLAB?
2. **`state_tensors()` itself** (`export_state_tensors.m` /
   `verify_state_tensors.py`) -- element-by-element, does catie_core.py's internal
   recursion match MATLAB's own loop variables, not just the final probability?

## 1. Validation of the bug correction

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

---

## 2. Element-by-element validation of `state_tensors()`

Everything above -- and every check in `../golden_test.py` before this section was
added -- validates only the **final choice probability**, end-to-end. None of it
looks at `state_tensors()`'s own outputs (`H`, `b`, `c_prev`, `s_prev`,
`sbar_prev`, `g`): the parameter-free recursion that everything else in this
project (mode decomposition, the closed-form re-fitting in Phase 3, the model
extensions in Phase 4) is built on top of. An end-to-end match doesn't strictly
rule out two compensating errors inside that recursion canceling out in the final
probability -- unlikely given how many independent subjects and schedules agree,
but not structurally impossible.

### Files

| File | Role |
|---|---|
| `CATIE_FIXED/COMPETITION_CATIE_schedule_choice_probability_INSTRUMENTED.m` | Copy of the FIXED per-k function, with six added lines exporting `H`, `b`, `c_prev`, `s_prev`, `sbar_prev`, `g` -- MATLAB's own loop variables, captured at the exact point they're read -- alongside the existing `p_decisions` output |
| `export_state_tensors.m` | Driver: runs the instrumented function for every subject × k∈{0,1,2} across all 12 schedules, writes `results/state_tensors_matlab.csv` (long format, one row per subject×k×trial) |
| `verify_state_tensors.py` | Loads that CSV, calls `catie_core.state_tensors()` for the same subjects/k, and compares every tensor element-by-element |

`results/state_tensors_matlab.csv` is **not tracked in git** (≈1M rows,
regenerable) -- see `.gitignore`.

### Result

```
compared 9,984 (subject, k) pairs, 998,400 trials (3,328 subjects x 3 k-values)
[PASS] H          max |dev| = 0.000e+00
[PASS] b          max |dev| = 0.000e+00
[PASS] c_prev     max |dev| = 0.000e+00
[PASS] s_prev     max |dev| = 5.551e-16
[PASS] sbar_prev  max |dev| = 5.551e-16
[PASS] g          max |dev| = 5.551e-16
```

`H`, `b`, `c_prev` are boolean/integer-valued and match **exactly** (0.0 deviation)
across all 998,400 trials. `s_prev`, `sbar_prev`, `g` are floating-point and match
to one ULP (5.55×10⁻¹⁶), the tightest deviation floating-point arithmetic can
produce. This is the entire sanitized population (all four splits) × all three
k values × all 100 trials each -- not a sample.

**Note on `g`:** MATLAB computes `p_choice_1_contingency_mode` unconditionally
every trial, including trial 1. `catie_core.py`'s `state_tensors()` computes the
same value at trial 1 but doesn't store it (`g_arr[t]` is only written inside its
`if t > 1` block), because trial 1's probability is hardcoded to 0.5 and never
reads `g`. `verify_state_tensors.py` compares indices 1..99 (MATLAB trials
2..100) for `g` and treats index 0 as a legitimate MATLAB value with no Python
counterpart -- not a mismatch. Same reasoning applies to `H`, `b`, `c_prev`,
`s_prev`, `sbar_prev` at trial 1, except there both sides are 0 by convention, so
no exclusion was needed for those five.

### How to run

```
matlab -batch "run('my_code/catie_calibration/matlab/export_state_tensors.m')"
python my_code/catie_calibration/matlab/verify_state_tensors.py
```

The MATLAB export takes a few minutes (comparable to the bug-comparison run,
slightly cheaper since it computes only one model × 3 k-agents per subject,
not two models × 3). It only prints once per *schedule* (not per file), so a
multi-minute silence while it works through a large schedule is expected, not a
hang -- confirmed by checking the process's CPU time was still climbing during
one such stretch.
