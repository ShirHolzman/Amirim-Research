# MATLAB validation

Four independent things are validated here, in the subsections below:
1. **The bug correction** (`run_bug_comparison_all_schedules.m`) -- does fixing the
   three index errors actually improve E[p]/E[log p], measured in real MATLAB?
2. **`state_tensors()` itself** (`export_state_tensors.m` /
   `verify_state_tensors.py`) -- element-by-element, does catie_core.py's internal
   recursion match MATLAB's own loop variables, not just the final probability?
3. **The k-mixture / BMA step** (`export_bma_mixing_inputs.m` /
   `verify_bma_mixing.py`) -- given MATLAB's own per-agent probabilities, does
   `mix_agents()` combine them the way `hetro.m` does?
4. **The headline E[log p]** (`report_original_elogp.m`) -- what does the paper's
   own aggregation function report, called directly?

Together these cover the three stages of the likelihood pipeline separately --
state recursion (2), per-trial probability (1), k-mixture (3) -- plus the
end-to-end aggregate (4). `../golden_test.py` check 1 covers the composition of
all three; these exist because a composition check alone cannot localise a
fault, and could in principle be passed by two errors that cancel.

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
| `results/matlab_per_subject_metrics.csv` | Per-subject E[p], E[log p] under both models (3,332 rows) |
| `results/matlab_per_schedule_metrics.csv` | Per-schedule aggregates |

The original `base2dec.m` and `getExploreProb.m` helpers are used unmodified by
both the original and fixed functions — nothing in the helpers changed.

## Result

Pooled, all 12 schedules, 3,332 subjects (identical to the Python run in
`../01_bug_correction/`):

| | Published | Fixed | Δ |
|---|---|---|---|
| E[p] | 0.6190 | 0.6293 | +0.0103 |
| E[log p] | −0.6733 | −0.6611 | +0.0122 |

The published E[p] of 0.6190 matches the paper's reported 0.619 exactly.

Paired t-test (subject-clustered, n=3332): E[p] t(3331)=39.02, p=1.14×10⁻²⁷⁴;
E[log p] t(3331)=27.64, p=1.46×10⁻¹⁵¹. Every per-schedule value matches the
Python reproduction to the digits reported. **This is agreement between two
independent implementations (a from-scratch Python port and a minimally-edited
copy of the original MATLAB), not a rerun of the same code** — it rules out an
implementation-specific artifact in either one.

## A bug caught by a sanity check (historical)

*The check described here has since been retired — `golden_test.py`'s check 1 now
compares against a live all-12-schedule MATLAB reference, which covers strictly more.
The episode is kept because its lesson still applies.*

The driver used to include a self-check: for schedules 4/5/7, the original model's
output on the raw per-subject CSVs should reproduce
`my_code/initial_investigation/processing/eda_with_catie_probabilities.csv` (a completely
independent, previously-generated file) to floating-point precision, since both
were produced by the same unmodified `.m` files.

The first run failed this check by 0.9 — nowhere close to floating-point noise.
Before trusting any headline number, that gap was run down rather than
dismissed. It was **not** a data problem: a full row-by-row diff of all raw
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
compared 9,996 (subject, k) pairs, 999,600 trials (3,332 subjects x 3 k-values)
[PASS] H          max |dev| = 0.000e+00
[PASS] b          max |dev| = 0.000e+00
[PASS] c_prev     max |dev| = 0.000e+00
[PASS] s_prev     max |dev| = 5.551e-16
[PASS] sbar_prev  max |dev| = 5.551e-16
[PASS] g          max |dev| = 5.551e-16
```

`H`, `b`, `c_prev` are boolean/integer-valued and match **exactly** (0.0 deviation)
across all 999,600 trials. `s_prev`, `sbar_prev`, `g` are floating-point and match
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

---

## 3. Isolated validation of the k-mixture (Bayesian model averaging)

Sections 1-2 and `../golden_test.py` leave one gap. golden_test compares the
**end** of the pipeline; `verify_state_tensors.py` covers the **start** (the
state recursion). Neither isolates the final stage: the k∈{0,1,2} mixture in
`COMPETITION_CATIE_schedule_choice_probability_hetro.m:20-25`. A mixing error
compensated by a recursion error would pass both.

This section closes that gap by handing Python the per-agent probability matrix
**MATLAB itself built**, so the only thing compared is the mixing arithmetic.

### The substantive point: the two implementations mix in different spaces

`COMPETITION_CATIE_schedule_choice_probability.m` returns **P(choice actually
made)**, not P(alt 1) — its lines 136-139. So `hetro.m:25` forms its weighted
average in *choice* space. `catie_core.mix_agents()` instead keeps P(alt 1) and
mixes in *alt-1* space, converting afterwards via `p_of_observed_choice()`.

These agree only because the BMA weights are a convex combination (they sum to
1 per column), which makes "mix then flip" equal "flip then mix":

```
sum_a w_a (1 - P_a) = 1 - sum_a w_a P_a     iff  sum_a w_a = 1
```

That identity is the translation's load-bearing assumption, so it is checked
numerically here rather than argued on paper — including a direct check that
the weights really do sum to 1.

### Files

| File | Role |
|---|---|
| `export_bma_mixing_inputs.m` | Calls the unmodified original functions and dumps both the **input** to the mixing (the 3×100 per-k probability matrix, reproducing `hetro.m`'s own `K = 0:2` loop only to capture what `hetro.m` computes internally but does not return) and its **output** |
| `verify_bma_mixing.py` | Feeds that exact matrix to `mix_agents()` and compares only the mixing; also runs four mutation controls |
| `results/bma_mixing_verification.txt` | Result transcript |
| `results/bma_mixing_reference.csv` | The exported matrix (6.9 MB, gitignored — regenerable) |

The exporter writes `%.17g`, **not** `writetable`, which serialises doubles at
~15 significant digits. At `writetable` precision the comparison would have a
~1e-16 floor — exactly the scale being tested, so it could mask or manufacture
a discrepancy. `%.17g` round-trips an IEEE double exactly.

### Result

600 subjects × 100 trials, sampled evenly across all 12 schedules:

```
  [OK] mix_agents(published) -> choice space  vs  MATLAB    2.00e-15
  [OK] literal choice-space transcription     vs  MATLAB    2.00e-15
  [OK] alt-1-space route  vs  choice-space route            5.55e-16
  [OK] per-trial weight columns sum to 1                    3.33e-16
  [OK] time-averaged w_bar sums to 1                        4.44e-16
```

### Mutation controls

A pass only means something if the test could have failed, so four plausible
mistranslations are run through the same comparison. All four are caught:

| Mutation | Max deviation |
|---|---|
| cumprod over P(alt 1) instead of P(choice made) | 2.41e-01 |
| per-trial weights (`hetro.m:26`, the commented-out line) | 1.71e-01 |
| uniform 1/3 weights (no BMA at all) | 1.43e-01 |
| un-lagged weights | 1.81e-03 |

**Two findings worth recording.**

*"Forgot the prior column" and "forgot the `1:end-1` shift" are the same error.*
These were initially written as two independent controls and returned
bit-identical deviations. Prepending `ones(n_sub_agents,1)` at `hetro.m:20` and
then dropping the last column at `hetro.m:22` is precisely what lags the weight
matrix by one trial — the prior column *is* the lag. This is now asserted in the
script (0.000e+00), so the control list cannot silently double-count.

*That lag is worth only 1.81e-03* — roughly 100× smaller than the other three
mutations. Because the published code time-averages the weights over all 100
trials (`mean(...,2)`), shifting by one only swaps the flat-prior column for the
final column and divides the difference by 100. So the same quirk that discards
the per-trial adaptivity `hetro.m`'s own comment describes also makes the code
nearly insensitive to the off-by-one its `1:end-1` was guarding against. This is
consistent with the separately measured result that the "intended" per-trial
variant scores slightly *worse* (E[log p] −0.6992 → −0.7042).

### How to run

```
matlab -batch "run('my_code/catie_calibration/matlab/export_bma_mixing_inputs.m')"
python my_code/catie_calibration/matlab/verify_bma_mixing.py
```

The export takes a couple of minutes. It scores each subject six times: three
explicit per-k calls to capture the mixing input, plus `hetro`'s own three
internal ones. That duplication is deliberate — `hetro.m` does not return its
intermediate matrix, and instrumenting it would mean modifying an original file.

---

## 4. The headline E[log p], from the paper's own functions

`report_original_elogp.m` reports the aggregate E[log p] by **calling** the
original unmodified MATLAB for both steps, rather than reimplementing either:

- `COMPETITION_CATIE_schedule_choice_probability_hetro.m` — per-subject
  P(choice made), the same function `COMPETITION_empirical_decisions_probabilities.m:58`
  calls.
- `permutation_test_and_bootstrap.m` — the aggregation, the same function
  `COMPETITION_main.m:202` feeds `log(catie_probabilities)` into.

`permutation_test_and_bootstrap` has no return value (it only `fprintf`s), so it
is called as `(all_log_p, all_log_p)` and the mean is regexp'd back out of its
own printed report via `evalc`. The number is produced by *its* code, not ours.

`COMPETITION_empirical_decisions_probabilities.m` itself cannot be called: it
hardcodes an absolute path to another machine's disk (`C:\Users\ojd5\...`). The
per-subject loop over our schedule folders is necessary glue for that, but every
actual computation is delegated to the original functions.

### Result

All 12 schedules, 3,332 subjects, 333,200 trials:

| | E[log p] |
|---|---|
| Original unmodified MATLAB, published | **−0.6733** |
| `catie_core.py` port, published | −0.6733 |
| Paper, Tables S1/S2 | −0.678 |

The port and real MATLAB agree. The ~+0.0047 residual against the paper is real,
now reproduced two independent ways, and remains unexplained — see
`../REVIEW_PLAN.md`. It does not affect the published-vs-corrected comparison,
which is paired on identical data either way.

### How to run

```
matlab -batch "run('my_code/catie_calibration/matlab/report_original_elogp.m')"
```

Requires the Statistics and Machine Learning Toolbox (`datasample`, used by
`permutation_test_and_bootstrap.m`'s bootstrap). Takes about a minute.
