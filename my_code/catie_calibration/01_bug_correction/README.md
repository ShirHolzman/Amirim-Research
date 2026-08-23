# Phase 1 — Reproduction and correction of the CATIE likelihood

## Research question

Does the published CATIE trial-level likelihood implement the model as specified? And if
not, how much of CATIE's E[log p] deficit is attributable to the implementation rather
than to the model?

## What was found

`COMPETITION_CATIE_schedule_choice_probability.m:108-109` reads `pays(trial)`, which is
still `NaN` (preallocated line 16, assigned line 147). MATLAB evaluates `NaN > x` and
`NaN < x` as `false`, so the trend/heuristic branch can never select alternative 1 —
while line 114 still reduces the remaining mass to `1 − τ`. On every trend-testable
trial, τ = 0.29 of the probability mass goes to alternative 2 regardless of the trend.

The intended semantics are unambiguous in the generative simulator
(`CATIE_schedule_1.m:152-157`): payoff rose → repeat, payoff fell → switch. Correcting
the three index errors is the only change made.

## Key findings

Parameters held at published values (τ=0.29, ε=0.30, φ=0.71, k∈{0,1,2}). Nothing fitted.

| Pooled, 12 schedules, 3,332 subjects | Published | Corrected | Δ |
|---|---|---|---|
| E[p] | 0.6190 | 0.6293 | +0.0103 |
| E[log p] | −0.6733 | −0.6611 | +0.0122 |

Subject-clustered paired tests: E[p] t(3331)=39.02, p=1.14e-274, dz=0.68;
E[log p] t(3331)=27.64, p=1.46e-151, dz=0.48.

- E[log p] improves on **all 12 schedules** individually (+0.0027 to +0.0205).
- 16.63% of trials are trend-testable; the corrected branch selects alternative 1 on
  7.44% of all trials.
- Mean |ΔP(alt 1)| = 0.023, max 0.329; bimodal, with 7.44% of trials differing by >0.20.
- The correction closes ~13% of the gap to the best Q-Learning model's E[log p].

**Scope.** Affects only the trial-level model comparison (Tables S1/S2, Fig. S4). The
competition result is unaffected — schedule optimisation and bias prediction use the
*simulator*, which reads `pays(trial)` only after assignment.

**Open discrepancy.** The port reproduces the paper's E[p] (0.6190 vs 0.619) but E[log p]
differs by ~0.005 (−0.6733 vs −0.678), which is unexplained. This is *not* a cohort-size
artifact: the pipeline migrated onto the competition's organized data release (see
`../../completed_issues/SCHEDULE_N_RECONCILIATION.md`), and the cohort now matches the
paper's 3,332 exactly on all 12 schedules -- yet the residual is essentially unchanged.
That migration therefore *rules out* the subject-count-mismatch hypothesis rather than
explaining the gap. It does not affect this chapter's conclusion, since the
published-vs-corrected comparison is paired on identical data either way. Raised with
supervisors in `VERIFICATION_MEMO.md`.

## Validation

The Python port reproduces ORIGINAL, unmodified MATLAB, called live, to **2.33×10⁻¹⁵**
across all 12 schedules (3,332 subjects, 333,200 trials) -- see `../golden_test.py`
check 1. Per-schedule subject counts reproduce the paper's reported N exactly on all
12 schedules. Run `../golden_test.py` — it is a blocking gate.

**Independently confirmed in real MATLAB**, not just the Python port: `../matlab/` runs
the unmodified original `.m` file against a minimally-patched copy across all 3,332
subjects and reproduces every number in this document exactly (pooled and per-schedule).
See `../matlab/README.md` — its own sanity check first caught a bug in the *driver*
script (not the model), worth reading as an example of not trusting a first run.

## Output

| File | Contents |
|---|---|
| `VERIFICATION_MEMO.md` | Memo for supervisors |
| `figures/output.txt` | Full transcript |
| `figures/fig1_probability_distributions.png` | p and log p distributions, both models |
| `figures/fig2_trend_trial_scatter.png` | Published vs corrected P(alt 1), trend trials |
| `figures/fig3_per_schedule_impact.png` | E[log p] by schedule, both models |
| `figures/headline_metrics.csv` | Per-split metrics |
| `figures/per_schedule_metrics.csv` | Per-schedule metrics |
| `figures/bug_comparison_per_trial.csv.gz` | Per-trial output (gitignored) |

## How to run

```bash
python my_code/catie_calibration/golden_test.py
python my_code/catie_calibration/sanitize_splits.py
python my_code/catie_calibration/01_bug_correction/bug_benchmark.py
```

`pip install pandas numpy scipy matplotlib`

## Data dependency

```
my_code/{Training_set,Test_set,schedule_0}/**/*.csv
   -> catie_calibration/sanitize_splits.py
   -> catie_calibration/data/cleaned_{training,test,schedule_0}.csv
                                                                  \
data/cleaned_eda.csv ------------------------------------+-> bug_benchmark.py
                                                                  /
catie_calibration/catie_core.py (+ metrics.py) --------------------
```
