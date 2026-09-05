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
| E[p] | 0.6192 | 0.6295 | +0.0103 |
| E[log p] | −0.6776 | −0.6654 | +0.0122 |

Subject-clustered paired tests: E[p] t(3331)=39.16, p=3.2e-276, dz=0.68;
E[log p] t(3331)=27.29, p=3.3e-148, dz=0.47.

The published row now reproduces the paper's pooled values (E[p] 0.619,
E[log p] −0.678) to the rounding floor — see the resolved-discrepancy note below.

- E[log p] improves on **all 12 schedules** individually (+0.0027 to +0.0205).
- 16.63% of trials are trend-testable; the corrected branch selects alternative 1 on
  7.44% of all trials.
- Mean |ΔP(alt 1)| = 0.023, max 0.355; bimodal, with 7.44% of trials differing by >0.20.
- The correction closes ~11% of the gap to the best Q-Learning model's E[log p].

**Scope.** Affects only the trial-level model comparison (Tables S1/S2, Fig. S4). The
competition result is unaffected — schedule optimisation and bias prediction use the
*simulator*, which reads `pays(trial)` only after assignment.

**Resolved discrepancy (2026-09).** Earlier versions of this chapter showed a systematic
~+0.005 E[log p] residual against the paper (−0.6733 vs −0.678, same sign on every
schedule). Root cause: the *shipped* k-mixture code (`hetro.m:25`) time-averages the BMA
weights (`mean(P' * W, 2)`), whereas the paper's reported numbers were produced by the
per-trial weighting on its own commented-out line 26. Verified per schedule against the
paper's Fig S5 tables: per-trial weights + the heuristic bug + trial 1 included
reproduces every schedule to ≤0.0005 (the 3-decimal rounding floor), with mixed-sign
residuals. All numbers in this chapter now use the per-trial weighting
(`catie_core.mix_agents(weighting="per_trial")`, the project default); the shipped
time-averaged variant is retained as `"shipped_time_avg"` only for golden tests against
live MATLAB. The published-vs-corrected comparison is paired on identical data under
either weighting, so this chapter's conclusion never depended on the choice.

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
