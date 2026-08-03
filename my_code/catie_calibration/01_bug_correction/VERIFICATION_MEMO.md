# Verification memo: the trend/heuristic branch in the CATIE likelihood code

**To:** Prof. Yonatan Loewenstein, Dr. Ohad Dan
**From:** Shir Holzman (Amirim honours project)
**Subject:** Reproduction of the CATIE trial-level likelihood, and one discrepancy found

---

## Summary

While building an independent Python reimplementation of the CATIE choice-probability
model as the baseline for my calibration work, I reproduced
`COMPETITION_CATIE_schedule_choice_probability.m` to machine precision and then found
what appears to be an indexing error in its trend/heuristic branch.

The branch reads `pays(trial)` before that element has been assigned, so it is always
`NaN`. Because MATLAB evaluates `NaN > x` and `NaN < x` as `false`, the branch can never
select alternative 1 — while the line below it still reduces the remaining probability
mass to `1 − τ`. The net effect is that on trend-testable trials, τ = 0.29 of the
probability mass is assigned to alternative 2 regardless of what the trend indicates.

Correcting the indices — to match the generative simulator, which is unaffected —
improves both scores with **no parameter re-fitting**:

| Pooled, all 12 schedules, 3,328 subjects | Published | Corrected | Δ |
|---|---|---|---|
| E[p] | 0.6191 | 0.6293 | **+0.0103** |
| E[log p] | −0.6732 | −0.6610 | **+0.0122** |

Both differences are significant under subject-clustered paired tests:
E[p] t(3327) = 39.09, p = 2.4×10⁻²⁷⁵, dz = 0.68;
E[log p] t(3327) = 27.74, p = 1.7×10⁻¹⁵², dz = 0.48.

**Scope.** This affects only the trial-level model comparison (Tables S1/S2, Fig. S4).
The competition result itself is unaffected — schedule optimisation and the bias
predictions run through the *simulator* (`CATIE_schedule_1.m`), which computes the same
condition correctly because `pays(trial)` is already assigned by the time it is read
there.

**Independently confirmed in MATLAB.** The numbers above were first obtained from a
from-scratch Python reimplementation. I then ran the correction a second, independent
way: a MATLAB copy of the original function with only the flagged lines edited, executed
by real MATLAB against the raw per-subject CSVs directly (no Python involved). It
reproduces every number above exactly. See `my_code/catie_calibration/matlab/README.md`
for the files and a self-check the driver ran against itself, which is worth reading —
it caught a bug of my own (a redundant probability-flip in the driver, not in either
version of the model) before I trusted the first MATLAB run's numbers.

---

## 1. The discrepancy

`Data_resources/competition_analysis-main/CATIE/COMPETITION_CATIE_schedule_choice_probability.m`

```matlab
 16    pays = NaN(nTrials,1);
 ...
106    is_test_trend = (trial>2) && (is_choice_1(trial-1) == is_choice_1(trial-2)) && ((pays(trial-1) ~= pays(trial-2)));
107    if is_test_trend
108        if ((is_choice_1(trial-1) && (pays(trial) > pays(trial-1))) ||...
109                (~is_choice_1(trial) && (pays(trial) < pays(trial-1))))
110            p_choose_1_heuristic_mode = pHeuristic;
111        else
112            p_choose_1_heuristic_mode = 0;
113        end
114        p_try_explore = 1-pHeuristic;
 ...
147        pays(trial) = rewards_1(trial);     % <-- first assignment of pays(trial)
```

`pays(trial)` is preallocated as `NaN` on line 16 and not assigned until line 147, which
is *after* the block above. So at line 108 it is always `NaN`, both comparisons are
`false`, and `p_choose_1_heuristic_mode` is identically 0 — while line 114 still applies
the τ discount. On a trend-testable trial the model therefore caps P(alternative 1) at
1 − τ = 0.71 and hands the remaining 0.29 to alternative 2, irrespective of the trend.

Line 109 additionally reads `is_choice_1(trial)` — the choice being predicted — where
lines 106 and 108 use `trial-1`. That would be leakage if `pays(trial)` were not `NaN`.

**Intended semantics**, from the simulator `CATIE_schedule_1.m:152-157`, which decides
for `trial+1` using trials `trial` and `trial-1`:

```matlab
152    if (rand(1) < pHeuristic) && (decs_b(trial) == decs_b(trial-1)) && (pays(trial) ~= pays(trial-1))
153        if pays(trial) > pays(trial-1)
154            decs_b(trial+1) = decs_b(trial);      % payoff rose -> repeat
155        else
156            decs_b(trial+1) = 1-decs_b(trial);    % payoff fell -> switch
```

Shifted one trial back, the likelihood file's condition should read
`pays(trial-1)` vs `pays(trial-2)`, and `is_choice_1(trial-1)`. Three index errors in
total. This is the only change made in the "corrected" model below.

## 2. Reproduction

An independent Python port (`my_code/catie_calibration/catie_core.py`) of the per-k
likelihood and the `_hetro` k∈{0,1,2} mixture reproduces the stored MATLAB output
(`eda_with_catie_probabilities.csv`, 49,200 trials) to a maximum absolute deviation of
**2.2×10⁻¹⁵**, i.e. floating-point noise. The port preserves the details that matter:
the `base2dec` window ordering, the CAB-k gate `(trial-1) > k`, the stale
`expected_reward` on the insufficient-history branch, the uniform-over-distinct-rows
confusion draw, and the time-averaged (rather than per-trial) posterior weighting in
`_hetro.m:25`.

Against the published figures, pooled over all 12 static schedules:

| | This port (published mode) | Paper (Tables S1/S2) |
|---|---|---|
| E[p] | 0.6191 | 0.619 |
| E[log p] | −0.6732 | −0.678 |

E[p] agrees to the paper's reported precision. **E[log p] differs by 0.005, which I have
not fully accounted for.** My cohort is 3,328 subjects (332,800 probabilities) against
the paper's 333,200, a difference of 4 subjects; per-schedule counts otherwise match the
paper exactly for all nine schedules I can check (0→549, 1→595, 2→538, 3→607, 6→144,
8→116, 9→107, 10→93, 11→87). I checked alternative pooling conventions — mean of
per-subject means (−0.6732), excluding trial 1 (−0.6730), mean of per-schedule means
(−0.7200) — and none produces −0.678.

I want to flag this rather than gloss it. It does not affect the conclusion below,
because the published-vs-corrected comparison is **paired on identical data**: whatever
cohort or convention difference explains the 0.005, it applies equally to both arms.

## 3. Effect of the correction

Parameters held at their published values throughout (τ=0.29, ε=0.30, φ=0.71, k∈{0,1,2}).
Nothing is fitted.

| Split | Subjects | E[p] pub | E[p] fix | Δ | E[log p] pub | E[log p] fix | Δ |
|---|---|---|---|---|---|---|---|
| training (2,3,6,9,11) | 1,483 | 0.6056 | 0.6216 | +0.0161 | −0.6842 | −0.6688 | +0.0154 |
| EDA (4,5,7) | 492 | 0.6013 | 0.6146 | +0.0134 | −0.7141 | −0.6992 | +0.0149 |
| test (1,8,10) | 804 | 0.6159 | 0.6191 | +0.0031 | −0.6900 | −0.6789 | +0.0111 |
| schedule 0 | 549 | 0.6762 | 0.6784 | +0.0022 | −0.5826 | −0.5799 | +0.0027 |
| **pooled** | **3,328** | **0.6191** | **0.6293** | **+0.0103** | **−0.6732** | **−0.6610** | **+0.0122** |

The correction improves E[log p] on **every one of the 12 schedules** individually
(range +0.0027 to +0.0205).

**How often the branch is live.** Pooled, 16.63% of trials are trend-testable (same
choice on t−1 and t−2 with differing payoffs). Under the corrected code the branch
selects alternative 1 on 7.44% of all trials. Schedule 0 is the outlier at only 3.74%
testable — expected, since its long unbroken reward blocks mean consecutive same-choice
trials frequently share a payoff and so fail the `pays(t−1) ≠ pays(t−2)` condition.

**Magnitude.** Mean |ΔP(alt 1)| = 0.023, max 0.329. The distribution is bimodal: 7.44%
of trials differ by more than 0.20 (the directly affected trend trials), while most of
the remaining differences are small and arise indirectly, through the k-mixture weights.

For context, the corrected model closes about 13% of the gap between CATIE's published
E[log p] and the best Q-Learning model's −0.569.

## 4. What I would like to check with you

1. Whether you read lines 108–109 the same way, or whether there is an intended reading
   I have missed.
2. Whether the 0.005 E[log p] gap in §2 is explained by a known exclusion or cohort
   detail on your side — that would let me close the reproduction cleanly.
3. Whether you would like the corrected model used as the baseline for the remainder of
   my project. My inclination is to report both throughout, with the corrected version
   as the reference, since the calibration analysis that follows is sensitive to
   probability mass in exactly the region this affects.

## Reproducing this memo

Python:

```bash
python my_code/catie_calibration/golden_test.py                        # port validation
python my_code/catie_calibration/sanitize_splits.py                    # build splits
python my_code/catie_calibration/01_bug_correction/bug_benchmark.py    # this analysis
```

MATLAB (independent, no Python):

```
matlab -batch "run('my_code/catie_calibration/matlab/run_bug_comparison_all_schedules.m')"
```

Full transcripts: `01_bug_correction/figures/output.txt` (Python),
`matlab/results/` (MATLAB).
