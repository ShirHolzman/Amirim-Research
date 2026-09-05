# Corrections to the CATIE trial-level likelihood

Two bugs in the CATIE likelihood code are fixed here. Both are in the code that scores
CATIE against participants' choices, neither touches the generative simulator, so the
schedule-optimisation and bias-prediction results are unaffected. The corrected model
predicts participants better than the published one, and the ranking of the models
against each other is essentially unchanged.

## Bug 1: the heuristic branch reads a payoff that has not been assigned yet

In `COMPETITION_CATIE_schedule_choice_probability.m`, the trend test read:

```matlab
if ((is_choice_1(trial-1) && (pays(trial) > pays(trial-1))) ||...
        (~is_choice_1(trial) && (pays(trial) < pays(trial-1))))
```

`pays` is preallocated as `NaN(nTrials,1)` and `pays(trial)` is assigned further down the
same iteration, at `pays(trial) = rewards_1(trial);`. At the point above it is therefore
always `NaN`, and MATLAB evaluates both `NaN > x` and `NaN < x` as false. The branch could
never select alternative 1. The line immediately below still ran
`p_try_explore = 1-pHeuristic`, so on every trend-testable trial 29% (since τ = 0.29) of the probability mass was removed from the other three modes and handed to alternative 2. Overall, this affected 16.6% of all trials.

The second clause also used `is_choice_1(trial)`, the choice being predicted, where it should use `is_choice_1(trial-1)`.

The intended rule is unambiguous in the simulator (`CATIE_schedule_1.m`, lines 152-157):
if the trend of the last two outcomes was positive, repeat the last choice. If it was negative, switch. Shifted back by one trial
for a likelihood function, that is `pays(trial-1)` against `pays(trial-2)`, with
`is_choice_1(trial-1)` in both clauses. That is the correction applied.

## Bug 2: the sub-agent mixture used time-averaged weights

In `COMPETITION_CATIE_schedule_choice_probability_hetro.m`, the CAB-k sub-agents were
combined with:

```matlab
p_decisions = mean(agents_choice_probabilities' * normalized_..._all_but_last_trial,2);
```

The matrix product builds every trial-by-trial cross term and the `mean` collapses it, which is equivalent to weighting every trial by the posterior averaged over all 100 trials. 
Two consequences: the sequential adaptivity that the Bayesian model average exists for is discarded, and each trial's prediction is influenced by choices the participant had not yet made.

The correct element-wise form was already present in the file, on the following line, commented out. It is now the active line:

```matlab
p_decisions = sum(agents_choice_probabilities .* normalized_..._all_but_last_trial);
```

The published results were computed with this per-trial form, not with the shipped time-averaged one: as shipped, the code returns E[log p] = −0.6733, while the per-trial form returns −0.6776, which matches the −0.678 reported in the paper. 
Reinstating it therefore also makes the repository reproduce its own published numbers.

## Effect on the results

Pooled over all 12 schedules, 3,332 participants, 333,200 trials, with all parameters at their published values (τ = 0.29, ε = 0.30, φ = 0.71, k ∈ {0,1,2}), nothing was refitted.

| CATIE | E[p] | E[log p] |
|---|---|---|
| Published (Fig. 4, Tables S1-S2) | 0.619 | −0.678 |
| Corrected | 0.6295 | −0.6654 |

The change comes from bug 1, which improves E[p] by +0.0103 and E[log p] by +0.0122
E[log p] improves on all 12 schedules individually, by +0.0027 to +0.0205.

**The conclusions of the paper do not change.** 
CATIE is still the best model by E[p], and
is still clearly beaten on E[log p] by models 10 and 6. 
The correction closes about 11% of the gap between CATIE and model 10, the best model on that measure. 
Every correction here moves CATIE's score in the favourable
direction, so no result that CATIE underperforms is affected.