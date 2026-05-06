# CATIE Results — EDA Choice Probability Predictions

## What This File Is

`eda_with_catie_probabilities.csv` is the EDA SET (`cleaned_eda_data.csv`) enriched with a
new column, `catie_choice_probability`. It contains one row per trial for each human
participant in the EDA SET (492 subjects × 100 trials = 49,200 rows total).

## Why It Was Produced

As part of the original "Choice Engineering Competition" study, the CATIE model's ability
to predict human behavior was evaluated by calculating the probability it assigns to the
*actual* choice made by a participant at each trial, given that participant's full history
of choices and rewards up to that point. This file replicates that evaluation on the EDA
SET (schedules 4, 5, and 7), keeping the result trial-level for downstream analysis.

## How `catie_choice_probability` Was Calculated

**No model logic was reimplemented.** All computation is delegated to the original
MATLAB source files from the competition repository:

| File | Role |
|------|------|
| `COMPETITION_CATIE_schedule_choice_probability_hetro.m` | Entry point — called once per participant |
| `COMPETITION_CATIE_schedule_choice_probability.m` | Core CATIE algorithm (called internally) |
| `CATIE_implementation_helpers/getExploreProb.m` | Exploration probability helper |
| `CATIE_implementation_helpers/base2dec.m` | Contingency encoding helper |

The orchestrator script is `my_code/EDA_set/processing/compute_eda_catie_probabilities.m`.

### Data Mapping (from original study's `COMPETITION_empirical_decisions_probabilities.m`)

| EDA column | CATIE argument | Meaning |
|---|---|---|
| `biased_reward` | `rewards_1` | Reward scheduled for the biased alternative at each trial |
| `unbiased_reward` | `rewards_2` | Reward scheduled for the unbiased alternative at each trial |
| `is_biased_choice` | `is_choice_1` | `true` when the subject chose the biased alternative |

The function is called once per subject with the full 100-trial sequence sorted by
`trial_number`, and returns one probability value per trial.

### What the Probability Represents

`catie_choice_probability` is the probability that the CATIE heterogeneous model
assigns to the choice that participant *actually made* at that trial, given their
complete prior history (choices + rewards up to, but not including, that trial).

- A value close to **1** means the model confidently predicted the participant's choice.
- A value close to **0.5** means the model was uncertain between the two options.
- A value close to **0** means the model was surprised by the participant's choice.

The first trial of every subject is always **0.5** because the CATIE model has no prior
history and assigns equal probability to both alternatives.

### CATIE Heterogeneous Model Summary

The heterogeneous variant integrates across three sub-agents with contingency depths
k = 0, 1, 2. At each trial, each sub-agent's likelihood of having produced the observed
choices up to that point is used to weight its prediction for the current trial (Bayesian
belief updating). This allows the model to adapt its effective contingency depth to the
observed behavior without fitting per-participant parameters.

Fixed model parameters (pre-fitted, not adjusted in this study):
- τ (trend/heuristic probability): 0.29
- ε (exploration base rate): 0.30
- φ (inertia probability): 0.71

## Derived Metrics

To compute the standard model comparison metrics reported in the paper:

```matlab
data = readtable('eda_with_catie_probabilities.csv');

% E[p]: mean choice probability across all trials and participants
E_p = mean(data.catie_choice_probability);

% E[log(p)]: mean log-probability (log-likelihood per trial)
E_logp = mean(log(data.catie_choice_probability));

% Per-subject E[p]
subjects = unique(data.subject_file);
for i = 1:numel(subjects)
    mask = strcmp(data.subject_file, subjects{i});
    subject_E_p(i) = mean(data.catie_choice_probability(mask));
end
```

Expected values (based on published paper results for CATIE-scheduled participants):
- E[p] ≈ 0.619
- E[log(p)] ≈ −0.678

Note: the EDA SET covers schedules 4, 5, and 7 only, so exact values will differ from
the paper's full-competition figures.

## Column Dictionary

| Column | Type | Values / Format | Description |
|--------|------|-----------------|-------------|
| `trial_number` | integer | 0 – 99 | Trial index within the session (0-indexed) |
| `time` | string | MM:SS.S | Elapsed session time at trial onset |
| `is_biased_choice` | string | TRUE / FALSE | Whether the subject chose the biased (rewarded) alternative |
| `side_choice` | string | LEFT / RIGHT | Physical side chosen by the subject |
| `RT` | float | ms | Raw reaction time from stimulus onset to response |
| `observed_reward` | integer | 0 / 1 | Reward actually received by the subject on this trial |
| `unobserved_reward` | integer | 0 / 1 | Reward that would have been received had the other option been chosen |
| `biased_reward` | integer | 0 / 1 | Reward scheduled for the biased alternative on this trial (used as `rewards_1`) |
| `unbiased_reward` | integer | 0 / 1 | Reward scheduled for the unbiased alternative on this trial (used as `rewards_2`) |
| `BLUE_RIGHT_LEFT_RED` | string | TRUE / FALSE | Screen layout flag (TRUE = blue button on right, red on left) |
| `RT_net` | float | ms | Net reaction time after subtracting a baseline offset |
| `RT_zscore` | float | z-score | Reaction time standardized within subject |
| `subject_file` | string | Unix timestamp_XX.csv | Unique participant identifier (original filename) |
| `schedule` | string | schedule_4 / 5 / 7 | Experimental reward schedule assigned to this participant |
| `catie_choice_probability` | float | (0, 1] | **New column.** CATIE heterogeneous model's probability for the actual choice made at this trial |
