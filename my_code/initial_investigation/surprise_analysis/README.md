# CATIE Surprise Point Analysis

## Research Question

Why does the CATIE model sometimes completely fail to predict human behavior?
A "surprise point" is any trial where `catie_choice_probability < 0.15` — CATIE assigned less
than 15% probability to the choice the participant actually made.

This script identifies what contextual and historical patterns reliably precede those failures.

---

## Surprise Threshold

| Criterion | Label | Count (out of 49,200) |
|-----------|-------|----------------------|
| `catie_choice_probability < 0.15` | Surprise | ~4,454 (9.1%) |
| All other trials | Baseline | ~44,746 (90.9%) |

---

## Features Engineered

All features are derived solely from **observed** data — the partial-feedback paradigm means
participants and CATIE only see `observed_reward`. The columns `biased_reward`,
`unbiased_reward`, and `unobserved_reward` are **never used as features**.

All lag and rolling operations are computed within each subject's 100-trial session
(via `groupby("subject_file")`) to prevent information leaking across participants.

| Feature | Description |
|---------|-------------|
| `prev_choice_biased_1/2/3` | Whether the biased option was chosen at t−1, t−2, t−3 |
| `prev_reward_1/2/3` | Observed reward received at t−1, t−2, t−3 |
| `prev_side_switch` | 1 if participant switched physical side (L/R) on the current trial |
| `streak_same_choice` | Consecutive trials with same biased/unbiased choice ending here |
| `streak_same_side` | Consecutive trials on the same physical side ending here (motor inertia) |
| `recent_reward_rate` | Mean observed reward over last 5 trials |
| `recent_switch_rate` | Fraction of last 5 trials that were side switches |
| `cumulative_bias_rate` | Running proportion of biased choices so far in the session |
| `catie_p_prev` | CATIE's confidence at t−1 (was it autopilot just before the break?) |
| `trial_number` | Session progress (0–99) |

---

## Methods

### Statistical comparison
Welch's t-test (unequal variance) for each feature, comparing Surprise vs Baseline.
Bonferroni correction for k = 14 pairwise comparisons. Effect size = Cohen's d
(unequal-variance pooled form).

### Random Forest feature importance
`RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced")` trained
on all 14 features. Feature importances (mean decrease in impurity) rank which variables
CATIE-failure trials rely on most.

### Shallow decision tree
`DecisionTreeClassifier(max_depth=4)` fitted for human-readable decision rules.
Printed as text via `export_text`.

---

## Key Findings (from a representative run)

1. **Side switching is the overwhelmingly dominant predictor** (Cohen's d ≈ 2.15, p ≈ 0).
   `prev_side_switch = 1.0` for **every** surprise trial — CATIE is only surprised when
   the participant physically switches to the other side. This makes sense: CATIE learns
   sequence patterns, so continuation is always more probable than a switch given long streaks.

2. **Streak length is the second strongest signal** (Cohen's d ≈ −0.75 for both
   `streak_same_choice` and `streak_same_side`). Surprise trials have streak = 1 (just
   broke a run), while baseline trials average streak ~6.8. The longer the habitual
   repetition before the break, the more surprised CATIE is.

3. **Choice history strongly predicts surprise direction** (Cohen's d ≈ 0.69 for
   `prev_choice_biased_1`). Before a surprise, participants overwhelmingly chose the biased
   option (mean ≈ 0.87 vs 0.58 baseline). Pattern-breaking tends to happen after a string
   of consistent biased choices — participants "rebel" against their own habit.

4. **Recent reward rate is slightly lower before surprises** (d ≈ −0.18–−0.25 for
   `prev_reward_1/2`). Mild evidence that losing streaks contribute to pattern breaks, but
   the effect is modest compared to the switching signal.

5. **CATIE confidence at t−1 is slightly lower before surprises** (d ≈ −0.13). Contrary to
   the "autopilot → sudden break" hypothesis, surprise points are *not* predominantly
   preceded by high CATIE confidence. The effect is weak; side-switching dominates.

6. **Surprise rate increases slightly later in the session** (higher `trial_number`
   in surprise trials). Fatigue or boredom accumulation may contribute marginally.

---

## Output Figures

All figures saved to `figures/` in PNG (150 dpi) and SVG format.

| File | Description |
|------|-------------|
| `fig1_feature_comparison.png/svg` | Cohen's d for top 8 features (Surprise vs Baseline). Red bars = higher in surprise trials. |
| `fig2_reward_history_heatmap.png/svg` | Surprise rate as a function of reward at t−1 and t−2. Shows whether loss sequences drive pattern-breaking. |
| `fig3_streak_vs_surprise.png/svg` | Surprise rate vs consecutive same-choice / same-side streak length. Confirms that habit breaks follow long runs. |
| `fig4_trial_position.png/svg` | Surprise rate per trial quintile, broken down by reward schedule (4, 5, 7). |
| `fig5_prior_catie_confidence.png/svg` | KDE of CATIE's confidence at t−1 for surprise vs baseline trials. |
| `fig6_rf_feature_importance.png/svg` | Random Forest feature importances, confirming side-switching dominates. |

---

## How to Run

```bash
cd my_code/EDA_set/surprise_analysis
python surprise_analysis.py
```

### Dependencies

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn
```

### Prerequisites

The input file must exist:
```
my_code/EDA_set/processing/eda_with_catie_probabilities.csv
```

Run the MATLAB wrapper `compute_eda_catie_probabilities.m` first if it is missing.

---

## Data Dependency Diagram

```
my_code/EDA_set/
│
├── early_proccessing/
│   └── sanitize.py               ──► cleaned_eda_data.csv
│
├── processing/
│   ├── compute_eda_catie_probabilities.m
│   └── eda_with_catie_probabilities.csv  ◄── INPUT for this script
│
└── surprise_analysis/
    ├── surprise_analysis.py       ◄── THIS SCRIPT
    ├── README.md                  ◄── this file
    └── figures/                   ◄── output (auto-created)
```
