# RT Analysis — CATIE Prediction Confidence vs. Reaction Time

## Research Rationale

The original Choice Engineering Competition study deliberately ignored reaction time (RT) data because of a hardware constraint: a mandatory 1.5-second button deactivation after every choice. This floor makes raw RT uninformative (every response is at least 1 500 ms). However, this same constraint creates a natural signal:

- If a participant is on **autopilot** — mechanically repeating a learned pattern — they simply wait for the lock to expire, pressing as soon as the button reactivates. RT will be close to the floor.
- If a participant is **consciously deliberating** — pausing to reconsider their strategy — they will react substantially later than their own baseline, even accounting for the floor.

Two derived measures correct for this floor:

| Column | Formula | Meaning |
|--------|---------|---------|
| `RT_net` | `RT − 1 500 ms` | Excess time above the hardware lock, per trial |
| `RT_zscore` | `(RT_net − μᵢ) / σᵢ` per subject | RT relative to each participant's own mean; removes individual baseline speed differences |

The analysis uses `RT_zscore` throughout. Because it is normalised within each participant, group differences reflect within-subject patterns, not between-subject speed variation.

The CATIE model's `catie_choice_probability` provides a trial-level readout of how "expected" a participant's choice was, given their history. This analysis tests whether that readout correlates with how fast the participant responded.

---

## Three Probability Categories

| Category | Criterion | Interpretation |
|----------|-----------|---------------|
| **Surprise** | `catie_choice_probability < 0.05` | CATIE was nearly certain the participant would choose the other option; the participant broke their own statistical pattern |
| **Routine** | `0.05 ≤ catie_choice_probability ≤ 0.95` | Typical decision — model assigns moderate probability to both options |
| **Autopilot** | `catie_choice_probability > 0.95` | CATIE was highly confident in predicting the choice; the participant was following a learned, consistent pattern |

**Expected RT direction**: Surprise > Routine > Autopilot (a staircase in `RT_zscore`).

---

## How to Run

### Prerequisites

The input file must exist first. It is produced by the MATLAB wrapper in `processing/`:

```
my_code/EDA_set/processing/eda_with_catie_probabilities.csv
```

If this file is missing, run `compute_eda_catie_probabilities.m` in MATLAB before continuing.

### Dependencies

```bash
pip install pandas numpy scipy matplotlib seaborn
```

All four are standard scientific Python packages available on PyPI.

### Execution

```bash
cd my_code/EDA_set/rt_analysis
python rt_analysis.py
```

---

## Expected Console Output

```
============================================================
RT vs. CATIE Choice Probability — EDA SET Analysis
============================================================

Loading: …/processing/eda_with_catie_probabilities.csv
Loaded 49,200 rows | 492 subjects | 100 trials each
Schedules: ['schedule_4', 'schedule_5', 'schedule_7']

── Descriptive Statistics (RT_zscore) ─────────────────────────────
Category          n       mean      std      SEM   95% CI lower   95% CI upper
Surprise       XXXX     +0.XXX   …        …        …              …
Routine       XXXXX     +0.XXX   …        …        …              …
Autopilot      XXXX     -0.XXX   …        …        …              …

── Welch's t-tests (Bonferroni corrected, k=3) ─────────────────────
…

── Saving figures ──────────────────────────────────────────────────
  fig1_bar_chart.png / .svg
  fig2_kde_overlay.png / .svg
  fig3_scatter_trend.png / .svg
  fig4_schedule_bars.png / .svg
  fig5_trial_trajectory.png / .svg

All figures saved to: …/rt_analysis/figures/
Done.
```

If the hypothesis holds, the Surprise group will show a positive mean `RT_zscore` and the Autopilot group a negative mean, with the Routine group near zero.

---

## Output Files

All figures are saved to `rt_analysis/figures/` in both PNG (150 dpi, for reports) and SVG (vector, for editing).

| File | Description |
|------|-------------|
| `fig1_bar_chart.png/svg` | Mean `RT_zscore` ± SEM for each category. The "staircase" pattern is visible here if the hypothesis holds. |
| `fig2_kde_overlay.png/svg` | Full RT_zscore distributions for all three categories overlaid on one axes. Dashed vertical lines mark group means. |
| `fig3_scatter_trend.png/svg` | Scatter plot of `RT_zscore` vs `catie_choice_probability` (5 000-point subsample for clarity) with a linear regression line and a LOWESS smooth curve, both fitted on all 49 200 trials. |
| `fig4_schedule_bars.png/svg` | Same staircase, broken down per reward schedule (4, 5, 7). Tests whether the RT pattern is driven by CATIE probability rather than schedule-specific reward structure. |
| `fig5_trial_trajectory.png/svg` | Mean RT_zscore across five trial quintiles (T0–19, T20–39, …, T80–99) for each category. Tests whether the autopilot effect builds up as participants habituate to the reward schedule. |

---

## Statistical Methods

**Pairwise comparison**: Welch's t-test (does not assume equal variance; appropriate for groups of very different sizes).

**Multiple comparisons**: Bonferroni correction across 3 pairwise tests (Surprise–Routine, Surprise–Autopilot, Routine–Autopilot).

**Effect size**: Cohen's d using the unequal-variance pooled form:

```
d = (μ_a − μ_b) / √((σ²_a + σ²_b) / 2)
```

---

## Additional Exploratory Analyses

### Figure 4 — RT Staircase Across Reward Schedules

**What it shows**: The three-tier bar chart reproduced separately for each of the three reward schedules (4, 5, 7).

**Scientific rationale**: The EDA SET covers three different reward contingency schedules. If the RT–probability relationship is a genuine cognitive effect (autopilot vs. deliberation) rather than an artefact of a specific schedule's reward structure, the staircase pattern should replicate within each schedule independently. Convergent replication across schedules strengthens the claim.

### Figure 5 — Trial-Position Trajectory

**What it shows**: Mean RT_zscore for each category, plotted across five trial quintiles (blocks of 20 trials each), with ±1 SEM shaded bands.

**Scientific rationale**: Participants are expected to learn the reward contingency progressively over the 100-trial session. If autopilot behaviour emerges through learning, the gap between the Autopilot and Surprise categories should *widen* in later quintiles compared to earlier ones. Conversely, if the gap is stable, the effect is present from the start and may reflect trait-level rather than session-level habituation. This plot lets us distinguish those two hypotheses visually.

---

## Data Dependency Diagram

```
my_code/EDA_set/
│
├── early_proccessing/
│   └── sanitize.py           ──► cleaned_eda_data.csv
│
├── processing/
│   ├── compute_eda_catie_probabilities.m
│   └── eda_with_catie_probabilities.csv  ◄── INPUT for this script
│
└── rt_analysis/
    ├── rt_analysis.py         ◄── THIS SCRIPT
    ├── README.md              ◄── this file
    └── figures/               ◄── output (auto-created)
        ├── fig1_bar_chart.*
        ├── fig2_kde_overlay.*
        ├── fig3_scatter_trend.*
        ├── fig4_schedule_bars.*
        └── fig5_trial_trajectory.*
```
