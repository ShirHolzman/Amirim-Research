# CATIE Calibration

Thesis codebase: diagnosing, correcting and extending the CATIE model's trial-level
choice probabilities on the Choice Engineering Competition dataset.

## Research question

CATIE won the competition on **E[p] = 0.619** but *lost* on **E[log p] = −0.678** to a
Q-Learning model (−0.569). The source paper (Dan, Plonsky & Loewenstein, *Nat. Commun.*
2025) describes this as "a relatively large fraction of actions to which the model
assigned a relatively small probability" but does not explain why, where, or whether it
is fixable. This project answers that.

## Layout

```
catie_calibration/
├── catie_core.py         exact port of the competition CATIE likelihood
├── metrics.py            E[p], E[log p], ECE, MCE, Brier, subject-clustered bootstrap
├── sanitize_splits.py    raw subject CSVs -> one tidy CSV per split
├── golden_test.py        BLOCKING correctness gate (run this first)
├── data/                 generated, gitignored
├── 01_bug_correction/    Phase 1 -- the published likelihood bug
├── 02_mode_calibration/  Phase 2 -- which partition explains the miscalibration
├── 03_parameter_fitting/ Phase 3 -- re-fitting vs post-hoc calibration
├── 04_model_extension/   Phase 4 -- asymmetric inertia, lapse, soft CA
└── 05_nn_ceiling/        Phase 5 -- how much predictable structure remains
```

## Getting started

```bash
python my_code/catie_calibration/golden_test.py        # must pass before anything else
python my_code/catie_calibration/sanitize_splits.py    # builds data/cleaned_*.csv
python my_code/catie_calibration/01_bug_correction/bug_benchmark.py
```

Dependencies: `pandas numpy scipy matplotlib scikit-learn` (all present in the repo
`.venv`). No `torch`, no GPU — see "Why no autodiff" below.

## Data splits

Schedule-level splits, so held-out evaluation tests generalisation to **unseen reward
schedules** rather than unseen participants. Fit on training, select on EDA, report once
on test.

| Split | Subjects | Schedules |
|---|---|---|
| training | 1,483 | 2, 3, 6, 9, 11 |
| EDA | 492 | 4, 5, 7 |
| test | 804 | 1, 8, 10 |
| schedule_0 | 549 | 0 (the 69.0% empirically-tuned benchmark) |

Per-schedule subject counts reproduce the paper's reported N exactly for all nine
schedules, which independently validates the exclusion rule (drop a subject who chose
one side fewer than 5 times).

## Two things that are easy to get wrong

**`p_choice` vs `p_alt1`.** `p_choice` is P(the choice actually made) — what E[p] and
E[log p] score, and what the stored `catie_choice_probability` column holds. `p_alt1` is
P(choose the biased alternative) — a forecast of a *fixed* event, and the only quantity
for which calibration is meaningful. A reliability curve over `p_choice` is degenerate,
because its outcome is 1 by construction. `metrics.score_all` takes both.

**`subject_id`, never `subject_file`.** Filenames are timestamp-derived and *do* collide
across schedules (`1609192765_75.csv` is a different participant in `schedule_6` and in
`schedule_11`). Grouping on `subject_file` silently merges them into a 200-trial subject.
`sanitize_splits.py` emits `subject_id = "<schedule>/<file>"` and asserts uniqueness.

## Why no autodiff

The likelihood is conditioned on the participant's observed choices, so CATIE never
samples. Every internal state variable — reward means, observed SDs, surprise,
contingency tables, `g`, `H`, `c_prev` — is a function of the **data only**, independent
of τ, ε and φ. Given cached state, the choice probability is closed-form:

```
P(alt 1) = τ·H·b + (1 − τ·H) · [ p_exp/2 + (1 − p_exp)·(φ·c_prev + (1 − φ)·g) ]
p_exp    = ε · (1 + s_prev + s̄_prev) / 3
```

So parameter fitting is vectorised arithmetic over cached tensors, not backprop through
a recurrence. Only `K` alters the state recursion, and it is handled by enumeration.
`state_tensors()` computes the data-only part; `probability_from_state()` applies
parameters. The direct-port path calls exactly those two functions in sequence, so the
cached path cannot silently diverge from the port.

## Closed directions — do not reopen

**Reaction time.** The U-shape at extreme p is an artifact of z-scoring subjects with
near-zero personal SD against the 1.5 s hardware floor. At the robust threshold (X=10%)
the Surprise–Routine contrast is n.s. (d=0.02, p≈0.82). The X=5% bins are literally the
195 floor-clipped and 509 ceiling-clipped rows, nearly all at trials 1–3 where every
participant is slow. See `my_code/EDA_set/rt_analysis/`.

**Current-choice side switches.** `is_surprise ⊂ {switch trials}` by construction, since
φ=0.71 means a repeated choice can never receive p < 0.15. Measured
P(surprise | no switch) = 0.000 exactly. Additionally `BLUE_RIGHT_LEFT_RED` is constant
within every subject, so `side_choice ≡ is_biased_choice` up to relabelling — motor and
value inertia are not separable in this dataset. Any feature reading the current trial's
choice is circular. See `my_code/EDA_set/surprise_analysis/`.

## Conventions

Each analysis directory holds `<script>.py`, `README.md`, and `figures/` containing PNGs
plus an `output.txt` transcript (via `metrics.Tee`). Scripts set `matplotlib.use("Agg")`
before importing pyplot, assert expected row counts at load, seed at 42, and save at
`dpi=150, bbox_inches="tight"`.

**Never open the generated CSVs in Excel.** The existing `cleaned_eda_data.csv` had its
`time` column destroyed that way (`05-25-2020 15:11:02.738300` → `11:02.7`) and its
booleans re-cased to `TRUE`/`FALSE`; that damage is unrecoverable from the cleaned file.
