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
├── matlab/               independent MATLAB validation -- see matlab/README.md
├── build_cache.py        precompute parameter-free state tensors -> cache/ (Phase 3+)
├── 01_bug_correction/    Phase 1 -- the published likelihood bug (done)
├── 02_mode_calibration/  Phase 2 -- which partition explains the miscalibration (done)
├── 03_parameter_fitting/ Phase 3 -- re-fitting vs post-hoc calibration (done)
├── 04_model_extension/   Phase 4 -- asymmetric inertia, lapse, soft CA
└── 05_nn_ceiling/        Phase 5 -- how much predictable structure remains
```

## Getting started

```bash
python my_code/catie_calibration/golden_test.py        # must pass before anything else
python my_code/catie_calibration/sanitize_splits.py    # builds data/cleaned_*.csv
python my_code/catie_calibration/01_bug_correction/bug_benchmark.py
```

## Correctness -- what's actually been checked

`golden_test.py` and `matlab/` together give five independent layers of validation,
not one check repeated five times:

1. The `"published"` port reproduces ORIGINAL, unmodified MATLAB, called live (not a
   stored CSV), across all 12 schedules -- hetero mixture and each single-`k` model
   (2.331e-15, pooled). The independent monolithic reference (item 2) is checked
   against that same live MATLAB directly too (6.661e-16).
2. A from-scratch, single-pass monolithic reimplementation (`golden_test.py`,
   `_reference_catie_probability`) that never splits state from parameters agrees
   with the production split (`state_tensors` + `probability_from_state`), for
   both modes and all three `k` values (3.3e-16).
3. `mode_contributions()`'s four terms sum to `probability_from_state()`'s output
   (3.3e-16) -- checked at runtime, not just true by algebraic construction.
4. A second, completely independent implementation -- a minimally-patched copy of
   the real MATLAB source, run in real MATLAB, not Python -- reproduces every
   published-vs-fixed E[p]/E[log p] number exactly across all 3,332 subjects
   (`matlab/README.md` §1).
5. `state_tensors()`'s individual outputs (`H`, `b`, `c_prev`, `s_prev`,
   `sbar_prev`, `g`) match MATLAB's own internal loop variables **element by
   element** -- not just the final probability -- across all 998,400 trials in
   the sanitized population (`matlab/README.md` §2). Exact match (0.0) on the
   boolean/integer tensors, one ULP (5.55e-16) on the floating-point ones.

Layer 5 is the one worth knowing about specifically: layers 1-4 all compare final
probabilities, so a pair of bugs inside the state recursion that happened to
cancel out could in principle survive all four. Layer 5 compares the intermediate
values those bugs would have to hide in, which closes that gap.

Dependencies: `pandas numpy scipy matplotlib scikit-learn` (all present in the repo
`.venv`). No `torch`, no GPU — see "Why no autodiff" below.

## Data splits

Schedule-level splits, so held-out evaluation tests generalisation to **unseen reward
schedules** rather than unseen participants. Fit on training, select on EDA, report once
on test.

| Split | Subjects | Schedules |
|---|---|---|
| training | 1,483 | 2, 3, 6, 9, 11 |
| EDA | 496 | 4, 5, 7 |
| test | 804 | 1, 8, 10 |
| schedule_0 | 549 | 0 (the 69.0% empirically-tuned benchmark) |

Per-schedule subject counts reproduce the paper's reported N exactly for **all twelve**
schedules (3,332 total, the paper's own figure). Data comes from the competition's
organized release (`Data_resources/.../simple_format_data/`); the exclusion rule is the
curators' own `..._INVALID_BIAS.csv` tag rather than an inferred threshold. See
`../completed_issues/SCHEDULE_N_RECONCILIATION.md` and
`../completed_issues/REORGANIZATION_PLAN.md`.

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
parameters. This split was cross-checked against an independent monolithic
implementation and against MATLAB's own internal variables — see "Correctness" above —
not just assumed correct because the production code always calls the two functions in
sequence.

## Closed directions — do not reopen

**Reaction time.** The U-shape at extreme p is an artifact of z-scoring subjects with
near-zero personal SD against the 1.5 s hardware floor. At the robust threshold (X=10%)
the Surprise–Routine contrast is n.s. (d=0.02, p≈0.82). The X=5% bins are literally the
195 floor-clipped and 509 ceiling-clipped rows, nearly all at trials 1–3 where every
participant is slow. See `my_code/initial_investigation/rt_analysis/`.

**Current-choice side switches.** `is_surprise ⊂ {switch trials}` by construction, since
φ=0.71 means a repeated choice can never receive p < 0.15. Measured
P(surprise | no switch) = 0.000 exactly. Additionally `BLUE_RIGHT_LEFT_RED` is constant
within every subject, so `side_choice ≡ is_biased_choice` up to relabelling — motor and
value inertia are not separable in this dataset. Any feature reading the current trial's
choice is circular. See `my_code/initial_investigation/surprise_analysis/`.

## Conventions

Each analysis directory holds `<script>.py`, `README.md`, and `figures/` containing PNGs
plus an `output.txt` transcript (via `metrics.Tee`). Scripts set `matplotlib.use("Agg")`
before importing pyplot, assert expected row counts at load, seed at 42, and save at
`dpi=150, bbox_inches="tight"`.

**Never open the generated CSVs in Excel.** The existing `cleaned_eda_data.csv` had its
`time` column destroyed that way (`05-25-2020 15:11:02.738300` → `11:02.7`) and its
booleans re-cased to `TRUE`/`FALSE`; that damage is unrecoverable from the cleaned file.
