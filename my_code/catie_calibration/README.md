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
├── CLAUDE.md              rules for every Claude session -- read first
├── PLAN.md                the current week plan
├── catie/                 the validated base (import from here, nothing else)
│   ├── core.py            exact port of the competition CATIE likelihood (MATLAB-verified)
│   ├── likelihood.py      StateCache + closed-form P(alt1) over cached state tensors
│   ├── cache.py           precompute parameter-free state tensors -> cache/
│   ├── splits.py          raw subject CSVs -> one tidy CSV per split -> data/
│   └── metrics.py         E[p], E[log p], ECE, reliability tables, cluster bootstrap, paired test
├── tests/                 mirrors the code tree; <file>_test.py
│   └── catie/             core_test.py (golden gate), likelihood_test.py (vs paper), metrics_test.py
├── matlab/                independent MATLAB validation -- see matlab/README.md
├── 01_bug_correction/     Phase 1 -- the published likelihood bug (done, kept as is)
├── 02_reliability/        reliability diagrams on Training            (stage README inside)
├── 03_recalibration/      maps fitted on Training, scored on EDA
├── 04_refit/              M2 (tau, eps, phi) and M3 (run-length phi)
├── 05_ladder/             one table, paired tests, Pareto figure
├── 06_test/               Test split, once
├── data/, cache/          generated, gitignored
└── (archive lives in ../previous_errors/ -- superseded, off-limits, see its README)
```

Each stage holds one script (≤150 lines, no prints), a README, `figures/`, and one CSV.
Every number quoted anywhere points to one CSV row from one script.

## Getting started

```bash
cd my_code/catie_calibration
python -m pytest tests -q                       # metrics unit checks
python tests/catie/core_test.py                 # golden gate -- run this first
python tests/catie/likelihood_test.py training  # paper reproduction (also: eda, schedule_0)
python -m catie.splits                          # builds data/cleaned_*.csv
python -m catie.cache                           # builds cache/
```

The golden gate and the paper reproduction are scripts with a `main()`, not `pytest`
test functions, so `pytest` collects them without running anything — run them directly.

## Correctness — what has actually been checked

`tests/catie/core_test.py` and `matlab/` together give five independent layers of
validation, not one check repeated five times:

1. The `"published"` port reproduces ORIGINAL, unmodified MATLAB, called live (not a
   stored CSV), across all 12 schedules — hetero mixture and each single-`k` model
   (2.331e-15, pooled). The independent monolithic reference (item 2) is checked
   against that same live MATLAB directly too (6.661e-16).
2. A from-scratch, single-pass monolithic reimplementation (`_reference_catie_probability`
   in `core_test.py`) that never splits state from parameters agrees with the production
   split (`state_tensors` + `probability_from_state`), for both modes and all three `k`
   values (3.3e-16).
3. `mode_contributions()`'s four terms sum to `probability_from_state()`'s output
   (3.3e-16) — checked at runtime, not just true by algebraic construction.
4. A second, completely independent implementation — a minimally-patched copy of the
   real MATLAB source, run in real MATLAB — reproduces every published-vs-fixed
   E[p]/E[log p] number exactly across all 3,332 subjects (`matlab/README.md` §1).
5. `state_tensors()`'s individual outputs (`H`, `b`, `c_prev`, `s_prev`, `sbar_prev`, `g`)
   match MATLAB's own internal loop variables **element by element** across all 998,400
   trials in the sanitized population (`matlab/README.md` §2). Exact on the
   boolean/integer tensors, one ULP (5.55e-16) on the floating-point ones.

Layer 5 is the one worth knowing about: layers 1–4 compare final probabilities, so a
pair of bugs inside the state recursion that cancelled could survive all four. Layer 5
compares the intermediate values those bugs would have to hide in.

`tests/catie/likelihood_test.py` closes the loop one level up: the closed-form
likelihood over cached tensors reproduces the paper's per-split E[p] and E[log p].

Dependencies: `pandas numpy scipy matplotlib scikit-learn` (repo `.venv`). No `torch`,
no GPU — see "Why no autodiff".

## Data splits

Schedule-level splits, so held-out evaluation tests generalisation to **unseen reward
schedules** rather than unseen participants. Fit on Training, score once on EDA, report
once on Test.

| Split | Subjects | Schedules |
|---|---|---|
| training | 1,483 | 2, 3, 6, 9, 11 |
| EDA | 496 | 4, 5, 7 |
| test | 804 | 1, 8, 10 |
| schedule_0 | 549 | 0 (the 69.0% empirically-tuned benchmark) |

Per-schedule subject counts reproduce the paper's reported N exactly for all twelve
schedules (3,332 total). Data comes from the competition's organized release
(`Data_resources/.../simple_format_data/`); the exclusion rule is the curators' own
`..._INVALID_BIAS.csv` tag rather than an inferred threshold.

## One thing that is easy to get wrong

**`p_choice` vs `p_alt1`.** `p_choice` is P(the choice actually made) — what E[p] and
E[log p] score. `p_alt1` is P(choose the biased alternative) — a forecast of a *fixed*
event, and the only quantity for which calibration is meaningful. A reliability curve
over `p_choice` is degenerate, because its outcome is 1 by construction.

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
a recurrence. Only `K` alters the state recursion, and it is handled by enumeration
(`K` is drawn uniformly from {0, 1, 2} in the paper's own code, so the {0,1,2} mixture
*is* the paper's model).

## Closed directions — do not reopen

**Reaction time.** The U-shape at extreme p is an artifact of z-scoring subjects with
near-zero personal SD against the 1.5 s hardware floor. At the robust threshold the
Surprise–Routine contrast is n.s. (d=0.02, p≈0.82).

**Current-choice side switches.** `is_surprise ⊂ {switch trials}` by construction, since
φ=0.71 means a repeated choice can never receive p < 0.15. `BLUE_RIGHT_LEFT_RED` is
constant within every subject, so `side_choice ≡ is_biased_choice` up to relabelling.
Any feature reading the current trial's choice is circular.

**E-step / posterior partitions.** Conditioning on a responsibility computed from the
observed choice reads the outcome into the predictor.

(The evidence for the first two lives in `../previous_errors/initial_investigation/`, kept
for the record only; the conclusions above are what matters.)

## Conventions

Scripts set `matplotlib.use("Agg")` before importing pyplot, assert expected row counts
at load, seed at 42, save at `dpi=150, bbox_inches="tight"`, and **do not print** —
outputs are CSVs and figures.

**Never open the generated CSVs in Excel.** A `time` column was once destroyed that way
(`05-25-2020 15:11:02.738300` → `11:02.7`); that damage is unrecoverable.
