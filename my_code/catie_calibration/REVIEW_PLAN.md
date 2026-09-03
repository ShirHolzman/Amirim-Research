# Independent Review Plan — Phases 0–3

A structured guide for reviewing this codebase yourself, file by file, for **both code
correctness and methodological soundness**.

## How to use this document

Work top to bottom. Phase 0 first — every later number is computed on top of it, so a
defect there invalidates everything downstream, while a defect in Phase 3 is contained.

Each file entry has four parts:

- **Does** — what the file is for.
- **Method** — the algorithm or statistic actually used.
- **Check** — specific things to verify. These are questions with answers, not vague
  prompts. Where a number is given, it is what the code currently produces.
- **Known issues** — defects already identified. Confirm these rather than rediscover them.

Three labels appear throughout:

- **[VERIFIED]** — independently reproduced, by MATLAB, by a second implementation, or
  by re-derivation from raw data. Safe to accept, but spot-check the reasoning.
- **[UNVERIFIED]** — produced by one code path only, or asserted by an adversarial audit
  whose mathematics you have not checked. **Treat as a claim, not a fact.**
- **[OPEN]** — a known discrepancy with no resolution yet.

> **A note on documentation.** Every README in this tree, including this file, is a
> claim. Comments go stale and printed labels can describe something other than what was
> computed. Where a check matters, read the executable path or recompute the number.
> The `scratchpad/pareto.py` incident below is a live example of exactly this failure.

---

## Cross-cutting checks — apply to every phase

Before the per-file review, these are the failure modes that recur across the project.
Keep them in mind while reading any file.

| # | Check | Why it matters |
|---|---|---|
| X1 | **`p_choice` vs `p_alt1`.** `p_choice` = P(the action actually taken); `p_alt1` = P(choose the biased alternative). E[p] and E[log p] score `p_choice`. Calibration (ECE, reliability, gap, R²) is only meaningful on `p_alt1`, because `p_choice`'s outcome is 1 by construction. | Silently invalidates any calibration result. The single highest-risk confusion in the project. |
| X2 | **Subject-level, never trial-level, inference.** ~100 trials per subject are strongly correlated. Every CI and p-value must resample or cluster on **subjects**. | Trial-level independence inflates significance by roughly √(trials per subject) ≈ 10×. |
| X3 | **Circularity.** Any feature or partition built from the current trial's own choice — or from a posterior/E-step that uses it — is circular. Only history-only (prospective) features may be scored. | Already caused a spurious R² = 0.967 in Phase 2. |
| X4 | **Subject-boundary leakage in lagged features.** `c_prev`, `streak`, and any `.shift()` must reset at each subject. Matrices are (subject × trial), so rows are independent by construction — verify that shape assumption holds wherever a lag is computed. | Manufactures effects out of nothing. |
| X5 | **In-sample vs held-out.** Fit on Training, select on EDA, report once on Test. Check which split each number was computed on, and whether any selection touched the split it is reported on. | Selection on the evaluation set biases every "held-out" claim. |
| X6 | **Which baseline?** *published model* (reproduces the paper, contains the bug), *corrected model* (Phase 1 fix), and *published parameters* (τ=.29, ε=.30, φ=.71) are three different things. A number is meaningless without saying which. | See the 0.6158 incident in §Open Items — this has already gone wrong once. |
| X7 | **Partition cardinality.** Comparing partitions with 2 vs 20 groups on in-sample R² flatters the finer partition. | Affects the whole Phase 2 adjudication table. |

---

## Phase 0 — Infrastructure

Everything downstream is arithmetic over what this phase produces. Review it hardest.

### `catie_core.py` (409 lines)

**Does.** The reference Python port of the competition CATIE likelihood. Two modes:
`"published"` (bug-for-bug faithful to the MATLAB) and `"fixed"` (Phase 1 correction).

**Method.** Splits the model into a **parameter-free state recursion** and a **closed-form
probability**. The enabling fact: the likelihood is conditioned on the participant's
observed choices, so CATIE never samples, so every state variable (reward means, observed
SDs, surprise, contingency tables, `g`, `H`, `c_prev`) depends on the **data only** —
independent of τ, ε, φ. Only `K` alters the recursion, and is handled by enumeration.

```
P(alt 1) = τ·H·b + (1 − τ·H)·[ p_exp/2 + (1 − p_exp)·(φ·c_prev + (1 − φ)·g) ]
p_exp    = ε·(1 + s_prev + s̄_prev)/3
```

Key API: `state_tensors()` (data-only part), `probability_from_state()` (applies
parameters), `mode_contributions()`, `mode_weights()`, `mix_agents()`, `catie_hetero()`,
`p_of_observed_choice()`.

**Check.**

- [x] **The parameter-free claim.** This is the load-bearing assumption of the entire
  project. `state_tensors()` takes no τ/ε/φ argument — confirm that is structurally true
  and that nothing parameter-dependent leaks in. If it were false, the cache would be
  invalid and every Phase 3 number with it.
- [x] **`published` vs `fixed` differ ONLY in `b`** (`CatieState.as_published()` zeroes
  it). Verify `b` never feeds back into the state recursion — that is what makes one
  state pass serve both models.
- [x] The closed form against
  `Data_resources/competition_analysis-main/CATIE/COMPETITION_CATIE_schedule_choice_probability.m`,
  and `p_exp` against `getExploreProb.m`.
- [x] **Fiddly details that must be preserved** (each one is a plausible silent bug):
  - `_base2dec` weights the **oldest** window element by base⁰ (`base2dec.m:5-9`).
  - CAB-k gate is `(trial-1) > k && trial < nTrials` — the last trial always falls back
    to grand means.
  - On the insufficient-history branch, `expected_reward` is **not** updated (retains its
    prior value).
  - Confusion draws are uniform over **distinct encountered rows**, not count-weighted;
    the `meshgrid` marginalisation reproduces this. Do not count-weight it.
  - Sample SD uses (n−1) with an `isnan` guard; surprise uses SD updated *through* trial
    t but `expected_reward` computed at the *start* of t.
- [x] **`mix_agents(weighting="shipped_time_avg")`** reproduces `..._hetro.m:25`, which
  computes `mean(agents' * W, 2)` — collapsing to **time-averaged** posterior weights
  applied uniformly to every trial. **The MATLAB file's own comment describes per-trial
  weighting, which is not what the code does.** Verify against the MATLAB, not the comment.
  ⚠ **The parenthetical that used to sit here was wrong, and instructively so.** It read:
  *"the 'intended' per-trial variant was measured as slightly worse: E[log p] −0.6992 →
  −0.7042. Footnote, not a finding."* Scoring lower on our data is not evidence about
  which variant produced the **paper's** numbers — those are two different questions, and
  the reasoning silently substituted one for the other. Checked properly in 2026-09
  (against the paper's per-schedule Fig S5 values rather than against our own preference),
  **per-trial is the paper's weighting**: it reproduces every schedule to ≤5e-4 while the
  time-average leaves a same-signed +0.005 residual. It was the finding, not the footnote.
  The project default is now `per_trial`; see `validate_against_paper.py` section 2.
- [x] Trial 1 is fixed at p = 0.5 by the reference implementation. Check every metric
  either drops it or includes it **consistently**.

### `sanitize_splits.py` (~140 lines) — rewritten 2026-08-22

**Does.** The competition's organized per-subject CSVs → one tidy CSV per split.

**Method.** Reads `my_code/{Training_set, Test_set, EDA_set, Schedule0_set}/schedule_N/`
(copies of `Data_resources/.../simple_format_data/`), drops the curator-tagged
`..._INVALID_BIAS.csv` files, renames the 4 source columns to this project's canonical
names, derives `observed_reward`, and emits `subject_id = "<schedule>/<file>"`.

**Check.**

- [x] **`subject_id`, never `subject_file`.** *(Historical: with the old timestamp
  filenames, `1609192765_75.csv` existed in both `schedule_6` and `schedule_11`.)* The
  organized filenames are globally unique, so the hazard is gone by construction — but the
  uniqueness assertion is kept, since it now costs nothing and guards the invariant.
- [x] Per-schedule counts reproduce the paper's reported *n*. **[VERIFIED]** exact on
  **all 12 schedules**, total 3,332 — matching the paper's own stated total. Previously
  verified on the 5 training schedules only.
- [x] **Where does the EDA split come from?** **[RESOLVED]** `data/cleaned_eda.csv`, produced
  by this script from `my_code/EDA_set/` like every other split. The old special case —
  reading `EDA_set/processing/eda_with_catie_probabilities.csv`, an Excel-damaged file that
  bypassed this script entirely — is gone, along with its `[OPEN]` caveat.
- [x] Every subject has exactly 100 trials (asserted here *and* in `build_cache.load_split`).
- [x] **The `MIN_CHOICES_PER_SIDE = 5` rule is now an assertion, not a filter.** The
  curators' `_INVALID_BIAS` tag is authoritative and proved to be a superset: zero subjects
  trip the min-5 rule on any split. If it ever fires, the two criteria have diverged — a
  finding, not noise.
- [x] **`observed_reward` is derived, not read** (`where(chose_biased, biased_reward,
  unbiased_reward)`), because the organized release does not store it. Verified exact
  against all 148,300 old training rows before the migration. Phase 2's
  `recent_reward_rate` is the only consumer.

### `metrics.py` (221 lines)

**Does.** Scoring and inference helpers.

**Method.** `e_p`, `e_log_p`, `brier`, `accuracy`, `reliability_table` (equal-width and
equal-mass), `ece`, `mce`, `score_all`, `bootstrap_ci`, `paired_subject_test`, `Tee`.

**Check.**

- [x] Each metric matches its textbook definition. ECE = Σ (nᵢ/n)·|acc(i) − conf(i)|;
  MCE = max over bins (note `min_count=30` — check how sparse bins are handled).
- [x] **`bootstrap_ci` and `paired_subject_test` resample SUBJECTS**, whole, with
  replacement — never trials (cross-cutting check X2). Verify `_subject_means` collapses
  to one value per subject *before* resampling.
- [x] `n_boot=10_000`; CI construction is percentile — confirm, and note it is not BCa.
- [x] `score_all` takes both `p_alt1` and `p_choice` — confirm each metric receives the
  right one (cross-cutting check X1).

### `golden_test.py` (290 lines) — the blocking correctness gate

**Does.** Seven checks that must pass before any analysis is trusted.

**Method.** The important one is `_reference_catie_probability()`: an **independent,
monolithic, single-pass reimplementation** that never separates state from parameters, and
that re-derives the bug from literal `NaN` propagation rather than by construction.

**Check.**

- [x] Current results: published port vs LIVE, unmodified MATLAB (all 12 schedules)
  **2.331e-15**; monolithic reference vs that same live MATLAB **6.661e-16**;
  monolithic reference vs split implementation **3.331e-16**; `mode_contributions`
  sums to p(alt 1) **3.331e-16**; `mode_weights` sums to 1 **2.220e-16**; published
  heuristic contribution **exactly 0.0**; fixed heuristic contribution **2.9e-01**;
  trend-testable trials **16.89%** (496-subject EDA, post-migration).
- [x] **Why the monolithic reference matters.** It is the only check that could catch a
  pair of cancelling bugs inside the state/parameter split. Confirm it is genuinely
  independent — that it does not import the production functions it is meant to check.
- [x] "published heuristic contribution is exactly 0.0" is the bug, expressed as a test.
  Confirm the assertion is `== 0`, not a tolerance.
- [x] Are all seven checks **blocking** (assert / non-zero exit), or merely printed?

### `build_cache.py` (117 lines)

**Does.** Precomputes the parameter-free state tensors → `cache/state_{split}.npz`.

**Method.** For k ∈ {0,1,2,3}, stores `H, b, c_prev, s_prev, sbar_prev, g` each
(n_subjects × 100) float64, plus `y` (int8), `subject_id`, `schedule`. Built in
`mode="fixed"`; published mode is obtained downstream by zeroing `b`
(`p_choice_matrix(..., published_b=True)`).

**Check.**

- [x] **Cache integrity.** Recomputing p from the cache at published parameters must
  reproduce the direct `catie_core` port. **[UNVERIFIED by you]** — an audit reported
  0.000e+00 across four parameter settings and all four k values, but re-run it yourself;
  it is cheap and it underwrites all of Phase 3.
- [x] **Test exclusion.** `SPLITS = ("training", "eda", "schedule_0")` and `load_split`
  asserts no `schedule_{1,8,10}` is present. Confirm the assertion is reachable and would
  fire. Note the guarantee is *absence from the cache*, not an assertion inside Phase 3.
- [x] `K_VALUES = (0,1,2,3)` but Phase 3's default mixture is `ks=(0,1,2)`. Confirm k=3 is
  only used where intended.

### `matlab/` — independent cross-validation

**Does.** A second, fully independent implementation: minimally-patched copies of the real
MATLAB, run in **real MATLAB**, not Python.

**Files.** `CATIE_FIXED/*.m` (`_FIXED`, `_hetro_FIXED`, `_INSTRUMENTED`),
`run_bug_comparison_all_schedules.m` (275 lines), `export_state_tensors.m` (181),
`verify_state_tensors.py`, and `results/*.csv`.

**Check.**

- [ ] **The element-by-element state check is the most valuable artifact in the repo.**
  `export_state_tensors.m` dumps MATLAB's *internal loop variables* and
  `verify_state_tensors.py` compares them to `state_tensors()` across all 998,400 trials.
  Reported: exact (0.0) on boolean/integer tensors, 5.55e-16 on floats. Layers that only
  compare final probabilities cannot catch cancelling bugs inside the recursion; this one
  can.
- [ ] `run_bug_comparison_all_schedules.m` includes `paired_ttest_manual` implemented via
  `betainc`/`betaincinv` because the Statistics Toolbox is unavailable. It was validated
  against a known case (t=2, df=20 → p≈0.0591). **Re-check this**: a hand-rolled
  t-distribution CDF is exactly the kind of thing that is subtly wrong.
- [ ] Note the history: this driver once **double-applied** the choice-probability
  conversion, producing a 0.9 sanity-check failure. It was traced by diffing all 492 raw
  files (zero mismatches) and fixed, giving 5.55e-16. Confirm the conversion happens
  exactly once.

---

## Phase 1 — Bug reproduction and correction

`01_bug_correction/` — `bug_benchmark.py` (299 lines), `README.md` (94), `figures/`.

**Does.** Quantifies the published likelihood bug on all four splits.

**The bug.** In `COMPETITION_CATIE_schedule_choice_probability.m:106-114`, the trend
branch reads `pays(trial)` — still `NaN` from the preallocation until it is assigned at
line 147. In MATLAB `NaN > x` and `NaN < x` are **both false**, so the branch never fires
and `b ≡ 0`. But `p_try_explore = 1-pHeuristic` still executes, so τ = 29% of the
probability mass is withheld from the ordinary computation and never delivered to
alternative 1 — effectively handed to alternative 2 — on every trend-testable trial
(**16.89%** of trials), regardless of what the trend indicated.

Three index errors sit on those two lines: `pays(trial)` should be `pays(trial-1)`; the
`pays(trial-1)` it compares against should be `pays(trial-2)`; and `~is_choice_1(trial)`
should be `~is_choice_1(trial-1)` (it reads the current trial's choice — the very thing
being predicted).

**Check.**

- [x] **The intended semantics.** The generative simulator
  `Data_resources/Choice engineering - Models/static_models/CATIE_schedule_1.m:152-157`
  implements the same rule with correct indices — deciding `decs_b(trial+1)` from
  `pays(trial)` vs `pays(trial-1)`. Shift the frame by one and you get exactly the fix.
  **Verify this yourself**; it is the sole evidence that this is a bug rather than a
  design choice, and the whole chapter rests on it.
- [x] **Scope.** The bug affects only the trial-level likelihood path (Tables S1/S2,
  Fig S4/S5). The competition result used the *simulator*, which is correct, so the
  paper's headline finding is untouched. Confirm this scoping — it is what makes the
  chapter a reproduction rather than a critique.
- [x] Headline numbers in `figures/headline_metrics.csv`:

  | split | E[p] pub | E[p] fix | E[log p] pub | E[log p] fix |
  |---|---|---|---|---|
  | eda | 0.60110 | 0.61432 | −0.71410 | −0.69950 |
  | training | 0.60555 | 0.62165 | −0.68417 | −0.66878 |
  | test | 0.61594 | 0.61905 | −0.69000 | −0.67886 |
  | schedule_0 | 0.67618 | 0.67842 | −0.58256 | −0.57985 |

- [x] **⚠ METHODOLOGICAL ITEM: Phase 1 computed metrics on the TEST split.** The plan
  says Test is touched exactly once, at the end. It was planned ("quantify the bug on all
  four data sets") and no selection was performed on it, so this is arguably benign — but
  **it is a decision you should make consciously and record**, not discover later. Decide
  now whether Phase 1's test row stays in the thesis.
- [x] **[VERIFIED]** MATLAB agreement: 6.66e-16 across 48 numbers, all 3,332 subjects
  (re-measured 2026-08-23 -- the "3,328"/"7.77e-16" this line previously cited predated
  the schedule_7 data migration; the 48-number count itself was already correct).
- [x] Pooled paired test (subject-level, `metrics.paired_subject_test`): E[log p]
  t(3331)=27.64, p=1.456e-151, dz=0.479; E[p] t(3331)=39.02, p=1.138e-274, dz=0.676.
  Confirmed subject-level by reading the function: it collapses to per-subject means
  before `ttest_rel`, and bootstraps by resampling subject indices, not trial indices
  (cross-cutting check X2). *(The previously-cited "t=12.83, p=1.07e-32 on EDA" is not
  reproduced by any current script -- `bug_benchmark.py` only reports the pooled test,
  never an EDA-only one -- and is almost certainly a pre-implementation planning
  estimate; replaced with the real, current, reproducible figure above.)*

---

## Phase 2 — Conditional calibration (the core chapter)

`02_mode_calibration/` — `conditional_calibration.py` (654 lines), `responsibility.py`
(159), `verify_cprev_separation.py` (263), `README.md` (303), `figures/`.

**Does.** Adjudicates between competing partitions of the calibration gap, with `c_prev`
as the baseline every richer partition must beat.

**Data.** EDA + Training + schedule_0 = 2,524 subjects, 252,400 trials → 249,876 after
dropping trial 1. Test excluded.

**Method.** For a partition `L`, `R²(L) = 1 − E[Var(gap|L)] / Var(gap)` where
`gap = y − p_alt1`. Plus a variance decomposition
`Var(gap|L) = Var(y|L) + Var(p|L) − 2Cov(y,p|L)`, a noise-ceiling estimate, and reliability
curves.

**The central arithmetic.**

| c_prev | n | predicted | empirical | gap |
|---|---|---|---|---|
| 0 | 91,548 | 0.249 | 0.454 | **+0.205** |
| 1 | 158,328 | 0.848 | 0.735 | **−0.114** |
| aggregate | 249,876 | — | — | **+0.003** |

Near-total cancellation in the mean. Present under the published model too
(+0.213 / −0.087, aggregate +0.023), so it does not depend on the Phase 1 fix.

**Headline partition.** `c_prev × streak_bin` R² = 0.180 (vs `c_prev` alone 0.104), and
the gap **reverses sign inside both strata**: c_prev=0 runs +0.347 → −0.072;
c_prev=1 runs −0.258 → +0.019.

**Check.**

- [ ] **X1 applies hardest here.** Every calibration quantity must use `p_alt1` against a
  binary `y`, while E[p]/E[log p] use `p_choice`.
- [ ] **X7: the R² table compares partitions of 2 to 21 groups with no complexity penalty
  and no cross-validation.** Does the +0.076 increment of `c_prev × streak_bin` (10 groups)
  over `c_prev` (2 groups) survive an out-of-fold computation? This is the most important
  statistical question in the chapter and, as far as I know, it has not been answered.
- [ ] **Is the R² statistic even the right adjudicator?** The decomposition shows a
  partition scores by **either** predicting `y` **or** homogenising `p`. `c_prev` removes
  **89%** of the variance in CATIE's own forecast (0.0934 → 0.0101), so "`c_prev` explains
  the gap" is substantially a statement about CATIE's architecture (φ=0.71 dominates `p`),
  not purely about human behaviour. The README acknowledges this — judge whether the
  acknowledgement is adequate or whether the chapter needs a different statistic.
  The out-of-fold q̂ row (best `y`-predictor, R² = 0.024) is the cleanest illustration.
- [ ] **X3: circularity.** `soft_argmax` (the responsibility posterior in
  `responsibility.py`) is an E-step built from the trial's own observed choice, and gave a
  spurious **R² = 0.967**. It was removed. **Verify nothing circular survives** — in the
  scored partitions, the figures, or the CSVs. Confirm the posterior is used only as
  *retrospective* error attribution, never as a forecasting partition.
- [ ] **X4: `streak` must be prospective** — run length of identical choices ending at
  t−1, using nothing from trial t, resetting at subject boundaries. **[OPEN]** an audit
  reported an off-by-one at sequence start: `c_prev[:,0] = 0` is a placeholder for "no
  previous choice", so the ~50% of subjects who opened on alternative 2 have their first
  run inflated by 1 (0.79% of cells; reported effect on E[log p] ≈ +0.0002). Confirm and
  decide whether to fix.
- [ ] Were the `streak_bin` edges {1, 2, 3, 4-5, 6+} chosen before or after seeing the
  result?
- [ ] **X2:** are the sign-reversing end cells (n = 15,977 and 53,003) distinguishable
  from zero under a **subject-clustered** interval?
- [ ] **The mechanistic claim** — "CATIE's constant φ over-predicts perseveration after
  short runs and under-predicts after long ones" — has a serious alternative explanation:
  subjects who produce long runs are self-selected perseverators, so a **between-subject
  composition effect** could produce this with no within-subject mechanism.
  **[UNVERIFIED]** an audit put the composition share at ~59% using within-subject mean
  centring. That figure is a **ratio of two swings, not a variance decomposition**, and the
  centring attenuates the genuine within-subject effect because run length is unbalanced
  within subject. **The direction is plausible; the magnitude is not established.** A
  subject-fixed-effects or mixed model would settle it. Phases 3 and 4 build on this, so
  it deserves your own scrutiny.
- [ ] **Noise ceiling ≈ 0.26.** [UNVERIFIED] The chapter's own history here is instructive:
  an audit proposed 0.182, and it was falsified because `c_prev × streak_bin` (0.180)
  nearly exceeded it — *a bound an observed partition surpasses is not a bound*. The
  current estimate uses a CV gradient booster with the bound `ceiling ≥ 1 − OOF-Brier/Var(gap)`.
  Check: are the folds grouped **by subject**? Is the plug-in (0.261) being lower than the
  "rigorous lower bound" (0.268) consistent, or a contradiction?
- [ ] `schedule` R² ≈ 0 is presented as a **positive** result (CATIE tracks a 0.185
  between-schedule spread to within 0.018), not as evidence schedule is irrelevant.
  Verify both numbers.
- [ ] The `p=7.062e-281` and `p=2.307e-122` values for published-vs-fixed: what test, and
  at what level? Any test treating ~250k correlated trials as independent is invalid
  (X2).
- [ ] 13 partitions are ranked and the winner reported — any selection-inference concern?
- [ ] Read each figure's plotting code and confirm it plots what the README says.
- [ ] **History worth knowing.** Two conclusions in the first pass were wrong and were
  corrected: (a) "miscalibration is one-dimensional in `c_prev`" — an artifact of crossing
  `c_prev` with only 2 of 5 candidates; (b) a pre-written conclusion claimed crossings were
  dropped as sparse bins when they were not (n = 1,273 / 254). Check the current text
  contains no residue of either.

---

## Phase 3 — Re-fitting, with recalibration as the control

`03_parameter_fitting/` — `catie_likelihood.py` (197), `fit_parameters.py` (487),
`validate_against_paper.py` (362), `README.md` (251), `figures/`.

**Does.** Asks whether CATIE's E[log p] deficit is a *parameter* problem or a *structure*
problem, using post-hoc recalibration as the control that distinguishes them.

**Method.** Vectorised likelihood over the cache; ML fit of (τ, ε, φ) in logit space;
K by enumeration; Hessian-based identifiability; temperature / Platt / isotonic
recalibration as controls; a run-length-dependent φ extension.

**Headline.** Fitted τ .29→.107, ε .30→.628, φ .71→.314, gaining +0.108 E[log p] on EDA.
But isotonic recalibration — zero psychological content — captures **98%** of that
(+0.106). Conclusion: *the deficit is predominantly a calibration deficit.*

**Check.**

- [ ] **The mixture-weight subtlety.** The BMA weights are built from the cumulative
  product of per-agent choice probabilities, which depend on the parameters — so they
  must be recomputed **inside every likelihood evaluation**. Verify no memoisation on
  the path `negloglik_factory.f → mean_log_p → p_choice_matrix → mix_agents_3d →
  mix_per_trial → _posterior_weights`. Frozen weights would give a subtly wrong optimum.
  *(Renamed 2026-09: the single `mix_published` became `mix_per_trial` /
  `mix_shipped_time_avg` behind the `mix_agents_3d` dispatcher.)*
- [ ] **X5 — the control is where leakage would hide.** Verify temperature, Platt and
  isotonic are each `.fit()` on **Training** and only `.predict()` on EDA. Isotonic is
  non-parametric and overfits spectacularly in-sample. Trace the actual data objects.
  **[UNVERIFIED by you]** an audit reported no leakage and computed the counterfactual
  (isotonic fit-on-EDA = −0.5898, only 0.0034 better).
- [ ] **The 98% figure** is a ratio of two differences, both estimated with error. Is it
  distinguishable from 85%? From 100%? A subject-clustered interval on the ratio would
  settle it.
- [ ] **The interpretive leap.** "Fitted parameters buy calibration, not psychology"
  requires that the re-fit be reachable by a monotone transform of `p_alt1`. **[UNVERIFIED]**
  an audit reports Spearman(published, re-fitted `p_alt1`) = 0.93 with max deviation 0.364
  — i.e. the re-fit **does** reorder trials, so the stated argument does not hold as
  written even if the conclusion does. Check this yourself before relying on either form.
- [ ] **⚠ The SE scaling — verify this yourself; it is subtler than it first looks.**
  `fit_parameters.py:222-223` is `n_eff = tr.n_subjects; cov = inv(Hm) / n_eff`, printed as
  *"approximate subject-clustered SEs (delta method)"*. `Hm` is the Hessian of the
  **per-trial mean** nll, so the iid divisor would be `n_trials` (1,483 × 99 = 146,817).
  Dividing by `n_subjects` instead is a **deliberate** conservative gesture toward
  clustering — but it amounts to assuming ICC = 1 (each subject contributes exactly one
  independent observation), which over-corrects. So this is not a simple error: it is a
  crude bound where a cluster-robust sandwich (bread = `inv(H_sum)`, meat = Σ per-subject
  score outer products) belongs. Reported τ±0.048 / ε±0.055 / φ±0.035.
  **The file is internally inconsistent** (verified): line 366 computes the LR statistic
  as `2 * tr.n_subjects * 99 * Δ` — i.e. the **trial** count — for the same likelihood
  whose SEs use the **subject** count. Both cannot be right. Derive the correct scaling
  and decide; note the SE direction is conservative, so no conclusion flips either way.
- [ ] **X5 — threshold selection.** The run-length threshold ≥4 was chosen by scoring all
  five thresholds **on EDA** and picking the best — selection on the selection set. Note
  the *training* column is minimised at ≥6, not ≥4. Report the train-selected value.
  (Reported optimism is small, ~0.0003, but the principle matters.)
- [ ] **φ_long = 1.000 at threshold ≥6 is a boundary solution.** Verify whether the
  optimiser hit a bound and what that implies for the smooth-φ extension planned in
  Phase 4. **[UNVERIFIED]** an audit attributes it to an ε-imposed ceiling on attainable
  P(repeat); the algebra is unchecked.
- [ ] **"Isotonic is the best possible monotone transform"** — as written this is false,
  because that isotonic was fitted on **Training**, making it the best monotone transform
  *of Training*, not of EDA. The comparison between two transferred models is still fair;
  the justifying sentence is not. Decide the correct phrasing.
- [ ] **⚠ E[p] appears nowhere in the Phase 3 README.** The chapter optimises and reports
  E[log p] only, while CATIE's claim to fame in the paper is E[p]. Every rung degrades it
  relative to the corrected baseline (re-fit −0.013, isotonic −0.018, temperature −0.041 on
  EDA). Given your stated goal — improve E[log p] *while keeping* the strong E[p] — decide
  whether this omission misleads and add E[p] columns.
- [ ] **Finding #7 ("the paper says K=2 but the code mixes") is refuted.**
  `CATIE_single_schedule_score.m:5` is `k = randi([0,2])` — the competition simulator
  itself draws k uniformly from {0,1,2}, and `hetro.m:6` is `K = 0:2`. "K = 2" in the
  Methods denotes the **upper bound of a uniform draw**, which is exactly what both
  implement. **Verify this yourself and remove the discrepancy framing.** The enumeration
  table itself is fine and worth keeping.
- [ ] Per-schedule fits: φ range 0.227–0.364 vs ε range 0.442–0.926, used to argue "φ
  transfers, ε does not". **Raw ranges are not comparable across parameters on different
  scales, and the per-schedule n vary 12-fold (87 to 607).** A proper heterogeneity test
  (Cochran Q, or range scaled by each parameter's own SE) is needed. **[UNVERIFIED]** an
  audit reports Q(φ)=6.3, p=0.175 (n.s.) and Q(ε)=149.6, p=2e-31, which would support the
  claim — but check it.
- [ ] `fig2` plots ladder gains against a red "gap to best QL" line at 0.109, with M3's
  bar crossing it. The README's prose caveat says this is a statement of *scale*, not a
  claim to beat Q-Learning. Judge whether the figure undermines its own caveat.
- [ ] **[VERIFIED] The four-optimiser table in the README has no backing code.**
  `fit_parameters.py` contains `optimize.minimize(..., method="Nelder-Mead")` at line 93
  and two `minimize_scalar` calls (profile likelihood at 247, temperature at 283) — and
  nothing else. No L-BFGS-B, no Powell, no `differential_evolution`, anywhere in the repo.
  The README's "verified against four independent optimisers, including a *global* one"
  claim cannot be regenerated by running the code, and it sits inside the "Correctness
  checks performed" section. Either commit the script that produced it or delete the table.
  (Its *substance* was independently reproduced in an audit, so this is a reproducibility
  defect rather than a false claim — but in a thesis that distinction will not protect you.)

### `validate_against_paper.py` (432 lines) — added 2026-08-21, hardened 2026-09

**Does.** Validates the port against the paper's **per-schedule** reported values
(`Data_resources/extracted_data/{E_p.md, E_log_p.md}`, from Fig S5).
Run: `... validate_against_paper.py [training|eda|schedule_0]`.

**Method.** Parses the reported tables; **verifies rather than assumes** the schedule
mapping (paper labels 1–12, repo labels 0–11) by matching reported *n*; scores the
published port, the corrected port and the re-fitted parameters per schedule.

Section 2 **determines the paper's two undocumented conventions empirically** rather
than assuming them: it scores all four {`per_trial`, `shipped_time_avg`} × {trial 1
kept, dropped} combinations against the reported per-schedule values, picks the winner
by worst-case error, and asserts it before sections 3–4 use it. This is the check that
localised the ~0.005 E[log p] residual to the shipped k-weighting.

The design decision that matters: the paper's numbers came from the original MATLAB,
**which contains the bug**, so the variant that should reproduce them is the *published*
port. Comparing the paper to the corrected model would manufacture a discrepancy that is
really the bug.

**Check.**

- [x] **[RESOLVED 2026-09] Both metrics now reproduce, on all three splits.** Under the
  paper's own conventions (per-trial k-weighting, trial 1 kept — both determined
  empirically in the script's section 2, not assumed): max |ΔE[p]| = 0.0005 / 0.0003 /
  0.0002 and max |ΔE[log p]| = 0.0004 / 0.0003 / 0.0004 on training / EDA / schedule_0.
  All at the 3-decimal rounding floor, with **mixed-sign** residuals (3+/2− and 2+/1−).
- [x] **[RESOLVED] The systematic same-signed E[log p] residual** (+0.0056, +0.0029,
  +0.0065, +0.0064, +0.0040 on training; mean +0.0051) was the shipped `hetro.m:25`
  time-averaged k-weighting, not a porting error. It disappears under the per-trial rule.
  The script now warns loudly if a same-signed pattern ever returns — but only when there
  are ≥3 schedules, since with fewer a uniform sign is not evidence.
- [x] **[RESOLVED] Sample sizes: exact on every split** — 5/5 training, 3/3 EDA, 1/1
  schedule_0. The `schedule_7` 4-subject gap (115 vs 119) is gone; the organized-data-
  release migration fixed it, and the script asserts the counts on every run.
- [x] **[RESOLVED] `MATCH_TOL`** tightened 0.002 → **0.0006**. The old value was 4× the
  0.0005 rounding floor, which is precisely why it certified E[p] as a "match" while a
  real systematic error sat underneath. All three splits pass at the tight tolerance.
- [x] **[RESOLVED] Section 2 no longer ignores its own result.** It now scores all four
  {per_trial, shipped_time_avg} × {keep, drop trial 1} combinations, picks the winner by
  worst-case error, asserts it, and feeds it to sections 3–4. The earlier
  "inconclusive" reading was an artifact of testing trial 1 alone while holding the
  weighting fixed at the wrong value — with the weighting free, the answer is
  unambiguous (worst error 0.0005 vs 0.0073).

---

## Open items and known incidents

Things already known to be wrong or unresolved. Confirm rather than rediscover.

1. **[RESOLVED 2026-09] The systematic E[log p] residual (~+0.005).** Root cause: the
   shipped `hetro.m:25` time-averages the k-mixture (BMA) weights; the paper's numbers
   were produced by the per-trial weighting on its commented-out line 26. Verified per
   schedule against Fig S5: per-trial + heuristic bug + trial 1 kept reproduces all 9
   non-Test schedules to ≤0.0005 (rounding floor, mixed-sign residuals). The project
   default is now `weighting="per_trial"`; `"shipped_time_avg"` is kept only for golden
   tests against live MATLAB. Phases 1–3 rerun accordingly (pooled published now
   0.6192/−0.6776 vs paper 0.619/−0.678).
2. **[RESOLVED 2026-09] The 4-subject gap** in `schedule_7` (115 local vs 119 reported)
   is closed: the split now holds 119, matching the paper exactly, as do all other
   schedules on all three cached splits. `validate_against_paper.py` section 1 asserts
   this on every run, so a regression cannot pass silently.
3. **⚠ The 0.6158 mislabelling — a live example of why documentation is untrustworthy.**
   `scratchpad/pareto.py` labels a row `"M0 published"` where `PUB` means published
   **parameters** applied to the **corrected** model. That row (EDA E[p] = 0.6158) was
   quoted as if it were the published *model*, and propagated into an audit summary. The
   correct EDA baselines are:

   | EDA baseline | E[p] | E[log p] |
   |---|---|---|
   | paper reported (published model) | 0.6010 | −0.7185 |
   | our published port | 0.6023 | −0.7143 |
   | our corrected port | 0.6158 | −0.6992 |
   | re-fitted (held out) | 0.6027 | −0.5911 |

   *(Historical table — computed under the shipped time-averaged weighting. Under the
   per-trial default adopted 2026-09: published port 0.6020/−0.7194, corrected
   0.6153/−0.7046; see the Phase 1/3 READMEs for current numbers.)*

   Consequence: "the re-fit loses E[p]" is true against the **corrected** baseline; against
   the paper's own reported 0.6010 it is roughly flat. **Decide which baseline the thesis
   quotes, and state it every time.**
4. **[OPEN] Phase 1 computed metrics on the Test split.** Planned, no selection performed,
   but decide consciously whether it stays.
5. **[UNVERIFIED] All Phase-3 audit mathematics** — the isotonic/monotone-transform
   argument, the ε/6 boundary algebra, the oracle-isotonic comparison, the φ_long ceiling
   analysis, and the ~59% composition figure. Produced by one adversarial pass, not
   independently checked. Do not build on them until reviewed.
6. **Permanently closed directions — do not reopen.** Reaction time (artifact of z-scoring
   against a 1.5 s hardware floor; at X=10% the contrast is n.s., d=0.02, p≈0.82).
   Current-trial-choice features (circular; `P(surprise | no switch) = 0.000` exactly, and
   `BLUE_RIGHT_LEFT_RED` is constant within subject so `side_choice ≡ is_biased_choice`).
   Any feature built from an E-step or the current trial's outcome.

---

## Verification recipes

```bash
cd "c:/Users/shirh/OneDrive - huji.ac.il/Amirim Research"
PY=./.venv/Scripts/python.exe

# Phase 0 — blocking gate. Run this before trusting anything else.
$PY my_code/catie_calibration/golden_test.py

# Cache integrity: p from cache must equal the direct port.
# (worth writing yourself rather than trusting a prior audit)

# Phase 1
$PY my_code/catie_calibration/01_bug_correction/bug_benchmark.py

# Phase 2  (slow: responsibility posteriors over 2,524 subjects)
$PY my_code/catie_calibration/02_mode_calibration/conditional_calibration.py
$PY my_code/catie_calibration/02_mode_calibration/verify_cprev_separation.py

# Phase 3  (~15-20 min; the 784-point profile surface dominates and prints nothing)
$PY my_code/catie_calibration/03_parameter_fitting/fit_parameters.py

# Paper validation (fast)
$PY my_code/catie_calibration/03_parameter_fitting/validate_against_paper.py training
$PY my_code/catie_calibration/03_parameter_fitting/validate_against_paper.py eda
```

Every script writes a `figures/output.txt` transcript via `metrics.Tee`. Diff the
transcript against the README's quoted numbers — divergence between them is itself a
finding.

**Never open the generated CSVs in Excel.** That is how `cleaned_eda_data.csv` lost its
`time` column.

---

## Sign-off checklist

Phase 0 must pass before the rest is worth reviewing.

- [ ] **Phase 0** — golden test passes; the parameter-free claim holds; cache reproduces
  the direct port; MATLAB state tensors match element by element; splits are clean and
  subject IDs unique.
- [ ] **Phase 1** — the bug is real (simulator confirms intended semantics); scope is
  correctly limited to the trial-level path; the Test-split decision is made.
- [ ] **Phase 2** — no circular partition survives; `p_alt1` used throughout for
  calibration; the R² increment survives out-of-fold; the run-length mechanism is
  distinguished from composition; inference is subject-clustered.
- [ ] **Phase 3** — no leakage in the control; mixture weights recomputed; SEs correctly
  scaled; threshold selected on Training; the K claim removed; E[p] reported.
- [ ] **Cross-cutting** — one baseline convention chosen and stated everywhere; Test
  touched only as decided; every README number traced to a committed script.
