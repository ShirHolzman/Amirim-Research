# Phase 3 — Re-fitting CATIE's parameters, with recalibration as the control

## Research question

CATIE loses to Q-Learning on E[log p] (−0.678 vs −0.569, Supplementary Table S2).
**Is that because its parameter *values* are wrong for this population, or because
its *structure* is wrong?** These demand different responses, and the distinction is
only visible with the right control.

## Why this tests something the paper itself foregrounds

**Methods:** *"the model is characterized by four free parameters (one for each mode:
τ, ε, φ, K). The parameters of the model were estimated from behavior[10], and these
estimations were used in all simulations: τ = 0.29, ε = 0.30, φ = 0.71, K = 2."*

**Discussion:** *"Particularly remarkable is the fact that the parameters of this
model were **not fitted to this particular competition**. Rather, they were taken
from previous experiments."*

The paper presents the non-fitting as evidence of CATIE's robustness. Re-fitting
therefore tests a highlighted claim, and either outcome is informative: a large gain
means CATIE competed handicapped; a small gain means the parameters genuinely
transfer, which *strengthens* the paper's point.

**Protocol.** Fit on Training (1,483 subj, schedules 2/3/6/9/11), select on EDA (496,
schedules 4/5/7), schedule_0 (549) as out-of-distribution check. All numbers use the
**per-trial** k-mixture weighting (the paper's; project default since 2026-09 -- see
the Phase 1 README's resolved-discrepancy note). The rerun moved every value at the
2nd–3rd decimal; no conclusion changed. **Test (1/8/10) is
not merely unused — it is not present in the cache**, so accidental use is
structurally impossible.

---

## Key findings

### 1. The fit moves the parameters enormously — and the optimum is real

| parameter | published | fitted | change |
|---|---|---|---|
| τ (trend) | 0.29 | **0.115** | −0.175 |
| ε (exploration) | 0.30 | **0.702** | +0.402 |
| φ (inertia) | 0.71 | **0.339** | −0.371 |

| split | published | fitted | gain |
|---|---|---|---|
| training | −0.6731 | −0.5867 | +0.086 |
| EDA (select) | −0.7046 | −0.5975 | **+0.107** |
| schedule_0 (OOD) | −0.5818 | −0.5076 | +0.074 |

**Verified against four independent optimisers**, including a *global* one with no
start point:

| optimiser | τ | ε | φ | nll |
|---|---|---|---|---|
| Nelder-Mead (8 restarts) | 0.1150 | 0.7024 | 0.3387 | 0.586735 |
| L-BFGS-B | 0.1150 | 0.7025 | 0.3387 | 0.586735 |
| Powell | 0.1150 | 0.7021 | 0.3390 | 0.586735 |
| **differential evolution** (global) | **0.1150** | **0.7024** | **0.3387** | **0.586735** |
| coarse grid (15³) | 0.1029 | 0.6821 | 0.3179 | 0.586939 |

### 2. No ε/φ ridge — contrary to the project plan's prediction

The plan anticipated an identifiability ridge (both parameters push mass toward
"repeat"). There isn't one. Hessian condition number **8.3**; the profile surface
(`fig1`) is cleanly unimodal. Subject-clustered sandwich SEs (1,483 subjects,
G/(G−1) applied): τ 0.115±0.0067, ε 0.702±0.0159, φ 0.339±0.0101 (iid SEs, which
ignore within-subject correlation: ±0.0048 / ±0.0055 / ±0.0037). Sandwich
correlations: corr(ε,φ) = **−0.42**, corr(τ,ε) = +0.43, corr(τ,φ) = −0.21 — moderate,
and well below the |corr(ε,φ)| > 0.7 that would indicate a ridge. (An earlier version
of this section divided the inverse Hessian of the *mean* per-trial nll by n_subjects
instead of N_trials, inflating the SEs ~√99 to ±0.048/0.054/0.037, and quoted the
Hessian-implied correlations of < 0.11; the clustered sandwich replaces both.)

### 3. THE CONTROL — re-fitting is *calibration*, not better psychology

This is the chapter's central result.

| model (fit on Training, scored on EDA) | #par | E[log p] | ECE | gain |
|---|---|---|---|---|
| published CATIE | 0 | −0.7046 | 0.174 | — |
| + temperature (T=2.03) | 1 | −0.6235 | 0.084 | +0.081 (76%) |
| + Platt | 2 | −0.6193 | 0.071 | +0.085 |
| **+ isotonic (non-parametric)** | np | **−0.5988** | 0.023 | **+0.106 (99%)** |
| re-fitted (τ, ε, φ) | 3 | −0.5975 | 0.029 | +0.107 |

**The isotonic row is the decisive comparison, not temperature.** Isotonic is the
best *possible* monotone recalibration of `p_alt1` — it has zero psychological
content and can use nothing beyond the ordering of CATIE's own forecast. It captures
**99%** of the 3-parameter re-fit's gain.

> **The fitted parameters are buying calibration, not revised psychology. They should
> not be reported as new estimates of exploration or inertia tendencies.**

### 4. The mechanism: ε *is* CATIE's built-in temperature knob

Why does re-fitting reduce to recalibration? Because ε controls the probability
floor. Since `p_exp = ε(1 + s + s̄)/3` and the exploration mode contributes
`p_exp/2` toward 0.5, ε sets the attainable range at roughly [ε/6, 1−ε/6]:

| | min | median | max | SD | % extreme (>0.9 or <0.1) |
|---|---|---|---|---|---|
| published (ε=0.30) | 0.057 | 0.699 | 0.955 | 0.314 | **37.6%** |
| fitted (ε=0.702) | 0.131 | 0.635 | 0.883 | 0.192 | **0.0%** |

Fitting shrank the spread by 39% and **eliminated every extreme prediction**.
Maximum likelihood turns ε into a pure shrinkage parameter. *(This confirms a
hypothesis raised during project planning — that ε conflates exploration rate with
the probability floor — which is now demonstrated rather than speculated.)*

### 5. Per-schedule fits corroborate this directly

| | range across 5 schedules |
|---|---|
| φ (inertia) | **0.260 – 0.387** — tight, stable |
| τ (trend) | 0.080 – 0.208 |
| ε (exploration) | **0.496 – 0.975** — enormous |

**φ, the genuinely psychological parameter, transfers across schedules; ε does not.**
ε swings by a factor of two because it is absorbing schedule-specific
*miscalibration*, not measuring a schedule-specific exploration rate (`fig3`).

### 6. Run-length inertia: real structure, tested against non-parametric recalibration

Phase 2 found the calibration gap *reverses sign* with perseveration run length,
which no constant φ (and no asymmetric φ, one per side of `c_prev`) can represent.
Fitting φ_short / φ_long at a threshold (τ, ε free). **The threshold is selected on
Training** (highest train E[log p]); EDA is then used once, to score the selected
model. The table is regenerated by every run (`runlength_inertia.csv`). The values
below are from the full 5-restart rerun with the corrected streak (2026-09-06):

| threshold | φ_short | φ_long | train | EDA |
|---|---|---|---|---|
| ≥2 | 0.193 | 0.530 | −0.5801 | −0.5916 |
| ≥3 (EDA-selected, for reference only) | 0.235 | 0.685 | −0.5786 | −0.5913 |
| **≥4 (selected on Training)** | **0.259** | **0.831** | **−0.5781** | **−0.5915** |
| ≥5 | 0.277 | 0.937 | −0.5782 | −0.5917 |
| ≥6 | 0.289 | 1.000 | −0.5782 | −0.5917 |

Three notes on the selection. First, the Training objective is flat across ≥4–≥6
(they differ in the 4th–5th decimal), so a rerun could select any of them; the
transcript prints which, and the prose here holds for each. Second, the threshold
that scores best *on EDA* is printed for reference only — it is *not* a held-out
number — and it scores within ~0.0002 of the Training-selected one (the transcript
prints this optimism explicitly). An earlier version of this section selected the
threshold on EDA and reported that EDA score as held out; that has been corrected.
Third, at ≥6 the fit is a **boundary solution: φ_long = 1.000**, the edge of the
(0,1) logit parameterisation — the model says repetition is deterministic after six
identical choices, softened only by the ε exploration floor. The transcript flags
this whenever the selected threshold's φ_long exceeds 0.999. It is a sign the
threshold form is too crude at long runs, not a psychological estimate of exactly 1.

φ_long exceeds φ_short at every threshold, and the gap widens — **inertia
strengthens sharply with run length** (0.259 → 0.831 at ≥4, saturating at ≥6).
Phase 2 predicted this from calibration analysis; Phase 3 confirms it by
independent likelihood fitting.

**Does it beat isotonic?** Isotonic is the best possible monotone transform of
`p_alt1`, so it *cannot* represent a run-length-dependent effect; any reliable gain
over it is structural information that recalibration cannot capture. The gain is
small (the two rows differ in the 3rd decimal), so it is tested with a **paired
subject-level test on EDA** (per-subject mean log p; paired t and a subject-cluster
bootstrap 95% CI, `metrics.paired_subject_test`), against both the
isotonic-recalibrated published model and the 3-parameter re-fit (M2). Results are
in `runlength_tests.csv` and the transcript; in the current transcript (Training-selected
≥4, n = 496 EDA subjects): vs isotonic +0.0073 [+0.0039, +0.0108], p = 4e−5; vs M2
+0.0060 [+0.0038, +0.0084], p = 5e−7. The structural claim rests on the CI against
isotonic excluding zero, not on the point difference.

### 7. K sensitivity: single-k agents vs the {0,1,2} mixture

| K | E[log p] train | E[log p] EDA |
|---|---|---|
| K=0 | −0.7103 | −0.7432 |
| K=1 | −0.6888 | −0.7233 |
| K=2 | −0.6904 | −0.7192 |
| K=3 | −0.6847 | −0.7143 |
| **mix {0,1,2} (the paper's model)** | **−0.6731** | **−0.7046** |
| mix {0,1,2,3} | −0.6692 | −0.7006 |

The paper's "K = 2" is the upper bound of a uniform draw, not a single fixed value:
the competition simulator draws `k = randi([0,2])` on every run
(`CATIE_single_schedule_score.m:5`), and the likelihood code evaluates the matching
`K = 0:2` sub-agents (`..._hetro.m:6`). The {0,1,2} mixture is therefore the paper's
model, and the published E[log p] = −0.678 is the mixture's value. The single-k rows
are a sensitivity check on how much each k-agent contributes: every single k is worse
than the mixture (K=2 alone is −0.7192 vs −0.7046 on EDA), and larger k helps
monotonically. Extending the draw to {0,1,2,3} gives a small further improvement
(−0.7006 vs −0.7046 on EDA; −0.6692 vs −0.6731 on training).

### 8. The fit repairs Phase 2's `c_prev` split

| model | gap c_prev=0 | gap c_prev=1 | aggregate |
|---|---|---|---|
| published | +0.2132 | −0.1412 | −0.0008 |
| re-fitted | **+0.0048** | **−0.0271** | −0.0145 |
| + run-length φ (Training-selected threshold) | +0.0009 | −0.0253 | −0.0149 |

Both strata shrink substantially — the fit is not merely rebalancing one against the
other. But per finding #3, most of this repair is calibration.

---

## The model ladder

| model | #par | E[log p] EDA | vs M0 |
|---|---|---|---|
| M0 published params | 0 | −0.7046 | — |
| M1 + temperature (control) | 1 | −0.6235 | +0.081 |
| M2 re-fitted (τ, ε, φ) | 3 | −0.5975 | +0.107 |
| **M3 + run-length φ (threshold selected on Training; ≥4 in the current transcript)** | 5 (4 continuous + 1 selected threshold) | **−0.5915** | **+0.113** |

**Scale caveat.** The paper's CATIE-vs-QL gap of 0.109 is pooled over all 12
schedules; the numbers above are EDA only (schedules 4/5/7). "+0.113 ≈ 104% of the
gap" is a **statement of scale, not a like-for-like claim that this model beats
Q-Learning.** A proper comparison needs QL scored on the same subjects, which this
repo does not compute.

## Interpretation — what this means for the paper's claim

The paper's "remarkable" framing survives, but in a more precise form:

- **What transfers:** the *ordering* of CATIE's probabilities, and φ, the inertia
  parameter (stable at ~0.26–0.39 across schedules). Since the competition result
  depends on simulated bias, not on log-likelihood calibration, **the paper's
  headline finding is unaffected.**
- **What does not transfer:** the *confidence scale*. Parameters imported from a
  different task and population left CATIE systematically over-confident (roughly a third of
  predictions beyond [0.1, 0.9], ECE 0.174). In the model-comparison tables — and
  only there — CATIE was competing handicapped.
- **The honest headline:** *CATIE's E[log p] deficit is predominantly a calibration
  deficit, not a mechanism deficit.* Roughly 99% of what parameter re-fitting
  achieves is reachable by a content-free monotone recalibration. The one genuinely
  structural defect found is the **constant inertia assumption**, which run-length
  dependence corrects beyond anything recalibration can reach.

## Correctness checks performed

- **Golden check (blocking):** the vectorised likelihood reproduces Phase 1's
  E[p] and E[log p] to 4 dp on all three splits, at published parameters. Asserted
  before any fitted number is computed.
- **Reproduces the paper per schedule** (`validate_against_paper.py`): under the
  paper's own conventions the published port matches every reported Fig S5 value to
  ≤ 0.0005 in E[p] and ≤ 0.0004 in E[log p] — the 3-decimal rounding floor — on all
  9 non-Test schedules, with mixed-sign residuals. Subject counts match exactly
  (5/5, 3/3, 1/1). The tolerance is 0.0006, just above the rounding floor.
- **Four optimisers**, including global differential evolution with no start point,
  agree on the optimum to 4 dp (§1).
- **Identifiability:** Hessian eigenvalues, condition number, subject-clustered
  sandwich SEs (iid SEs alongside, delta method to the probability scale), sandwich
  correlation matrix, and a 28×28 profile surface. The sandwich SEs were cross-checked
  against a 200-resample subject bootstrap of the MLE.
- **Held-out validation:** every headline gain is reported on EDA, which is not used
  for fitting. The run-length threshold is selected on Training; EDA is used once,
  to score the selected model. The run-length-vs-isotonic comparison is a paired
  subject-level test on EDA (`runlength_tests.csv`), not a single-number difference.
- **The k-mixture weights are recomputed inside every likelihood evaluation**, since
  they depend on the parameters through the per-agent probabilities; treating them as
  fixed would give a subtly wrong optimum.

## Files

| File | Role |
|---|---|
| `../build_cache.py` | Precomputes parameter-free state tensors → `../cache/` (~1 min) |
| `catie_likelihood.py` | Vectorised likelihood over the cache; logit-space optimiser helpers |
| `fit_parameters.py` | The analysis: K enumeration, fitting, identifiability, control, run-length, per-schedule, ladder |
| `validate_against_paper.py` | Per-schedule reproduction of the paper's Fig S5 values; determines the paper's k-weighting and trial-1 conventions empirically |
| `figures/output.txt` | Full transcript |
| `figures/validation_output{,_eda,_schedule_0}.txt`, `paper_validation_*.csv`, `fig_validation_{Ep,Elogp}*` | Validation transcripts, tables and figures |
| `fig1_profile_likelihood` | (ε, φ) profile surface, published vs fitted |
| `fig2_model_ladder` | Gains of each ladder rung vs the paper's QL gap |
| `fig3_per_schedule_fits` | φ stable, ε unstable — the calibration-knob signature |
| `k_enumeration.csv`, `recalibration_control.csv`, `runlength_inertia.csv`, `runlength_tests.csv`, `per_schedule_fits.csv`, `model_ladder.csv`, `profile_surface.npz` | Backing tables |

## How to run

```bash
python my_code/catie_calibration/build_cache.py                       # once, ~1 min
python my_code/catie_calibration/03_parameter_fitting/fit_parameters.py
python my_code/catie_calibration/03_parameter_fitting/validate_against_paper.py [training|eda|schedule_0]
```

The profile surface (784 nested optimisations) dominates runtime — expect ~15–20 min
total. It prints nothing while computing; that is expected, not a hang.

## Implications for Phase 4

1. **Run-length-dependent inertia is confirmed as the extension to build**, by two
   independent routes (Phase 2 calibration analysis, Phase 3 likelihood fitting).
   The threshold form used here is deliberately crude — Phase 4 should try a smooth
   parameterisation (e.g. φ(run) = φ∞ − (φ∞ − φ₁)·exp(−run/λ)), which may do better
   with the same parameter count.
2. **Do not present re-fitted τ, ε, φ as psychological findings.** Report them, but
   frame per finding #3: they are absorbing miscalibration. Only φ, which is stable
   across schedules, has a defensible psychological reading.
3. **A lapse parameter is now redundant.** The planned M4 (explicit lapse λ) would
   duplicate what ε already does — finding #4 shows ε *is* the lapse parameter.
   Better: *decouple* them — fix ε at a psychologically motivated value and add λ
   separately, which would let ε mean "exploration" again.
4. **Consider widening the k draw to {0,1,2,3}** as a candidate: the K sensitivity check
   (finding #7) shows a small, consistent gain over the paper's {0,1,2}.
