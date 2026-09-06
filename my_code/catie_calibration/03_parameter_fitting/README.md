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

**Protocol.** Fit on Training (1,483 subj, schedules 2/3/6/9/11), score once on EDA
(496, schedules 4/5/7) with no selection on EDA, schedule_0 (549) as
out-of-distribution check. The only discrete choice made anywhere in Phase 3 -- the
run-length threshold of §6 -- is made on Training. All numbers use the
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
| EDA (held out) | −0.7046 | −0.5975 | **+0.107** |
| schedule_0 (OOD) | −0.5818 | −0.5076 | +0.074 |

**Verified against four independent optimisers**, including a *global* one with no
start point (transcript §2b; regenerated on every run into `optimiser_agreement.csv`):

| optimiser | τ | ε | φ | nll |
|---|---|---|---|---|
| Nelder-Mead (8 restarts) | 0.1150 | 0.7024 | 0.3387 | 0.586735 |
| L-BFGS-B | 0.1150 | 0.7025 | 0.3387 | 0.586735 |
| Powell | 0.1150 | 0.7021 | 0.3390 | 0.586735 |
| **differential evolution** (global; logit bounds ±5, seed 42, no start point) | **0.1150** | **0.7024** | **0.3387** | **0.586735** |
| coarse grid (15³ on the probability scale, no refinement) | 0.1143 | 0.6929 | 0.3714 | 0.587003 |

The three gradient/global solvers agree with Nelder-Mead to 3×10⁻⁸ in nll; the grid
is only a sanity check that no other basin exists.

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

**What those SEs are conditional on.** They cluster by *subject*, so they answer
“how precisely is this parameter pinned down for another **participant** drawn from
these five Training schedules?” They do not answer “… for another **schedule**” —
there the five schedules are themselves the sampling unit, and §5 shows the
parameters drift a long way between them. The same sandwich clustered by schedule
(G = 5) gives τ ±0.036, ε ±0.112, φ ±0.010; every run prints it as the
`sched-clust` column beside the subject-clustered one. **τ and ε are 5–7× less
certain for a new schedule; φ is not (1.03×).** A DerSimonian–Laird random-effects
summary of the five per-schedule fits says the same thing: τ 0.140 ± 0.032,
ε 0.814 ± 0.105, φ 0.328 ± 0.013 (computed from the committed
`per_schedule_fits.csv`; its ε line uses probability-scale inverse-variance
weights, which §5 explains are unreliable for ε, so ±0.105 is the softest of the
three). Five clusters is few, so read the schedule-clustered figures as magnitudes
rather than intervals — but the ordering is unambiguous and it is the reason τ and
ε must not be carried to a new schedule.

### 3. THE CONTROL — re-fitting is *calibration*, not better psychology

This is the chapter's central result.

| model (fit on Training, scored on EDA) | #par | E[log p] | ECE | gain | E[p] |
|---|---|---|---|---|---|
| published CATIE | 0 | −0.7046 | 0.174 | — | 0.6153 |
| + temperature (T=2.03) | 1 | −0.6235 | 0.084 | +0.081 (76%) | 0.5735 |
| + Platt | 2 | −0.6193 | 0.071 | +0.085 | 0.5778 |
| **+ isotonic (non-parametric)** | np | **−0.5988** | 0.022 | **+0.106 (98.8%)** | 0.5937 |
| re-fitted (τ, ε, φ) | 3 | −0.5975 | 0.029 | +0.107 | 0.5953 |

All three maps are fitted on Training with trial 1 (p = 0.5 by construction, never
scored) excluded; they are applied to every trial and scored with trial 1 dropped.
Every recalibrated row *loses* E[p] relative to the published model (0.6153). The
algebra usually quoted for this is exact only for a **linear** shrinkage of p: for
p → 0.5 + α(p − 0.5), ΔE[p] = (α − 1)(E[p] − 0.5), negative whenever α < 1 and
E[p] > 0.5. None of the rows above is a linear shrinkage — temperature is linear in
the *logit*, isotonic is an arbitrary monotone map, and the re-fit is a different
model rather than a transform of this one — so the losses are **consistent with**
that mechanism rather than derived from it. They are measured, not predicted.

**The isotonic row is the decisive comparison, not temperature.** Isotonic is the
best monotone recalibration of `p_alt1` *that Training can fit* — it has zero
psychological content and can use nothing beyond the ordering of CATIE's own
forecast. Empirically it captures **98.8%** of the 3-parameter re-fit's gain:
isotonic/re-fit gain ratio 0.988, subject-cluster bootstrap 95% CI **[0.968, 1.009]**
(2,000 resamples of the 496 EDA subjects), and the re-fit is not reliably better
than isotonic on EDA (paired subject-level test, M2 − isotonic = +0.0013
[−0.0009, +0.0036], p = 0.28; `isotonic_vs_refit.csv`).

> **The fitted parameters are buying calibration, not revised psychology. They should
> not be reported as new estimates of exploration or inertia tendencies.**

### 4. The mechanism: ε *is* CATIE's built-in temperature knob

Why does re-fitting reduce to recalibration? Because ε controls the probability
floor. Since `p_exp = ε(1 + s + s̄)/3` and the exploration mode contributes
`p_exp/2` toward 0.5, ε sets the attainable range at roughly [ε/6, 1−ε/6]
(`p_alt1` on EDA, trial 1 dropped; transcript §4b, `forecast_distribution.csv`):

| | min | median | max | SD | % extreme (>0.9 or <0.1) |
|---|---|---|---|---|---|
| published (ε=0.30) | 0.057 | 0.699 | 0.955 | 0.314 | **37.6%** |
| fitted (ε=0.702) | 0.131 | 0.635 | 0.883 | 0.192 | **0.0%** |

Fitting shrank the spread by 39% and **eliminated every extreme prediction**.
Maximum likelihood turns ε into a pure shrinkage parameter. *(This confirms a
hypothesis raised during project planning — that ε conflates exploration rate with
the probability floor — which is now demonstrated rather than speculated.)*

### 5. Per-schedule fits corroborate this directly

Each Training schedule is fitted alone (`per_schedule_fits.csv`, with
subject-clustered sandwich SEs per schedule), and the between-schedule spread is
tested against those SEs with a Cochran Q heterogeneity test (inverse-variance
weights, df = 4; `per_schedule_heterogeneity.csv`). **Q is computed on the logit
scale**, for the reason boxed below:

| | range across 5 schedules | Q | p | I² |
|---|---|---|---|---|
| φ (inertia) | **0.260 – 0.387** | 4.70 | **0.32** | 15% |
| τ (trend) | 0.080 – 0.208 | 76.65 | 9e−16 | 95% |
| ε (exploration) | **0.496 – 0.975** | 79.51 | 2e−16 | 95% |

> **Why the logit scale, and what it fixes.** The model is parameterised in logit
> space and the sandwich covariance is computed there; the probability-scale SE is a
> delta-method image, se_p = se_z · p(1−p). That linearisation is valid only where
> the likelihood is locally quadratic, and for ε on **schedules 9 and 11** it is not:
> ε sits on a flat plateau at ≈ 0.975 with a **logit-scale SE of ≈ 3**, i.e. it is
> essentially unidentified there. The delta method turns that into se_p ≈ 0.072 — a
> deceptively *small* number — which on the probability scale gives those two cells
> an inverse-variance weight of ≈ 190 and lets two unidentified estimates dominate
> the test. That is what produced the **Q_ε = 156.15** previously reported here
> (against 79.5 on the valid scale) and `fig3` error bars reaching **1.116 and
> 1.120**, impossible for a probability. On the logit scale a flat parameter
> correctly receives almost no weight. Every run now prints a **“PARAMETER NOT
> IDENTIFIED”** warning for any per-schedule parameter with se_z > 1 and records it
> in `per_schedule_fits.csv` (`se_z_*`, `unidentified`, `eps_unidentified`);
> `per_schedule_heterogeneity.csv` carries `scale = logit` and reports
> `weighted_mean` back-transformed through the sigmoid. `fig3`'s intervals are now
> built in logit space and mapped back, so they lie inside [0, 1] by construction.

**φ is the only parameter that is homogeneous across schedules**: its spread is what
the within-schedule SEs predict (p = 0.32). ε swings by a factor of two, and τ is
*as heterogeneous as ε* once each is set against its own SEs — on the valid scale
they are **roughly equally so** (Q = 76.65 vs 79.51, both p < 10⁻¹⁵), where
the old probability-scale test made ε look twice as heterogeneous as τ purely
through the two plateau cells. ε's drift is consistent with it absorbing schedule-specific
*miscalibration* rather than measuring a schedule-specific exploration rate
(finding #4, `fig3`); τ's drift shows the trend weight does not transfer either,
so only φ should be read as a population estimate.

### 6. Run-length inertia: real structure, tested against non-parametric recalibration

Phase 2 found the calibration gap *reverses sign* with perseveration run length,
which no constant φ (and no asymmetric φ, one per side of `c_prev`) can represent.
Fitting φ_short / φ_long at a threshold (τ, ε free). **The threshold is selected on
Training** (highest train E[log p]); EDA is then used once, to score the selected
model. The table is regenerated by every run (`runlength_inertia.csv`). The values
below are from the full 5-restart rerun with the corrected streak (2026-09-06):

| threshold | φ_short | φ_long | train | EDA | EDA E[p] |
|---|---|---|---|---|---|
| ≥2 | 0.193 | 0.530 | −0.5801 | −0.5916 | 0.6017 |
| ≥3 (EDA-selected, for reference only) | 0.235 | 0.685 | −0.5786 | −0.5913 | 0.6020 |
| **≥4 (selected on Training)** | **0.259** | **0.831** | **−0.5781** | **−0.5915** | **0.6014** |
| ≥5 | 0.277 | 0.937 | −0.5782 | −0.5917 | 0.6005 |
| ≥6 | 0.289 | 1.000 | −0.5782 | −0.5917 | 0.5998 |

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

**Does it beat isotonic?** Isotonic is a monotone transform of `p_alt1` (the best
one Training can fit), so it *cannot* represent a run-length-dependent effect: a
reliable gain over it is information that no recalibration of this forecast can
reach. The gain is small (the two rows differ in the 3rd decimal), so it is tested
with a **paired subject-level test on EDA** (per-subject mean log p; paired t and a
subject-cluster bootstrap 95% CI, `metrics.paired_subject_test`), against both the
isotonic-recalibrated published model and the 3-parameter re-fit (M2). Every run
also breaks the same test down **by EDA schedule**, because three schedules pooled
can hide a sign flip (`runlength_tests.csv`, `schedule` column):

| EDA schedule | n | M3 − isotonic | p | M3 − M2 | p |
|---|---|---|---|---|---|
| **pooled (EDA)** | 496 | **+0.0073 [+0.0039, +0.0107]** | 4.5e−5 | **+0.0060 [+0.0038, +0.0084]** | 5.3e−7 |
| schedule_4 | 201 | +0.0074 [+0.0026, +0.0126] | 0.0044 | +0.0063 [+0.0030, +0.0098] | 0.00032 |
| schedule_5 | 176 | −0.0001 [−0.0050, +0.0052] | 0.96 | +0.0014 [−0.0011, +0.0041] | 0.29 |
| schedule_7 | 119 | +0.0180 [+0.0098, +0.0268] | 7.1e−5 | +0.0124 [+0.0062, +0.0195] | 0.00039 |

**What the breakdown licenses — and what it does not.** The pooled effect is real
and the CI against isotonic excludes zero, but it is **carried by schedules 4 and 7
and absent on schedule 5** (−0.0001, p = 0.96 — a flat null, not a small positive),
and schedule 7 alone is 2.5× the pooled effect. So the defensible claim is *not*
that recalibration fundamentally cannot capture run-length inertia in general; it is
that **on two of these three EDA schedules** the run-length model carries
information the best Training-fitted monotone map of `p_alt1` does not, and on the
third it carries none. Per-schedule n is 119–201, so schedule 5's null is weak
evidence of absence rather than evidence of a true zero — but it is enough that the
effect must be reported as schedule-dependent rather than as a property of the
model class.

**How “held out” M3 really is.** Its parameters and its threshold were fitted and
selected on Training alone, so the EDA numbers are held out *in that sense*. The
**form** of the extension, though — a run-length-dependent φ with a threshold
somewhere in 2–6 — was suggested by Phase 2, whose analysis pooled EDA + Training +
schedule_0 (`02_mode_calibration/conditional_calibration.py:95-98`). The model
*class* has therefore seen EDA even though none of M3's numbers were fitted on it.
**Test (schedules 1/8/10) is the first genuinely clean evaluation of M3**, and of
whether the schedule-dependence above is noise or real. This caveat applies wherever
the M3 rung is called held out, including the ladder below.

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

| model | #par | E[log p] EDA | vs M0 | E[p] EDA | vs M0 |
|---|---|---|---|---|---|
| M0 published params | 0 | −0.7046 | — | 0.6153 | — |
| M1 + temperature (control) | 1 | −0.6235 | +0.081 | 0.5735 | −0.042 |
| M2 re-fitted (τ, ε, φ) | 3 | −0.5975 | +0.107 | 0.5953 | −0.020 |
| **M3 + run-length φ (threshold selected on Training; ≥4 in the current transcript)** | 5 (4 continuous + 1 selected threshold) | **−0.5915** | **+0.113** | 0.6014 | −0.014 |

Every rung loses E[p] relative to M0 (0.6153 on EDA). The algebra usually quoted
for this is exact only for a **linear** shrinkage: for p → 0.5 + α(p − 0.5),
ΔE[p] = (α − 1)(E[p] − 0.5) < 0 whenever α < 1 and E[p] > 0.5. No rung here is one
— M1 is linear in the *logit*, and M2/M3 are different models rather than transforms
of M0 — so the identity is the mechanism the pattern is **consistent with**, not a
derivation of it. Every rung does empirically shrink CATIE's over-confident
forecasts toward 0.5, and every rung's E[p] loss above is measured.

**On M3's “held out”.** Its 4 continuous parameters and its threshold come from
Training only, but the *form* of the extension came from Phase 2, which pooled EDA
with Training and schedule_0 (§6). Test is the first clean evaluation of this rung.

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
  deficit, not a mechanism deficit.* 98.8% (95% CI 96.8–100.9%) of what parameter
  re-fitting achieves is reachable by a content-free monotone recalibration. The one
  candidate structural defect found is the **constant inertia assumption**: making φ
  run-length-dependent beats the best Training-fitted monotone recalibration on EDA
  overall, though §6's breakdown shows that gain is carried by schedules 4 and 7 and
  is absent on schedule 5. Test is the first clean check of whether it generalises.

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
  agree on the optimum to 4 dp (§1; `optimiser_agreement.csv`, regenerated by every
  run, as is the §4 forecast-distribution table, `forecast_distribution.csv`).
- **Identifiability:** Hessian eigenvalues, condition number, subject-clustered
  sandwich SEs (with iid and **schedule**-clustered SEs alongside, delta method to
  the probability scale), sandwich correlation matrix, and a 28×28 profile surface.
- **The sandwich SEs are cross-checked against a subject bootstrap of the MLE.**
  `fit_parameters.py --bootstrap-se N` (off by default) resamples the 1,483 Training
  subjects with replacement N times and refits (τ, ε, φ) on each, L-BFGS-B
  warm-started at the full-sample optimum. At **N = 120** (120/120 converged, ~5 min)
  the bootstrap SDs are **0.0065 / 0.0144 / 0.0095** for τ / ε / φ, against sandwich
  SEs of 0.0067 / 0.0159 / 0.0101 — ratios 0.98 / 0.90 / 0.94, so the sandwich is if
  anything mildly conservative and its asymptotics hold at this sample size
  (`bootstrap_se.csv`, which carries both the probability and the logit scale). An
  earlier version of this README claimed a 200-resample cross-check for which no
  code existed; this is the real one, and the CSV is committed so it is checkable.
- **Held-out validation:** every headline gain is reported on EDA, which is not used
  for fitting. All recalibration maps are fitted on Training with trial 1 excluded.
  The run-length threshold is selected on Training; EDA is used once,
  to score the selected model. The run-length-vs-isotonic comparison is a paired
  subject-level test on EDA (`runlength_tests.csv`), reported pooled **and per EDA
  schedule**, not a single-number difference. One caveat is recorded in §6 rather
  than claimed away: the *form* of M3 was suggested by a Phase 2 analysis that
  pooled EDA, so Test is the first clean evaluation of that model class.
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
| `fig3_per_schedule_fits` | Per-schedule τ/ε/φ with 95% CIs built in logit space from the clustered sandwich SEs and mapped back through the sigmoid (so every bar lies inside [0, 1]; the bars are asymmetric because the sigmoid is) — φ's spread fits inside its error bars, τ's and ε's do not |
| `../tests/test_metrics.py` | Unit tests for `metrics.py`: fixed bootstrap bin edges in `ece_ci` / `reliability_table_ci`, and `cluster_bootstrap`'s two reductions to `bootstrap_ci` |
| `k_enumeration.csv`, `optimiser_agreement.csv`, `recalibration_control.csv`, `isotonic_vs_refit.csv`, `forecast_distribution.csv`, `runlength_inertia.csv`, `runlength_tests.csv`, `per_schedule_fits.csv`, `per_schedule_heterogeneity.csv`, `model_ladder.csv`, `bootstrap_se.csv`, `profile_surface.npz` | Backing tables (every table in this README is regenerated by `fit_parameters.py`) |

## How to run

```bash
python my_code/catie_calibration/build_cache.py                       # once, ~1 min
python my_code/catie_calibration/03_parameter_fitting/fit_parameters.py
python my_code/catie_calibration/03_parameter_fitting/fit_parameters.py --bootstrap-se 120   # + §3b
python my_code/catie_calibration/03_parameter_fitting/validate_against_paper.py [training|eda|schedule_0]
python my_code/catie_calibration/tests/test_metrics.py                # unit tests, ~10 s
```

The profile surface (784 nested optimisations) dominates runtime, followed by the
run-length and per-schedule fits and the §2b global search — expect ~40 min on an
idle machine (the 2026-09-06 run took 1h50 while sharing the CPU with other work).
The surface prints nothing while computing; that is expected, not a hang.

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
