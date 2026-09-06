# Final analysis week — plan (2026-09-06 → 2026-09-12), revised after review

Goal: close the empirical part of the project in one week with results that answer the
supervisor's questions from the last meeting directly, then move to writing. Everything
reuses the existing pipeline (`cache/`, `catie_likelihood.py`, `metrics.py`). Each day is
one script, one figure set, one short results note.

Revision note (2026-09-05): an independent methods review found that two of the planned
diagrams were the same diagram, that the symmetric-map construction was wrong, that the
Phase 3 standard-error bug runs in the opposite direction from what was stated, and that
the meeting framing was tactically backwards. All of that is folded in below. Numbers
quoted as "verified" were computed by the reviewer on the current cache.

## 0. What the supervisor reacted to, and how each point maps to a task

The figure he saw (`initial_investigation/calibration_analysis/fig1_reliability_diagram.png`)
is the **old** one: EDA only, *published* (buggy) likelihood, shipped time-averaged
k-weighting, 492 subjects from the Excel-damaged file, 10 uniform bins on P(biased). New
figures come from the `catie_calibration` pipeline (corrected likelihood, per-trial
weighting, Training+EDA). The "published" overlay will not reproduce the old figure
exactly; say so once and move on.

| His remark | Technical meaning | Task |
|---|---|---|
| "More buckets?" | Bin-count sensitivity of the curve and of ECE; more bins expose the 0.35–0.65 region, which is heuristic-only (H=1) and sparse | Day 1 A |
| "Not symmetric, that is a problem" | P(biased) is not a label-equivariant coordinate: the biased option is chosen 62 % of the time, so a label-equivariant model's curve on that axis is expected to sit off the diagonal in a base-rate-dependent way. The asymmetry pools calibration error with the experimenter's knowledge of which option was engineered | Day 1 B, explanation paragraph |
| "p(action) and p(not action)"; "double the points with p and 1−p, forces symmetry" | Reliability diagram over the doubled set (p, y) ∪ (1−p, 1−y); point-symmetric about (0.5, 0.5) by construction | Day 1 B |
| "Manually calibrate CATIE from the graph" | Histogram-binning / isotonic recalibration, fitted on Training and scored on EDA. This is the **first positive result of the week**, not a foil: Phase 3's control already showed it recovers most of the E[log p] deficit held-out. At 147k training trials a 10–100-bin map barely overfits, so the overfitting worry is measured rather than assumed. The genuine limits are that every calibration map lowers E[p], and that a map is a post-processor rather than a mechanism | Day 2 |
| "Project should be about calibrating and improving CATIE" | Ladder: published → his symmetric calibration → run-length inertia, all placed on one E[p]–E[log p] Pareto figure, then Test once | Days 3–5 |

## 1. Facts that shape the design (verified)

- **`sign(p − 0.5)` and `c_prev` agree on 99.5 % of EDA trials** (Phase 2 §6; disagreement
  only on H=1 crossings, 0.52 %). Hence the folded confidence coordinate max(p, 1−p) and
  the "P(repeat previous choice)" coordinate are the same diagram, and a symmetric map on p
  and a map on P(repeat) are the same map. Plan one diagram and one map, not two.
- **A c_prev-stratified symmetric map recovers what the unconstrained P(biased) isotonic
  recovers**, for the same reason. So the bias label is worth ≈0; that is a cleaner
  statement than any admissibility argument.
- **Phase 3's reported SEs are 3.4–7× too large, not too small.** `fit_parameters.py:222-224`
  divides the inverse Hessian of the *mean* per-trial NLL by n_subjects. Subject-clustered
  sandwich SEs at the Phase 3 optimum: τ 0.0067, ε 0.0159, φ 0.0101 (Phase 3 README says
  0.048 / 0.054 / 0.037). Fix and update the Phase 3 README.
- **`build_streak` (`fit_parameters.py:114-122`) has an off-by-one**: at t=1 it compares
  c_prev[:,1] against the c_prev[:,0]=0 placeholder, so subjects who opened on alternative 2
  start trial 2 with run length 2. Fix before any run-length work; rerun Phase 3's
  run-length table (15 min).
- **M3b (ε pinned at 0.30 + threshold φ) on EDA: E[p] 0.6329, E[log p] −0.6212**, versus
  M0 0.6153 / −0.7046 and M3 (ε free) 0.6013 / −0.5916. M3b improves both metrics, but its
  E[log p] is only temperature-level. M4 (ε pinned + free lapse λ=0.264) lands at
  0.6018 / −0.5901 ≈ M3: as soon as any shrinkage knob is free the fit re-buys calibration
  and E[p] falls. Conclusion: run-length φ adds discrimination at *any* calibration level;
  the E[p]/E[log p] trade is set by the shrinkage knob alone. That is a frontier, not a
  winner.
- **Phase 3's isotonic fit included trial 1** (p fixed at 0.5, 1,483 points). Every map
  fitted this week excludes column 0.
- **Test is not untouched**: `01_bug_correction/figures/headline_metrics.csv` has a Test
  row (E[log p] corrected −0.6830). Say "never used for fitting or selection". Use that row
  as the golden check when the Test cache is built.
- Likelihood evaluation costs ~0.1 s on Training; a 4-parameter fit with restarts ~2.5 min.
  Compute is not the constraint; plotting and CI code is.

## 2. Day-by-day

### Day 0 (Sat morning, ~3 h) — Phase 3 repairs, from the 2026-09-06 code audit

The audit found no blocking error: every headline number is right or moves at the 4th
decimal. Seven things must change before Phase 4 imports Phase 3 code or quotes its
tables. Fix in `fit_parameters.py`, rerun once end-to-end (~35 min), regenerate the CSVs
and README.

1. `build_streak` (lines 119-121): start the loop at t=2. Currently 745 Training subjects
   who opened on alternative 2 get run length 2 at trial 2; 0.79 % of cells wrong. Refit
   moves EDA E[log p] at threshold ≥4 from −0.59160 to −0.59150. Phase 2's streak is
   correct, so Phases 2 and 3 currently disagree on those cells.
2. SEs (lines 228-235): `inv(Hm)/n_subjects` is √99 × the iid SE. Replace with a
   subject-clustered sandwich (bread = inv(N_trials·Hm), meat = Σ per-subject score outer
   products). Correct values τ 0.0067, ε 0.0159, φ 0.0101 (README says 0.048 / 0.054 /
   0.037). Sandwich correlations are corr(ε,φ) = −0.42, corr(τ,ε) = +0.43, so README §2's
   "all correlations < 0.11" is wrong; "no ridge" (|corr| < 0.7) survives.
3. Run-length threshold (lines 365-366) is selected on EDA. Training selects ≥6, EDA ≥4;
   optimism 0.0002. Report ≥6 as headline, ≥4 as footnote, count the threshold as a
   parameter, and state that at ≥6 φ_long = 1.000 is a boundary solution (deterministic
   repetition after 6 identical choices, softened only by the ε floor). Replace the
   "beats isotonic" line with `metrics.paired_subject_test`: +0.0072, 95 % CI
   [0.0039, 0.0106], p = 5e-5.
4. LR statistic (lines 373-376): 2·n_subj·99·Δ = 2501 treats trials as independent and
   ignores threshold selection. Replace with the clustered paired t (M3 vs M2 on
   Training: t = 11.3, p = 1e-28).
5. README §7 K framing: "K = 2 (paper's stated value) is not the best one" misreads the
   paper. `CATIE_single_schedule_score.m:5` draws k uniformly from {0,1,2}; the mixture
   *is* the paper's model. Keep the table as a K-sensitivity analysis, drop the
   discrepancy framing (also in `fit_parameters.py:22-23,177` and
   `catie_likelihood.py:148-150`). This correction was already recorded in the 2026-08-21
   audit and never made it into the README.
6. Drop column 0 (trial 1, p fixed at 0.5) from the isotonic and Platt fits (lines
   298-303); temperature already drops it. Effect 6e-5, but one section must be
   consistent with itself.
7. Add E[p] columns to every Phase 3 table (M0 0.6153, M2 0.5953, M3 0.6013 on EDA).

Smaller items to do in the same pass: commit code for the four-optimiser table and the
§4 min/median/max table (both true, neither regenerable today; DE ≈ 60 s); add Cochran Q
for the per-schedule fits (φ: Q = 5.0, p = 0.28; ε: p = 7e-33; τ: Q = 81.4, p = 9e-17, so
τ is as heterogeneous as ε and §5 must say so); rewrite "isotonic is the best possible
monotone recalibration" as "iso/refit gain ratio 0.988, cluster-bootstrap 95 % CI
[0.966, 1.009]; M2 vs isotonic n.s., p = 0.26"; fix fig3's string-sorted schedule axis;
remove the QL line from fig2 or move the caveat into the figure; fix the 492/496
docstring and the Phase 2 attribution at line 382.

What can be reused unchanged: M2's fitted parameters (nll 0.586735, four optimisers
agree), the recalibration control (no leakage, all maps fitted on Training), the K
table, the per-schedule fits, the profile surface, `validate_against_paper.py` (it does
use the empirically chosen conventions; the earlier "hardcoded" suspicion is refuted).

Also in Day 0: the shared cluster-bootstrap helper for binned quantities (per-bin
empirical rate, ECE, E[p], E[log p]); `metrics.bootstrap_ci` only handles per-subject
means and the old figure's binomial SE ignored clustering. Every diagram and table this
week uses it.

### Day 1 (Sat afternoon + Sun morning, ~5 h) — Reliability diagrams, the supervisor's versions

Script: `04_calibration/reliability_redux.py`. Training + EDA caches, trial 1 dropped.
Corrected model by default; published model as one overlay.

- **A. Bin count.** P(biased) with 10 and 20 uniform bins plus 20 quantile bins, cluster
  CIs, per-bin n, points coloured by the H=1 fraction so the non-monotone 0.35–0.65 stretch
  is visibly the heuristic-only region. Table: ECE vs bin count, Training and EDA.
- **B. Doubled / action-based.** Every trial contributes (p_alt1, y) and (1−p_alt1, 1−y).
  Symmetric by construction. Also drawn with the two c_prev strata overlaid: the doubled
  plot *pools* the strata, it does not erase them.
- **C. Folded confidence = P(repeat).** x = max(p, 1−p) ∈ [0.5, 1], y = predicted
  alternative chosen (≡ repeated, 99.5 %). Overlay by run-length bin {1, 2, 3, 4–5, 6+}.
  This is the supervisor-facing form of the sign-reversal finding and the coordinate the
  Day 2 map lives on.
- **D. One paragraph**: why A is asymmetric (base rate ≠ 0.5, no label-asymmetry in the
  model); why B is symmetric and pools c_prev; why C is the informative symmetric
  coordinate.

Output: `fig1_bins.png`, `fig2_doubled.png`, `fig3_folded_by_runlength.png`,
`ece_vs_bins.csv`, `reliability_tables.csv`.

### Day 2 (Sun afternoon + Mon, ~5 h) — "Calibrate from the graph", done properly

Script: `04_calibration/recalibration_maps.py`. Every map fitted on **Training** (trial 1
excluded), scored on **EDA**; rows report E[p], E[log p], ECE, and a paired subject-level
test against M0.

| map | how it is fitted | reads |
|---|---|---|
| R0 corrected CATIE, no map | — | baseline |
| **R1 symmetric "from the graph", 10 bins** | histogram binning on the doubled set; the literal version of his request | headline calibration row |
| R1-iso symmetric isotonic | isotonic on the doubled set (p,y)∪(1−p,1−y); reflection-invariant, so f(1−p)=1−f(p) and f(0.5)=0.5 hold automatically. Do **not** fit on the folded diagram and unfold: that can dip below 0.5 near the centre and produce a non-monotone f | best label-equivariant map |
| R1-strat symmetric isotonic stratified by c_prev | same, one map per stratum | ≈ R4 by the 99.5 % fact; shows the label is worth ≈0 |
| R1-run stratified by c_prev × run-length bin | non-parametric twin of run-length φ | sets up Day 3 |
| R4 unconstrained isotonic on P(biased) | Phase 3 control | reference only |

Bin sweep 10/20/50/100 for R1: Training vs EDA E[log p]. Expected: the gap barely opens
with bins at this N; whatever gap there is comes from schedule shift. Report that plainly.

E[p] column everywhere, with the algebra: for p → 0.5 + α(p−0.5), ΔE[p] = (α−1)(E[p]−0.5)
< 0 whenever E[p] > 0.5. Every calibration row loses E[p]; the competition margin is 0.001.

### Day 3 (Mon, ~6 h) — Run-length inertia and the Pareto frontier

Script: `04_calibration/fit_extension.py` (reuses `fit_parameters.fit`, `build_streak`,
`p_alt1_single_k`).

| model | free | note |
|---|---|---|
| M0 corrected CATIE, published params | 0 | |
| M2 re-fit (τ, ε, φ) | 3 | Phase 3 |
| M3 threshold φ(run) | 4 | threshold chosen on Training (≥6; ≥4 in a footnote) |
| M3b threshold φ(run), ε pinned at 0.30 | 3 | the high-E[p] end of the frontier |
| M4 ε pinned + lapse λ | 4 | ≈ M3 reparametrised; one row, no "decoupling" story |
| M3 with K mixture {0,1,2,3} | 4 | 5-min sensitivity row |
| **ε sweep**: ε ∈ {0.30, 0.40, …, 0.70} pinned, (τ, φ, φ_long) refit | 3 each | ~10 min compute; draws the frontier as a curve |

Deliverables:
- **Pareto figure, E[p] vs E[log p], one point per model and per Day 2 map.** This is the
  thesis in one picture: calibration maps and re-fits lie on a curve; run-length φ shifts
  the whole curve up and right. Both numbers already exist for every row.
- Ladder table on EDA with schedule_0 as OOD column; subject-clustered sandwich SEs
  (fix `fit_parameters.py:222-224` here, update the Phase 3 README).
- Day 1 diagram C redrawn for M0 vs M3 vs R1-iso: the run-length fan under M0 should
  collapse under M3.

Smooth φ(run) is dropped: threshold results span 0.0002 across thresholds ≥3, so a fifth
parameter cannot buy anything measurable. If revived, reparametrise ρ = exp(−1/λ) ∈ (0,1)
so the logit machinery in `negloglik_factory` works unchanged.

### Day 4 (Tue, ~2 h) — Per-schedule robustness, then buffer

Ladder gains on each EDA schedule (4, 5, 7) and schedule_0: scoring only, one table, sign
consistency is the point. The per-subject-φ heterogeneity check is cut: per-subject φ from
49 trials is too noisy for a held-out comparison to answer the question, and the "59 %
between-subject" figure is a ratio of swings, not an established magnitude
(`REVIEW_PLAN.md:358-362`). Mention it in the discussion as a limitation, not as a result.
Rest of the day is buffer for anything that spilled from Days 1–3.

### Day 5 (Wed) — Phase 4 README and mentor summary

- `04_calibration/README.md` in the phase format (question, findings, correctness checks,
  files, how to run).
- `mentor_meeting_summary_phase0-3.md` → phase 0–4, opening with the Day 1 diagrams, then
  the calibration row, then the Pareto figure.
- Freeze the ladder: M0, R1, R1-iso, M2, M3, M3b. No further choices after this point.

### Day 6 (Thu morning) — Test, once

- Add an `--include-test` flag to `build_cache.py` (it currently excludes and asserts Test
  absent at lines 49 and 58); do not edit the assert away. Build the Test cache.
- Golden check: M0 corrected on Test must reproduce the Phase 1 headline row (E[log p]
  −0.6830) before anything else is scored.
- Score the frozen ladder once. One table: E[p], E[log p], ECE with cluster CIs. Add the
  Test points to the Pareto figure as hollow markers.

Test runs *after* the buffer so nothing gets re-touched afterwards.

### Days 6 (afternoon)–7 — Report skeleton

Chapter outline mapping phases to sections: background → reproduction and the two
likelihood bugs (supporting chapter) → where CATIE is miscalibrated (Phase 2 + Day 1
diagrams) → calibrating CATIE (Phase 3 + Day 2) → run-length inertia and the frontier
(Day 3) → Test → discussion. Figure list with final filenames and one caption each.

## 3. De-scoping order

Drop in this order if behind: ε sweep → K{0,1,2,3} row → R1-run → per-schedule table →
20-quantile bins. Never drop: Day 0 fixes, Day 1 A/B/C, Day 2 R1 and R1-iso, Day 3 M3 and
the Pareto figure, Day 6 Test table.

## 4. Meeting framing

Lead with his idea as the first positive result, not with the extension.

1. The symmetric diagrams (his request) show where the miscalibration is: confidence is
   too high after short runs and too low after long ones.
2. Calibrating CATIE from that graph, fitted on Training and scored on EDA, recovers most
   of the E[log p] deficit. Overfitting was measured, not feared: the 10-bin map and the
   100-bin map score the same held-out.
3. Every calibration map lowers E[p], by the algebra above. That is a property of
   calibration, not a flaw in his idea.
4. Run-length inertia is the one change that moves the whole E[p]–E[log p] frontier,
   because it adds discrimination rather than adjusting confidence. Shown on one Pareto
   figure with his calibration on it.
5. Re-fitted τ, ε, φ are not psychological estimates (Phase 3 §3); φ is the only one that
   transfers across schedules.

Vocabulary: say "label-equivariant", never "not admissible" or "information the model does
not have" (CATIE has the labels; it lacks label-asymmetry). Say the doubled plot "pools"
strata, not "averages them out". Do not lead with M3b.
