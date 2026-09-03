# Phase 2 — Conditional calibration: which partition explains the miscalibration?

## Research question

CATIE's *aggregate* calibration gap is small, yet it loses badly to Q-Learning on
E[log p]. Hypothesis: the aggregate conceals large, oppositely-signed *conditional*
miscalibration that cancels in a mean but compounds in a log. This chapter does not
presuppose which partition reveals that structure — it **adjudicates** between
candidates, with the previous choice (`c_prev`) as the baseline every richer
partition must beat, and reports the result honestly either way.

Model: the **corrected** ("fixed") CATIE likelihood, k∈{0,1,2} **per-trial** mixture
(the k-weighting the paper's own numbers match; the shipped time-averaged variant
is kept only for MATLAB golden tests -- see `../catie_core.py:mix_agents`).
§2 re-runs the central split under the *published* model too, so the robustness
claim is shown rather than asserted.

Data: **EDA + Training + schedule_0** (2,528 subjects, 250,272 trials after
dropping trial 1). **Test is excluded and asserted absent** — touched exactly once,
at the end of the project.

> **This chapter was substantially revised after an adversarial audit.** The audit
> confirmed every number reproduced, and found the *conclusions* were wrong in two
> ways: the partition search was incomplete (missing the strongest partition), and
> one section was built on a circular quantity. Both are fixed below; §"What the
> audit changed" records what the earlier version claimed.

> **Recomputed 2026-09 under the per-trial k-mixture weighting**, after the
> discovery that the shipped `hetro.m` time-averages the BMA weights while the
> paper's numbers use the per-trial rule (see the Phase 1 README's resolved-
> discrepancy note). Every number moved at the 2nd–3rd decimal; no conclusion
> changed.

## Partitions considered

Each row groups every trial into buckets by a different rule, then asks (§"Key
findings" #2) how much of the calibration gap that grouping explains. `c_prev` is
the mandatory baseline; every partition below is also tested crossed with it
(`c_prev × X`).

| partition | groups trials by |
|---|---|
| `c_prev` | whether the **previous** trial's choice was the biased option (0) or the unbiased one (1) — the baseline every richer partition must beat |
| `hard_argmax` | which of CATIE's four internal regimes (heuristic / exploration / inertia / contingent-average) contributed the most probability mass to "choose alt 1" this trial |
| `schedule` | which of the 9 reward schedules (sequences of payouts) the subject was run on |
| `trial_phase` | which fifth of the 100-trial session the trial falls in (trials 1–20, 21–40, …, 81–99) — i.e. how far into the session the subject is |
| `streak_bin` | how many trials in a row (ending the trial before this one) the subject made the *same* choice, bucketed into {1, 2, 3, 4–5, 6+} |
| `reward_rate_bin` | the subject's average reward over their last 5 trials (excluding the current one), split into ~4 equal-sized groups |
| `p_alt1_bin` | CATIE's own predicted probability of choosing alt 1 this trial, split into 20 equal-sized groups |

## Key findings

### 1. The central arithmetic

| c_prev | n | predicted | empirical | gap |
|---|---|---|---|---|
| 0 (prior choice unbiased) | 91,752 | 0.246 | 0.454 | **+0.208** |
| 1 (prior choice biased) | 158,520 | 0.848 | 0.734 | **−0.113** |
| **aggregate** | 250,272 | — | — | **+0.004** |

Near-total cancellation in the mean, from two large, oppositely-signed strata.
Present under the published model too (+0.216 / −0.087, aggregate +0.024), so it
does not depend on the Phase 1 bug fix.

### 2. The strongest partition is `c_prev × run length`, and the gap reverses sign

**This supersedes the earlier claim that the miscalibration is "one-dimensional in
`c_prev`", which was an artifact of an incomplete search.**

| partition | R² | vs `c_prev` |
|---|---|---|
| **`c_prev × streak_bin`** | **0.178** | **+0.072** |
| `c_prev × p_alt1_bin` | 0.144 | +0.039 |
| `p_alt1_bin` (20 quantiles) | 0.144 | +0.039 |
| `c_prev × trial_phase` | 0.125 | +0.019 |
| `hard_argmax` | 0.120 | +0.015 |
| `c_prev` (baseline) | 0.106 | — |
| `streak_bin` / `trial_phase` / `reward_rate_bin` / `schedule` alone | ≤0.005 | — |

The substance (`fig3`, `gap_by_cprev_streak.csv`):

```
c_prev=0:  run 1 → +0.348   run 2 → +0.215  ...  run 6+ → −0.060
c_prev=1:  run 1 → −0.256   run 2 → −0.168  ...  run 6+ → +0.017
```

Monotone in run length, and it **reverses sign inside both strata**. So the
headline +0.208/−0.113 is *itself* an average over oppositely-signed sub-strata —
the same failure mode this chapter is about, one level down. Mechanically:
**CATIE's constant φ over-predicts perseveration after short runs and
under-predicts it after long ones.**

### 3. The noise ceiling is ≈0.26 — and the obvious way to estimate it is wrong

Without a ceiling, R²=0.106 is uninterpretable. Deriving one (full mathematics in
the next section): for a partition `L`, R²(L) = 1 − E[Var(gap|L)]/Var(gap), and the
finest history partition maximises this, giving

> **ceiling = 1 − E[q(1−q)] / Var(gap)**,  where q = P(y=1 | history)

Estimating `q` is the whole difficulty, and **the error is directional**. An
underfit q̂ is shrunk toward 0.5, so q̂(1−q̂) *overstates* the noise and
*understates* the ceiling. A logistic regression gives ≈0.17–0.18 — and
`c_prev × streak_bin` scores **0.178**, which **exceeds it**. A bound an observed
partition surpasses is not a bound; that is how the error was caught.

With a cross-validated gradient booster (ECE of q̂ = 0.004, i.e. well calibrated):

| estimator | value |
|---|---|
| plug-in, 1 − E[q̂(1−q̂)]/Var(gap) | **0.257** |
| rigorous bound, 1 − OOF-Brier/Var(gap) | **0.265** (ceiling ≥ this) |
| logistic-regression estimate (**too low**) | ~0.17 |

Against ≈0.26: `c_prev` alone reaches **41%** of achievable; `c_prev × streak_bin`
reaches **69%**. So roughly a third of the *explainable* structure is still
unaccounted for — there is more to find, and the earlier "one-dimensional" framing
was doubly wrong.

### 4. What this R² actually rewards — and the tautology question

`Var(gap|L) = Var(y|L) + Var(p|L) − 2Cov(y,p|L)`, so a partition scores **either**
by predicting the choice `y` **or** by homogenising the model's own forecast `p`.
These are very different achievements:

| partition | R² | E Var(y) | E Var(p) | E Cov |
|---|---|---|---|---|
| `c_prev` | 0.106 | 0.2144 | 0.0107 | +0.0110 |
| `c_prev × streak_bin` | 0.178 | 0.1912 | 0.0098 | +0.0072 |
| `p_alt1_bin` | 0.144 | 0.1952 | 0.0006 | +0.0008 |
| out-of-fold q̂, 50 bins (**best y-predictor**) | **0.020** | **0.1669** | 0.0547 | −0.0004 |
| (unconditional) | 0.000 | 0.2327 | 0.0949 | +0.0503 |

The q̂ row is the point: it predicts `y` better than anything else and scores
almost nothing, because it leaves `p` heterogeneous.

**`c_prev` removes 89% of the variance in CATIE's own forecast** (0.0949 → 0.0107).
Combined with finding #6 — that `c_prev` nearly *determines* which side of 0.5 the
forecast falls on — this means **"`c_prev` explains the calibration gap" is
substantially a statement about CATIE's architecture (φ=0.71 dominates `p`), not
purely a discovery about human behaviour.** It is not fully tautological: if CATIE
were calibrated *within* each stratum, R² would be 0 regardless. But the horse race
was structurally tilted, because `c_prev` is essentially the only single feature
that splits `p` bimodally, so every rival competed on `E[y|group]` variation alone.

Honest formulation: *conditional on CATIE's dominant internal axis, the residual
miscalibration is largest along that same axis, and grows with run length.* Not:
*human miscalibration is one-dimensional.*

### 5. Schedule's near-zero R² is a positive result, not a null one

Empirical biased-choice rate ranges **0.507–0.692** across the nine schedules
(spread 0.185), yet max |gap| is only **0.018**. Schedule scores R²=0.0004 because
**CATIE tracks between-schedule variation almost perfectly** — not because schedule
is behaviourally irrelevant. The R² table alone invites the opposite reading.

### 6. The `c_prev`/0.5 separation is structural (`verify_cprev_separation.py`)

With φ=0.71 on the previous choice and exploration capped at ε=0.30, when the trend
branch is not testable (H=0, **83.4%** of trials) the model *cannot* produce
P(alt1) > 0.353 if `c_prev=0`, nor < 0.647 if `c_prev=1`. **0 violations in 208,757
H=0 trials.** Crossings exist (1,779 = 0.71%) and every one has H=1 — the heuristic
is the only mechanism able to outvote inertia (1,354 upward all with b=1; 425
downward all with b=0). Falsification test: the published model must show *zero*
upward crossings since b≡0 there — confirmed 0 (it shows *more* downward
crossings, 860 vs 425: its dead heuristic branch diverts τ to alternative 2
unconditionally whenever H=1).

### 7. The E[p] vs E[log p] mechanism — tested, not asserted (§8)

The earlier version *asserted* that the cancellation is what splits E[p] from
E[log p]. It is now tested by recalibrating and observing the trade:

| forecast | E[p] | E[log p] | ECE |
|---|---|---|---|
| CATIE as-is | 0.6341 | −0.6595 | 0.149 |
| `c_prev` stratum rate | 0.5712 | −0.6191 | 0.000 |
| `c_prev × streak_bin` rate | 0.6175 | **−0.5631** | 0.000 |
| isotonic on `p_alt1` (in-sample) | 0.6088 | −0.5728 | 0.000 |

Correcting the conditional gap **improves E[log p] while lowering E[p]** — exactly
the trade the disagreement consists of.

**Two caveats, both load-bearing.** These recalibrations are **in-sample** (the
ECE=0.000 column is by construction, not an achievement), so they bound the
available gain optimistically rather than demonstrating it. And a full *causal*
claim about why CATIE loses to Q-Learning would need QL's own conditional
calibration profile, **which is computed nowhere in this repo**. Stated as a
mechanism consistent with the data, not a demonstrated cause. Worth noting: the
in-sample figures land near the project's quoted QL value of −0.569, which hints
CATIE's log-loss deficit may be largely a *calibration* deficit rather than a
*mechanism* deficit — a stronger thesis claim than the one made here, and one that
needs a held-out test before it can be asserted.

---

## The noise-ceiling mathematics

**Setup.** For each trial let `y ∈ {0,1}` be the choice, `p` CATIE's forecast, and
`gap = y − p`. For a partition `L` of trials, the R² used throughout is that of the
saturated group-mean model:

```
R²(L) = 1 − E[Var(gap | L)] / Var(gap)
```

**Step 1 — refining a partition never hurts.** By the law of total variance, for
any refinement `L' ⊇ L`, `E[Var(gap|L')] ≤ E[Var(gap|L)]`. So R² is maximised by
the *finest* partition available.

**Step 2 — the finest history partition.** Let `X` be the full history (all choices
and rewards through trial t−1). Since `p` is a deterministic function of history,
`p` is constant within a cell of `X`. Hence

```
Var(gap | X) = Var(y − p | X) = Var(y | X) = q(1 − q),    q := P(y = 1 | X)
```

using Var(Bernoulli(q)) = q(1−q).

**Step 3 — the ceiling.** Combining,

```
ceiling  =  max over history partitions of R²  =  1 − E[q(1−q)] / Var(gap)
```

The residual `E[q(1−q)]` is irreducible Bernoulli noise: no partition of the
history can explain a coin flip's variance.

**Step 4 — why the estimator's error is directional.** `q` is unknown and must be
estimated. For any q̂,

```
E[(y − q̂)²] = E[ Var(y|X) + (q − q̂)² ] = E[q(1−q)] + E[(q − q̂)²]  ≥  E[q(1−q)]
```

Two consequences:

- **Plug-in** `1 − E[q̂(1−q̂)]/Var(gap)` can err either way. An *underfit* q̂ is
  shrunk toward 0.5; since `x(1−x)` is maximised at 0.5, shrinkage inflates
  `q̂(1−q̂)`, overstating noise and **understating the ceiling**. An *overfit* q̂ is
  too extreme and overstates the ceiling — cross-validation guards this direction.
- **Rigorous bound.** The inequality above gives `E[q(1−q)] ≤ OOF-Brier(q̂)` for
  *any* q̂, calibrated or not, so
  ```
  ceiling ≥ 1 − OOF-Brier(q̂) / Var(gap)
  ```
  and this bound *tightens* as q̂ improves. It requires no assumption about q̂.

**Step 5 — the numbers.** `Var(gap) = 0.2270`. With 5-fold subject-grouped CV
(`GroupKFold`, so no subject spans folds) on history-only features:

| q̂ model | E[q̂(1−q̂)] | plug-in | OOF-Brier bound |
|---|---|---|---|
| constant base rate | 0.2327 | −0.025 | — |
| logistic regression | 0.1888 | 0.168 | 0.165 |
| gradient boosting | 0.1687 | **0.257** | **0.265** |

Monotone increase with model capacity, exactly as Step 4 predicts. The logistic
estimate is **falsified empirically**: `c_prev × streak_bin` achieves R² = 0.178,
above the 0.168 it implies. The gradient booster's q̂ has ECE = 0.004, so its
plug-in is trustworthy, and its rigorous bound (0.265) agrees. **Ceiling ≈ 0.26.**

**Caveat.** Step 2 assumes `p` is exactly history-measurable. Under the per-trial
k-weighting used throughout since 2026-09, it is: the mixture weight at trial t is
built from choices strictly before t. (Under the legacy shipped time-averaged
weighting this held only approximately, mean |Δp| = 0.0099 — one more reason the
per-trial rule is the right default. The Step-4 bound never relied on Step 2.)

---

## What the audit changed

| Earlier claim | Status |
|---|---|
| "Miscalibration is close to one-dimensional in `c_prev`" | **Wrong.** `c_prev × streak_bin` scores 0.180 vs 0.104, with a sign reversal inside each stratum. The search had crossed `c_prev` with only 2 of 5 candidates. |
| "No partition tested meaningfully beats `c_prev`" | **Literally true, materially misleading.** Three untested partitions beat the reported best. |
| §7 retrospective error attribution by `soft_argmax` (+ fig3, + a README finding) | **Deleted.** Not caveatable — see below. |
| Noise ceiling ≈0.182 (raised during the audit) | **Also wrong**, and in the same direction: an underfit q̂. True ceiling ≈0.26, so `c_prev × streak_bin` reaches 69%, not "99%". |
| E[p]/E[log p] causal link | Was asserted; now **tested** in §8, with caveats. |
| Phase 4 → asymmetric inertia | **Revised** — see below. |

**Why §7/fig3 were deleted rather than caveated.** `soft_argmax` is provably the
repeat/switch indicator. Regime *r*'s unnormalised responsibility is `w_r·P(y|r)`;
for inertia `P(alt1|inertia) = c_prev ∈ {0,1}`, so its term is exactly `w_I` when
`y == c_prev` and exactly 0 otherwise. And
`w_I = (1−τH)(1−p_exp)·φ ≥ 0.71·0.70·0.71 = 0.3529` strictly exceeds every other
regime's maximum (heuristic ≤ τ = 0.29; contingent ≤ (1−τH)(1−p_exp)·0.29 < w_I;
exploration ≤ (1−τH)·0.15 < w_I). The inequality is per-k, so it survives the convex
k-mixture. Therefore **`soft_argmax == inertia` ⟺ `y == c_prev`, identically** —
verified at **0 mismatches in 250,272 trials**. Every "empirical" number in the old
§7 was reconstructible from a 2×4 count table, and fig3 plotted four steeply
*decreasing* reliability curves — the visual grammar of catastrophic miscalibration
— for what is actually a deterministic identity. A caption cannot fix that. The
proof is retained in §5 as the finding; the derived numbers are gone.

## Implications for Phase 4

**The planned asymmetric inertia (one φ per side of `c_prev`) is no longer the
right minimal extension.** It targets only the main effect and *cannot* represent a
within-stratum sign reversal — finding #2 shows the gap runs from +0.347 to −0.072
*inside* `c_prev=0` alone. What the data demand is **run-length-dependent or
recency-weighted inertia**: φ that strengthens with the length of the current
perseveration run, rather than a constant (or a constant per side). Still one or two
parameters, and it addresses the effect that is actually there. Phase 3's parameter
re-fitting should include a run-length term in its candidate set.

## Files

| File | Role |
|---|---|
| `conditional_calibration.py` | Main analysis: partitions, R², ceiling, decomposition, recalibration test, figures |
| `responsibility.py` | Bayesian E-step over the four regimes (used only for §5's proof) |
| `verify_cprev_separation.py` | Standalone check backing finding #6 |
| `figures/output.txt` | Full transcript, all 8 sections |
| `fig1_reliability_aggregate_vs_cprev` | Aggregate reliability vs. split by `c_prev` |
| `fig2_r2_partition_comparison` | R² of all prospective partitions, with the ceiling marked |
| `fig3_gap_by_cprev_and_runlength` | **The headline**: sign reversal with run length |
| `fig4_ece_by_schedule` | ECE per schedule |
| `partition_r2.csv`, `variance_decomposition.csv`, `gap_by_cprev_streak.csv`, `metrics_by_schedule.csv` | Backing tables |
| `trial_level.csv.gz` | Per-trial frame (gitignored, regenerable) |

## How to run

```bash
python my_code/catie_calibration/02_mode_calibration/conditional_calibration.py
python my_code/catie_calibration/02_mode_calibration/verify_cprev_separation.py
```

`pip install pandas numpy scipy matplotlib scikit-learn`. A few minutes for 2,528
subjects × 3 k-agents, plus ~1 min for the 5-fold ceiling estimate.
