# Phase 2 — Conditional calibration: which partition explains the miscalibration?

## Research question

CATIE's *aggregate* calibration gap is small, yet it loses badly to Q-Learning on
E[log p]. The working hypothesis: the aggregate conceals large, oppositely-signed
*conditional* miscalibration that cancels in a mean but compounds in a log. This
chapter doesn't presuppose which partition of the data reveals that structure — it
**adjudicates** between candidates, with the previous choice (`c_prev`) as the
mandatory baseline every richer partition must beat, and reports the result
honestly in either direction.

Model: the **corrected** ("fixed") CATIE likelihood, k∈{0,1,2} published mixture —
the project's default baseline per the memo. A brief published-vs-fixed check
(§2 of the output) confirms the chapter's conclusions don't depend on which one
is used.

Data: **EDA + Training + schedule_0** (2,524 subjects, 249,876 trials after
dropping trial 1, schedules 0,2,3,4,5,6,7,9,11). **Test is deliberately excluded
and asserted absent** (`load_frame` raises if any Test schedule appears) — Test
is touched exactly once, at the very end of the project timeline, never during
exploratory work.

## Key findings

**1. The central arithmetic** (re-derived on the full k-mixture and population,
not the single-k/EDA-only planning-phase estimate):

| c_prev | n | predicted | empirical | gap |
|---|---|---|---|---|
| 0 (prior choice unbiased) | 91,548 | 0.249 | 0.454 | **+0.205** |
| 1 (prior choice biased) | 158,328 | 0.848 | 0.735 | **−0.114** |
| **aggregate** | 249,876 | — | — | **+0.003** |

Confirms the planning-phase pattern robustly: near-total cancellation in the
mean, from two large, oppositely-signed strata. This is the arithmetic behind
CATIE's E[p]/E[log p] split decision.

**2. Partition adjudication (prospective partitions only — see caveat below):**
`c_prev` alone explains **R²=0.104** of the trial-level calibration-gap variance.
Every other *prospective* candidate — schedule, trial-position quintile,
streak length, recent reward rate — explains essentially none (R² < 0.006).
The hard-argmax mode attribution (mass toward "choosing alt 1", built only from
historical state — see `catie_core.mode_contributions`) adds a marginal **+0.016**.
No partition tested meaningfully beats `c_prev`. This is the "cleaner result"
outcome the project plan flagged as acceptable if mode structure didn't survive
scrutiny as an independent signal: **the miscalibration is close to
one-dimensional in `c_prev`**, which directly motivates Phase 4's asymmetric-inertia
extension (one parameter, targeted exactly at this split).

**3. A methodological trap, found and fixed during this analysis, not before it.**
The plan called for a Bayesian responsibility posterior over CATIE's four regimes
(`responsibility.py`) to fix a known flaw in the planning-phase's hard-argmax mode
attribution (which is provably forced to equal `c_prev`). The posterior *does* fix
that — `P(c_prev=1 | mode)` moves from the degenerate {0,0,0,1} to {0.78, 0.89,
0.30, 0.70} (§5 of the output; `fig5`). But a first version of this analysis then
asked "how much variance in the calibration gap does this new mode partition
explain" and got **R²=0.967** for `c_prev × soft_argmax` — implausibly high, and
traced (not assumed) to a real problem: the responsibility posterior is computed
*from the observed choice* `y(t)` (that's what makes it a proper E-step), so
grouping by it silently leaks the answer. Verified directly: within every
`(c_prev, soft_argmax)` cell, `chose_biased` comes out **exactly** 0 or 1 — the
same circularity that discredited `prev_side_switch` in the original EDA work,
one level more indirect. `soft_argmax` was removed from the R² adjudication table
entirely and reframed as what it legitimately is: **retrospective error
attribution** (§7) — "of the choices CATIE got most wrong, which regime's implicit
confidence was responsible" — useful for targeting Phase 4's fixes, not usable as
a forecasting feature. See §5 of `figures/output.txt` for the full derivation and
the circularity proof baked into the script's own reproducible output.

**4. Retrospective attribution (diagnostic, not calibration) still tells a
coherent mechanistic story:** trials the E-step attributes to `exploration` are
dramatically overconfident (predicted 0.816, empirical 0.110) — these are
essentially "surprise" trials where the other regimes' predictions failed and
exploration becomes the residual explanation. `contingent_avg`-attributed trials
are underconfident in the other direction (predicted 0.437, empirical 0.701).
`inertia`, the largest group by far (166,310 of 249,876 trials), is nearly
well-calibrated on its own (+0.033) — consistent with finding #2: almost all of
the *real* miscalibration lives in the `c_prev` split, not in mode identity.

**5. Per-schedule ECE** (never reported before, including in the source paper)
ranges 0.14–0.21 across the 9 non-held-out schedules — consistent with finding #2
that schedule barely explains the gap; miscalibration is not schedule-specific.

**6. Why `fig1`'s right panel looks cleanly split at p=0.5**
(`verify_cprev_separation.py`). The `c_prev` split isn't merely *correlated* with
which side of 0.5 the forecast falls on — for most trials it is a **theorem**.
With φ=0.71 loaded on the previous choice and the exploration term capped at
ε=0.30, when the trend branch is not testable (H=0, **83.4%** of trials) the model
*cannot* produce P(alt1) > 0.353 if `c_prev=0`, nor < 0.647 if `c_prev=1`. Verified:
**0 violations in 208,429 H=0 trials.**

The separation is nonetheless not absolute. **1,527 crossings (0.61%)** occur, and
every one is an H=1 trial where the heuristic points opposite to the previous
choice — the only mechanism in CATIE able to outvote inertia. The prediction that
*all* crossings must have H=1 was checked exhaustively and holds at 100% (1,273
upward, all with b=1; 254 downward, all with b=0).

Two things this rules out: the crossings are **not** hidden by the `min_count=30`
bin filter (their bins hold 1,273 and 254 trials and are drawn in `fig1` as the
slight overhang past 0.5 at each curve's inner end), and the pattern is **not** a
plotting artifact. As a falsification test, the published model must show *zero*
upward crossings, since b≡0 identically under the Phase 1 bug — confirmed: 0.
(It shows *more* downward crossings than the corrected model, 565 vs 254, because
its dead heuristic branch diverts τ=0.29 to alternative 2 unconditionally.)

The practical implication for Phase 3/4: φ is not just the largest parameter, it
is close to *deterministic* of the forecast's side of 0.5. That is why the
miscalibration is one-dimensional in `c_prev` (finding #2), and why asymmetric
inertia is the natural minimal fix.

## Methods

- `responsibility.py`: exact Bayesian E-step over CATIE's four regimes
  (heuristic/exploration/inertia/contingent-average), mixed across k using the
  identical per-trial weight matrix `catie_core.mix_agents` uses internally
  (`return_weights=True`, not recomputed) so probability- and
  responsibility-mixing cannot silently diverge. Validated at runtime on every
  subject: responsibilities sum to 1 to 8.2×10⁻¹⁵.
- `conditional_calibration.py`: loads data, builds non-circular partition
  features (all built from trial *t−1* and earlier — `streak_prev` and
  `recent_reward_rate` use `.shift(1)` before any rolling/run-length
  computation), runs the per-subject model, adjudicates partitions by R², full
  metric suite overall/per-schedule/per-stratum, figures.
- `r2_explained(gap, labels)`: R² of the saturated group-mean model
  (between-group SS / total SS) — deliberately excludes `soft_argmax` (see
  finding #3).
- `verify_cprev_separation.py`: standalone check backing finding #6 — derives the
  analytic P(alt1) bounds per (c_prev, H, b), counts every crossing in the raw
  data with no binning, confirms all crossings are H=1, checks whether the
  `min_count=30` filter hid anything, and runs the published-model falsification
  test. Run it after `conditional_calibration.py` (it reads `trial_level.csv.gz`).

## Output

| File | Contents |
|---|---|
| `figures/output.txt` | Full transcript, all 7 sections |
| `fig1_reliability_aggregate_vs_cprev` | Aggregate reliability curve vs. split by `c_prev` — the central figure |
| `fig2_r2_partition_comparison` | R² bar chart, prospective partitions only |
| `fig3_reliability_by_soft_mode` | Retrospective attribution curves (near-vertical lines are the visual signature of the circularity documented in finding #3 — expected, not a bug) |
| `fig4_ece_by_schedule` | ECE per schedule |
| `fig5_hard_vs_soft_attribution` | `P(c_prev=1\|mode)`: hard (degenerate) vs. soft (not) |
| `partition_r2.csv`, `metrics_by_schedule.csv`, `gap_by_soft_mode.csv` | Tables backing the figures |
| `trial_level.csv.gz` | Full per-trial frame (gitignored, regenerable) |

## How to run

```bash
python my_code/catie_calibration/02_mode_calibration/conditional_calibration.py
```

`pip install pandas numpy scipy matplotlib`. Runs in a few minutes for 2,524
subjects × 3 k-agents × (responsibility posterior + published comparison pass).

## Data dependency

```
my_code/EDA_set/processing/eda_with_catie_probabilities.csv  ─┐
my_code/catie_calibration/data/cleaned_{training,schedule_0}.csv ─┼─> conditional_calibration.py
my_code/catie_calibration/catie_core.py (+ metrics.py) ────────────┘
                                          │
                                          └─> responsibility.py
```
