# Step 1 — `02_reliability`

## Context

`PLAN.md` step 1. The supervisor meeting of 2026-09-05 produced three asks about the
first reliability diagram: **more bins**; the P(biased alternative) axis is asymmetric
and forecasts a label CATIE does not carry, so redo it on p(action)/p(not action) by
entering each trial twice as `(p, y)` and `(1 − p, 1 − y)`; then "calibrate CATIE from
the graph".

This stage produces the graph. It makes no model change and fits nothing. It exists so
that step 2 (`03_recalibration`) can read a map off a curve that is already fixed,
already binned on Training, and already has a clustered confidence interval on every
bin. Because that map is *read off this diagram*, this diagram is part of a fit — so
every curve here is Training-only, with EDA appearing exactly once, at the end, as a
replication panel that no decision may follow.

The stage also delivers the empirical fact that motivates M3 in step 3: whether the
calibration gap changes sign with the length of the current run of repeated choices. A
single constant φ cannot represent a sign change, so if the gap does reverse, that is
the evidence for run-length-dependent inertia — measured here, not assumed.

## Scope

**In:** binned reliability curves of the corrected CATIE with the published parameters
(τ, ε, φ), per-trial k-mixture over k ∈ {0,1,2}, computed from `cache/state_training.npz`
and `cache/state_eda.npz`; subject-clustered bootstrap CIs; ECE/MCE/Brier/E[p]/E[log p]
per curve; figures drawn only from the CSVs.

**Out (explicitly not this step):** any recalibration map, any isotonic/Platt/temperature
fit, any re-fitted parameter, any comparison table or Pareto figure, any use of the Test
split, any per-schedule breakdown, any significance test between curves. No new code in
`catie/` — the base is frozen, and this stage imports from it read-only.

## Definitions (fixed here, reused unchanged in 03 and 04)

**Scored trials.** Trials 2…100 of every subject. Trial 1 is dropped because
`p_alt1_single_k` pins it to exactly 0.5 by construction, so it is not a forecast; the
paper's own E[log p] also drops it (`likelihood.mean_log_p(drop_first=True)`). Training:
1,483 × 99 = **146,817** trials. EDA: 496 × 99 = **49,104**.

**Recovering `p_alt1`.** `likelihood.p_choice_matrix` returns P(the choice actually
made). Since it is built as `where(y, p1, 1 − p1)`, the forecast is recovered exactly by
the same expression: `p_alt1 = where(y, p_choice, 1 − p_choice)`. No new function in
`catie/`, and the identity is asserted in the test (check 1).

**The doubled (label-equivariant) set.** Each scored trial contributes two rows:
`(p_alt1, y)` and `(1 − p_alt1, 1 − y)`. Training doubled: **293,634** rows. The
`subject_id` vector is doubled alongside, so both copies of a trial belong to the same
subject and the cluster bootstrap resamples them together — a doubled set with
independent-looking rows would halve every CI width, which would be wrong. Language:
this set is **label-equivariant** and it **pools** the two strata; never
"information the model does not have".

**Run length.** `run[i, t]` = number of consecutive identical choices ending at trial
`t − 1`, so it is a function of `y[i, :t]` only and never of the trial being scored.
Strata, fixed a priori: **1, 2, 3–4, 5–9, 10+**. These cut points are frozen here and
carried unchanged into 03 and 04; they are not re-chosen after seeing any result.

## Curves produced

| # | diagram | split | bins | why |
|---|---|---|---|---|
| 1 | `baseline_biased` — P(biased alt) vs outcome | training | 10, uniform | the version shown on 2026-09-05, kept as a before/after reference and labelled **superseded** in the README |
| 2 | `doubled` | training | 10, 20, 50, 100 uniform + 10, 20 quantile | the supervisor's symmetrised axis, and his "more bins" |
| 3 | `doubled_run{1,2,3-4,5-9,10+}` | training | 10 uniform | does the gap reverse sign with run length |
| 4 | `doubled_eda` | eda | 10 uniform | one replication panel, generated last, no decision may follow it |

Confidence intervals: subject-clustered bootstrap, seed 42, `n_boot = 2000` on curves
1, 3, 4 and on the 10-bin member of curve 2; `n_boot = 500` on the remaining sweep
members (bins ≥ 20), which exist to show where bins empty out, not to be quoted. Every
CI in the CSV records its own `n_boot`.

Quantile binning is included because CATIE's forecast is clumped — ε clamps it away from
0 and 1 — so uniform bins at 50 and 100 will contain empty or near-empty cells. Reporting
both makes the binning a stated choice rather than a hidden one.

## Files

```
02_reliability/
  reliability.py            cache/ -> the two CSVs. No matplotlib, no plotting.   (<=150 lines)
  figures.py                the two CSVs -> figures/*.png. No cache, no catie/.   (<=150 lines)
  reliability.csv           one row per bin
  reliability_summary.csv   one row per curve
  figures/                  png only
  README.md                 every number in it cites one CSV row
tests/02_reliability/
  reliability_test.py       the checks below
```

The compute/plot split is deliberate: a bug in `figures.py` can then only produce an
ugly picture, never a wrong number, and every plotted point is literally a row of
`reliability.csv` rather than something recomputed alongside it. Re-plotting for the
supervisor costs a second and no recomputation.

**`reliability.csv`** — `diagram, split, stratum, n_bins, strategy, bin, lo, hi, n,
predicted, empirical, gap, emp_lo, emp_hi, n_boot`

**`reliability_summary.csv`** — `diagram, split, stratum, n_bins, strategy, n,
n_subjects, ece, ece_lo, ece_hi, mce, brier, e_p, e_log_p, n_boot`

`e_p` / `e_log_p` are on `p_choice` over the same scored trials (they are properties of
the split, identical across binnings — carried on every row so a reader never has to
join tables).

## Reused, not rewritten

From `catie/metrics.py`: `reliability_table_ci` (bin edges computed once on the full
sample and held fixed across replicates — the property that makes a quantile-binned CI
meaningful), `ece_ci`, `mce`, `brier`, `e_p`, `e_log_p`, `cluster_bootstrap`.
From `catie/likelihood.py`: `StateCache`, `p_choice_matrix`, `mean_p`, `mean_log_p`.
New code in this stage: the doubling, the run-length tensor, the CSV assembly, the plots.

## Verification — `tests/02_reliability/reliability_test.py`

Checks that can actually fail, in order. Each is cheap enough to run on the full
Training cache.

1. **`p_alt1` round-trip.** `where(y, p_alt1, 1 − p_alt1)` equals
   `p_choice_matrix(...)` elementwise, max abs diff < 1e-12.
2. **Anchor to the validated base.** The stage's `e_p` / `e_log_p` on Training equal
   `likelihood.mean_p` / `mean_log_p` with `drop_first=True` to < 1e-12, and equal the
   values `tests/catie/likelihood_test.py` reproduces from the paper within its own
   tolerance. This ties every number in the stage to the MATLAB-validated layer.
3. **Doubling identities.** On the doubled set: mean predicted = 0.5 and mean empirical
   = 0.5, both exactly (to 1e-12), by construction; and the reliability table of the
   doubled set is invariant under a global label flip `(p, y) -> (1 − p, 1 − y)`. All
   three break if the doubling is built wrong.
4. **Binning conservation.** Per curve, `sum(n)` over bins = the number of rows scored
   (146,817; doubled 293,634; EDA 49,104), the n-weighted mean of `predicted` equals the
   overall mean forecast, and the n-weighted mean of `empirical` equals the overall base
   rate. Catches off-by-one bin edges and dropped tail bins.
5. **Run-length brute force.** The vectorised `run` tensor equals a naive Python
   double loop on 50 randomly chosen subjects (seed 42), exactly.
6. **Run-length has no leakage.** Flipping `y[i, t]` leaves `run[i, t]` unchanged and
   changes `run[i, t + 1]`. This is the mechanical proof that the stratifier is
   history-only, which is what keeps it clear of the closed "current-choice" direction.
7. **Strata partition.** The five strata are disjoint and their counts sum to 146,817;
   no stratum is empty; the smallest stratum has enough rows for a 10-bin diagram
   (recorded, and reported to the user if any bin falls below 200 rows).
8. **CI sanity.** For every bin row, `emp_lo <= empirical <= emp_hi`; CI width is
   negatively rank-correlated with `n` across bins of one curve.
9. **Determinism.** Rerunning `reliability.py` reproduces both CSVs byte-identically.
10. **README fidelity.** Every number quoted in `02_reliability/README.md` matches a
    row of one of the two CSVs (parsed and compared, not eyeballed).

## Self-assessment — what a correct result looks like

These are expectations, not targets. They are written down *before* the run so that
agreement is evidence and disagreement is a finding rather than an excuse.

- CATIE is known to be overconfident (it wins E[p], loses E[log p]). So on the doubled
  diagram the curve should sit **inside** the diagonal: `empirical < predicted` above
  0.5 and `empirical > predicted` below it. The gap should be a few percentage points,
  not fractions of one.
- ECE near 0 would be a **red flag**, not a success — it would contradict the E[log p]
  deficit that motivates the whole project and would most likely mean the outcome vector
  is aligned with the forecast it is supposed to be scored against.
- The doubled curve must be **exactly symmetric** about (0.5, 0.5); visible asymmetry
  means the doubling is broken, not that CATIE is asymmetric.
- Forecast mass will be concentrated between roughly 0.05 and 0.95 (ε clamps it), so
  extreme uniform bins will be thin. Thin bins with huge CIs are expected and are the
  reason quantile binning is reported alongside.
- Across the 10/20/50/100 sweep, ECE should rise with bin count (finer bins reveal
  miscalibration that coarse bins average away). ECE **falling** as bins get finer
  would indicate a binning bug.
- If the run-length strata show no sign change in the gap, that is a real negative
  result: it is reported as such and M3 in step 3 loses its motivation. It must not be
  rescued by re-choosing the cut points.

## Execution order

1. Write `sub_plans/01_reliability.md` (this file) and the run lines into
   `02_reliability/README.md` skeleton.
2. Write `reliability.py`. **Show it to the user before running it.**
3. Run it. Expected runtime: cache load and scoring are seconds; the bootstraps dominate
   — roughly 2–6 minutes total. If it exceeds ~2 minutes, rerun under
   `python -u 02_reliability/reliability.py > 02_reliability/figures/run.log 2>&1` in the
   background, per `CLAUDE.md`.
4. Write `tests/02_reliability/reliability_test.py`, show it, run it. Any failing check
   is reported to the user before anything is changed.
5. Write `figures.py`, show it, run it.
6. Write `02_reliability/README.md` with each number citing its CSV row.
7. Fill in the Outcome section below; ask for approval to commit.

Run lines:

```
cd my_code/catie_calibration
python 02_reliability/reliability.py
python -m pytest tests/02_reliability -q
python 02_reliability/figures.py
```

## Outcome

**Ran, in order:** `reliability.py` (16m19s — slower than the 2–6 min estimate; CPU
time confirmed it was genuinely computing, not stalled, so it was left to finish
rather than killed), `tests/02_reliability/reliability_test.py` directly and via
`python -m pytest tests/02_reliability -q` (11/11 pass both ways), `figures.py` (one
cosmetic fix mid-way: fig4's five narrow panels had overlapping titles when the
per-panel n/ECE/E[p]/E[log p] line was included — that annotation was dropped from
fig4 only; the CSVs were never touched by this).

**Two deviations from the plan, both flagged before being applied, not silently
fixed:**
- Check 2's second clause ("...and equal the values `likelihood_test.py` reproduces
  from the paper") does not apply: the paper has no reported number for the
  *corrected* model (only the published/buggy one), and `likelihood_test.py`'s own
  pooled corrected figure uses trial 1 kept, while this stage drops it — there is no
  external reference number a drop_first=True corrected E[p]/E[log p] could be
  checked against. Implemented as the exact half only: this stage's numbers equal
  `catie.likelihood.mean_p`/`mean_log_p(drop_first=True)` to < 1e-12.
- Check 9 (determinism) as written means rerunning the full 16-minute script and
  diffing the CSVs on every test invocation. Replaced with a check on the actual
  source of randomness (`cluster_bootstrap`'s seeded RNG): the same bootstrap call
  twice gives a byte-identical result. The full script was in effect run once (the
  actual production run above); it was not rerun a second time purely to diff bytes,
  since that would cost another ~16 minutes for a already-covered guarantee. Say if
  you want that second full rerun done anyway.
- The README-fidelity test itself had a bug on first run: its number-matching regex
  didn't recognise the typographic minus sign (−, U+2212) used throughout the prose,
  so it silently read negative numbers as positive and flagged 4 false mismatches.
  Fixed in the test, not the README; also removed one genuinely unanchored number
  from the README (a derived difference, "within 0.005", that wasn't itself a CSV
  cell).

**What the curves showed:**
- Baseline (superseded), doubled (10 bins), and EDA all confirm CATIE is
  overconfident, matching the self-assessment's predicted direction — but the
  **magnitude** is larger than "a few percentage points": 10-bin ECE = 0.144
  (training) / 0.169 (EDA), individual bins reach 20+ point gaps. This is flagged
  explicitly in the README as exceeding the pre-registered expectation, not smoothed
  over.
- ECE rises monotonically with bin count (0.144 → 0.152 → 0.160 → 0.162 at
  10/20/50/100 uniform bins), as predicted.
- The doubled diagram is exactly symmetric about (0.5, 0.5) to float precision, as
  required (test 3).
- **Run length: the calibration gap reverses sign**, cleanly and monotonically —
  strongly overconfident at run length 1 (ECE 0.297) shrinking through 2, 3–4, 5–9
  (ECE 0.066), then flipping to underconfident at 10+ (every bin's gap changes
  direction). This is the pre-registered positive case for M3 (run-length-dependent
  inertia); it was not rescued or adjusted after being observed.
- One numerical curiosity, checked rather than assumed: uniform-10 and quantile-10
  ECE coincide to 16 significant digits on this dataset despite different bin edges.
  Verified NOT to be a general property (arbitrary bin edges, including other
  mirror-symmetric ones, give ECE differing by ~0.0002–0.0004) and not a strategy
  silently falling back to another — both edge sets and bin tables were printed and
  are genuinely different. Noted in the README as an unexplained but verified
  coincidence, not chased further.

**Verification:** all 10 planned checks pass (11 test functions — the plan's check 2
was split into two functions). `python -m pytest tests -q` (whole project) still
green. Not yet committed — awaiting approval.
