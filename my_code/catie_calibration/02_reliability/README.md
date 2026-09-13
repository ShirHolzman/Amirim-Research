# 02_reliability

Reliability diagrams of corrected CATIE's forecast (published τ=0.29, ε=0.30, φ=0.71;
per-trial k-mixture over k ∈ {0,1,2}), on the Training split (1,483 subjects, 146,817
scored trials, trial 1 dropped) with one EDA replication panel (496 subjects, 49,104
scored trials). No fitting happens in this stage; it produces the diagram that
`03_recalibration` reads a map off. See `sub_plans/02_reliability.md` for the full
plan, every definition, and the ten verification checks (all pass — outcome recorded
there).

Script: `reliability.py` (cache → CSVs, no plotting) → `figures.py` (CSVs → figures/,
no cache). Outputs: `reliability.csv` (267 rows, one per bin), `reliability_summary.csv`
(13 rows, one per curve), `figures/fig1..fig5*.png`.

## 1. The diagram the supervisor originally reacted to is superseded

`fig1_baseline_biased.png` — P(biased alternative) vs outcome, 10 uniform bins:
ECE = 0.147, E[p] = 0.623, E[log p] = −0.673 (`reliability_summary.csv`, `baseline_biased`
row). Kept only as a before/after reference. The axis is asymmetric and forecasts a
label CATIE does not carry (see project memory); every other figure below uses the
label-equivariant doubled set instead.

## 2. Main result: CATIE is overconfident by several percentage points, not a fraction of one

`fig2_doubled_main.png`, 10 uniform bins, training: ECE = 0.144, MCE = 0.257,
Brier = 0.232, E[p] = 0.623, E[log p] = −0.673 (`doubled` row, `n_bins=10`,
`strategy=uniform`). The curve sits inside the diagonal at both ends — bin 0
(predicted 0.081, empirical 0.180, gap +0.100) and bin 9 (predicted 0.919, empirical
0.820, gap −0.100) — the qualitative direction expected of an overconfident model. The
**size** of the gap is larger than the "a few points" expected going in: individual
bins reach a 21-point gap (bin 7: predicted 0.749, empirical 0.541), and the 10-bin ECE
(0.144, i.e. 14.4 points on average) is itself an order of magnitude above the
competition's 0.001 margin on E[p]. This is the headline number for the report.

## 3. Bin-count sweep: ECE rises monotonically with resolution, as expected

`fig3_doubled_bin_sweep.png`. ECE (uniform): 0.144 (10 bins) → 0.152 (20) → 0.160 (50)
→ 0.162 (100) — finer bins keep revealing more miscalibration, never less, matching the
pre-registered expectation. Quantile bins agree closely at 10 bins (ECE = 0.144,
identical to uniform to high precision for this dataset — a property of these
particular edges, verified NOT to hold for arbitrary bin edges, not a general
invariance) and stay close to uniform at 20 bins (0.157 vs 0.152). At 50–100
uniform bins, CATIE's forecast clumping (ε keeps it away from 0/1) starts producing
bins with a few hundred trials and visibly wider error bars, which is why the 10-bin
uniform diagram is the one quoted as the headline.

## 4. Run length: the calibration gap reverses sign — the motivation for M3

`fig4_doubled_run_length.png`, five training strata, 10 uniform bins each
(`doubled_run*` rows). Fixed a priori, reused unchanged in `03_recalibration` and
`04_refit`:

| run length | n (undoubled) | ECE | E[p] | E[log p] | pattern |
|---|---:|---:|---:|---:|---|
| 1 (just switched) | 51,453 | 0.297 | 0.518 | −0.897 | strongly overconfident |
| 2 | 25,125 | 0.173 | 0.575 | −0.758 | overconfident |
| 3–4 | 24,352 | 0.108 | 0.630 | −0.654 | overconfident |
| 5–9 | 21,604 | 0.066 | 0.708 | −0.502 | mildly overconfident |
| 10+ (long streak) | 24,283 | 0.072 | 0.814 | −0.283 | **underconfident** |

Right after a switch (run length 1), CATIE is dramatically overconfident: its most
extreme bin claims 92% and reality delivers 64% (gap −0.274, and the mirrored low bin
claims 8% for a reality of 36%, gap +0.274). As the streak lengthens the gap shrinks
monotonically through run lengths 2, 3–4 and 5–9 — and then **flips sign** at 10+: every
bin's gap changes direction (e.g. predicted 0.081 vs empirical 0.055, gap −0.026; mirror
bin predicted 0.919 vs empirical 0.945, gap +0.026), meaning CATIE is now *underconfident*
— real behaviour on very long streaks is even more predictable than CATIE's constant φ
credits. A single constant φ cannot represent a sign change; this is the measured,
not assumed, motivation for run-length-dependent inertia (M3) in `04_refit`.

## 5. EDA replication — no decision follows this panel

`fig5_doubled_eda.png`, generated last: ECE = 0.169, MCE = 0.246, Brier = 0.244,
E[p] = 0.615, E[log p] = −0.705 (`doubled_eda` row). Same direction and similar
magnitude as Training's doubled result (ECE 0.144, E[p] 0.623, E[log p] −0.673) on
schedules the diagram was not built from — the overconfidence pattern is not an
artifact of the Training schedules.
