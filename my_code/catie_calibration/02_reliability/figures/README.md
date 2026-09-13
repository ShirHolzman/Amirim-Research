# figures/

Drawn by `../figures.py` from `../reliability.csv` and `../reliability_summary.csv`
only — no recomputation happens here. Full numbers and discussion: `../README.md`.

- **fig1_baseline_biased.png** — P(alt 1) vs whether alt 1 was chosen, training, 10
  bins. The original (asymmetric) diagram. Superseded by fig2; kept as a before/after
  reference.
- **fig2_doubled_main.png** — the label-equivariant diagram: each trial counted twice,
  as `(P(alt1), chose alt1)` and `(P(alt2), chose alt2)`. 10 uniform bins, training.
  The headline result — CATIE is overconfident.
- **fig3_doubled_bin_sweep.png** — the same doubled diagram at 10/20/50/100 bins
  (uniform: equal-width bins in probability) and 10/20 bins (quantile: equal-count
  bins). Shows the result isn't an artifact of one binning choice.
- **fig4_doubled_run_length.png** — the doubled diagram split into 5 panels by run
  length (the streak of identical choices just before the scored trial: 1, 2, 3-4,
  5-9, 10+). Shows the calibration gap shrinking, then reversing sign, on long streaks.
- **fig5_doubled_eda.png** — fig2's diagram rerun on the EDA split instead of
  Training. A replication check: everything was fixed on Training beforehand, EDA is
  scored once, and nothing was adjusted after seeing it.
