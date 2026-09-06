# Plan — rebuild and finish

Decision: the Phase 2/3 analyses are archived under `my_code/previous_errors/` and
rebuilt as five small stage scripts on top of the validated `catie/` package. Nothing
from the archive is cited. Each stage: one script ≤150 lines, no prints, one CSV,
figures, a README, and a test. One script at a time; the student reads each before it
runs.

## Order

| step | stage | what it delivers | est. |
|---|---|---|---|
| 0 | refactor imports to `catie/`; run `tests/` | green base | 2 h |
| 1 | `02_reliability` | bin-count sweep, doubled/action diagram, folded-by-run-length; all on Training; one EDA replication panel last | 4 h |
| 2 | `03_recalibration` | 10-bin histogram on the doubled set (the supervisor's "from the graph" map), symmetric isotonic, unconstrained isotonic, temperature; fitted on Training excl. trial 1, scored on EDA; bin sweep 10/20/50/100 | 4 h |
| 3 | `04_refit` | M2 (τ, ε, φ); M3 run-length φ with threshold chosen on Training; `build_streak` with brute-force test | 5 h |
| 4 | `05_ladder` | one table, one paired subject-level test per comparison, E[p]–E[log p] Pareto figure | 2 h |
| 5 | `05_ladder` | freeze; write stage READMEs; update supervisor summary | 3 h |
| 6 | `06_test` | Test cache with explicit flag; golden check vs `01_bug_correction`'s Test row; score the ladder once | 2 h |
| 7 | report skeleton | | rest |

## Split discipline
Every diagram and every fit is on Training. EDA is scored once per stage and never
selected on. Bin edges and thresholds are fixed on Training and reused. Test is
touched only in step 6.

## What is deliberately not done
Standard errors on parameters, per-schedule fits and heterogeneity tests, K and ε
sweeps, optimiser-agreement tables, profile surfaces. If asked: "one paired
subject-level test per comparison; parameters differ across schedules; precision not
estimated."

## Supervisor framing
Lead with the symmetric reliability diagram (his request) as the first result; then the
map read off it, fitted on Training and scored on EDA; then that every map lowers E[p]
(a property of calibrating an overconfident model, not a flaw in the idea); then
run-length inertia as the one change that moves both metrics. Say "label-equivariant",
never "information the model does not have".
