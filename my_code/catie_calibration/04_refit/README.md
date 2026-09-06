# 04_refit

Purpose: parameter re-fit on Training -- M2 (tau, eps, phi) and M3 (run-length-dependent phi, threshold chosen on Training) -- scored once on EDA.
Input: cache/. Output: refit.csv (parameters, E[p], E[log p] per model per split) and figures/. No prints.
Script: refit.py (to be written). Contains build_streak with its brute-force test. Test: tests/04_refit/refit_test.py.
