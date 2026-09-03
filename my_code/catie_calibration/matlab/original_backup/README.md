# Backup of original reference-repo files that were edited in place

`Data_resources/competition_analysis-main/` (the paper's reference MATLAB) is
**gitignored**, so edits to it leave no history. On 2026-08-30 a separate session
patched four files there so that `COMPETITION_main.m` could run end-to-end. This folder
holds byte-identical copies of those four files **as originally shipped**, rescued from
that session's temporary scratchpad (which is disposable) so the originals are never lost.

| File here (original) | Live location (patched) | What the patch changed |
|---|---|---|
| `COMPETITION_constants.m` | `COMPETITION_constants.m` | Paths anchored to `mfilename('fullpath')`; `DATA` → `COMPETITION_data`; `DATA_DYNAMIC` pointed at the restricted data drop |
| `COMPETITION_empirical_decisions_probabilities.m` | `helper_functions/…` | Hardcoded `C:\Users\ojd5\...` replaced by `COMPETITION_constants()`; folder names `model_1..12` → `schedule_0..11`; columns `biased_reward`/`unbiased_reward`/`is_biased_choice` → `reward_alternative_1`/`reward_alternative_2`/`is_choice_alternative_1` |
| `COMPETITION_empirical_simulated_bias_correlation.m` | `helper_functions/…` | Call to the name-colliding `COMPETITION_all_schedules_biases` replaced by an inline mean-per-schedule loop |
| `optimize_schedule_CATIE.m` | `helper_functions/optimize_schedule_CATIE_LEGACY_BROKEN.m.bak` | Renamed only (broken legacy script; the working one is `CATIE/optimize_schedule_CATIE.m`) |

**None of the CATIE or QL likelihood/simulator files were touched.** Every model
number in this project therefore still comes from unmodified original model code; only
path/glue code was patched. To restore the repo to its shipped state, copy these four
files back over the live ones and rename the `.m.bak` back to `.m`.

To re-verify at any time:

```
diff my_code/catie_calibration/matlab/original_backup/COMPETITION_constants.m Data_resources/competition_analysis-main/COMPETITION_constants.m
```
