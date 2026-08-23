# Project reorganization plan

Migrating the project onto the competition's **organized** data release
(`Data_resources/COMPETITION_data_restricted/simple_format_data/`), renaming the
first-investigations folder, and simplifying `sanitize_splits.py`.

Status: **EXECUTED 2026-08-22 — closed.** See "Execution record" at the bottom for what
actually happened, including two deviations from the plan.

> **Archived.** Moved to `my_code/completed_issues/` once the work was done. Unqualified
> file paths below (`sanitize_splits.py`, `01_bug_correction/README.md`, `REVIEW_PLAN.md`,
> …) are relative to **`my_code/catie_calibration/`**, where this document used to live.

---

## Why this is worth doing

The organized release is the source the competition's own README points to:

> "This data is provided as a redundant addendum to the full dataset provided with the
> competition (see Competition data). Please note that all the analyses presented in the
> manuscript can be achieved using **that full dataset which is better organized and suited
> for analysis**."

Three concrete wins, all verified during planning (not assumed):

1. **It resolves the `schedule_7` gap.** The organized release has 122 `schedule_7` files
   (vs 118 in our current raw copy), 3 of them explicitly tagged `..._INVALID_BIAS.csv` →
   **119 valid, exactly the paper's reported N**. The previously `[OPEN]` 4-subject
   discrepancy disappears.
2. **Exclusion becomes authoritative instead of inferred.** Applying only the curators'
   `_INVALID_BIAS` tag reproduces the paper's N on **all 12 schedules**, with
   `few_choices = 0` everywhere — our inferred `MIN_CHOICES_PER_SIDE = 5` heuristic becomes
   redundant (it stays as an assertion, see Phase C).
3. **Filenames are globally unique** (`schedule_7_n_42.csv`), so the timestamp-collision
   hazard (`1609192765_75.csv` in both `schedule_6` and `schedule_11`) is gone by
   construction.

| | current raw | organized release |
|---|---|---|
| total files | 3,382 | **3,386** (= paper's stated raw total) |
| total valid | 3,328 | **3,332** (= paper's stated final total) |
| schedule_7 valid | 115 | **119** (= paper's N) |
| exclusion basis | inferred min-5-per-side | curator `_INVALID_BIAS` tag |

### Verified during planning

- `reward_alternative_1` / `reward_alternative_2` are **bit-identical** to
  `biased_reward` / `unbiased_reward` (checked against independent raw subjects, all
  100 trials).
- Organized subjects are a **subset of** the raw subjects for schedules 2 and 4 (zero
  orphans); `schedule_7` is the only schedule where organized contains participants our
  raw copy lacks — precisely the known gap.
- `observed_reward` is **100% derivable** as
  `where(chose_biased, biased_reward, unbiased_reward)` — confirmed exact across all
  148,300 training rows. Nothing is lost by the narrower column set.

---

## Column-coverage audit (blocking question answered)

The organized files carry only 4 columns. Every column the **active** pipeline reads:

| column | in organized data? | resolution |
|---|---|---|
| `trial_number` | yes | direct |
| `is_biased_choice` | as `is_choice_alternative_1` | rename |
| `biased_reward` | as `reward_alternative_1` | rename |
| `unbiased_reward` | as `reward_alternative_2` | rename |
| `observed_reward` | no | **derived** (verified exact) |
| `subject_file`, `schedule`, `split`, `subject_id` | n/a | added by sanitize |
| `chose_biased`, `c1` | n/a | derived in consuming code |

**Columns that disappear and are never read by active code:** `time`, `RT`, `RT_net`,
`RT_zscore`, `side_choice`, `unobserved_reward`, `BLUE_RIGHT_LEFT_RED`,
`catie_choice_probability`.

- `RT*` / `time` — reaction time is a permanently closed direction.
- `side_choice` — only ever used for the exclusion count, now superseded by the
  `_INVALID_BIAS` tag.
- `BLUE_RIGHT_LEFT_RED` — already established as constant within subject, so
  `side_choice ≡ is_biased_choice` up to relabeling. Closed direction.
- `catie_choice_probability` — the old stored MATLAB output. Fully superseded by
  `matlab/results/original_reference_all_schedules.csv`, which is regenerable, covers all
  12 schedules (not just EDA's 3), and holds both the hetero mixture and each single-k model.

**Conclusion: no active code loses anything.** The archived investigation scripts keep
using their own untouched copies (Phase A).

---

## Phase A — Rename the investigation folder

`my_code/EDA_set/` → **`my_code/initial_investigation/`**

This folder holds the first round of work (RT analysis, surprise analysis, calibration
analysis, early processing) and is **not** part of the active pipeline. It moves as-is;
its internal scripts keep pointing at their own files, which move with it.

It keeps its own `schedule_4/5/7` copies and `processing/eda_with_catie_probabilities.csv`
so its historical outputs remain reproducible. The active pipeline stops reading from it
entirely (Phase D).

**Files needing a path edit after the move:** the folder's own scripts
(`early_proccessing/sanitize.py:7` hardcodes the absolute `EDA_set` path;
`rt_analysis/rt_analysis.py`, `processing/compute_eda_catie_probabilities.m`,
`early_proccessing/Verification_script.py`, and 4 READMEs reference it). These are
archive-only; updating them keeps the archive runnable but changes no active result.

---

## Phase B — Copy the organized schedules into the four sets

From `Data_resources/.../simple_format_data/1. COMPETITION_data_static/`:

| destination | schedules | valid subjects |
|---|---|---|
| `my_code/Training_set/` | 2, 3, 6, 9, 11 | 1,483 |
| `my_code/Test_set/` | 1, 8, 10 | 804 |
| `my_code/EDA_set/` *(new, schedules only)* | 4, 5, 7 | **496** (was 492) |
| `my_code/Schedule0_set/` *(see note)* | 0 | 549 |

The `_INVALID_BIAS` files are copied too — the sanitizer needs to see and count them, and
their presence is what makes the exclusion auditable rather than implicit.

**Note on `schedule_0`.** It is currently *flat* (`my_code/schedule_0/*.csv`) while the
others are *nested*. Proposing `my_code/Schedule0_set/schedule_0/` so all four sets share
one layout, which is what lets Phase C delete the `layout` parameter entirely. Easy to
revert to flat if you would rather not move it — say so and I will keep both layouts.

The existing raw files are replaced. They are gitignored and byte-identical to
`Data_resources/.../raw_data/` (verified earlier), so they remain fully recoverable.

---

## Phase C — Rewrite `sanitize_splits.py` (simplify)

Currently 205 lines carrying two column schemas, two directory layouts, and a
`schedule_7_organized` one-off. Target: **one schema, one layout, four splits.**

- **Delete** the `SCHEMAS` dict and `schema` parameter — a single column set now.
- **Delete** the `layout` parameter and the flat branch — all sets nested.
- **Delete** the `schedule_7_organized` entry (it was a probe; superseded).
- **Exclusion**: `_INVALID_BIAS` filename tag is authoritative.
- **Keep `MIN_CHOICES_PER_SIDE = 5` as an assertion, not a filter** — it must now find
  *zero* subjects to drop. If it ever fires, the two exclusion criteria have diverged and
  that is a finding worth surfacing, not silently absorbing.
- **Derive** `observed_reward` on write, so Phase 2's `recent_reward_rate` keeps working
  unchanged.
- **Canonical output columns**: `subject_id`, `schedule`, `split`, `subject_file`,
  `trial_number`, `is_biased_choice`, `biased_reward`, `unbiased_reward`, `observed_reward`.
- **Comments trimmed** to what is still true: drop the "generalises EDA sanitize.py" framing
  and the Excel-damage warning about a file the active pipeline no longer reads (keep a
  short "never open these in Excel" line — the hazard is real for the *outputs*).

**Output** → `my_code/catie_calibration/data/`:

```
cleaned_training.csv     1,483 subjects
cleaned_test.csv           804
cleaned_eda.csv            496   <- NEW (replaces the old EDA source)
cleaned_schedule_0.csv     549
```
and **delete** `cleaned_schedule_7_organized.csv`.

---

## Phase D — Update every consumer

**Python — EDA data source** (all currently read
`EDA_set/processing/eda_with_catie_probabilities.csv`, all switch to
`data/cleaned_eda.csv`):

| file | line | change |
|---|---|---|
| `build_cache.py` | 43 | `EDA_CSV` → `DATA_DIR/"cleaned_eda.csv"`; the special-case EDA branch in `load_split` collapses into the normal path |
| `01_bug_correction/bug_benchmark.py` | 52 | `EDA_CSV` |
| `02_mode_calibration/conditional_calibration.py` | 79 | `EDA_CSV` |
| `02_mode_calibration/verify_cprev_separation.py` | 69 | `EDA_CSV` |
| `matlab/verify_state_tensors.py` | 39 | `EDA_CSV` |

**Python — `golden_test.py`:**
- `RAW_SCHEDULE_DIRS` (lines 62, 63, 65) → new `EDA_set` schedule paths; plus the
  `schedule_0` path if Phase B's `Schedule0_set` move is accepted.
- `REFERENCE` (line 75) → `data/cleaned_eda.csv`.
- `load_reference()` — drop `assert len(df) == 49_200` (now 49,600) and the
  `catie_choice_probability` assertion (column gone). Derive `c1` from `is_biased_choice`
  as today.
- Checks 2–5 are otherwise untouched; they only need `r1`, `r2`, `c1`.

**MATLAB — schedule directory tables:**

| file | lines |
|---|---|
| `matlab/export_original_reference_all_schedules.m` | 68, 69, 71 (+ schedule_0 at 66) |
| `matlab/run_bug_comparison_all_schedules.m` | 23, 70, 71, 73, 85 |
| `matlab/export_state_tensors.m` | 59, 60, 62 |

These read the raw per-subject CSVs directly, so they also need the **column rename**
(`is_choice_alternative_1` → the `is_biased_choice` they currently expect) and their
`req` column lists updated. `run_bug_comparison_all_schedules.m` additionally loses its
EDA-reference sanity check (line 85) — that check compared against the stored CSV we are
retiring; the live all-schedule reference now covers it strictly better.

**Comment-only edits:** `catie_core.py:14`, `metrics.py:203`, `sanitize_splits.py:4,200`.

**`.gitignore`:** update the three `EDA_set/schedule_*` entries; add
`my_code/Schedule0_set/` if adopted.

---

## Phase E — Re-run order

Each step depends on the previous. Times are from this session's measurements.

| # | command | ~time | why |
|---|---|---|---|
| 1 | `sanitize_splits.py` | 30 s | produces the 4 cleaned CSVs |
| 2 | `matlab -batch export_original_reference_all_schedules.m` | 3 min | subject set changed → reference is stale |
| 3 | `golden_test.py` | 2 min | **blocking gate** — nothing downstream is trusted until this passes |
| 4 | `build_cache.py` | ~30 min | state tensors for the new subject set |
| 5 | `01_bug_correction/bug_benchmark.py` | 2 min | Phase 1 numbers |
| 6 | `02_mode_calibration/conditional_calibration.py`, `verify_cprev_separation.py` | ~10 min | Phase 2 |
| 7 | `03_parameter_fitting/fit_parameters.py` | 15–20 min | Phase 3 fits |
| 8 | `03_parameter_fitting/validate_against_paper.py training` / `eda` | 1 min | **the payoff check** — schedule_7 should now be 119/119 |
| 9 | `matlab -batch export_state_tensors.m` → `verify_state_tensors.py` | 5 min | element-wise MATLAB cross-check |
| 10 | `matlab -batch run_bug_comparison_all_schedules.m` | 5 min | MATLAB-native bug comparison |

Per your decision: results are **overwritten in place**; old values stay recoverable
through git history.

---

## Phase F — Verification checklist

- [ ] `sanitize_splits.py` reports `few_choices: 0` on every split (the min-5 assertion
      never fires) and per-schedule counts equal the paper's N on all 12.
- [ ] `golden_test.py`: all 7 checks PASS, deviations still ~1e-15/1e-16.
- [ ] `verify_state_tensors.py`: `H`/`b`/`c_prev` exact 0.0, floats ≤ 5.55e-16, now over
      **999,600** trials (3,332 × 3 k-values) instead of 998,400.
- [ ] `validate_against_paper.py eda`: `schedule_7` reports **115 → 119**, matching the
      reported n exactly; the "4-subject gap" line disappears.
- [ ] Does the systematic **~+0.005 E[log p] residual** change? It is the project's most
      concrete open defect. The schedule_7 fix was previously bounded at ≤0.0028 impact, so
      it likely persists — but this is the first clean opportunity to re-measure it against
      the corrected population. **Record the before/after either way.**
- [ ] Phase 2's `recent_reward_rate` still computes (depends on derived `observed_reward`).
- [ ] No remaining active-code reference to `EDA_set` or
      `eda_with_catie_probabilities.csv` (`grep` should return only archive + docs).

---

## Phase G — Documentation updates

| file | update |
|---|---|
| `SCHEDULE_N_RECONCILIATION.md` | The `schedule_7` gap moves `[OPEN]` → **resolved**; add the organized-release explanation and the 3,386/3,332 totals now matching exactly |
| `REVIEW_PLAN.md` | `sanitize_splits.py` entry rewritten; the EDA-provenance `[OPEN]` items close; Excel-damage caveat no longer applies to active data |
| `catie_calibration/README.md` | new data flow and split sizes |
| `01_bug_correction/README.md`, `matlab/README.md` | subject counts (492→496, 3,328→3,332) |
| `initial_investigation/README.md` *(new)* | one paragraph: what this folder was, why it is archived, that its numbers predate the organized-data migration |

---

## Impact and risks

**Every EDA-based number in the project changes** (492 → 496 subjects). Affected: Phase 1
bug quantification, Phase 2 calibration/partition tables, Phase 3 fits and recalibration
controls, and all figures derived from them. The Training/Test/schedule_0 splits are
**unchanged in size** (1,483 / 804 / 549), so only EDA-conditioned results move.

| risk | mitigation |
|---|---|
| Test set is touched during re-runs | `build_cache.py` still excludes Test by construction (`SPLITS` omits it, plus a reachable assertion). Phase 1's existing test-split row is a pre-existing decision, unchanged by this migration. |
| A consumer reads a dropped column at runtime, not import time | The audit above is exhaustive over `["col"]` access in active code, but Phase E step 3 is the real gate — and step 6 exercises the one derived column. |
| MATLAB drivers silently read the wrong column | They fail loudly: the `req` column check rejects files whose columns don't match. |
| Old results become unreproducible | Raw data is byte-identical to `Data_resources/raw_data` (verified), the archive folder keeps its own copies, and git history holds the old outputs. |

**Not addressed here** (unchanged, still open): the ~+0.005 E[log p] systematic residual,
the Phase 1 test-split decision, and the Phase 3 audit items in `REVIEW_PLAN.md`.

---

# Execution record (2026-08-22)

All seven phases executed. **Golden test passes; every verification item met.**

## Outcome against the checklist

| check | result |
|---|---|
| `sanitize_splits.py` per-schedule N vs paper | **12/12 exact**, 3,332 total |
| min-5-per-side assertion | never fired — the `_INVALID_BIAS` tag is a strict superset |
| `golden_test.py` | **PASS**, all 7 checks, 3,332 subjects, deviations ~1e-15/1e-16 |
| `verify_state_tensors.py` | **PASS**, now 999,600 trials (was 998,400); H/b/c_prev exact 0.0, floats 5.55e-16 |
| `validate_against_paper.py eda` | `schedule_7` **119 = 119**; sample sizes now **3/3** (was 2/3) |
| MATLAB `run_bug_comparison` | 3,332 subjects; published **E[p] = 0.6190**, matching the paper's reported 0.619 exactly |
| Phase 2 `recent_reward_rate` (derived `observed_reward`) | ran clean |
| active-code refs to `EDA_set` / `eda_with_catie_probabilities.csv` | none remain |

## Data change

Only the EDA split moved: **492 → 496 subjects** (`schedule_7` 115 → 119).
Training (1,483), Test (804) and schedule_0 (549) are unchanged. Phase 1 EDA numbers
shifted in the 4th decimal, e.g. E[p]_fix 0.6146 → 0.6143.

## Two deviations from the plan

**1. `golden_test.py:_load_raw_subject` was missed in the Phase D audit.** The
column-coverage audit covered `["col"]` access across active code, but this helper reads
the *raw* per-subject files rather than a cleaned CSV, so it still expected
`biased_reward` / `is_biased_choice`. It failed loudly on the first run
(`KeyError: 'biased_reward'`) and was fixed to read the organized column names. No silent
wrong answer was possible — exactly the failure mode the plan's risk table predicted, and
it surfaced at the gate as intended.

**2. `fit_parameters.py`'s golden check had hardcoded Phase 1 numbers**, which went stale
the moment EDA changed (it asserted `eda E[p] = 0.6146`, now 0.6143, and aborted). Rather
than update the constants, it now **reads `01_bug_correction/figures/headline_metrics.csv`
directly**, so it can never go stale again. This closes the "golden-check should read from
CSV" item already listed in `REVIEW_PLAN.md`'s recommended actions.

## The E[log p] residual: hypothesis ruled out

The systematic ~+0.005 E[log p] gap against the paper **persists after the migration**:
training mean +0.0051 (max 0.0065), EDA mean +0.0046 (max 0.0055), same sign on every
schedule, while E[p] matches to ≤0.0018.

This is a real result, not a null one. The missing `schedule_7` subjects were the most
plausible remaining explanation for the gap; the population now matches the paper's
exactly on all 12 schedules and the residual is unchanged. **That hypothesis is
eliminated**, and the defect remains open — see `SCHEDULE_N_RECONCILIATION.md`.

## Not re-run

`initial_investigation/` (archived; intentionally frozen on its original data) and Phase 4
/ Phase 5, which do not exist yet.
