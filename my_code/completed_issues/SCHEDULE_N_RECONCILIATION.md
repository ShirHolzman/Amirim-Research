# Subject-count (N) reconciliation against the paper

Companion to `sanitize_splits.py`. Compares the number of valid subjects our
pipeline keeps per schedule against the paper's own reported N.

> **RESOLVED (2026-08-22).** This document originally recorded a 4-subject gap on
> `schedule_7` (115 local vs 119 reported) as `[OPEN]`. The project has since migrated
> to the competition's **organized release**
> (`Data_resources/COMPETITION_data_restricted/simple_format_data/`), which contains
> 122 `schedule_7` files — 4 more than the raw dump — of which 3 are explicitly tagged
> `..._INVALID_BIAS.csv` by the curators. **All 12 schedules now match the paper
> exactly, with a 3,332 total.** The migration is documented in
> `REORGANIZATION_PLAN.md` (alongside this file); the analysis below is retained because
> it is what located the gap and because the arithmetic still explains *why* the raw dump
> was short.
>
> **Archived.** Moved to `my_code/completed_issues/` once the issue closed. Unqualified
> file paths below (`sanitize_splits.py`, `03_parameter_fitting/…`) are relative to
> **`my_code/catie_calibration/`**, where this document used to live. Note also that the
> "Sources" and "Full reconciliation" sections below describe the **pre-migration** raw
> dump — see "Post-migration result" at the end for the current state.

## Sources

- **Our raw data**: `Data_resources/COMPETITION_data_restricted/raw_data/
  1. COMPETITION_data_raw_static/schedule_{0..11}/`. Verified byte-for-byte
  identical (`diff -rq`) to `my_code/{Training_set,Test_set,schedule_0}` and
  `my_code/EDA_set/schedule_{4,5,7}` for all 12 schedules — the files our
  pipeline reads are unmodified copies of the canonical source, not something
  altered upstream of `sanitize.py` / `sanitize_splits.py`.
- **Our valid N**: `sanitize_splits.py`'s exclusion rule (drop a subject if
  `side_choice.value_counts().min() < 5`) reproduced directly against the raw
  files above (independent of the already-written `cleaned_*.csv` / EDA CSV,
  as a check against silent drift).
- **Paper's N**: `Data_resources/Supplementary Data 1. Reward schedules.docx`
  — the "N" column, per schedule ID.
- **Paper's totals**: `Data_resources/Behavior engineering using quantitative
  reinforcement learning models.pdf`, p.2 ("we tested the effectiveness of 11
  different reward schedules on **3386** human participants") and p.9,
  Methods §Sample size, Phase 3 ("This increased the total number of
  participants to **3332**"). `3386 − 3332 = 54`, which is exactly the
  exclusion count `sanitize_splits.py`'s docstring already cites ("54/3386 =
  1.6%") — so that docstring claim is now traced to a primary source, not
  just asserted.

## Indexing note — do not reuse the Phase-3 schedule mapping here

`Supplementary Data 1`'s schedule ID runs **0–11** and **already equals the
repo's `schedule_N` number directly** (ID 0 → `schedule_0`, … ID 11 →
`schedule_11`). This is a **different** table from Fig. S5 (`E_p.md` /
`E_log_p.md`, used in `03_parameter_fitting/validate_against_paper.py`),
which labels schedules **1–12** and requires the shift
`paper label k (1–12) → schedule_{k-1} (0–11)`.

Both directions re-confirmed directly, not assumed:
- **This table (0–11):** applying the 1–12 shift here breaks 11 of 12
  matches below; the direct (unshifted) correspondence is the one that
  reproduces the paper's numbers.
- **Fig. S5 (1–12):** cross-checked every row of `E_p.md` against this
  table's N column — `E_p.md`'s "Schedule 1" (n=549) through "Schedule 12"
  (n=87) match `schedule_0` (549) through `schedule_11` (87) exactly under
  the shift, confirming Fig. S5 genuinely uses the shifted 1–12 convention
  and is not a second instance of the direct 0–11 one.

## Full reconciliation

| Schedule | Split | Paper N (Supp. Data 1) | Our raw files | Excluded (ours) | Our valid N | Gap |
|---|---|---:|---:|---:|---:|---:|
| schedule_0 | schedule_0 | 549 | 556 | 7 | 549 | 0 |
| schedule_1 | test | 595 | 601 | 6 | 595 | 0 |
| schedule_2 | training | 538 | 547 | 9 | 538 | 0 |
| schedule_3 | training | 607 | 611 | 4 | 607 | 0 |
| schedule_4 | eda | 201 | 208 | 7 | 201 | 0 |
| schedule_5 | eda | 176 | 179 | 3 | 176 | 0 |
| schedule_6 | training | 144 | 148 | 4 | 144 | 0 |
| **schedule_7** | **eda** | **119** | **118** | **3** | **115** | **4** |
| schedule_8 | test | 116 | 119 | 3 | 116 | 0 |
| schedule_9 | training | 107 | 110 | 3 | 107 | 0 |
| schedule_10 | test | 93 | 95 | 2 | 93 | 0 |
| schedule_11 | training | 87 | 90 | 3 | 87 | 0 |
| **Total** | | **3332** | **3382** | **54** | **3328** | **4** |

Every drop reason is `few_choices` (the `min(choices per side) < 5` rule);
zero drops anywhere from missing columns, bad trial counts, or read errors.

**11 of 12 schedules match the paper's N exactly.** The excluded-count total
(54) matches the paper's own aggregate exclusion (3386 − 3332 = 54) exactly.
**The entire 4-subject discrepancy is confined to `schedule_7`.**

## `schedule_7` — the one gap

- Paper reports **N = 119** (Supp. Data 1, ID 7).
- Our valid N is **115** (492-subject EDA set = 201 + 176 + 115).
- Our raw archive holds **118 files** for `schedule_7` — already one short of
  119 before any exclusion is applied — and this 118 is confirmed
  byte-for-byte identical to the canonical raw-data folder, so it is not an
  artifact of our processing.
- Applying the standard exclusion rule to those 118 files drops 3 more
  (`few_choices = 3`), landing at 115.

**Arithmetic check (not an independently published per-schedule figure —
the paper gives only the aggregate raw total, 3386):** summing our raw file
counts across the other 11 schedules gives 3382 − 118 = 3264. The paper's
aggregate raw total is 3386, so it implies a `schedule_7` raw count of
3386 − 3264 = **122** — i.e., our archive is short by exactly 4 raw files
for this one schedule, and 4 is precisely the observed N gap. This is
consistent with (though not direct proof of) the interpretation that
**4 `schedule_7` subject files are missing from the archived raw data**
(cause undocumented — no `invalid_bias`-tagged files exist anywhere in the
static raw-data tree, so it is not an explicit-exclusion artifact), rather
than any divergence in the exclusion logic itself.

## Post-migration result (2026-08-22)

After migrating to the organized release and re-running the full pipeline:

| check | before | after |
|---|---|---|
| `validate_against_paper.py eda` sample sizes | 2/3 exact (schedule_7 115 vs 119) | **3/3 exact** |
| total valid subjects | 3,328 | **3,332** (= paper) |
| `verify_state_tensors.py` coverage | 998,400 trials | **999,600 trials**, still exact on H/b/c_prev, 5.55e-16 on floats |
| `golden_test.py` | PASS | **PASS** (all 12 schedules, ~1e-15) |

**The systematic E[log p] residual persists**, as anticipated: every EDA schedule is still
off by the same sign, mean **+0.0046** (was +0.0040 across the old 3-schedule set), against
E[p] matching to ≤0.0018. So the ~0.005 gap is *not* explained by the missing subjects —
this migration rules that hypothesis out rather than confirming it. It remains the
project's most concrete open defect.

## Downstream relevance

`schedule_7` is one of the three EDA schedules. Phase 3's
`validate_against_paper.py` already bounded this gap's impact on E[log p] at
≤0.0028 (`03_parameter_fitting/README.md`) — smaller than the ~0.005
systematic residual discussed there, so it does not explain that residual on
its own. This document exists to record *why* the gap exists and that it is
fully accounted for, not to reopen that analysis.
