# EDA Set — Early Preprocessing

This folder contains the scripts that transform the raw per-subject CSV files
(in `EDA_set/schedule_4/`, `schedule_5/`, `schedule_7/`) into the single
cleaned dataset used for all downstream analysis.

## Scripts

### `sanitize.py` — Main cleaning pipeline
Loads all raw CSV files from the three schedule folders and produces
`EDA_set/cleaned_eda_data.csv`. Steps applied:

1. **Exclusion**: participants who chose one side fewer than 5 times out of 100
   trials are excluded (likely ignored the reward schedule).
2. **RT floor**: reaction times below 1 500 ms are clipped to 1 500 ms (removes
   implausibly fast responses caused by pre-emptive key-holds).
3. **Hardware delay correction**: a fixed hardware offset is subtracted from each
   RT to obtain `RT_net`.
4. **Z-score normalisation**: `RT_zscore` is computed per participant so that
   reaction times are comparable across individuals.
5. **Consolidation**: all retained participants are stacked into a single CSV
   with a `subject_file` column identifying each participant and a `schedule`
   column identifying their assigned reward schedule.

### `Verification_script.py` — Audit / validation run
Runs the same pipeline as `sanitize.py` with additional verification logging:
tracks and prints each excluded file together with the reason for exclusion,
and spot-samples a few records from both normal and anomalous RT ranges to
validate that the corrections were applied correctly. Output is saved to
`cleaned_eda_master.csv` (same content as `cleaned_eda_data.csv`; kept
separately for auditing purposes).

## Output files

| File | Location | Description |
|------|----------|-------------|
| `cleaned_eda_data.csv` | `EDA_set/` | **Primary dataset used for all analysis.** 49 200 rows (492 subjects × 100 trials). |
| `cleaned_eda_master.csv` | this folder | Identical output produced by the verification run; kept for audit trail. |

## Column dictionary for `cleaned_eda_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `trial_number` | int (0–99) | Trial index within the session |
| `time` | string | Elapsed session time at trial onset (MM:SS.S) |
| `is_biased_choice` | bool | TRUE when the subject chose the reward-biased alternative |
| `side_choice` | string | Physical side chosen (LEFT / RIGHT) |
| `RT` | float (ms) | Raw reaction time (floored at 1 500 ms) |
| `observed_reward` | int (0/1) | Reward actually received |
| `unobserved_reward` | int (0/1) | Reward that would have been received for the unchosen option |
| `biased_reward` | int (0/1) | Reward scheduled for the biased alternative at this trial |
| `unbiased_reward` | int (0/1) | Reward scheduled for the unbiased alternative at this trial |
| `BLUE_RIGHT_LEFT_RED` | bool | Screen layout flag (TRUE = blue button on right) |
| `RT_net` | float (ms) | RT after hardware delay subtraction |
| `RT_zscore` | float | RT standardised within participant |
| `subject_file` | string | Unique participant identifier (original filename) |
| `schedule` | string | Reward schedule assigned to this participant (schedule_4 / 5 / 7) |

## How to re-run

```bash
cd my_code/EDA_set/early_proccessing
python sanitize.py          # produces ../cleaned_eda_data.csv
python Verification_script.py  # produces cleaned_eda_master.csv
```

A Python virtual environment (`.venv/`) is provided but excluded from version
control. Re-create it with:

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install pandas numpy
```
