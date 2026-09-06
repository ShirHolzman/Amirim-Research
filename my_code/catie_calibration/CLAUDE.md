# Rules for this codebase (read every session)

## Off-limits
- `my_code/previous_errors/` is archived. Never read, import, grep, run, quote or cite anything
  in it, and never use a number that came from it. If a fact is needed, recompute it in
  the current stage scripts. Numbers from the old Phase 2/3 work (E[p]/E[log p] ladders,
  streak facts, per-schedule fits, SEs) are unverified until a current script reproduces
  them.
- Never modify the original MATLAB under `Data_resources/competition_analysis-main/`.
  CATIE model logic is never re-implemented; `catie/core.py` is the one port, validated
  against those files (`matlab/README.md`, `tests/catie/core_test.py`).
- Never open generated CSVs in Excel.
- The Test split (schedules 1/8/10) is absent from `cache/` and is scored exactly once,
  in `06_test/`, after every other stage is frozen.
- Closed directions, do not reopen: reaction time; any feature reading the current
  trial's choice; any partition built from an E-step / posterior.

## Trusted base
`catie/` (core, likelihood, cache, splits, metrics), `matlab/`, `01_bug_correction/`,
`tests/`. Analysis stages `02_reliability/` … `06_test/` import from `catie/` only.

## How stage code is written
- One script per stage, ≤150 lines, one purpose, reads `cache/`, writes one CSV plus
  figures into its own `figures/`. **No `print`.** If a transcript is needed, write a
  file.
- Docstrings state only what the reader can verify from the code directly below;
  results and interpretation go in the stage README next to the CSV row they came from.
- Every stage script has a test in `tests/` mirroring its path, named `<file>_test.py`
  (`tests/02_reliability/reliability_test.py`), with checks that can actually fail.
- One script at a time. Show the user the script before it runs. Agents may check a
  script independently; agents do not write stage scripts. No parallel agents.
- Every number in a README points to one CSV row from one script.
- Fit on Training, score once on EDA, no selection on EDA. Bin edges and any threshold
  are fixed on Training and reused unchanged.

## Running
- Long runs: `python -u <script> > <stage>/figures/run.log 2>&1` in the background;
  stdout must go to a file or the process stalls. One heavy job at a time.
- Tests: `python -m pytest tests -q` from `my_code/catie_calibration/`.

## Framing
Dr. Ohad Dan is both the supervisor and the author of the MATLAB code in which bugs were
found. All findings are framed as reproduction, never critique.
