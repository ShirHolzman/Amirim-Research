# Step 0 — import refactor

Status: **done, 2026-09-09** — all seven checks passed. Outcome at the bottom.
Part of `../PLAN.md` step 0.

## Context

Commit `914c2a2` moved the shared modules into `catie/`, the two validation scripts into
`tests/catie/`, and the old Phase 2/3 work into `my_code/previous_errors/`. It
deliberately did not touch any import line, so **nothing in the project currently runs**.
This step makes the validated base runnable again and nothing else.

Two kinds of breakage, both caused by the move:

1. **Import names.** `catie_core` is now `catie/core.py`, `catie_likelihood` is
   `catie/likelihood.py`, `metrics` is `catie/metrics.py`.
2. **Path counts.** Several constants are written as a hand-counted
   `Path(__file__).parent.parent`. Files that moved a level down now resolve those to the
   wrong directory — silently, with no error. `catie/cache.py` would write to
   `catie/cache/` instead of `catie_calibration/cache/`; `catie/splits.py` would look for
   the raw subject folders inside `catie_calibration/`; `tests/catie/core_test.py` would
   look for `matlab/` and `data/` inside `tests/catie/`.

Not every path is wrong — four of them still resolve correctly and are listed below as
"no change" so nobody edits them by mistake.

## Scope

Mechanical only. **No logic, signature, output or behaviour change.** Nine files, about
twenty lines. No new modules, no new helpers, no shared paths file. Explicitly not in
this step: trimming `catie/metrics.py`, removing `print`/`Tee` from existing scripts (the
no-print rule is for the new stage scripts 02–06), touching `previous_errors/`, writing
any stage script.

## Changes, file by file

### `catie/likelihood.py`

| line | now | change to |
|---|---|---|
| 32 | `import sys` | delete — `sys` is used nowhere else in the file |
| 36 | `sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))` | delete |
| 37 | `from catie_core import EPSILON, PHI, TAU` | `from .core import EPSILON, PHI, TAU` |
| 39 | `CACHE_DIR = pathlib.Path(__file__).parent.parent / "cache"` | **no change** — already resolves to `catie_calibration/cache/` |

`import pathlib` stays (still used by `CACHE_DIR`).

### `catie/cache.py`

| line | now | change to |
|---|---|---|
| 31 | `import sys` | **keep** — `sys.exit(main())` at line 112 |
| 38 | `sys.path.insert(0, str(HERE))` | delete |
| 39 | `from catie_core import state_tensors` | `from .core import state_tensors` |
| 41 | `CACHE_DIR = HERE / "cache"` | `CACHE_DIR = HERE.parent / "cache"` |
| 42 | `DATA_DIR = HERE / "data"` | `DATA_DIR = HERE.parent / "data"` |

`HERE` keeps its meaning — the directory of this file — as in every other script.

### `catie/splits.py`

No import to fix (it never imported `catie_core`). Two path counts:

| line | now | change to |
|---|---|---|
| 32 | `MY_CODE = pathlib.Path(__file__).parent.parent` | `... .parent.parent.parent` |
| 33 | `OUT_DIR = pathlib.Path(__file__).parent / "data"` | `... .parent.parent / "data"` |

### `catie/core.py`, `catie/metrics.py`

No change. Neither imports anything from the project.

### Entry points

`cache.py` and `splits.py` keep their `main()` and `__main__` blocks, but because they
now use relative imports they are run as `python -m catie.cache` / `python -m catie.splits`
rather than as loose files. That is already what `README.md` documents.

### `tests/catie/core_test.py`

| line | now | change to |
|---|---|---|
| 51 | `sys.path.insert(0, str(pathlib.Path(__file__).parent))` | `sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))` |
| 52 | `from catie_core import (` | `from catie.core import (` |
| 60 | `PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent` | `PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[4]` |
| 61 | `MATLAB_DIR = pathlib.Path(__file__).parent / "matlab"` | `MATLAB_DIR = pathlib.Path(__file__).resolve().parents[2] / "matlab"` |
| 84 | `REFERENCE = pathlib.Path(__file__).parent / "data" / "cleaned_eda.csv"` | `... .resolve().parents[2] / "data" / "cleaned_eda.csv"` |

`parents[2]` is `catie_calibration/`, `parents[4]` is the repo root (`Amirim Research/`).
The 12-entry `PROJECT_ROOT / "my_code" / …` schedule map at lines 67–80 needs no edit
once `PROJECT_ROOT` is right.

### `tests/catie/likelihood_test.py`

| line | now | change to |
|---|---|---|
| 57–58 | two `sys.path.insert` calls (`HERE`, `HERE.parent`) | one call: `sys.path.insert(0, str(HERE.parents[1]))` |
| 60 | `from catie_likelihood import StateCache, mean_log_p, mean_p` | `from catie.likelihood import StateCache, mean_log_p, mean_p` |
| 61 | `from metrics import Tee` | `from catie.metrics import Tee` |
| 63 | `REPO = HERE.parent.parent.parent` | `REPO = HERE.parents[3]` |

`HERE` is `tests/catie/`, so `HERE.parents[1]` is `catie_calibration/` and
`HERE.parents[3]` is the repo root. `FIGDIR = HERE / "figures"` is left alone — its
output now lands in `tests/catie/figures/`, which is where it belongs.

### `tests/catie/metrics_test.py`

| line | now | change to |
|---|---|---|
| 34 | `sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))` | `... .resolve().parents[2]` |
| 35 | `import metrics as M` | `from catie import metrics as M` |

### `01_bug_correction/bug_benchmark.py`

| line | now | change to |
|---|---|---|
| 36 | `sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))` | **no change** — already `catie_calibration/` |
| 37 | `from catie_core import (` | `from catie.core import (` |
| 42 | `import metrics as M` | `from catie import metrics as M` |
| 51 | `DATA = HERE.parent / "data"` | **no change** — already correct |

`fig4_recreation.py` and `github_correction/make_figure_4.py` import nothing from the
project. Untouched.

### `matlab/verify_bma_mixing.py`

| line | now | change to |
|---|---|---|
| 51 | `sys.path.insert(0, str(HERE.parent))` | **no change** |
| 52 | `from catie_core import mix_agents, p_of_observed_choice` | `from catie.core import mix_agents, p_of_observed_choice` |

### `matlab/verify_state_tensors.py`

| line | now | change to |
|---|---|---|
| 34 | `sys.path.insert(0, str(HERE.parent))` | **no change** |
| 35 | `from catie_core import state_tensors` | `from catie.core import state_tensors` |
| 38 | `DATA_DIR = HERE.parent / "data"` | **no change** |

### Docstring `Run:` lines

Every module names its own invocation near the top of its docstring. Six of these still
point at the pre-move filenames and must be corrected in the same pass, or the file tells
the reader to run something that does not exist.

| file | line | now | change to |
|---|---|---|---|
| `catie/cache.py` | 25 | `Run:  python my_code/catie_calibration/build_cache.py` | `Run:  cd my_code/catie_calibration && python -m catie.cache` |
| `catie/splits.py` | 21 | `Run:  python my_code/catie_calibration/sanitize_splits.py` | `Run:  cd my_code/catie_calibration && python -m catie.splits` |
| `tests/catie/core_test.py` | 38–39 | `… /golden_test.py` (+ `--regenerate-matlab`) | `… /tests/catie/core_test.py` (+ `--regenerate-matlab`) |
| `tests/catie/likelihood_test.py` | 40 | `… /03_parameter_fitting/validate_against_paper.py` | `… /tests/catie/likelihood_test.py [training\|eda\|schedule_0]` |
| `tests/catie/metrics_test.py` | 8 | `… /tests/test_metrics.py` | `… /tests/catie/metrics_test.py`, and note `python -m pytest tests -q` from `catie_calibration/` as the usual route |

Already correct, no change: `01_bug_correction/bug_benchmark.py:21`,
`01_bug_correction/github_correction/make_figure_4.py:15-16`,
`matlab/verify_bma_mixing.py:40`, `matlab/verify_state_tensors.py:23`.

Note the one convention break: every other script is documented as run from the
repository root, but `catie/cache.py` and `catie/splits.py` are package modules, so
`-m` needs `catie_calibration/` as the working directory. Their `Run:` lines say so
explicitly rather than leaving the reader to discover it.

## Verification

In order; each must pass before the next.

1. `python -c "import catie, catie.core, catie.likelihood, catie.metrics, catie.cache, catie.splits"`
   from `catie_calibration/` — every module imports.
2. `python -c "from catie.cache import CACHE_DIR, DATA_DIR; print(CACHE_DIR, DATA_DIR)"`
   and the same for `catie.splits`' `MY_CODE` / `OUT_DIR` — the corrected anchors must
   print the real `catie_calibration/cache`, `catie_calibration/data` and `my_code/`,
   and those directories must exist. This is the specific silent bug being removed.
3. `python -m pytest tests -q` (system python — the venv has no pytest). Expect the 10
   `metrics_test.py` checks to pass.
4. `python tests/catie/core_test.py` — the golden gate, five validation layers. The
   45 MB `matlab/results/original_reference_all_schedules.csv` is present, so layer 1
   reuses it and MATLAB is not required. Tolerances must match what `README.md` quotes
   (2.331e-15, 6.661e-16, 3.3e-16).
5. `python tests/catie/likelihood_test.py training` and `… eda` — reproduces the paper's
   per-schedule E[p] / E[log p] from the existing `cache/state_*.npz`.
6. Every `Run:` line is now runnable as written: grep the tree for the pre-move names
   (`build_cache`, `sanitize_splits`, `golden_test`, `validate_against_paper`,
   `test_metrics`, `catie_core`, `catie_likelihood`) and expect zero hits outside
   `previous_errors/` and `.pyc` files.
7. **The decisive check.** `python 01_bug_correction/bug_benchmark.py`, then
   `git status --short 01_bug_correction/figures/`. Those seven files are tracked, so a
   clean status proves the refactor changed nothing numerically. Any diff is a failure
   and must be explained before this step is committed.

`catie/splits.py` and `catie/cache.py` are not rerun — they rebuild gitignored `data/`
and `cache/`, and check 5 already proves the existing cache is read correctly. Check 2
confirms they now point at the right places.

## Commit

One commit, after all seven checks pass and the user has approved:
`Refactor imports to the catie package`.

---

## Outcome (2026-09-09)

All seven checks passed; nothing about the science moved.

| check | result |
|---|---|
| 1 imports | pass |
| 2 path anchors | pass -- all five resolve to real directories |
| 3 `pytest tests -q` | 10 passed, all three files collect |
| 4 golden gate | pass -- pooled hetero 2.331e-15, per-k 7.772e-16, monolithic-vs-split 3.331e-16 |
| 5 paper reproduction | pass -- training 0.0005 / 0.0004, EDA 0.0003 / 0.0003 (tol 0.0006) |
| 6 stale names | zero outside `previous_errors/` |
| 7 `bug_benchmark.py` rerun | tracked figures byte-identical -- the refactor changed nothing numerically |

Two things the plan did not anticipate:

**`tests/catie/likelihood_test.py` read `sys.argv` at import time** (`SPLIT = sys.argv[1]
…`), which is harmless for a script but aborts pytest collection, since the file now sits
under `tests/` and ends in `_test.py`. Fixed by moving the read into a `select_split(argv)`
helper that `main()` calls; module-level defaults to `training`. Script behaviour unchanged.

**The `Run:` fix was too narrow.** Beyond the six run-lines, ~35 further references to the
pre-move filenames sat in docstrings and READMEs. All were corrected in the same pass
(39 replacements across 8 files) on the user's decision, so the grep in check 6 returns
clean rather than returning prose hits that have to be triaged by eye every time.

Two decisions taken during execution, both by the user:

- `README.md`'s "Getting started" claimed `pytest tests -q` ran the golden gate and the
  paper reproduction. It does not -- both are `main()` scripts with no `test_*` functions,
  so pytest collects them and runs nothing. All four commands are now listed separately.
- `tests/catie/figures/` (204 KB, 8 files) is **tracked**, matching
  `01_bug_correction/figures/`. It is the only durable artefact behind the "reproduces the
  paper to <=0.0006" claim, and it cannot be regenerated from a fresh clone (it needs the
  gitignored cache, and upstream of that the restricted participant data). Tracking it also
  gives the paper reproduction the same rerun-and-diff tripwire that made check 7 possible.
