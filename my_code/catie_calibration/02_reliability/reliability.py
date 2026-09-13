"""
Reliability diagrams and calibration metrics for CATIE's forecast on Training and EDA.

Writes reliability.csv (one row per bin) and reliability_summary.csv (one row per
curve). No matplotlib import, no plotting -- see figures.py, which reads only these
two CSVs. No prints; this file's only output is the two CSVs.

Curves (see sub_plans/02_reliability.md for why each exists):
    baseline_biased   P(biased alt) vs outcome, training, 10 bins       -- superseded
    doubled           label-equivariant (p, y) + (1-p, 1-y), training, bin/strategy sweep
    doubled_run*      the doubled set stratified by run length, training
    doubled_eda       the doubled set on EDA, one replication panel

Run length: run[i, t] = length of the streak of identical choices ending at trial
t-1, i.e. a function of y[i, :t] only -- it cannot see the trial it stratifies.

e_p / e_log_p on every row are computed on p_choice (NOT the doubled p/y), over
exactly the trials in that row's split+stratum -- see the module docstring in
catie/metrics.py on why calibration and E[p]/E[log p] are scored on different
quantities.

Run:  cd my_code/catie_calibration && python 02_reliability/reliability.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # -> catie_calibration/
from catie.likelihood import StateCache, p_choice_matrix  # noqa: E402
from catie import metrics as M  # noqa: E402

HERE = pathlib.Path(__file__).parent
SEED = 42
BASE_BOOT = 2000    # curves quoted in the README
SWEEP_BOOT = 500    # bin-count sweep members, shown only to locate where bins empty out
RUN_STRATA = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, None, "10+")]


def compute_run_length(y: np.ndarray) -> np.ndarray:
    """run[i, t] = length of the streak of identical choices in y[i, :t] ending at
    index t-1. run[:, 0] is unused (trial 1 is never scored)."""
    n, n_trials = y.shape
    run = np.zeros((n, n_trials), dtype=np.int64)
    if n_trials > 1:
        run[:, 1] = 1
    for t in range(2, n_trials):
        same = y[:, t - 1] == y[:, t - 2]
        run[:, t] = np.where(same, run[:, t - 1] + 1, 1)
    return run


def load_flat(split: str) -> dict:
    """Scored trials (columns 1..99) of one split, flattened to 1-D arrays."""
    cache = StateCache(split)
    pc = p_choice_matrix(cache)                        # P(choice made), (n, 100)
    y = cache.y.astype(np.float64)
    p1 = np.where(cache.y.astype(bool), pc, 1.0 - pc)   # P(alt 1) -- round-trip of pc
    run = compute_run_length(cache.y)
    subj = np.repeat(cache.subject_id[:, None], cache.n_trials, axis=1)

    sl = slice(1, None)  # drop trial 1 -- fixed at p=0.5 by construction, not a forecast
    return dict(p1=p1[:, sl].ravel(), y=y[:, sl].ravel(), pc=pc[:, sl].ravel(),
                run=run[:, sl].ravel(), subj=subj[:, sl].ravel())


def double(p, y, subj):
    """The label-equivariant set: (p, y) and (1-p, 1-y), subject ids carried along
    so the cluster bootstrap resamples both copies of a trial together."""
    return (np.concatenate([p, 1.0 - p]), np.concatenate([y, 1.0 - y]),
            np.concatenate([subj, subj]))


def build_curves(tr: dict, eda: dict) -> list[dict]:
    curves = [dict(
        diagram="baseline_biased", split="training", stratum="all",
        p=tr["p1"], y=tr["y"], subj=tr["subj"], p_choice=tr["pc"],
        bins=[(10, "uniform", BASE_BOOT)],
    )]

    p_d, y_d, subj_d = double(tr["p1"], tr["y"], tr["subj"])
    curves.append(dict(
        diagram="doubled", split="training", stratum="all",
        p=p_d, y=y_d, subj=subj_d, p_choice=tr["pc"],
        bins=[(10, "uniform", BASE_BOOT), (20, "uniform", SWEEP_BOOT),
              (50, "uniform", SWEEP_BOOT), (100, "uniform", SWEEP_BOOT),
              (10, "quantile", SWEEP_BOOT), (20, "quantile", SWEEP_BOOT)],
    ))

    for lo, hi, label in RUN_STRATA:
        mask = (tr["run"] >= lo) & ((tr["run"] <= hi) if hi is not None else True)
        pm, ym, sm = double(tr["p1"][mask], tr["y"][mask], tr["subj"][mask])
        curves.append(dict(
            diagram=f"doubled_run{label}", split="training", stratum=label,
            p=pm, y=ym, subj=sm, p_choice=tr["pc"][mask],
            bins=[(10, "uniform", BASE_BOOT)],
        ))

    p_de, y_de, subj_de = double(eda["p1"], eda["y"], eda["subj"])
    curves.append(dict(
        diagram="doubled_eda", split="eda", stratum="all",
        p=p_de, y=y_de, subj=subj_de, p_choice=eda["pc"],
        bins=[(10, "uniform", BASE_BOOT)],
    ))
    return curves


def score_curves(curves: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bin_rows, summary_rows = [], []
    for c in curves:
        for n_bins, strategy, n_boot in c["bins"]:
            table = M.reliability_table_ci(c["p"], c["y"], c["subj"], n_bins=n_bins,
                                           strategy=strategy, n_boot=n_boot, seed=SEED)
            table = table.copy()
            table.insert(0, "diagram", c["diagram"])
            table.insert(1, "split", c["split"])
            table.insert(2, "stratum", c["stratum"])
            table.insert(3, "n_bins", n_bins)
            table.insert(4, "strategy", strategy)
            table["n_boot"] = n_boot
            bin_rows.append(table)

            ece_pt, ece_lo, ece_hi = M.ece_ci(c["p"], c["y"], c["subj"], n_bins=n_bins,
                                              strategy=strategy, n_boot=n_boot, seed=SEED)
            summary_rows.append({
                "diagram": c["diagram"], "split": c["split"], "stratum": c["stratum"],
                "n_bins": n_bins, "strategy": strategy,
                "n": len(c["p"]), "n_subjects": int(len(np.unique(c["subj"]))),
                "ece": ece_pt, "ece_lo": ece_lo, "ece_hi": ece_hi,
                "mce": M.mce(c["p"], c["y"], n_bins=n_bins, strategy=strategy),
                "brier": M.brier(c["p"], c["y"]),
                "e_p": M.e_p(c["p_choice"]), "e_log_p": M.e_log_p(c["p_choice"]),
                "n_boot": n_boot,
            })
    return pd.concat(bin_rows, ignore_index=True), pd.DataFrame(summary_rows)


def main() -> None:
    tr = load_flat("training")
    eda = load_flat("eda")
    curves = build_curves(tr, eda)
    bins_df, summary_df = score_curves(curves)
    bins_df.to_csv(HERE / "reliability.csv", index=False)
    summary_df.to_csv(HERE / "reliability_summary.csv", index=False)


if __name__ == "__main__":
    main()
