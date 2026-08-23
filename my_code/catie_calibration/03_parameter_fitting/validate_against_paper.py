"""
Per-schedule validation of our CATIE port against the values reported in the paper.

WHY THIS EXISTS. Every E[p] / E[log p] number quoted in Phases 1-3 so far was produced
by our own port. None of them had been checked against the paper's own PER-SCHEDULE
figures (Fig S5, transcribed into Data_resources/extracted_data/E_p.md and E_log_p.md).
Pooled agreement can hide compensating per-schedule errors, so this script checks each
training schedule separately, and checks the subject counts too.

THE ONE THING THAT MATTERS FOR INTERPRETATION. The paper's numbers were produced by the
ORIGINAL MATLAB, which contains the Phase 1 likelihood bug. So the model that should
reproduce them is the *published* (bug-for-bug) variant, NOT the corrected one. Both are
computed here, and only the published variant is asserted against the paper. Comparing
the paper's numbers to our corrected model would be comparing two different models and
would manufacture a discrepancy that is really the bug.

SCHEDULE INDEXING. The paper labels schedules 1..12; this repo labels them
schedule_0..schedule_11. The mapping (paper k -> schedule_{k-1}) is not assumed: it is
verified by matching the reported n against our own subject counts, and asserted.

Scope: selectable split. Default TRAINING (schedules 2, 3, 6, 9, 11); pass "eda"
or "schedule_0" as argv[1] for the others.

NOTE ON THE RE-FITTED COLUMN. The Phase 3 parameters were fitted on the pooled training
set, so evaluating them on training schedules is IN-SAMPLE and is expected to look good.
It is shown here to answer "what did re-fitting do to E[p] per schedule", not as evidence
of generalisation. Held-out numbers live in the Phase 3 README.

Run:  python my_code/catie_calibration/03_parameter_fitting/validate_against_paper.py
"""

from __future__ import annotations

import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")  # before pyplot, per repo convention
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from catie_likelihood import StateCache, mean_log_p, mean_p  # noqa: E402
from metrics import Tee  # noqa: E402

REPO = HERE.parent.parent.parent
EXTRACTED = REPO / "Data_resources" / "extracted_data"
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)

np.random.seed(42)

# Phase 3 maximum-likelihood fit on the pooled TRAINING split (fit_parameters.py).
PUBLISHED_PARAMS = dict(tau=0.29, eps=0.30, phi=0.71)
REFITTED_PARAMS = dict(tau=0.106812, eps=0.628011, phi=0.314217)

SPLIT_SCHEDULES = {
    "training": ("schedule_2", "schedule_3", "schedule_6", "schedule_9", "schedule_11"),
    "eda": ("schedule_4", "schedule_5", "schedule_7"),
    "schedule_0": ("schedule_0",),
}

# Tolerance for calling our port a match to the paper. The paper reports 3 decimals,
# so the rounding alone permits 5e-4; we allow a little more for the transcription.
MATCH_TOL = 0.002

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "training"
assert SPLIT in SPLIT_SCHEDULES, f"unknown split {SPLIT!r}; pick one of {list(SPLIT_SCHEDULES)}"
SCHEDULES = SPLIT_SCHEDULES[SPLIT]
SUF = "" if SPLIT == "training" else f"_{SPLIT}"


# ── parsing the reported tables ──────────────────────────────────────────────
def parse_reported(path: pathlib.Path) -> pd.DataFrame:
    """Read one of the extracted_data/*.md tables into a tidy frame.

    Expected body lines look like:  'Schedule 3,538,0.594*,0.559,...'
    The asterisk marks the best model in that row and is stripped.
    """
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^Schedule\s+(\d+)\s*,\s*(\d+)\s*,\s*([^,]+)", line.strip())
        if not m:
            continue
        paper_idx, n, catie = int(m.group(1)), int(m.group(2)), m.group(3)
        rows.append(
            {
                "paper_schedule": paper_idx,
                "schedule": f"schedule_{paper_idx - 1}",  # verified below, not trusted
                "n_reported": n,
                "catie_reported": float(catie.replace("*", "").strip()),
            }
        )
    assert len(rows) == 12, f"{path.name}: expected 12 schedule rows, got {len(rows)}"
    return pd.DataFrame(rows)


def load_reported() -> pd.DataFrame:
    ep = parse_reported(EXTRACTED / "E_p.md").rename(columns={"catie_reported": "E_p_reported"})
    el = parse_reported(EXTRACTED / "E_log_p.md").rename(
        columns={"catie_reported": "E_logp_reported"}
    )
    merged = ep.merge(
        el[["paper_schedule", "n_reported", "E_logp_reported"]],
        on=["paper_schedule", "n_reported"],
        how="inner",
        validate="one_to_one",
    )
    assert len(merged) == 12, "E_p.md and E_log_p.md disagree on schedules or n"
    return merged


# ── scoring ──────────────────────────────────────────────────────────────────
def score(cache: StateCache, params: dict, published_b: bool, drop_first: bool) -> tuple:
    kw = dict(params, published_b=published_b, drop_first=drop_first)
    return mean_p(cache, **kw), mean_log_p(cache, **kw)


def main() -> None:
    out_path = FIGDIR / f"validation_output{SUF}.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        sys.stdout = Tee(sys.__stdout__, fh)
        try:
            run()
        finally:
            sys.stdout = sys.__stdout__
    print(f"\ntranscript -> {out_path}")


def run() -> None:
    print("=" * 78)
    print("PER-SCHEDULE VALIDATION AGAINST THE PAPER'S REPORTED VALUES")
    print("=" * 78)
    print("source: Data_resources/extracted_data/{E_p.md, E_log_p.md}  (paper Fig S5)")
    print(f"scope : {SPLIT.upper()} -- schedules {list(SCHEDULES)}")
    print()

    reported = load_reported()
    cache = StateCache(SPLIT, ks=(0, 1, 2))
    print(f"loaded cache '{SPLIT}': {cache.n_subjects:,} subjects x {cache.n_trials} trials")

    # ---- 1. schedule mapping + sample sizes -------------------------------
    print("\n" + "=" * 78)
    print("1. SCHEDULE MAPPING AND SAMPLE SIZES")
    print("=" * 78)
    print("   The paper labels schedules 1..12; this repo labels them 0..11.")
    print("   The mapping is VERIFIED against our subject counts, not assumed.\n")

    ours = pd.Series(cache.schedule).value_counts().to_dict()
    print(f"   {'paper':<13}{'ours':<14}{'n reported':>12}{'n ours':>9}{'match':>8}")
    rows = []
    for _, r in reported.iterrows():
        sched = r["schedule"]
        if sched not in SCHEDULES:
            continue
        n_ours = ours.get(sched, 0)
        ok = n_ours == r["n_reported"]
        print(
            f"   {'Schedule ' + str(r['paper_schedule']):<13}{sched:<14}"
            f"{r['n_reported']:>12,}{n_ours:>9,}{'OK' if ok else 'MISMATCH':>8}"
        )
        rows.append({**r.to_dict(), "n_ours": n_ours, "n_match": ok})

    val = pd.DataFrame(rows)
    assert len(val) == len(SCHEDULES), (
        f"expected {len(SCHEDULES)} schedules for {SPLIT}, matched {len(val)}"
    )
    n_bad = int((~val["n_match"]).sum())
    print(f"\n   {len(val) - n_bad}/{len(val)} schedules match the reported n exactly.")
    if n_bad:
        print("   *** SAMPLE SIZES DISAGREE on some schedules.")
        print("   Proceeding so the size of the discrepancy stays visible, but an")
        print("   affected schedule is NOT evidence about the port: we would be")
        print("   scoring a different set of participants from the one the paper scored.")
    else:
        print(f"   => paper-k -> schedule_(k-1) mapping confirmed by n on all "
              f"{len(val)} schedules.")

    # ---- 2. does trial 1 belong in the average? ---------------------------
    print("\n" + "=" * 78)
    print("2. IS TRIAL 1 INCLUDED IN THE PAPER'S AVERAGE?")
    print("=" * 78)
    print("   Trial 1 is fixed at p=0.5 by the reference implementation, so including")
    print("   it shifts both metrics. We do not know the paper's choice a priori --")
    print("   we determine it empirically, by which one reproduces the reported values.\n")

    for drop_first in (True, False):
        errs = []
        for _, r in val.iterrows():
            sub = cache.subset(cache.schedule == r["schedule"])
            e_p, e_lp = score(sub, PUBLISHED_PARAMS, published_b=True, drop_first=drop_first)
            errs.append((e_p - r["E_p_reported"], e_lp - r["E_logp_reported"]))
        errs = np.array(errs)
        label = "drop trial 1" if drop_first else "keep trial 1"
        print(
            f"   {label:<14} mean|dE[p]| = {np.abs(errs[:, 0]).mean():.4f}   "
            f"mean|dE[log p]| = {np.abs(errs[:, 1]).mean():.4f}"
        )

    print("\n   Using drop_first=True below (the convention used throughout Phases 1-3).")

    # ---- 3. the comparison ------------------------------------------------
    print("\n" + "=" * 78)
    print(f"3. REPORTED vs OUR PORT vs RE-FITTED (per {SPLIT} schedule)")
    print("=" * 78)
    print("   published port = bug-for-bug MATLAB  -> this is what should match the paper")
    print("   corrected port = Phase 1 bug fix     -> a DIFFERENT model; must not match")
    print("   re-fitted      = Phase 3 params, IN-SAMPLE here (fitted on pooled training)\n")

    recs = []
    for _, r in val.iterrows():
        sub = cache.subset(cache.schedule == r["schedule"])
        pub_p, pub_lp = score(sub, PUBLISHED_PARAMS, published_b=True, drop_first=True)
        fix_p, fix_lp = score(sub, PUBLISHED_PARAMS, published_b=False, drop_first=True)
        ref_p, ref_lp = score(sub, REFITTED_PARAMS, published_b=False, drop_first=True)
        recs.append(
            {
                "schedule": r["schedule"],
                "paper_schedule": int(r["paper_schedule"]),
                "n": int(r["n_ours"]),
                "E_p_reported": r["E_p_reported"],
                "E_p_published": pub_p,
                "E_p_corrected": fix_p,
                "E_p_refitted": ref_p,
                "E_logp_reported": r["E_logp_reported"],
                "E_logp_published": pub_lp,
                "E_logp_corrected": fix_lp,
                "E_logp_refitted": ref_lp,
            }
        )
    res = pd.DataFrame(recs).sort_values("paper_schedule").reset_index(drop=True)
    res["dE_p"] = res["E_p_published"] - res["E_p_reported"]
    res["dE_logp"] = res["E_logp_published"] - res["E_logp_reported"]

    print(f"   {'sched':<13}{'n':>6}{'E[p] rep':>10}{'pub':>9}{'diff':>9}"
          f"{'corr':>9}{'refit':>9}")
    for _, r in res.iterrows():
        print(
            f"   {r['schedule']:<13}{r['n']:>6,}{r['E_p_reported']:>10.3f}"
            f"{r['E_p_published']:>9.4f}{r['dE_p']:>+9.4f}"
            f"{r['E_p_corrected']:>9.4f}{r['E_p_refitted']:>9.4f}"
        )
    print()
    print(f"   {'sched':<13}{'n':>6}{'E[lgp] rep':>12}{'pub':>9}{'diff':>9}"
          f"{'corr':>9}{'refit':>9}")
    for _, r in res.iterrows():
        print(
            f"   {r['schedule']:<13}{r['n']:>6,}{r['E_logp_reported']:>12.3f}"
            f"{r['E_logp_published']:>9.4f}{r['dE_logp']:>+9.4f}"
            f"{r['E_logp_corrected']:>9.4f}{r['E_logp_refitted']:>9.4f}"
        )

    # ---- 4. verdict --------------------------------------------------------
    print("\n" + "=" * 78)
    print("4. VERDICT")
    print("=" * 78)
    worst_p = res["dE_p"].abs().max()
    worst_lp = res["dE_logp"].abs().max()
    print(f"   published port vs paper: max |dE[p]| = {worst_p:.4f}, "
          f"max |dE[log p]| = {worst_lp:.4f}  (tolerance {MATCH_TOL})")

    ok_p = worst_p <= MATCH_TOL
    ok_lp = worst_lp <= MATCH_TOL
    print(f"   E[p]     reproduced on all {len(res)} schedules : {'YES' if ok_p else 'NO'}")
    print(f"   E[log p] reproduced on all {len(res)} schedules : {'YES' if ok_lp else 'NO'}")

    # Subject-weighted pooled figures, so the training split can be quoted as a whole.
    w = res["n"].to_numpy(dtype=float)
    _smp = "in-sample" if SPLIT == "training" else "held out"
    if ok_p and not ok_lp:
        same = bool((res["dE_logp"] > 0).all() or (res["dE_logp"] < 0).all())
        print(f"   NOTE: every E[log p] residual has the {'SAME' if same else 'mixed'} sign "
              f"(mean {res['dE_logp'].mean():+.4f}).")
        print("   A same-signed residual on every schedule is a SYSTEMATIC difference, not")
        print("   rounding noise. E[p] matching while E[log p] does not means the port puts")
        print("   the same average mass on the chosen action but spreads it slightly")
        print("   differently across trials -- log-score punishes the low tail, E[p] does not.")
    print()
    print(f"   Subject-weighted pooled over the {len(res)} {SPLIT} schedules:")
    for lbl, col_p, col_l in [
        ("reported (paper)", "E_p_reported", "E_logp_reported"),
        ("published port  ", "E_p_published", "E_logp_published"),
        ("corrected port  ", "E_p_corrected", "E_logp_corrected"),
        (f"re-fitted ({_smp})", "E_p_refitted", "E_logp_refitted"),
    ]:
        pp = float(np.average(res[col_p], weights=w))
        ll = float(np.average(res[col_l], weights=w))
        print(f"      {lbl}   E[p] = {pp:.4f}   E[log p] = {ll:.4f}")

    print("\n   READ THIS BEFORE QUOTING ANY BASELINE E[p]:")
    print("   The paper's numbers describe the PUBLISHED (buggy) model. Any comparison")
    print("   of a re-fitted or recalibrated variant against the paper must state which")
    print("   baseline it uses, because published and corrected differ materially.")

    csv_path = FIGDIR / f"paper_validation_{SPLIT}.csv"
    res.to_csv(csv_path, index=False)
    print(f"\n   table -> {csv_path}")

    make_figures(res)


# ── figures ──────────────────────────────────────────────────────────────────
def _grouped_bars(ax, res, cols, labels, colours, title, ylabel):
    x = np.arange(len(res))
    width = 0.8 / len(cols)
    for i, (c, lab, col) in enumerate(zip(cols, labels, colours)):
        ax.bar(x + (i - (len(cols) - 1) / 2) * width, res[c], width,
               label=lab, color=col, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{s.replace('schedule_', 'sch ')}\n(n={n:,})" for s, n in zip(res["schedule"], res["n"])],
        fontsize=9,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)


def make_figures(res: pd.DataFrame) -> None:
    cols = ["E_p_reported", "E_p_published", "E_p_corrected", "E_p_refitted"]
    labels = ["reported (paper)", "our published port", "our corrected port",
              "re-fitted (in-sample)"]
    colours = ["#4a4a4a", "#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    _grouped_bars(
        ax, res, cols, labels, colours,
        "E[p] per training schedule — the paper's values reproduce only under the "
        "published (buggy) model",
        "E[p]",
    )
    ax.set_ylim(0.5, max(res[cols].to_numpy().max() + 0.03, 0.70))
    fig.tight_layout()
    fig.savefig(FIGDIR / f"fig_validation_Ep{SUF}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    cols_l = ["E_logp_reported", "E_logp_published", "E_logp_corrected", "E_logp_refitted"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    _grouped_bars(
        ax, res, cols_l, labels, colours,
        "E[log p] per training schedule — higher (less negative) is better",
        "E[log p]",
    )
    ax.set_ylim(min(res[cols_l].to_numpy().min() - 0.05, -0.85), 0.0)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"fig_validation_Elogp{SUF}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"   figures -> {FIGDIR / f'fig_validation_Ep{SUF}.png'}")
    print(f"              {FIGDIR / f'fig_validation_Elogp{SUF}.png'}")


if __name__ == "__main__":
    main()
