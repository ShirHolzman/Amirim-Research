"""
Phase 1 -- isolate and quantify the published CATIE likelihood bug.

Compares two models that differ in exactly one expression:

  published  Data_resources/competition_analysis-main/CATIE/
             COMPETITION_CATIE_schedule_choice_probability.m as written. Lines 108-109
             read `pays(trial)`, still NaN at that point (preallocated line 16,
             assigned line 147). MATLAB evaluates NaN comparisons as false, so the
             trend/heuristic branch can never select alternative 1 -- while line 114
             still reduces p_try_explore to (1 - tau). Net effect: on every
             trend-testable trial, tau of the probability mass is handed to
             alternative 2 regardless of what the trend actually indicates.

  fixed      The same code with the three index errors corrected, matching the
             generative simulator CATIE_schedule_1.m:152-157.

No parameters are re-fitted. tau, epsilon, phi are held at their published values
throughout, so every difference reported here is attributable to the bug alone.

Run:  python my_code/catie_calibration/01_bug_correction/bug_benchmark.py
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # must precede pyplot; keeps the script headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from catie_core import (  # noqa: E402
    EPSILON, PHI, TAU,
    mix_agents, mode_contributions, p_of_observed_choice,
    probability_from_state, state_tensors,
)
import metrics as M  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

RNG_SEED = 42
KS = (0, 1, 2)
HERE = pathlib.Path(__file__).parent
OUT_DIR = HERE / "figures"
DATA = HERE.parent / "data"
EDA_CSV = HERE.parent.parent / "EDA_set" / "processing" / "eda_with_catie_probabilities.csv"

PALETTE = {"published": "#d62728", "fixed": "#1f77b4"}


# ── data loading ─────────────────────────────────────────────────────────────────
def load_split(name):
    """Return a tidy frame with subject_id, schedule, choices and reward schedules."""
    if name == "eda":
        df = pd.read_csv(EDA_CSV)
        df["subject_id"] = df["schedule"] + "/" + df["subject_file"]
    else:
        df = pd.read_csv(DATA / f"cleaned_{name}.csv")
    df["c1"] = df["is_biased_choice"].astype(str).str.upper() == "TRUE"
    df = df.sort_values(["subject_id", "trial_number"]).reset_index(drop=True)
    sizes = df.groupby("subject_id").size()
    assert (sizes == 100).all(), f"{name}: expected 100 trials/subject"
    return df


def run_split(df):
    """Score published and fixed CATIE over one split. Returns a per-trial frame."""
    out = []
    for sid, d in df.groupby("subject_id", sort=False):
        r1 = d["biased_reward"].to_numpy()
        r2 = d["unbiased_reward"].to_numpy()
        c1 = d["c1"].to_numpy()

        # One state pass per k serves both models -- they differ only in `b`.
        states_fixed = [state_tensors(r1, r2, c1, k=k, mode="fixed") for k in KS]
        states_pub = [s.as_published() for s in states_fixed]

        P_fix = np.vstack([probability_from_state(s, TAU, EPSILON, PHI) for s in states_fixed])
        P_pub = np.vstack([probability_from_state(s, TAU, EPSILON, PHI) for s in states_pub])

        p1_fix = mix_agents(P_fix, c1)
        p1_pub = mix_agents(P_pub, c1)

        heur_fix = mode_contributions(states_fixed[2])[0]
        out.append(pd.DataFrame({
            "subject_id": sid,
            "schedule": d["schedule"].to_numpy(),
            "trial_number": d["trial_number"].to_numpy(),
            "chose_biased": c1.astype(int),
            "p1_pub": p1_pub,
            "p1_fix": p1_fix,
            "pc_pub": p_of_observed_choice(p1_pub, c1),
            "pc_fix": p_of_observed_choice(p1_fix, c1),
            "trend_testable": states_fixed[2].H,
            "heuristic_fires": (heur_fix > 0).astype(int),
        }))
    return pd.concat(out, ignore_index=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = open(OUT_DIR / "output.txt", "w", encoding="utf-8")
    sys.stdout = M.Tee(sys.__stdout__, log)

    print("=" * 78)
    print("PHASE 1 -- PUBLISHED vs BUG-FIXED CATIE LIKELIHOOD")
    print("=" * 78)
    print(f"parameters held fixed at published values: "
          f"tau={TAU}, epsilon={EPSILON}, phi={PHI}, k in {KS}")
    print("no re-fitting; every difference below is attributable to the bug alone\n")

    splits = ["eda", "training", "test", "schedule_0"]
    results = {}
    for name in splits:
        print(f"scoring {name} ...", flush=True)
        df = load_split(name)
        results[name] = run_split(df)
        print(f"   {results[name]['subject_id'].nunique()} subjects, "
              f"{len(results[name]):,} trials")

    allr = pd.concat([r.assign(split=s) for s, r in results.items()], ignore_index=True)
    allr.to_csv(OUT_DIR / "bug_comparison_per_trial.csv.gz", index=False,
                compression="gzip")

    # ── 1. headline metrics ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("1. HEADLINE METRICS BY SPLIT")
    print("=" * 78)
    print(f"{'split':<12}{'subj':>6}{'trials':>9}"
          f"{'E[p] pub':>10}{'E[p] fix':>10}{'d':>8}"
          f"{'Elogp pub':>11}{'Elogp fix':>11}{'d':>9}")
    rows = []
    for name in splits:
        r = results[name]
        ep_p, ep_f = M.e_p(r.pc_pub), M.e_p(r.pc_fix)
        el_p, el_f = M.e_log_p(r.pc_pub), M.e_log_p(r.pc_fix)
        print(f"{name:<12}{r.subject_id.nunique():>6}{len(r):>9,}"
              f"{ep_p:>10.4f}{ep_f:>10.4f}{ep_f - ep_p:>+8.4f}"
              f"{el_p:>11.4f}{el_f:>11.4f}{el_f - el_p:>+9.4f}")
        rows.append({"split": name, "E[p]_pub": ep_p, "E[p]_fix": ep_f,
                     "E[logp]_pub": el_p, "E[logp]_fix": el_f})
    pd.DataFrame(rows).to_csv(OUT_DIR / "headline_metrics.csv", index=False)

    print("\nreference (Dan, Plonsky & Loewenstein 2025, Tables S1/S2, pooled over all")
    print("12 schedules): CATIE E[p]=0.619, E[log p]=-0.678; best QL E[log p]=-0.569.")
    print("Per-split values above are NOT comparable to that pooled figure -- see the")
    print("pooled row below, which is.")

    # ── 2. pooled, comparable to the paper ───────────────────────────────────────
    print("\n" + "=" * 78)
    print("2. POOLED ACROSS ALL SCHEDULES (comparable to Tables S1/S2)")
    print("=" * 78)
    ep_p, ep_f = M.e_p(allr.pc_pub), M.e_p(allr.pc_fix)
    el_p, el_f = M.e_log_p(allr.pc_pub), M.e_log_p(allr.pc_fix)
    print(f"   subjects {allr.subject_id.nunique():,}   trials {len(allr):,}   "
          f"schedules {allr.schedule.nunique()}")
    print(f"   E[p]     published {ep_p:.4f}   fixed {ep_f:.4f}   delta {ep_f - ep_p:+.4f}")
    print(f"   E[log p] published {el_p:.4f}   fixed {el_f:.4f}   delta {el_f - el_p:+.4f}")

    # ── 3. subject-clustered significance ────────────────────────────────────────
    print("\n" + "=" * 78)
    print("3. PAIRED TESTS, CLUSTERED BY SUBJECT (10 000 bootstrap resamples)")
    print("=" * 78)
    print("   trials within a subject are highly dependent; resampling trials rather")
    print("   than subjects would overstate significance by roughly sqrt(100) = 10x\n")
    for label, a, b in [("E[p]", allr.pc_fix, allr.pc_pub),
                        ("E[log p]", np.log(np.clip(allr.pc_fix, 1e-12, 1)),
                         np.log(np.clip(allr.pc_pub, 1e-12, 1)))]:
        res = M.paired_subject_test(a, b, allr.subject_id, seed=RNG_SEED)
        print(f"   {label:<10} fixed - published = {res['mean_diff']:+.4f}  "
              f"95% CI [{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]  "
              f"t({res['n_subjects'] - 1})={res['t']:.2f}  "
              f"p={res['p']:.3e} {M.stars(res['p'])}  dz={res['cohens_dz']:.3f}")

    # ── 4. how often the branch fires ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("4. TREND/HEURISTIC BRANCH ACTIVITY")
    print("=" * 78)
    print(f"{'split':<12}{'testable':>10}{'% trials':>10}{'fires (fixed)':>15}{'% trials':>10}")
    for name in splits:
        r = results[name]
        t, f = int(r.trend_testable.sum()), int(r.heuristic_fires.sum())
        print(f"{name:<12}{t:>10,}{100 * t / len(r):>9.2f}%{f:>15,}{100 * f / len(r):>9.2f}%")
    t, f = int(allr.trend_testable.sum()), int(allr.heuristic_fires.sum())
    print(f"{'POOLED':<12}{t:>10,}{100 * t / len(allr):>9.2f}%{f:>15,}{100 * f / len(allr):>9.2f}%")
    print("\n   'testable' = same choice on t-1 and t-2 with differing payoffs, so the")
    print("   trend rule is applicable. Under the published code the branch contributes")
    print("   exactly zero to P(alt 1) on every one of these trials, while still")
    print("   diverting tau = 0.29 of the mass away from it.")

    # ── 5. where the two models disagree ─────────────────────────────────────────
    print("\n" + "=" * 78)
    print("5. MAGNITUDE OF DISAGREEMENT")
    print("=" * 78)
    d = (allr.p1_fix - allr.p1_pub).abs()
    print(f"   |delta P(alt 1)|   mean {d.mean():.4f}   median {d.median():.4f}   "
          f"max {d.max():.4f}")
    print(f"   trials with any disagreement (>1e-9): {(d > 1e-9).sum():,} "
          f"({100 * (d > 1e-9).mean():.2f}%)")
    for thr in (0.05, 0.10, 0.20):
        print(f"   |delta| > {thr:.2f}: {(d > thr).sum():,} ({100 * (d > thr).mean():.2f}%)")

    print("\n   low-probability tail (the quantity that drives E[log p]):")
    for thr in (0.05, 0.10, 0.15):
        a, b = (allr.pc_pub < thr).sum(), (allr.pc_fix < thr).sum()
        print(f"   p(choice) < {thr:.2f}:  published {a:,} ({100 * a / len(allr):.2f}%)"
              f"   fixed {b:,} ({100 * b / len(allr):.2f}%)   delta {b - a:+,}")

    # ── 6. per-schedule ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("6. PER-SCHEDULE BREAKDOWN")
    print("=" * 78)
    print(f"{'schedule':<14}{'subj':>6}{'E[p] pub':>10}{'E[p] fix':>10}{'delta':>8}"
          f"{'Elogp pub':>11}{'Elogp fix':>11}{'delta':>8}")
    sched_rows = []
    for sched, g in allr.groupby("schedule"):
        ep_p, ep_f = M.e_p(g.pc_pub), M.e_p(g.pc_fix)
        el_p, el_f = M.e_log_p(g.pc_pub), M.e_log_p(g.pc_fix)
        print(f"{sched:<14}{g.subject_id.nunique():>6}{ep_p:>10.4f}{ep_f:>10.4f}"
              f"{ep_f - ep_p:>+8.4f}{el_p:>11.4f}{el_f:>11.4f}{el_f - el_p:>+8.4f}")
        sched_rows.append({"schedule": sched, "n_subjects": g.subject_id.nunique(),
                           "E[p]_pub": ep_p, "E[p]_fix": ep_f,
                           "E[logp]_pub": el_p, "E[logp]_fix": el_f})
    pd.DataFrame(sched_rows).to_csv(OUT_DIR / "per_schedule_metrics.csv", index=False)

    # ── figures ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("FIGURES")
    print("=" * 78)

    fig1, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 1, 51)
    axes[0].hist(allr.pc_pub, bins=bins, alpha=0.55, label="published",
                 color=PALETTE["published"])
    axes[0].hist(allr.pc_fix, bins=bins, alpha=0.55, label="fixed", color=PALETTE["fixed"])
    axes[0].set_xlabel("p(choice made)")
    axes[0].set_ylabel("trials")
    axes[0].set_title("Distribution of p")
    axes[0].legend(frameon=False)
    lb = np.linspace(-6, 0, 61)
    axes[1].hist(np.log(np.clip(allr.pc_pub, 1e-12, 1)), bins=lb, alpha=0.55,
                 label="published", color=PALETTE["published"])
    axes[1].hist(np.log(np.clip(allr.pc_fix, 1e-12, 1)), bins=lb, alpha=0.55,
                 label="fixed", color=PALETTE["fixed"])
    axes[1].set_xlabel("log p(choice made)")
    axes[1].set_ylabel("trials")
    axes[1].set_title("Distribution of log p  (the tail drives E[log p])")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig1.tight_layout()

    fig2, ax = plt.subplots(figsize=(7.5, 4.5))
    sub = allr[allr.trend_testable > 0]
    ax.scatter(sub.p1_pub[::40], sub.p1_fix[::40], s=3, alpha=0.15,
               color=PALETTE["fixed"], edgecolors="none")
    ax.plot([0, 1], [0, 1], color="0.4", lw=1, ls="--")
    ax.set_xlabel("P(alt 1), published")
    ax.set_ylabel("P(alt 1), fixed")
    ax.set_title(f"Trend-testable trials only (n={len(sub):,}; every 40th plotted)")
    ax.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()

    fig3, ax = plt.subplots(figsize=(8.5, 4.5))
    sr = pd.DataFrame(sched_rows)
    sr["order"] = sr["schedule"].str.replace("schedule_", "").astype(int)
    sr = sr.sort_values("order")
    x = np.arange(len(sr))
    ax.bar(x - 0.2, sr["E[logp]_pub"], 0.4, label="published", color=PALETTE["published"])
    ax.bar(x + 0.2, sr["E[logp]_fix"], 0.4, label="fixed", color=PALETTE["fixed"])
    ax.set_xticks(x)
    ax.set_xticklabels(sr["order"])
    ax.set_xlabel("schedule")
    ax.set_ylabel("E[log p]")
    ax.set_title("Bug impact by reward schedule")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig3.tight_layout()

    for nm, fg in {"fig1_probability_distributions": fig1,
                   "fig2_trend_trial_scatter": fig2,
                   "fig3_per_schedule_impact": fig3}.items():
        fg.savefig(OUT_DIR / f"{nm}.png", dpi=150, bbox_inches="tight")
        print(f"  {nm}.png")

    print("\ndone.")
    sys.stdout = sys.__stdout__
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
