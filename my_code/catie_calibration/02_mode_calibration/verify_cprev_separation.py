"""
Why does fig1's right panel show near-perfect separation -- c_prev=0 living almost
entirely below p=0.5 and c_prev=1 above it? Is that an absolute structural rule of
the model, or an artifact of dropping sparse reliability bins (min_count=30)?

Answer, established analytically and then verified against the data below:

  It is a NEAR-rule with a precisely characterised exception. Writing the model as

      P(alt1) = tau*H*b + (1-tau*H) * [ p_exp/2 + (1-p_exp)*(phi*c_prev + (1-phi)*g) ]

  with tau=0.29, phi=0.71, and p_exp = eps*(1+s_prev+sbar_prev)/3 in [eps/3, eps]
  = [0.10, 0.30] (since surprise is bounded in [0,1]):

    H=0 (trend branch not testable) -- the inertia term phi*c_prev is the only
        thing that can move P(alt1) far from the exploration floor, so:
            c_prev=0  =>  P(alt1) in [0.050, 0.353]
            c_prev=1  =>  P(alt1) in [0.647, 0.950]
        These ranges do not overlap and 0.5 falls strictly between them.
        Separation is ABSOLUTE on H=0 trials. No data can violate it.

    H=1 (trend branch testable) -- the heuristic can pay out tau=0.29 to the side
        OPPOSITE the previous choice, which is the only mechanism in the model
        able to outvote inertia:
            c_prev=0, b=1 (trend says switch TO alt1)   => up to 0.541
            c_prev=1, b=0 (trend says switch AWAY)      => down to 0.459
        Crossings are possible here, but only narrowly, and only when g is
        extreme as well (g > ~0.72 or g < ~0.28 respectively).

  So every crossing trial in the data must have H=1. That is a falsifiable
  prediction, checked exhaustively below rather than asserted. Result: it holds
  at 100% -- all 1,273 upward crossings have H=1 and b=1, all 254 downward have
  H=1 and b=0, and there are 0 violations among the 208,429 H=0 trials.

  Answering the two hypotheses directly: the separation is NOT an absolute rule
  (1,527 crossings, 0.61% of trials), and it is NOT an artifact of dropping
  sparse bins either -- the crossing bins hold 1,273 and 254 trials, far above
  the min_count=30 threshold, and ARE drawn in fig1 as the slight overhang past
  0.5 at the inner end of each curve. fig1 simply looks cleanly split because
  crossings are rare and confined to a narrow band (max 0.525 / min 0.476).

Note H, b and c_prev are k-INDEPENDENT (they are functions of the observed choices
and payoffs only -- see catie_core.state_tensors), so a single cheap k=0 pass
recovers them exactly for the k-mixture. Only g, s_prev and sbar_prev vary with k.

A second consequence, worth stating because it bears on the published-vs-fixed
choice: under the PUBLISHED (buggy) model b == 0 identically, so the
"c_prev=0 crossing upward" case cannot occur at all -- the published model's
separation is one-sided in a way the corrected model's is not.

Run:  python my_code/catie_calibration/02_mode_calibration/verify_cprev_separation.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from catie_core import EPSILON, PHI, TAU, catie_hetero, state_tensors  # noqa: E402
import metrics as M  # noqa: E402

TRIAL_CSV = HERE / "figures" / "trial_level.csv.gz"
DATA_DIR = HERE.parent / "data"


def load_choices() -> pd.DataFrame:
    """Same population as conditional_calibration.py (EDA + Training + schedule_0)."""
    frames = []
    for name in ("eda", "training", "schedule_0"):
        df = pd.read_csv(DATA_DIR / f"cleaned_{name}.csv")
        frames.append(df[["subject_id", "trial_number", "biased_reward",
                          "unbiased_reward", "is_biased_choice"]])
    out = pd.concat(frames, ignore_index=True)
    out["chose_biased"] = (out["is_biased_choice"].astype(str).str.upper() == "TRUE").astype(int)
    return out.sort_values(["subject_id", "trial_number"]).reset_index(drop=True)


def main() -> int:
    print("=" * 78)
    print("IS THE c_prev / p=0.5 SEPARATION ABSOLUTE, OR A SPARSE-BIN ARTIFACT?")
    print("=" * 78)

    if not TRIAL_CSV.exists():
        print(f"missing {TRIAL_CSV} -- run conditional_calibration.py first")
        return 1

    tl = pd.read_csv(TRIAL_CSV)
    print(f"loaded {len(tl):,} trials from the Phase 2 run\n")

    # ── 1. Analytic bounds ───────────────────────────────────────────────────
    print("=" * 78)
    print("1. ANALYTIC BOUNDS on P(alt 1)")
    print("=" * 78)
    print(f"   tau={TAU}, phi={PHI}, eps={EPSILON} -> p_exp in [{EPSILON/3:.3f}, {EPSILON:.3f}]\n")
    print(f"   {'c_prev':<8}{'H':<4}{'b':<4}{'min P(alt1)':>13}{'max P(alt1)':>13}   crosses 0.5?")
    for cprev in (0, 1):
        for H, b in ((0, 0), (1, 0), (1, 1)):
            vals = []
            for pexp in (EPSILON / 3, EPSILON):
                for g in (0.0, 1.0):
                    th = TAU * H
                    vals.append(th * b + (1 - th) * (0.5 * pexp + (1 - pexp)
                                * (PHI * cprev + (1 - PHI) * g)))
            lo, hi = min(vals), max(vals)
            crosses = "YES" if lo < 0.5 < hi else "no"
            print(f"   {cprev:<8}{H:<4}{b:<4}{lo:>13.4f}{hi:>13.4f}   {crosses}")
    print("\n   => on H=0 trials the two c_prev ranges are disjoint and 0.5 lies")
    print("      strictly between them: separation is STRUCTURAL, not empirical.")
    print("      Only the heuristic branch (H=1) can outvote inertia.")

    # ── 2. Empirical crossing counts ─────────────────────────────────────────
    print("\n" + "=" * 78)
    print("2. EMPIRICAL CROSSING COUNTS (all trials, no bins, nothing dropped)")
    print("=" * 78)
    c0 = tl[tl["c_prev"] == 0]
    c1 = tl[tl["c_prev"] == 1]
    x0 = c0[c0["p_alt1"] > 0.5]
    x1 = c1[c1["p_alt1"] < 0.5]
    print(f"   c_prev=0 total {len(c0):>8,}   of which p>0.5: {len(x0):>6,} "
          f"({100*len(x0)/len(c0):.4f}%)")
    print(f"   c_prev=1 total {len(c1):>8,}   of which p<0.5: {len(x1):>6,} "
          f"({100*len(x1)/len(c1):.4f}%)")
    print(f"   total crossings: {len(x0)+len(x1):,} of {len(tl):,} "
          f"({100*(len(x0)+len(x1))/len(tl):.4f}%)")

    print(f"\n   observed ranges:")
    print(f"     c_prev=0: p_alt1 in [{c0['p_alt1'].min():.4f}, {c0['p_alt1'].max():.4f}]")
    print(f"     c_prev=1: p_alt1 in [{c1['p_alt1'].min():.4f}, {c1['p_alt1'].max():.4f}]")

    # ── 3. Do ALL crossings have H=1, as predicted? ──────────────────────────
    print("\n" + "=" * 78)
    print("3. TESTING THE PREDICTION: every crossing must have H=1")
    print("=" * 78)
    choices = load_choices()
    print("   recomputing H, b (k-independent; single k=0 pass) ...")
    recs = []
    for sid, d in choices.groupby("subject_id", sort=False):
        st = state_tensors(d["biased_reward"].to_numpy(),
                           d["unbiased_reward"].to_numpy(),
                           d["chose_biased"].to_numpy().astype(bool),
                           k=0, mode="fixed")
        recs.append(pd.DataFrame({"subject_id": sid,
                                  "trial_number": d["trial_number"].to_numpy(),
                                  "H": st.H, "b": st.b}))
    Hb = pd.concat(recs, ignore_index=True)
    merged = tl.merge(Hb, on=["subject_id", "trial_number"], how="left")
    assert merged["H"].notna().all(), "failed to recover H for some trials"

    m0 = merged[(merged["c_prev"] == 0) & (merged["p_alt1"] > 0.5)]
    m1 = merged[(merged["c_prev"] == 1) & (merged["p_alt1"] < 0.5)]
    print(f"\n   c_prev=0 & p>0.5 : n={len(m0):,}  H=1 in {int(m0['H'].sum()):,} "
          f"({100*m0['H'].mean() if len(m0) else float('nan'):.2f}%)  "
          f"b=1 in {int(m0['b'].sum()):,}")
    print(f"   c_prev=1 & p<0.5 : n={len(m1):,}  H=1 in {int(m1['H'].sum()):,} "
          f"({100*m1['H'].mean() if len(m1) else float('nan'):.2f}%)  "
          f"b=1 in {int(m1['b'].sum()):,}")

    ok0 = (len(m0) == 0) or bool((m0["H"] == 1).all() and (m0["b"] == 1).all())
    ok1 = (len(m1) == 0) or bool((m1["H"] == 1).all() and (m1["b"] == 0).all())
    print(f"\n   [{'PASS' if ok0 else 'FAIL'}] every c_prev=0 upward crossing has H=1 AND b=1")
    print(f"   [{'PASS' if ok1 else 'FAIL'}] every c_prev=1 downward crossing has H=1 AND b=0")

    nH0 = merged[merged["H"] == 0]
    v0 = nH0[(nH0["c_prev"] == 0) & (nH0["p_alt1"] > 0.5)]
    v1 = nH0[(nH0["c_prev"] == 1) & (nH0["p_alt1"] < 0.5)]
    print(f"\n   violations of the H=0 structural bound: {len(v0) + len(v1)} "
          f"(must be 0; {len(nH0):,} H=0 trials checked)")
    print(f"   H=0 observed max for c_prev=0: "
          f"{nH0[nH0['c_prev']==0]['p_alt1'].max():.4f}  (bound 0.3530)")
    print(f"   H=0 observed min for c_prev=1: "
          f"{nH0[nH0['c_prev']==1]['p_alt1'].min():.4f}  (bound 0.6470)")

    # ── 4. Was the figure's appearance a sparse-bin artifact? ────────────────
    print("\n" + "=" * 78)
    print("4. WAS fig1's APPEARANCE CAUSED BY DROPPING SPARSE BINS (min_count=30)?")
    print("=" * 78)
    for label, sub in (("c_prev=0", c0), ("c_prev=1", c1)):
        rel_all = M.reliability_table(sub["p_alt1"], sub["chose_biased"], n_bins=10, min_count=1)
        rel_shown = M.reliability_table(sub["p_alt1"], sub["chose_biased"], n_bins=10, min_count=30)
        dropped = rel_all[~rel_all["bin"].isin(rel_shown["bin"])]
        wrong_side = rel_all[(rel_all["predicted"] > 0.5) if label == "c_prev=0"
                             else (rel_all["predicted"] < 0.5)]
        print(f"\n   {label}: {len(rel_all)} non-empty bins, {len(rel_shown)} shown at min_count=30")
        if len(dropped):
            print(f"     bins dropped as sparse: "
                  f"{[(round(r.lo,2), round(r.hi,2), int(r.n)) for r in dropped.itertuples()]}")
        else:
            print("     bins dropped as sparse: none")
        print(f"     bins on the 'wrong' side of 0.5: "
              f"{[(round(r.lo,2), round(r.hi,2), int(r.n)) for r in wrong_side.itertuples()] or 'none'}")

    # ── 5. Published model: the same test, as a falsifiable cross-check ──────
    print("\n" + "=" * 78)
    print("5. CROSS-CHECK -- the PUBLISHED model must show ZERO upward crossings")
    print("=" * 78)
    print("   Upward crossings (c_prev=0, p>0.5) require b=1. Under the published")
    print("   model b == 0 identically (the Phase 1 bug), so there must be exactly")
    print("   none. A non-zero count here would falsify the whole account above.\n")
    up = dn = tot = 0
    for sid, d in choices.groupby("subject_id", sort=False):
        # NB: named `choices_1`, not `c1` -- `c1` is the c_prev==1 DataFrame in the
        # enclosing scope (used above to build x1). Shadowing it worked only because
        # x0/x1 are captured before this loop; one edit away from a silent bug.
        choices_1 = d["chose_biased"].to_numpy().astype(bool)
        p = catie_hetero(d["biased_reward"].to_numpy(), d["unbiased_reward"].to_numpy(),
                         choices_1, mode="published")
        cp = np.concatenate([[np.nan], choices_1[:-1].astype(float)])
        m = ~np.isnan(cp)
        m[0] = False
        up += int(((cp == 0) & (p > 0.5) & m).sum())
        dn += int(((cp == 1) & (p < 0.5) & m).sum())
        tot += int(m.sum())
    print(f"   published, {tot:,} trials:  upward {up:,}   downward {dn:,}")
    print(f"   fixed,     {len(tl):,} trials:  upward {len(x0):,}   downward {len(x1):,}")
    print(f"\n   [{'PASS' if up == 0 else 'FAIL'}] published upward crossings == 0")
    print("   Note the published model has MORE downward crossings than the fixed one")
    print("   ({} vs {}): whenever H=1 its dead heuristic branch diverts tau=0.29 to".format(dn, len(x1)))
    print("   alternative 2 unconditionally, dragging P(alt1) down across the 0.5 line.")

    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    print("   The separation is NOT absolute, and NOT a sparse-bin artifact either.")
    print("   Both proposed explanations are wrong; the real one is structural:")
    print()
    print("   (a) On the 83.4% of trials where the trend branch is not testable")
    print("       (H=0), separation is a THEOREM, not a tendency. phi=0.71 on the")
    print("       previous choice, against an exploration term capped at eps=0.30,")
    print("       forces P(alt1) <= 0.353 when c_prev=0 and >= 0.647 when c_prev=1.")
    print("       0 violations in 208,429 such trials, exactly as predicted.")
    print()
    print("   (b) Crossings do occur -- 1,527 of them (0.61%) -- and every single")
    print("       one is an H=1 trial where the heuristic points opposite to the")
    print("       previous choice. That is the only mechanism in CATIE able to")
    print("       outvote inertia.")
    print()
    print("   (c) Those crossings are NOT hidden by the min_count=30 filter: their")
    print("       bins hold 1,273 and 254 trials and ARE plotted in fig1. They are")
    print("       the slight overhang past 0.5 at the inner end of each curve.")
    print("       fig1 looks cleanly split because crossings are rare and confined")
    print("       to a narrow band (max 0.525 / min 0.476), not because anything")
    print("       was dropped.")
    print()
    print("   Asymmetry worth noting: upward crossings (c_prev=0, n=1,273) outnumber")
    print("   downward ones (c_prev=1, n=254) 5:1. Upward crossings require b=1, which")
    print("   is identically impossible under the PUBLISHED model (b == 0 everywhere --")
    print("   that is the Phase 1 bug), so this asymmetry exists only in the corrected")
    print("   model and is itself a downstream consequence of the fix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
