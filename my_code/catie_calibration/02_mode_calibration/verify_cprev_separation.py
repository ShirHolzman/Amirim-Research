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
import metrics  # noqa: E402

TRIAL_CSV = HERE / "figures" / "trial_level.csv.gz"
DATA_DIR = HERE.parent / "data"


def load_choices() -> pd.DataFrame:
    """Same population as conditional_calibration.py (EDA + Training + schedule_0)."""
    split_frames = []
    for name in ("eda", "training", "schedule_0"):
        split_df = pd.read_csv(DATA_DIR / f"cleaned_{name}.csv")
        split_frames.append(split_df[["subject_id", "trial_number", "biased_reward",
                                      "unbiased_reward", "is_biased_choice"]])
    combined_df = pd.concat(split_frames, ignore_index=True)
    combined_df["chose_biased"] = (combined_df["is_biased_choice"].astype(str).str.upper() == "TRUE").astype(int)
    return combined_df.sort_values(["subject_id", "trial_number"]).reset_index(drop=True)


def main() -> int:
    print("=" * 78)
    print("IS THE c_prev / p=0.5 SEPARATION ABSOLUTE, OR A SPARSE-BIN ARTIFACT?")
    print("=" * 78)

    if not TRIAL_CSV.exists():
        print(f"missing {TRIAL_CSV} -- run conditional_calibration.py first")
        return 1

    trial_level_df = pd.read_csv(TRIAL_CSV)
    print(f"loaded {len(trial_level_df):,} trials from the Phase 2 run\n")

    # ── 1. Analytic bounds ───────────────────────────────────────────────────
    print("=" * 78)
    print("1. ANALYTIC BOUNDS on P(alt 1)")
    print("=" * 78)
    print(f"   tau={TAU}, phi={PHI}, eps={EPSILON} -> p_exp in [{EPSILON/3:.3f}, {EPSILON:.3f}]\n")
    print(f"   {'c_prev':<8}{'H':<4}{'b':<4}{'min P(alt1)':>13}{'max P(alt1)':>13}   crosses 0.5?")
    for cprev in (0, 1):
        for H, b in ((0, 0), (1, 0), (1, 1)):
            p_alt1_bounds = []
            for p_explore in (EPSILON / 3, EPSILON):
                for g in (0.0, 1.0):
                    tau_times_H = TAU * H
                    p_alt1_bounds.append(tau_times_H * b + (1 - tau_times_H) * (0.5 * p_explore + (1 - p_explore)
                                * (PHI * cprev + (1 - PHI) * g)))
            min_p_alt1, max_p_alt1 = min(p_alt1_bounds), max(p_alt1_bounds)
            crosses_half = "YES" if min_p_alt1 < 0.5 < max_p_alt1 else "no"
            print(f"   {cprev:<8}{H:<4}{b:<4}{min_p_alt1:>13.4f}{max_p_alt1:>13.4f}   {crosses_half}")
    print("\n   => on H=0 trials the two c_prev ranges are disjoint and 0.5 lies")
    print("      strictly between them: separation is STRUCTURAL, not empirical.")
    print("      Only the heuristic branch (H=1) can outvote inertia.")

    # ── 2. Empirical crossing counts ─────────────────────────────────────────
    print("\n" + "=" * 78)
    print("2. EMPIRICAL CROSSING COUNTS (all trials, no bins, nothing dropped)")
    print("=" * 78)
    c_prev_0_trials = trial_level_df[trial_level_df["c_prev"] == 0]
    c_prev_1_trials = trial_level_df[trial_level_df["c_prev"] == 1]
    c_prev_0_above_half = c_prev_0_trials[c_prev_0_trials["p_alt1"] > 0.5]
    c_prev_1_below_half = c_prev_1_trials[c_prev_1_trials["p_alt1"] < 0.5]
    print(f"   c_prev=0 total {len(c_prev_0_trials):>8,}   of which p>0.5: {len(c_prev_0_above_half):>6,} "
          f"({100*len(c_prev_0_above_half)/len(c_prev_0_trials):.4f}%)")
    print(f"   c_prev=1 total {len(c_prev_1_trials):>8,}   of which p<0.5: {len(c_prev_1_below_half):>6,} "
          f"({100*len(c_prev_1_below_half)/len(c_prev_1_trials):.4f}%)")
    print(f"   total crossings: {len(c_prev_0_above_half)+len(c_prev_1_below_half):,} of {len(trial_level_df):,} "
          f"({100*(len(c_prev_0_above_half)+len(c_prev_1_below_half))/len(trial_level_df):.4f}%)")

    print(f"\n   observed ranges:")
    print(f"     c_prev=0: p_alt1 in [{c_prev_0_trials['p_alt1'].min():.4f}, {c_prev_0_trials['p_alt1'].max():.4f}]")
    print(f"     c_prev=1: p_alt1 in [{c_prev_1_trials['p_alt1'].min():.4f}, {c_prev_1_trials['p_alt1'].max():.4f}]")

    # ── 3. Do ALL crossings have H=1, as predicted? ──────────────────────────
    print("\n" + "=" * 78)
    print("3. TESTING THE PREDICTION: every crossing must have H=1")
    print("=" * 78)
    choice_df = load_choices()
    print("   recomputing H, b (k-independent; single k=0 pass) ...")
    state_records = []
    for subject_id, subject_df in choice_df.groupby("subject_id", sort=False):
        state = state_tensors(subject_df["biased_reward"].to_numpy(),
                              subject_df["unbiased_reward"].to_numpy(),
                              subject_df["chose_biased"].to_numpy().astype(bool),
                              k=0, mode="fixed")
        state_records.append(pd.DataFrame({"subject_id": subject_id,
                                           "trial_number": subject_df["trial_number"].to_numpy(),
                                           "H": state.H, "b": state.b}))
    state_H_b_df = pd.concat(state_records, ignore_index=True)
    trial_level_with_Hb = trial_level_df.merge(state_H_b_df, on=["subject_id", "trial_number"], how="left")
    assert trial_level_with_Hb["H"].notna().all(), "failed to recover H for some trials"

    upward_crossings = trial_level_with_Hb[(trial_level_with_Hb["c_prev"] == 0) & (trial_level_with_Hb["p_alt1"] > 0.5)]
    downward_crossings = trial_level_with_Hb[(trial_level_with_Hb["c_prev"] == 1) & (trial_level_with_Hb["p_alt1"] < 0.5)]
    print(f"\n   c_prev=0 & p>0.5 : n={len(upward_crossings):,}  H=1 in {int(upward_crossings['H'].sum()):,} "
          f"({100*upward_crossings['H'].mean() if len(upward_crossings) else float('nan'):.2f}%)  "
          f"b=1 in {int(upward_crossings['b'].sum()):,}")
    print(f"   c_prev=1 & p<0.5 : n={len(downward_crossings):,}  H=1 in {int(downward_crossings['H'].sum()):,} "
          f"({100*downward_crossings['H'].mean() if len(downward_crossings) else float('nan'):.2f}%)  "
          f"b=1 in {int(downward_crossings['b'].sum()):,}")

    upward_crossings_valid = (len(upward_crossings) == 0) or bool((upward_crossings["H"] == 1).all() and (upward_crossings["b"] == 1).all())
    downward_crossings_valid = (len(downward_crossings) == 0) or bool((downward_crossings["H"] == 1).all() and (downward_crossings["b"] == 0).all())
    print(f"\n   [{'PASS' if upward_crossings_valid else 'FAIL'}] every c_prev=0 upward crossing has H=1 AND b=1")
    print(f"   [{'PASS' if downward_crossings_valid else 'FAIL'}] every c_prev=1 downward crossing has H=1 AND b=0")

    h_zero_trials = trial_level_with_Hb[trial_level_with_Hb["H"] == 0]
    h_zero_upward_violations = h_zero_trials[(h_zero_trials["c_prev"] == 0) & (h_zero_trials["p_alt1"] > 0.5)]
    h_zero_downward_violations = h_zero_trials[(h_zero_trials["c_prev"] == 1) & (h_zero_trials["p_alt1"] < 0.5)]
    print(f"\n   violations of the H=0 structural bound: {len(h_zero_upward_violations) + len(h_zero_downward_violations)} "
          f"(must be 0; {len(h_zero_trials):,} H=0 trials checked)")
    print(f"   H=0 observed max for c_prev=0: "
          f"{h_zero_trials[h_zero_trials['c_prev']==0]['p_alt1'].max():.4f}  (bound 0.3530)")
    print(f"   H=0 observed min for c_prev=1: "
          f"{h_zero_trials[h_zero_trials['c_prev']==1]['p_alt1'].min():.4f}  (bound 0.6470)")

    # ── 4. Was the figure's appearance a sparse-bin artifact? ────────────────
    print("\n" + "=" * 78)
    print("4. WAS fig1's APPEARANCE CAUSED BY DROPPING SPARSE BINS (min_count=30)?")
    print("=" * 78)
    for stratum_label, stratum_df in (("c_prev=0", c_prev_0_trials), ("c_prev=1", c_prev_1_trials)):
        reliability_all_bins = metrics.reliability_table(stratum_df["p_alt1"], stratum_df["chose_biased"], n_bins=10, min_count=1)
        reliability_shown_bins = metrics.reliability_table(stratum_df["p_alt1"], stratum_df["chose_biased"], n_bins=10, min_count=30)
        dropped_bins = reliability_all_bins[~reliability_all_bins["bin"].isin(reliability_shown_bins["bin"])]
        wrong_side_bins = reliability_all_bins[(reliability_all_bins["predicted"] > 0.5) if stratum_label == "c_prev=0"
                             else (reliability_all_bins["predicted"] < 0.5)]
        print(f"\n   {stratum_label}: {len(reliability_all_bins)} non-empty bins, {len(reliability_shown_bins)} shown at min_count=30")
        if len(dropped_bins):
            print(f"     bins dropped as sparse: "
                  f"{[(round(r.lo,2), round(r.hi,2), int(r.n)) for r in dropped_bins.itertuples()]}")
        else:
            print("     bins dropped as sparse: none")
        print(f"     bins on the 'wrong' side of 0.5: "
              f"{[(round(r.lo,2), round(r.hi,2), int(r.n)) for r in wrong_side_bins.itertuples()] or 'none'}")

    # ── 5. Published model: the same test, as a falsifiable cross-check ──────
    print("\n" + "=" * 78)
    print("5. CROSS-CHECK -- the PUBLISHED model must show ZERO upward crossings")
    print("=" * 78)
    print("   Upward crossings (c_prev=0, p>0.5) require b=1. Under the published")
    print("   model b == 0 identically (the Phase 1 bug), so there must be exactly")
    print("   none. A non-zero count here would falsify the whole account above.\n")
    upward_crossing_count = downward_crossing_count = total_trials = 0
    for subject_id, subject_df in choice_df.groupby("subject_id", sort=False):
        # Named `choices_1_bool` (not `c_prev_1_trials`) so it can't be confused with
        # `c_prev_1_trials`, the c_prev==1 DataFrame from section 2.
        choices_1_bool = subject_df["chose_biased"].to_numpy().astype(bool)
        p_alt1_published = catie_hetero(subject_df["biased_reward"].to_numpy(), subject_df["unbiased_reward"].to_numpy(),
                         choices_1_bool, mode="published")
        c_prev_published = np.concatenate([[np.nan], choices_1_bool[:-1].astype(float)])
        valid_trial_mask = ~np.isnan(c_prev_published)
        valid_trial_mask[0] = False
        upward_crossing_count += int(((c_prev_published == 0) & (p_alt1_published > 0.5) & valid_trial_mask).sum())
        downward_crossing_count += int(((c_prev_published == 1) & (p_alt1_published < 0.5) & valid_trial_mask).sum())
        total_trials += int(valid_trial_mask.sum())
    print(f"   published, {total_trials:,} trials:  upward {upward_crossing_count:,}   downward {downward_crossing_count:,}")
    print(f"   fixed,     {len(trial_level_df):,} trials:  upward {len(c_prev_0_above_half):,}   downward {len(c_prev_1_below_half):,}")
    print(f"\n   [{'PASS' if upward_crossing_count == 0 else 'FAIL'}] published upward crossings == 0")
    print("   Note the published model has MORE downward crossings than the fixed one")
    print("   ({} vs {}): whenever H=1 its dead heuristic branch diverts tau=0.29 to".format(downward_crossing_count, len(c_prev_1_below_half)))
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
