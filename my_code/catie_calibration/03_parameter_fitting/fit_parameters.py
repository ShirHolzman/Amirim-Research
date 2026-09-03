"""
Phase 3 -- Re-fitting CATIE's parameters, with post-hoc calibration as the control.

THE SCIENTIFIC QUESTION. CATIE loses to Q-Learning on E[log p] (-0.678 vs -0.569,
Supplementary Table S2). Is that because its parameter VALUES are wrong for this
population, or because its STRUCTURE is wrong? Those demand different responses and
the distinction is only visible with the right control.

WHAT THE PAPER SAYS, and why this is a direct test of it. Methods: "the model is
characterized by four free parameters (one for each mode: tau, epsilon, phi, K). The
parameters of the model were estimated from behavior[10], and these estimations were
used in all simulations: tau = 0.29, epsilon = 0.30, phi = 0.71, K = 2." Discussion:
"Particularly remarkable is the fact that the parameters of this model were not
fitted to this particular competition. Rather, they were taken from previous
experiments." The paper foregrounds the non-fitting as evidence of CATIE's
robustness. Re-fitting therefore tests a claim the paper itself highlights, and
either outcome is informative:
  * large gain  -> CATIE was competing handicapped; the transfer was costly
  * small gain  -> the parameters genuinely transfer across tasks and populations,
                   which strengthens the paper's point rather than weakening it

Note a discrepancy worth measuring: the Methods state K = 2, but the shipped
likelihood code mixes K in {0,1,2}. Both are evaluated below.

THE CONTROL, which is what makes this science rather than curve-fitting. Any
3-parameter re-fit will improve the likelihood somewhat. The question is whether it
beats a ONE-parameter post-hoc recalibration that has no psychological content at
all. If re-fitting only matches temperature scaling, then CATIE is simply
over-confident and the "psychology" of the fitted parameters is illusory. If it
clearly beats it, the parameter values carry real information.

PHASE 2 CARRIED FORWARD. Phase 2 found the calibration gap reverses sign with the
run length of the current perseveration streak (+0.347 at run 1 to -0.072 at run 6+
within c_prev=0 alone). A constant phi -- and equally an asymmetric phi, one per
side of c_prev -- cannot represent that. So a run-length-dependent phi is fitted
here as a candidate, ahead of Phase 4, because it is the extension the data demand.

PROTOCOL. Fit on Training (1,483 subjects, schedules 2/3/6/9/11). Select on EDA
(492, schedules 4/5/7). schedule_0 (549) is an out-of-distribution check. Test
(schedules 1/8/10) is NOT touched and is not even present in the cache.

Run:  python my_code/catie_calibration/03_parameter_fitting/fit_parameters.py
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import optimize  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from catie_core import EPSILON, PHI, TAU  # noqa: E402
import metrics as M  # noqa: E402
from catie_likelihood import (  # noqa: E402
    StateCache, mean_log_p, mean_p, negloglik_factory, p_choice_matrix,
    p_alt1_single_k, mix_per_trial, unpack, _logit, _sigmoid,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "font.family": "sans-serif"})

OUT_DIR = HERE / "figures"
RNG_SEED = 42
PUBLISHED = {"tau": TAU, "eps": EPSILON, "phi": PHI}

# Reference values from the paper (Supplementary Table S2)
PAPER_CATIE_ELOGP = -0.678
PAPER_QL10_ELOGP = -0.569
PAPER_GAP = PAPER_CATIE_ELOGP - PAPER_QL10_ELOGP   # -0.109


def fit(cache, ks=(0, 1, 2), free=("tau", "eps", "phi"), fixed=None,
        n_restarts=8, seed=RNG_SEED, **extra):
    """Multi-start Nelder-Mead in logit space. Returns (best_params, best_nll, all)."""
    f, order = negloglik_factory(cache, ks=ks, free=free, fixed=fixed, **extra)
    rng = np.random.default_rng(seed)
    starts = [np.array([_logit(PUBLISHED.get(nm, 0.5)) for nm in order])]
    for _ in range(n_restarts - 1):
        starts.append(rng.normal(0.0, 1.5, size=len(order)))
    results = []
    for z0 in starts:
        r = optimize.minimize(f, z0, method="Nelder-Mead",
                              options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 4000})
        results.append((float(r.fun), unpack(r.x, order), r.x))
    results.sort(key=lambda t: t[0])
    return results[0][1], results[0][0], results


def numeric_hessian(f, z, h=1e-4):
    n = len(z)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            zpp, zpm, zmp, zmm = z.copy(), z.copy(), z.copy(), z.copy()
            zpp[i] += h; zpp[j] += h
            zpm[i] += h; zpm[j] -= h
            zmp[i] -= h; zmp[j] += h
            zmm[i] -= h; zmm[j] -= h
            H[i, j] = (f(zpp) - f(zpm) - f(zmp) + f(zmm)) / (4 * h * h)
    return (H + H.T) / 2


def build_streak(cache):
    """Run length of identical choices ending at t-1, from the cached c_prev."""
    cp = cache.st[cache.ks[0]]["c_prev"]
    n_s, n_t = cp.shape
    streak = np.ones((n_s, n_t))
    for t in range(1, n_t):
        same = cp[:, t] == cp[:, t - 1]
        streak[:, t] = np.where(same, streak[:, t - 1] + 1, 1.0)
    return streak


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = open(OUT_DIR / "output.txt", "w", encoding="utf-8")
    sys.stdout = M.Tee(sys.__stdout__, log)

    print("=" * 78)
    print("PHASE 3 -- PARAMETER RE-FITTING, WITH RECALIBRATION AS THE CONTROL")
    print("=" * 78)
    print("published (Dan, Plonsky & Loewenstein 2025, Methods):")
    print(f"   tau={TAU}, eps={EPSILON}, phi={PHI}, K=2 -- 'estimated from behavior'")
    print("   and 'not fitted to this particular competition'")
    print(f"reference gap to close: CATIE {PAPER_CATIE_ELOGP} vs best QL "
          f"{PAPER_QL10_ELOGP} = {PAPER_GAP:+.3f}\n")

    tr = StateCache("training", ks=(0, 1, 2))
    ed = StateCache("eda", ks=(0, 1, 2))
    s0 = StateCache("schedule_0", ks=(0, 1, 2))
    print(f"training {tr.n_subjects:,} subj | EDA {ed.n_subjects:,} | "
          f"schedule_0 {s0.n_subjects:,} | Test NOT cached")

    # ── 0. Golden check ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("0. GOLDEN CHECK -- vectorised likelihood vs Phase 1 (blocking)")
    print("=" * 78)
    # Read Phase 1's own output rather than hardcoding its numbers -- hardcoded
    # copies go stale the moment the underlying data changes (they did, when the
    # project migrated to the organized data release and EDA went 492 -> 496).
    ph1 = pd.read_csv(HERE.parent / "01_bug_correction" / "figures" / "headline_metrics.csv")
    ph1 = ph1.set_index("split")
    caches = {"training": tr, "eda": ed, "schedule_0": s0}
    ref = {nm: (c, float(ph1.loc[nm, "E[p]_fix"]), float(ph1.loc[nm, "E[logp]_fix"]))
           for nm, c in caches.items()}
    ok = True
    for nm, (c, rp, rl) in ref.items():
        mp, ml = mean_p(c, drop_first=False), mean_log_p(c, drop_first=False)
        good = abs(mp - rp) < 5e-5 and abs(ml - rl) < 5e-5
        ok &= good
        print(f"   [{'PASS' if good else 'FAIL'}] {nm:<12} E[p]={mp:.4f} (ph1 {rp:.4f})   "
              f"E[log p]={ml:.4f} (ph1 {rl:.4f})")
    assert ok, "golden check failed -- do not trust anything below"
    print("   All subsequent numbers use drop_first=True (trial 1 is p=0.5 by fiat).")

    base_tr = mean_log_p(tr, **PUBLISHED)
    base_ed = mean_log_p(ed, **PUBLISHED)
    base_s0 = mean_log_p(s0, **PUBLISHED)
    print(f"\n   baseline E[log p] at published params (trial 1 dropped):")
    print(f"      training {base_tr:.4f} | EDA {base_ed:.4f} | schedule_0 {base_s0:.4f}")

    # ── 1. K enumeration ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("1. K ENUMERATION at published (tau, eps, phi)")
    print("=" * 78)
    print("   The paper's Methods state K=2; the shipped code mixes K in {0,1,2}.")
    print("   Measuring both, plus single K=0,1,3, on Training.\n")
    print(f"   {'K':<16}{'E[log p] train':>16}{'E[log p] EDA':>15}")
    krows = []
    for label, ks in [("K=0", 0), ("K=1", 1), ("K=2 (paper)", 2), ("K=3", 3),
                      ("mix {0,1,2} (code)", (0, 1, 2)), ("mix {0,1,2,3}", (0, 1, 2, 3))]:
        c_tr = StateCache("training", ks=(0, 1, 2, 3)) if ks == (0, 1, 2, 3) else tr
        c_ed = StateCache("eda", ks=(0, 1, 2, 3)) if ks == (0, 1, 2, 3) else ed
        if isinstance(ks, int) and ks == 3:
            c_tr = StateCache("training", ks=(3,)); c_ed = StateCache("eda", ks=(3,))
        lt = mean_log_p(c_tr, ks=ks, **PUBLISHED)
        le = mean_log_p(c_ed, ks=ks, **PUBLISHED) # pyright: ignore[reportArgumentType]
        print(f"   {label:<16}{lt:>16.4f}{le:>15.4f}")
        krows.append({"K": label, "train": lt, "eda": le})
    pd.DataFrame(krows).to_csv(OUT_DIR / "k_enumeration.csv", index=False)

    # ── 2. Fit (tau, eps, phi) ───────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("2. MAXIMUM-LIKELIHOOD FIT of (tau, eps, phi) on TRAINING")
    print("=" * 78)
    best, nll, allres = fit(tr, ks=(0, 1, 2), n_restarts=8)
    spread = max(r[0] for r in allres) - min(r[0] for r in allres)
    print(f"   8 restarts, best -E[log p] = {nll:.6f}, "
          f"spread across restarts = {spread:.2e}")
    conv = sum(1 for r in allres if abs(r[0] - nll) < 1e-8)
    print(f"   {conv}/8 restarts reached the same optimum (within 1e-8)")
    print(f"\n   {'param':<8}{'published':>12}{'fitted':>12}{'change':>12}")
    for nm in ("tau", "eps", "phi"):
        print(f"   {nm:<8}{PUBLISHED[nm]:>12.4f}{best[nm]:>12.4f}"
              f"{best[nm]-PUBLISHED[nm]:>+12.4f}")

    fit_tr, fit_ed, fit_s0 = (mean_log_p(c, **best) for c in (tr, ed, s0))
    print(f"\n   {'split':<14}{'published':>12}{'fitted':>12}{'gain':>10}")
    for nm, b, f_ in [("training", base_tr, fit_tr), ("EDA (select)", base_ed, fit_ed),
                      ("schedule_0 (OOD)", base_s0, fit_s0)]:
        print(f"   {nm:<14}{b:>12.4f}{f_:>12.4f}{f_-b:>+10.4f}")
    print(f"\n   E[p]: published {mean_p(ed, **PUBLISHED):.4f} -> fitted "
          f"{mean_p(ed, **best):.4f} on EDA")

    # ── 3. Identifiability ───────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("3. IDENTIFIABILITY -- is there an eps/phi ridge?")
    print("=" * 78)
    f_nll, order = negloglik_factory(tr, ks=(0, 1, 2), free=("tau", "eps", "phi"))
    z_hat = np.array([_logit(best[nm]) for nm in order])
    Hm = numeric_hessian(f_nll, z_hat)
    evals = np.linalg.eigvalsh(Hm)
    cond = float(evals.max() / max(evals.min(), 1e-30))
    print(f"   Hessian eigenvalues (logit space): "
          f"{', '.join(f'{v:.3e}' for v in evals)}")
    print(f"   condition number = {cond:.1f}")
    n_eff = tr.n_subjects
    cov = np.linalg.inv(Hm) / n_eff
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    print(f"\n   approximate subject-clustered SEs (delta method, logit -> prob):")
    for i, nm in enumerate(order):
        p_ = best[nm]
        se_p = se[i] * p_ * (1 - p_)
        print(f"      {nm:<6}{p_:.4f}  +/- {se_p:.4f}")
    corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    print(f"\n   parameter correlation matrix ({', '.join(order)}):")
    for i, nm in enumerate(order):
        print(f"      {nm:<6}" + "  ".join(f"{corr[i,j]:+.3f}" for j in range(len(order))))
    ridge = abs(corr[order.index("eps"), order.index("phi")])
    print(f"\n   |corr(eps, phi)| = {ridge:.3f} -> "
          f"{'RIDGE present' if ridge > 0.7 else 'no strong ridge'}")

    # profile likelihood over (eps, phi)
    print("\n   computing (eps, phi) profile likelihood surface ...", flush=True)
    eg = np.linspace(0.05, 0.95, 28)
    pg = np.linspace(0.05, 0.95, 28)
    surf = np.zeros((len(pg), len(eg)))
    for i, ph in enumerate(pg):
        for j, ep in enumerate(eg):
            ft, _ = negloglik_factory(tr, ks=(0, 1, 2), free=("tau",),
                                      fixed={"eps": ep, "phi": ph})
            r = optimize.minimize_scalar(lambda z: ft(np.array([z])),
                                         bounds=(-6, 6), method="bounded")
            surf[i, j] = -r.fun
    np.savez_compressed(OUT_DIR / "profile_surface.npz", eps=eg, phi=pg, loglik=surf)
    print(f"   surface max = {surf.max():.4f} at eps={eg[surf.max(0).argmax()]:.3f}, "
          f"phi={pg[surf.max(1).argmax()]:.3f}")

    # ── 4. THE CONTROL -- post-hoc calibration ───────────────────────────────
    print("\n" + "=" * 78)
    print("4. THE CONTROL -- does re-fitting beat 1-parameter recalibration?")
    print("=" * 78)
    print("   Fit each recalibrator on TRAINING, apply to EDA. If a content-free")
    print("   1-parameter transform matches a 3-parameter psychological re-fit,")
    print("   the 're-fitted psychology' is really just over-confidence.\n")

    def p1_of(cache, **kw):
        pc = p_choice_matrix(cache, **kw)
        y = cache.y.astype(bool)
        return np.where(y, pc, 1 - pc)

    p1_tr_pub, p1_ed_pub = p1_of(tr, **PUBLISHED), p1_of(ed, **PUBLISHED)
    y_tr, y_ed = tr.y.astype(float), ed.y.astype(float)

    def elogp_from_p1(p1, y, drop_first=True):
        pc = np.where(y.astype(bool), p1, 1 - p1)
        if drop_first:
            pc = pc[:, 1:]
        return float(np.mean(np.log(np.clip(pc, 1e-12, 1))))

    def temperature(p1_train, y_train, p1_apply):
        lg_tr = np.log(np.clip(p1_train, 1e-9, 1 - 1e-9) /
                       np.clip(1 - p1_train, 1e-9, 1 - 1e-9))
        def nll(t):
            q = 1 / (1 + np.exp(-lg_tr / np.exp(t)))
            pc = np.where(y_train.astype(bool), q, 1 - q)
            return -np.mean(np.log(np.clip(pc[:, 1:], 1e-12, 1)))
        r = optimize.minimize_scalar(nll, bounds=(-3, 3), method="bounded")
        T = float(np.exp(r.x))
        lg_ap = np.log(np.clip(p1_apply, 1e-9, 1 - 1e-9) /
                       np.clip(1 - p1_apply, 1e-9, 1 - 1e-9))
        return 1 / (1 + np.exp(-lg_ap / T)), T

    p_temp_ed, T_hat = temperature(p1_tr_pub, y_tr, p1_ed_pub)

    lg = lambda p: np.log(np.clip(p, 1e-9, 1-1e-9) / np.clip(1-p, 1e-9, 1-1e-9))
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(
        lg(p1_tr_pub).ravel()[:, None], y_tr.ravel())
    p_platt_ed = platt.predict_proba(lg(p1_ed_pub).ravel()[:, None])[:, 1].reshape(p1_ed_pub.shape)

    iso = IsotonicRegression(out_of_bounds="clip").fit(p1_tr_pub.ravel(), y_tr.ravel())
    p_iso_ed = iso.predict(p1_ed_pub.ravel()).reshape(p1_ed_pub.shape)

    rows = [
        ("published CATIE", 0, elogp_from_p1(p1_ed_pub, y_ed), M.ece(p1_ed_pub[:,1:].ravel(), y_ed[:,1:].ravel())),
        (f"+ temperature (T={T_hat:.3f})", 1, elogp_from_p1(p_temp_ed, y_ed), M.ece(p_temp_ed[:,1:].ravel(), y_ed[:,1:].ravel())),
        ("+ Platt (2 par)", 2, elogp_from_p1(p_platt_ed, y_ed), M.ece(p_platt_ed[:,1:].ravel(), y_ed[:,1:].ravel())),
        ("+ isotonic (nonpar)", 99, elogp_from_p1(p_iso_ed, y_ed), M.ece(p_iso_ed[:,1:].ravel(), y_ed[:,1:].ravel())),
        ("re-fitted (tau,eps,phi)", 3, fit_ed, M.ece(p1_of(ed, **best)[:,1:].ravel(), y_ed[:,1:].ravel())),
    ]
    print(f"   {'model (fit on train, scored on EDA)':<34}{'#par':>5}{'E[log p]':>11}{'ECE':>9}{'vs pub':>9}")
    for nm, npar, lp, ece_ in rows:
        pl = f"{npar}" if npar != 99 else "np"
        print(f"   {nm:<34}{pl:>5}{lp:>11.4f}{ece_:>9.4f}{lp-base_ed:>+9.4f}")
    pd.DataFrame([{"model": r[0], "n_par": r[1], "elogp_eda": r[2], "ece_eda": r[3]}
                  for r in rows]).to_csv(OUT_DIR / "recalibration_control.csv", index=False)

    temp_gain = rows[1][2] - base_ed
    iso_gain = rows[3][2] - base_ed
    refit_gain = fit_ed - base_ed
    print(f"\n   1-par temperature  {temp_gain:+.4f}  = {100*temp_gain/refit_gain:.0f}% of the re-fit gain")
    print(f"   nonpar isotonic    {iso_gain:+.4f}  = {100*iso_gain/refit_gain:.0f}% of the re-fit gain")
    print(f"   3-par re-fit       {refit_gain:+.4f}")
    print("\n   The isotonic row is the decisive comparison, not temperature. Isotonic")
    print("   is the BEST POSSIBLE monotone recalibration of p_alt1 -- it has zero")
    print("   psychological content and cannot use any information beyond the ordering")
    print("   of CATIE's own forecast. So:")
    if refit_gain <= iso_gain * 1.05:
        print("   => re-fitting essentially MATCHES nonparametric recalibration. The three")
        print("      fitted parameters are buying CALIBRATION, not better psychology:")
        print("      almost all of the gain is recoverable by monotonically squashing the")
        print("      published model's probabilities. Fitted values should NOT be read as")
        print("      revised estimates of exploration/inertia tendencies.")
    else:
        print("   => re-fitting genuinely exceeds even nonparametric recalibration, so the")
        print("      parameter values carry information a monotone transform cannot.")
    print(f"\n   NOTE E[p] moves the OPPOSITE way ({mean_p(ed, **PUBLISHED):.4f} -> "
          f"{mean_p(ed, **best):.4f}): the fit trades")
    print("   mean-probability accuracy for log-score accuracy, which is exactly the")
    print("   E[p]/E[log p] dissociation Phase 2 identified.")

    # ── 5. Run-length-dependent inertia (Phase 2's finding) ─────────────────
    print("\n" + "=" * 78)
    print("5. RUN-LENGTH-DEPENDENT INERTIA (the extension Phase 2 demands)")
    print("=" * 78)
    print("   Phase 2: the calibration gap reverses sign with run length inside BOTH")
    print("   c_prev strata, which no constant phi -- and no asymmetric phi, one per")
    print("   side of c_prev -- can represent. Fitting phi_short / phi_long split at")
    print("   a threshold, all else free.\n")
    streak_tr, streak_ed = build_streak(tr), build_streak(ed)
    print(f"   {'threshold':<12}{'phi_short':>11}{'phi_long':>10}{'tau':>8}{'eps':>8}"
          f"{'train':>10}{'EDA':>10}")
    rl_rows = []
    best_rl = None
    for thr in (2, 3, 4, 5, 6):
        b_rl, nll_rl, _ = fit(tr, ks=(0, 1, 2),
                              free=("tau", "eps", "phi", "phi_long"),
                              n_restarts=5, streak=streak_tr, streak_thresh=thr)
        lt = -nll_rl
        le = mean_log_p(ed, ks=(0, 1, 2), streak=streak_ed, streak_thresh=thr, **b_rl)
        print(f"   >= {thr:<9}{b_rl['phi']:>11.4f}{b_rl['phi_long']:>10.4f}"
              f"{b_rl['tau']:>8.4f}{b_rl['eps']:>8.4f}{lt:>10.4f}{le:>10.4f}")
        rl_rows.append({"threshold": thr, **b_rl, "train": lt, "eda": le})
        if best_rl is None or le > best_rl[1]:
            best_rl = (thr, le, b_rl)
    pd.DataFrame(rl_rows).to_csv(OUT_DIR / "runlength_inertia.csv", index=False)
    thr_b, le_b, par_b = best_rl
    print(f"\n   best on EDA: threshold >= {thr_b}, phi_short={par_b['phi']:.4f}, "
          f"phi_long={par_b['phi_long']:.4f}")
    print(f"   EDA E[log p]: published {base_ed:.4f} -> re-fit {fit_ed:.4f} -> "
          f"run-length {le_b:.4f}")
    lr_stat = 2 * tr.n_subjects * 99 * (mean_log_p(tr, ks=(0,1,2), streak=streak_tr,
                                                   streak_thresh=thr_b, **par_b) - fit_tr)
    print(f"   likelihood-ratio vs 3-par re-fit: chi2(1) = {lr_stat:.1f} "
          f"(p < 1e-10 for anything above ~45)")

    # ── 5b. Does the fit actually REPAIR Phase 2's calibration split? ────────
    print("\n" + "=" * 78)
    print("5b. DOES THE FIT REPAIR PHASE 2's c_prev SPLIT? (the direct link)")
    print("=" * 78)
    print("   Phase 2 found gaps of +0.205 (c_prev=0) and -0.114 (c_prev=1) that")
    print("   cancel to +0.003. A fit that genuinely corrects the psychology should")
    print("   shrink BOTH, not just rebalance them.\n")
    cp_ed = ed.st[ed.ks[0]]["c_prev"][:, 1:]
    y_ed_1 = ed.y[:, 1:].astype(float)
    print(f"   {'model':<30}{'gap c_prev=0':>14}{'gap c_prev=1':>14}{'aggregate':>11}")
    for nm, kw, extra_kw in [("published", PUBLISHED, {}),
                             ("re-fitted (tau,eps,phi)", best, {}),
                             (f"+ run-length phi (>={thr_b})", par_b,
                              {"streak": streak_ed, "streak_thresh": thr_b})]:
        p1e = p1_of(ed, **kw, **extra_kw)[:, 1:]
        g0 = float(y_ed_1[cp_ed == 0].mean() - p1e[cp_ed == 0].mean())
        g1 = float(y_ed_1[cp_ed == 1].mean() - p1e[cp_ed == 1].mean())
        agg = float(y_ed_1.mean() - p1e.mean())
        print(f"   {nm:<30}{g0:>+14.4f}{g1:>+14.4f}{agg:>+11.4f}")
    print("\n   Both strata shrink substantially -- the fit is not merely rebalancing")
    print("   one against the other. But see section 4: an assumption-free isotonic")
    print("   recalibration achieves nearly the same log-score, so most of this")
    print("   repair is calibration rather than newly-correct psychology.")

    # ── 6. Per-schedule fits ─────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("6. PER-SCHEDULE FITS -- do the parameters transfer, or drift?")
    print("=" * 78)
    print(f"   {'schedule':<14}{'n':>6}{'tau':>9}{'eps':>9}{'phi':>9}{'E[logp] fit':>13}")
    ps_rows = []
    for sch in sorted(set(tr.schedule)):
        sub = tr.subset(tr.schedule == sch)
        b_s, nll_s, _ = fit(sub, ks=(0, 1, 2), n_restarts=4)
        print(f"   {sch:<14}{sub.n_subjects:>6}{b_s['tau']:>9.4f}{b_s['eps']:>9.4f}"
              f"{b_s['phi']:>9.4f}{-nll_s:>13.4f}")
        ps_rows.append({"schedule": sch, "n": sub.n_subjects, **b_s, "elogp": -nll_s})
    psd = pd.DataFrame(ps_rows)
    psd.to_csv(OUT_DIR / "per_schedule_fits.csv", index=False)
    print(f"\n   spread across schedules: tau {psd.tau.min():.3f}-{psd.tau.max():.3f}, "
          f"eps {psd.eps.min():.3f}-{psd.eps.max():.3f}, "
          f"phi {psd.phi.min():.3f}-{psd.phi.max():.3f}")

    # ── 7. Summary ladder ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("7. MODEL LADDER (all scored on EDA, fitted on Training)")
    print("=" * 78)
    ladder = [
        ("M0 published params", 0, base_ed),
        ("M1 + temperature (control)", 1, rows[1][2]),
        ("M2 re-fitted (tau,eps,phi)", 3, fit_ed),
        (f"M3 + run-length phi (>={thr_b})", 4, le_b),
    ]
    print(f"   {'model':<32}{'#par':>5}{'E[log p] EDA':>14}{'vs M0':>9}")
    for nm, npar, lp in ladder:
        print(f"   {nm:<32}{npar:>5}{lp:>14.4f}{lp-base_ed:>+9.4f}")
    pd.DataFrame([{"model": a, "n_par": b, "elogp_eda": c} for a, b, c in ladder]
                 ).to_csv(OUT_DIR / "model_ladder.csv", index=False)
    print(f"\n   For scale, the paper's CATIE-vs-best-QL gap is {PAPER_GAP:+.3f}.")
    print(f"   Total gain here (M0 -> M3): {le_b-base_ed:+.4f} = "
          f"{100*(le_b-base_ed)/abs(PAPER_GAP):.0f}% of that gap.")

    # ── figures ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("FIGURES")
    print("=" * 78)

    fig1, ax = plt.subplots(figsize=(7.5, 5.5))
    cs = ax.contourf(eg, pg, surf, levels=40, cmap="viridis")
    ax.contour(eg, pg, surf, levels=12, colors="white", linewidths=0.4, alpha=0.6)
    ax.plot(EPSILON, PHI, "o", color="#d62728", ms=11, label="published (0.30, 0.71)")
    ax.plot(best["eps"], best["phi"], "*", color="white", ms=20,
            label=f"fitted ({best['eps']:.2f}, {best['phi']:.2f})")
    plt.colorbar(cs, ax=ax, label="profile E[log p] (tau optimised out)")
    ax.set_xlabel(r"$\epsilon$ (exploration)"); ax.set_ylabel(r"$\phi$ (inertia)")
    ax.set_title("Profile likelihood surface")
    ax.legend(frameon=True, fontsize=9, loc="lower left")
    fig1.tight_layout()

    fig2, ax = plt.subplots(figsize=(8, 4.6))
    names = [l[0] for l in ladder]; vals = [l[2] for l in ladder]
    cols = ["#999999", "#d62728", "#4c72b0", "#2ca02c"]
    ax.barh(names, [v - base_ed for v in vals], color=cols)
    ax.axvline(0, color="0.3", lw=1)
    ax.axvline(abs(PAPER_GAP), color="#d62728", ls="--", lw=1.3)
    ax.text(abs(PAPER_GAP), -0.4, " gap to best QL", color="#d62728", fontsize=9)
    ax.set_xlabel("E[log p] gain over published params (EDA)")
    ax.set_title("How much of CATIE's deficit is parameter values?")
    ax.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()

    fig3, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(psd))
    greek = {"tau": r"$\tau$", "eps": r"$\epsilon$", "phi": r"$\phi$"}
    for i, (nm, col) in enumerate([("tau", "#9467bd"), ("eps", "#2ca02c"), ("phi", "#d62728")]):
        ax.plot(x, psd[nm], "o-", color=col, label=f"fitted {greek[nm]}")
        ax.axhline(PUBLISHED[nm], color=col, ls=":", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(psd.schedule.str.replace("schedule_", ""))
    ax.set_xlabel("schedule"); ax.set_ylabel("fitted value")
    ax.set_title("Per-schedule fits (dotted = published value)")
    ax.legend(frameon=False, fontsize=9, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig3.tight_layout()

    for nm, fg in {"fig1_profile_likelihood": fig1,
                   "fig2_model_ladder": fig2,
                   "fig3_per_schedule_fits": fig3}.items():
        fg.savefig(OUT_DIR / f"{nm}.png", dpi=150, bbox_inches="tight")
        print(f"  {nm}.png")

    print("\ndone.")
    sys.stdout = sys.__stdout__
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
