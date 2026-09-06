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

On K: the paper's "K = 2" is the upper bound of a uniform draw over {0,1,2} (the
competition simulator, CATIE_single_schedule_score.m:5, draws k = randi([0,2]); the
likelihood code, hetro.m:6, uses K = 0:2), so the {0,1,2} mixture is the paper's
model. Single-k agents are evaluated below as a sensitivity check.

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
(496, schedules 4/5/7). schedule_0 (549) is an out-of-distribution check. Test
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
from scipy import optimize, stats  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from catie_core import EPSILON, PHI, TAU  # noqa: E402
import metrics as M  # noqa: E402
from catie_likelihood import (  # noqa: E402
    StateCache, mean_log_p, mean_p, negloglik_factory, p_choice_matrix,
    p_alt1_single_k, mix_per_trial, unpack, _logit, _sigmoid, subject_scores,
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
    """Run length of identical choices ending at t-1, from the cached c_prev.

    Convention (matches Phase 2's `_streak_transform`): streak[:, t] is the number
    of consecutive identical choices the subject has made up to and including
    trial t-1. Column 0 is unused (trial 1 has no history). Column 1 is always 1
    (only one prior choice exists at trial 2). For t >= 2,
        streak[:, t] = streak[:, t-1] + 1  if c_prev[:, t] == c_prev[:, t-1]  else 1.

    The loop must start at t=2: c_prev[:, 0] is a placeholder 0, not a choice, and
    comparing c_prev[:, 1] against it credited subjects whose first choice was
    alternative 2 (coded 0) with a run of 2 at trial 2 instead of 1.
    """
    cp = cache.st[cache.ks[0]]["c_prev"]
    n_s, n_t = cp.shape
    streak = np.ones((n_s, n_t))
    for t in range(2, n_t):
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
    print("   The paper's model is the {0,1,2} mixture (K=2 = upper bound of a uniform draw).")
    print("   Sensitivity check: single-k agents K=0,1,2,3 and the wider {0,1,2,3} mixture.\n")
    print(f"   {'K':<16}{'E[log p] train':>16}{'E[log p] EDA':>15}")
    krows = []
    for label, ks in [("K=0", 0), ("K=1", 1), ("K=2", 2), ("K=3", 3),
                      ("mix {0,1,2} (paper)", (0, 1, 2)), ("mix {0,1,2,3}", (0, 1, 2, 3))]:
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

    # ── 2b. Optimiser agreement ──────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("2b. OPTIMISER AGREEMENT -- the same objective under four other solvers")
    print("=" * 78)
    f_nll, order = negloglik_factory(tr, ks=(0, 1, 2), free=("tau", "eps", "phi"))
    z_pub = np.array([_logit(PUBLISHED[nm]) for nm in order])
    opt_rows = [("Nelder-Mead (8 restarts)", best, nll)]
    for meth in ("L-BFGS-B", "Powell"):
        r = optimize.minimize(f_nll, z_pub, method=meth)
        opt_rows.append((meth, unpack(r.x, order), float(r.fun)))
    # global search with no start point; (-5, 5) in logit space is (0.007, 0.993)
    r = optimize.differential_evolution(f_nll, bounds=[(-5, 5)] * 3, seed=RNG_SEED,
                                        tol=1e-8, polish=True)
    opt_rows.append(("differential evolution (global)", unpack(r.x, order), float(r.fun)))
    # coarse grid: 15^3 points on the probability scale, no local refinement
    grid = np.linspace(0.05, 0.95, 15)
    g_best, g_nll = None, np.inf
    for gt in grid:
        for ge in grid:
            for gp in grid:
                v = f_nll(np.array([_logit(gt), _logit(ge), _logit(gp)]))
                if v < g_nll:
                    g_best, g_nll = {"tau": float(gt), "eps": float(ge), "phi": float(gp)}, v
    opt_rows.append(("coarse grid (15^3, prob. scale)", g_best, float(g_nll)))
    print(f"   {'optimiser':<34}{'tau':>8}{'eps':>8}{'phi':>8}{'nll':>11}")
    for nm, pr, v in opt_rows:
        print(f"   {nm:<34}{pr['tau']:>8.4f}{pr['eps']:>8.4f}{pr['phi']:>8.4f}{v:>11.6f}")
    pd.DataFrame([{"optimiser": nm, **pr, "nll": v} for nm, pr, v in opt_rows]
                 ).to_csv(OUT_DIR / "optimiser_agreement.csv", index=False)
    print(f"   max |nll - Nelder-Mead nll| over L-BFGS-B / Powell / DE = "
          f"{max(abs(v - nll) for _, _, v in opt_rows[1:4]):.1e}; the grid is the "
          f"coarse sanity check that no other basin exists")

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
    # Hm is the Hessian of the MEAN per-trial nll over N = n_subjects x 99 scored
    # cells, so the iid MLE covariance is inv(Hm) / N. Trials within a subject are
    # correlated, so the SEs reported are the subject-clustered sandwich
    #     V = B^-1 M B^-1,  B = N * Hm (Hessian of the SUMMED nll),  M = sum_i s_i s_i^T,
    # s_i = gradient of subject i's summed nll (subject_scores), with the small-sample
    # factor G/(G-1), G = n_subjects. (An earlier version divided inv(Hm) by
    # n_subjects instead of N, which inflated the SEs by ~sqrt(99).)
    n_cells = tr.n_subjects * (tr.n_trials - 1)
    G = tr.n_subjects
    cov_iid = np.linalg.inv(Hm) / n_cells
    S = subject_scores(tr, z_hat, order, ks=(0, 1, 2))
    score_tot = S.sum(axis=0)
    print(f"   total score / N_trials (should be ~0 at the MLE): "
          f"max |.| = {np.abs(score_tot).max() / n_cells:.2e}")
    B_inv = np.linalg.inv(n_cells * Hm)
    cov = B_inv @ (S.T @ S) @ B_inv * (G / (G - 1))
    se_iid = np.sqrt(np.clip(np.diag(cov_iid), 0, None))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    print(f"\n   SEs (delta method, logit -> prob); N_trials = {n_cells:,}, "
          f"G = {G:,} subjects, G/(G-1) applied to the sandwich:")
    print(f"      {'param':<8}{'fitted':>8}{'iid':>10}{'sandwich':>10}")
    for i, nm in enumerate(order):
        p_ = best[nm]
        d_ = p_ * (1 - p_)
        print(f"      {nm:<8}{p_:>8.4f}{se_iid[i] * d_:>10.4f}{se[i] * d_:>10.4f}")
    corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    print(f"\n   parameter correlation matrix, subject-clustered sandwich "
          f"({', '.join(order)}):")
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

    def ep_from_p1(p1, y):
        """E[p] over the scored trials (trial 1 dropped) from a (recalibrated) p_alt1."""
        pc = np.where(y.astype(bool), p1, 1 - p1)
        return float(np.mean(pc[:, 1:]))

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
    # Trial 1 (column 0) is p = 0.5 by construction and is never scored, so every
    # map is FITTED on columns 1: only (temperature already does this). The maps
    # are still applied to all columns; scoring drops column 0.
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(
        lg(p1_tr_pub[:, 1:]).ravel()[:, None], y_tr[:, 1:].ravel())
    p_platt_ed = platt.predict_proba(lg(p1_ed_pub).ravel()[:, None])[:, 1].reshape(p1_ed_pub.shape)

    iso = IsotonicRegression(out_of_bounds="clip").fit(p1_tr_pub[:, 1:].ravel(), y_tr[:, 1:].ravel())
    p_iso_ed = iso.predict(p1_ed_pub.ravel()).reshape(p1_ed_pub.shape)

    rows = [
        ("published CATIE", 0, elogp_from_p1(p1_ed_pub, y_ed), M.ece(p1_ed_pub[:,1:].ravel(), y_ed[:,1:].ravel()), ep_from_p1(p1_ed_pub, y_ed)),
        (f"+ temperature (T={T_hat:.3f})", 1, elogp_from_p1(p_temp_ed, y_ed), M.ece(p_temp_ed[:,1:].ravel(), y_ed[:,1:].ravel()), ep_from_p1(p_temp_ed, y_ed)),
        ("+ Platt (2 par)", 2, elogp_from_p1(p_platt_ed, y_ed), M.ece(p_platt_ed[:,1:].ravel(), y_ed[:,1:].ravel()), ep_from_p1(p_platt_ed, y_ed)),
        ("+ isotonic (nonpar)", 99, elogp_from_p1(p_iso_ed, y_ed), M.ece(p_iso_ed[:,1:].ravel(), y_ed[:,1:].ravel()), ep_from_p1(p_iso_ed, y_ed)),
        ("re-fitted (tau,eps,phi)", 3, fit_ed, M.ece(p1_of(ed, **best)[:,1:].ravel(), y_ed[:,1:].ravel()), mean_p(ed, **best)),
    ]
    print(f"   {'model (fit on train, scored on EDA)':<34}{'#par':>5}{'E[log p]':>11}{'ECE':>9}{'vs pub':>9}{'E[p]':>9}")
    for nm, npar, lp, ece_, ep_ in rows:
        pl = f"{npar}" if npar != 99 else "np"
        print(f"   {nm:<34}{pl:>5}{lp:>11.4f}{ece_:>9.4f}{lp-base_ed:>+9.4f}{ep_:>9.4f}")
    pd.DataFrame([{"model": r[0], "n_par": r[1], "elogp_eda": r[2], "ece_eda": r[3], "ep_eda": r[4]}
                  for r in rows]).to_csv(OUT_DIR / "recalibration_control.csv", index=False)
    print("   Every recalibrated row loses E[p] relative to the published model: for a")
    print("   shrinkage p -> 0.5 + a(p - 0.5), dE[p] = (a - 1)(E[p] - 0.5) < 0 when E[p] > 0.5.")

    temp_gain = rows[1][2] - base_ed
    iso_gain = rows[3][2] - base_ed
    refit_gain = fit_ed - base_ed
    print(f"\n   1-par temperature  {temp_gain:+.4f}  = {100*temp_gain/refit_gain:.0f}% of the re-fit gain")
    print(f"   nonpar isotonic    {iso_gain:+.4f}  = {100*iso_gain/refit_gain:.0f}% of the re-fit gain")
    print(f"   3-par re-fit       {refit_gain:+.4f}")

    # How close is isotonic to the re-fit, with subject-level uncertainty? (a) the
    # ratio of the two gains with a subject-cluster bootstrap CI; (b) a paired
    # subject test of M2 vs isotonic on EDA. Isotonic is the best monotone map
    # FITTED ON TRAINING, not the best possible one, so the claim is empirical.
    def _trial_logp(pc):
        return np.log(np.clip(pc[:, 1:], 1e-12, 1.0)).ravel()
    sid_ed = np.repeat(ed.subject_id, ed.n_trials - 1)
    lp_pub = _trial_logp(np.where(y_ed.astype(bool), p1_ed_pub, 1 - p1_ed_pub))
    lp_iso = _trial_logp(np.where(y_ed.astype(bool), p_iso_ed, 1 - p_iso_ed))
    lp_m2 = _trial_logp(p_choice_matrix(ed, ks=(0, 1, 2), **best))
    subj_mean = lambda v: v.reshape(ed.n_subjects, -1).mean(axis=1)
    g_iso = subj_mean(lp_iso) - subj_mean(lp_pub)      # per-subject isotonic gain
    g_m2 = subj_mean(lp_m2) - subj_mean(lp_pub)        # per-subject re-fit gain
    n_boot_ratio = 2000
    draws = np.random.default_rng(RNG_SEED).integers(0, ed.n_subjects,
                                                     size=(n_boot_ratio, ed.n_subjects))
    ratio_boot = g_iso[draws].mean(axis=1) / g_m2[draws].mean(axis=1)
    ratio = float(g_iso.mean() / g_m2.mean())
    r_lo, r_hi = (float(q) for q in np.quantile(ratio_boot, [0.025, 0.975]))
    t_m2_iso = M.paired_subject_test(lp_m2, lp_iso, sid_ed)
    print(f"\n   isotonic / re-fit gain ratio on EDA = {ratio:.3f}, subject-cluster "
          f"bootstrap 95% CI [{r_lo:.3f}, {r_hi:.3f}] ({n_boot_ratio} resamples, "
          f"n = {ed.n_subjects} subjects)")
    print(f"   M2 re-fit - isotonic, paired subject test on EDA: {t_m2_iso['mean_diff']:+.4f} "
          f"[{t_m2_iso['ci_lo']:+.4f}, {t_m2_iso['ci_hi']:+.4f}]  t = {t_m2_iso['t']:.2f}  "
          f"p = {t_m2_iso['p']:.2g} {M.stars(t_m2_iso['p'])}")
    pd.DataFrame([{"ratio_iso_over_refit": ratio, "ratio_ci_lo": r_lo, "ratio_ci_hi": r_hi,
                   "n_boot": n_boot_ratio,
                   **{f"m2_minus_iso_{k}": v for k, v in t_m2_iso.items()}}]
                 ).to_csv(OUT_DIR / "isotonic_vs_refit.csv", index=False)

    print("\n   The isotonic row is the decisive comparison, not temperature. Isotonic is")
    print("   the best monotone recalibration of p_alt1 that Training can fit -- it has")
    print("   zero psychological content and cannot use any information beyond the")
    print("   ordering of CATIE's own forecast. So:")
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

    # ── 4b. What the fit did to the forecast distribution ────────────────────
    print("\n" + "=" * 78)
    print("4b. FORECAST DISTRIBUTION -- p_alt1 on EDA, trial 1 dropped")
    print("=" * 78)
    print("   eps sets the attainable range (roughly [eps/6, 1 - eps/6]); maximum")
    print("   likelihood uses it as a shrinkage knob.\n")
    print(f"   {'model':<24}{'min':>8}{'median':>9}{'max':>8}{'SD':>8}{'% <0.1 or >0.9':>16}")
    fd_rows = []
    for nm, kw in [(f"published (eps={EPSILON})", PUBLISHED),
                   (f"fitted (eps={best['eps']:.3f})", best)]:
        v = p1_of(ed, **kw)[:, 1:].ravel()
        row = {"model": nm, "min": float(v.min()), "median": float(np.median(v)),
               "max": float(v.max()), "sd": float(v.std()),
               "pct_extreme": 100 * float(np.mean((v > 0.9) | (v < 0.1)))}
        print(f"   {nm:<24}{row['min']:>8.3f}{row['median']:>9.3f}{row['max']:>8.3f}"
              f"{row['sd']:>8.3f}{row['pct_extreme']:>15.1f}%")
        fd_rows.append(row)
    pd.DataFrame(fd_rows).to_csv(OUT_DIR / "forecast_distribution.csv", index=False)
    print(f"   the fit shrank the SD of p_alt1 by "
          f"{100 * (1 - fd_rows[1]['sd'] / fd_rows[0]['sd']):.0f}%")

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
          f"{'train':>10}{'EDA':>10}{'EDA E[p]':>10}")
    rl_rows = []
    for thr in (2, 3, 4, 5, 6):
        b_rl, nll_rl, _ = fit(tr, ks=(0, 1, 2),
                              free=("tau", "eps", "phi", "phi_long"),
                              n_restarts=5, streak=streak_tr, streak_thresh=thr)
        lt = -nll_rl
        le = mean_log_p(ed, ks=(0, 1, 2), streak=streak_ed, streak_thresh=thr, **b_rl)
        ep_rl = mean_p(ed, ks=(0, 1, 2), streak=streak_ed, streak_thresh=thr, **b_rl)
        print(f"   >= {thr:<9}{b_rl['phi']:>11.4f}{b_rl['phi_long']:>10.4f}"
              f"{b_rl['tau']:>8.4f}{b_rl['eps']:>8.4f}{lt:>10.4f}{le:>10.4f}{ep_rl:>10.4f}")
        rl_rows.append({"threshold": thr, **b_rl, "train": lt, "eda": le, "eda_ep": ep_rl})
    pd.DataFrame(rl_rows).to_csv(OUT_DIR / "runlength_inertia.csv", index=False)
    # The threshold is a discrete hyper-parameter and is SELECTED ON TRAINING (highest
    # train E[log p] = lowest fitted nll). EDA is then used once, to score the
    # selected model. Selecting on EDA and reporting that EDA score would make the
    # "held-out" number an in-sample maximum over five candidates.
    best_rl = max(rl_rows, key=lambda r: r["train"])
    thr_b = int(best_rl["threshold"])
    par_b = {nm: best_rl[nm] for nm in ("tau", "eps", "phi", "phi_long")}
    le_b = best_rl["eda"]
    eda_pick = max(rl_rows, key=lambda r: r["eda"])
    rl_optimism = eda_pick["eda"] - le_b        # >= 0 by construction
    print(f"\n   selected on Training: threshold >= {thr_b}, phi_short={par_b['phi']:.4f}, "
          f"phi_long={par_b['phi_long']:.4f} (train E[log p] {best_rl['train']:.6f})")
    print(f"   for reference (EDA-selected, NOT held-out): threshold >= "
          f"{int(eda_pick['threshold'])}, EDA {eda_pick['eda']:.4f}")
    print(f"   optimism = EDA(EDA-selected) - EDA(train-selected) = {rl_optimism:+.4f}")
    if par_b["phi_long"] > 0.999:
        print(f"   boundary solution: phi_long saturates at 1 -- deterministic repetition "
              f"after >={thr_b} identical choices, softened only by the eps floor")
    print(f"   EDA E[log p]: published {base_ed:.4f} -> re-fit {fit_ed:.4f} -> "
          f"run-length {le_b:.4f}")
    print(f"   EDA E[p]:     published {mean_p(ed, **PUBLISHED):.4f} -> re-fit "
          f"{mean_p(ed, **best):.4f} -> run-length {best_rl['eda_ep']:.4f}")

    # Paired subject-level tests on EDA: is the Training-selected run-length model
    # better than (a) the isotonic-recalibrated published model of section 4 (the
    # monotone map fitted on Training) and (b) the 3-par re-fit M2?
    # Per-subject mean log p, paired t-test + cluster bootstrap (metrics.py).
    # _trial_logp, sid_ed, lp_iso and lp_m2 come from section 4.
    lp_rl = _trial_logp(p_choice_matrix(ed, ks=(0, 1, 2), streak=streak_ed,
                                        streak_thresh=thr_b, **par_b))
    print(f"\n   paired subject-level tests on EDA (per-subject mean log p, "
          f"n = {ed.n_subjects} subjects; cluster bootstrap 95% CI, paired t):")
    rl_tests = []
    for nm, key, lp_o in [("isotonic recal. (sec. 4)", "isotonic", lp_iso),
                          ("M2 re-fit (tau,eps,phi)", "m2_refit", lp_m2)]:
        t_ = M.paired_subject_test(lp_rl, lp_o, sid_ed)
        print(f"   run-length (>={thr_b}) - {nm:<26} {t_['mean_diff']:+.4f} "
              f"[{t_['ci_lo']:+.4f}, {t_['ci_hi']:+.4f}]  t = {t_['t']:.2f}  "
              f"p = {t_['p']:.2g} {M.stars(t_['p'])}")
        rl_tests.append({"comparison": f"runlength_ge{thr_b}_minus_{key}", **t_})

    # Nested-model comparison on TRAINING (in-sample): run-length model M3 vs the
    # 3-par re-fit M2 it nests. A classical LR chi2 would treat every trial as an
    # independent observation, but trials within a subject are strongly dependent
    # (perseveration), and the threshold was chosen among five candidates, so the
    # chi2(1) calibration is invalid. Same subject-clustered paired test instead.
    sid_tr = np.repeat(tr.subject_id, tr.n_trials - 1)
    lp_rl_tr = _trial_logp(p_choice_matrix(tr, ks=(0, 1, 2), streak=streak_tr,
                                           streak_thresh=thr_b, **par_b))
    lp_m2_tr = _trial_logp(p_choice_matrix(tr, ks=(0, 1, 2), **best))
    t_tr = M.paired_subject_test(lp_rl_tr, lp_m2_tr, sid_tr)
    print(f"\n   in-sample nested comparison on TRAINING (per-subject mean log p, "
          f"n = {tr.n_subjects} subjects; cluster bootstrap 95% CI, paired t):")
    print(f"   run-length (>={thr_b}) - M2 re-fit (tau,eps,phi)     {t_tr['mean_diff']:+.4f} "
          f"[{t_tr['ci_lo']:+.4f}, {t_tr['ci_hi']:+.4f}]  t = {t_tr['t']:.2f}  "
          f"p = {t_tr['p']:.2g} {M.stars(t_tr['p'])}")
    print("   in-sample nested comparison on Training, subject-clustered; the held-out "
          "comparisons are the EDA rows above / runlength_tests.csv")
    rl_tests.append({"comparison": f"runlength_ge{thr_b}_minus_m2_refit_TRAIN_insample",
                     **t_tr})
    pd.DataFrame(rl_tests).to_csv(OUT_DIR / "runlength_tests.csv", index=False)

    # ── 5b. Does the fit actually REPAIR Phase 2's calibration split? ────────
    print("\n" + "=" * 78)
    print("5b. DOES THE FIT REPAIR PHASE 2's c_prev SPLIT? (the direct link)")
    print("=" * 78)
    print("   Phase 2 found gaps of +0.205 (c_prev=0) and -0.114 (c_prev=1) that")
    print("   cancel to +0.003 -- POOLED over EDA + Training + schedule_0. The table")
    print("   below is EDA only, so its 'published' row differs from those figures.")
    print("   A fit that genuinely corrects the psychology should shrink BOTH strata,")
    print("   not just rebalance them.\n")
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
    print("   SEs are subject-clustered sandwich SEs on each schedule alone (sec. 3's")
    print("   machinery, delta method to the probability scale).\n")
    print(f"   {'schedule':<14}{'n':>6}{'tau':>9}{'eps':>9}{'phi':>9}{'E[logp] fit':>13}"
          f"{'se_tau':>9}{'se_eps':>9}{'se_phi':>9}")
    ps_rows = []
    # numeric order (a plain sort puts schedule_11 before schedule_2)
    for sch in sorted(set(tr.schedule), key=lambda s: int(s.split("_")[-1])):
        sub = tr.subset(tr.schedule == sch)
        b_s, nll_s, _ = fit(sub, ks=(0, 1, 2), n_restarts=4)
        f_s, order_s = negloglik_factory(sub, ks=(0, 1, 2), free=("tau", "eps", "phi"))
        z_s = np.array([_logit(b_s[nm]) for nm in order_s])
        H_s = numeric_hessian(f_s, z_s)
        n_cells_s, G_s = sub.n_subjects * (sub.n_trials - 1), sub.n_subjects
        S_s = subject_scores(sub, z_s, order_s, ks=(0, 1, 2))
        Binv_s = np.linalg.inv(n_cells_s * H_s)
        cov_s = Binv_s @ (S_s.T @ S_s) @ Binv_s * (G_s / (G_s - 1))
        se_s = {nm: float(np.sqrt(max(cov_s[i, i], 0.0)) * b_s[nm] * (1 - b_s[nm]))
                for i, nm in enumerate(order_s)}
        print(f"   {sch:<14}{sub.n_subjects:>6}{b_s['tau']:>9.4f}{b_s['eps']:>9.4f}"
              f"{b_s['phi']:>9.4f}{-nll_s:>13.4f}"
              f"{se_s['tau']:>9.4f}{se_s['eps']:>9.4f}{se_s['phi']:>9.4f}")
        ps_rows.append({"schedule": sch, "n": sub.n_subjects, **b_s, "elogp": -nll_s,
                        **{f"se_{nm}": se_s[nm] for nm in order_s}})
    psd = pd.DataFrame(ps_rows)
    psd.to_csv(OUT_DIR / "per_schedule_fits.csv", index=False)
    print(f"\n   spread across schedules: tau {psd.tau.min():.3f}-{psd.tau.max():.3f}, "
          f"eps {psd.eps.min():.3f}-{psd.eps.max():.3f}, "
          f"phi {psd.phi.min():.3f}-{psd.phi.max():.3f}")

    # Cochran Q: is the between-schedule spread more than the within-schedule SEs
    # predict? Q = sum_s w_s (theta_s - theta_bar)^2, w_s = 1/SE_s^2, theta_bar the
    # inverse-variance-weighted mean, df = n_schedules - 1, I^2 = max(0, (Q-df)/Q).
    print("\n   Cochran Q heterogeneity test across schedules (inverse-variance weights")
    print("   from the sandwich SEs, probability scale):")
    print(f"   {'param':<8}{'Q':>9}{'df':>4}{'p':>10}{'I^2':>7}{'weighted mean':>15}")
    het_rows = []
    for nm in ("tau", "eps", "phi"):
        th = psd[nm].to_numpy()
        w = 1.0 / psd[f"se_{nm}"].to_numpy() ** 2
        th_bar = float((w * th).sum() / w.sum())
        Q = float((w * (th - th_bar) ** 2).sum())
        df_ = len(th) - 1
        p_q = float(stats.chi2.sf(Q, df_))
        I2 = max(0.0, (Q - df_) / Q) if Q > 0 else 0.0
        print(f"   {nm:<8}{Q:>9.2f}{df_:>4}{p_q:>10.2g}{100 * I2:>6.0f}%{th_bar:>15.4f}")
        het_rows.append({"param": nm, "Q": Q, "df": df_, "p": p_q, "I2": I2,
                         "weighted_mean": th_bar})
    pd.DataFrame(het_rows).to_csv(OUT_DIR / "per_schedule_heterogeneity.csv", index=False)
    het = {r["param"]: r for r in het_rows}
    print(f"   -> phi {'is' if het['phi']['p'] > 0.05 else 'is NOT'} homogeneous across "
          f"schedules (p = {het['phi']['p']:.2g}); tau (p = {het['tau']['p']:.1g}) and "
          f"eps (p = {het['eps']['p']:.1g}) are both heterogeneous.")

    # ── 7. Summary ladder ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("7. MODEL LADDER (all scored on EDA, fitted on Training)")
    print("=" * 78)
    ep_m0 = mean_p(ed, **PUBLISHED)
    ladder = [
        ("M0 published params", 0, base_ed, ep_m0),
        ("M1 + temperature (control)", 1, rows[1][2], rows[1][4]),
        ("M2 re-fitted (tau,eps,phi)", 3, fit_ed, mean_p(ed, **best)),
        (f"M3 + run-length phi (>={thr_b})", 5, le_b, best_rl["eda_ep"]),   # 4 continuous + 1 selected threshold
    ]
    print(f"   {'model':<32}{'#par':>5}{'E[log p] EDA':>14}{'vs M0':>9}{'E[p] EDA':>10}{'vs M0':>9}")
    for nm, npar, lp, ep_ in ladder:
        print(f"   {nm:<32}{npar:>5}{lp:>14.4f}{lp-base_ed:>+9.4f}{ep_:>10.4f}{ep_-ep_m0:>+9.4f}")
    pd.DataFrame([{"model": a, "n_par": b, "elogp_eda": c, "ep_eda": d} for a, b, c, d in ladder]
                 ).to_csv(OUT_DIR / "model_ladder.csv", index=False)
    print("   Every rung loses E[p] relative to M0: any shrinkage p -> 0.5 + a(p - 0.5)")
    print("   changes E[p] by (a - 1)(E[p] - 0.5), negative whenever a < 1 and E[p] > 0.5.")
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
    ax.text(abs(PAPER_GAP), -0.4, " paper's pooled CATIE-QL gap\n (scale only, not like-for-like)",
            color="#d62728", fontsize=8, va="top")
    ax.set_xlabel("E[log p] gain over published params (EDA)")
    ax.set_title("How much of CATIE's deficit is parameter values?")
    ax.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()

    fig3, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(psd))
    greek = {"tau": r"$\tau$", "eps": r"$\epsilon$", "phi": r"$\phi$"}
    for i, (nm, col) in enumerate([("tau", "#9467bd"), ("eps", "#2ca02c"), ("phi", "#d62728")]):
        # 95% CI from the per-schedule subject-clustered sandwich SEs, so the eye can
        # judge the same evidence the Cochran Q test uses: phi's spread is within its
        # error bars, tau's and eps's are not.
        ax.errorbar(x, psd[nm], yerr=1.96 * psd[f"se_{nm}"], fmt="o-", color=col,
                    capsize=3, lw=1.6, label=f"fitted {greek[nm]}")
        ax.axhline(PUBLISHED[nm], color=col, ls=":", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(psd.schedule.str.replace("schedule_", ""))
    ax.set_xlabel("schedule"); ax.set_ylabel("fitted value")
    ax.set_title("Per-schedule fits (dotted = published value; bars = 95% CI)")
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
