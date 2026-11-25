# -*- coding: utf-8 -*-
"""Unified recomputation for the bpfix relock (2026-11-23): bpfix fix round + bpfixdrop18
(drop 18 boundary code-2 trials), scipy-default Wilcoxon, BH within manuscript families.
Derived from segfix-bh-recalc.py; inputs changed to the bpfix / bpfixdrop18 CSVs.
Adds: balanced-accuracy chance family (drop18 leaves 11 subjects with unequal class
counts, so the sensitivity chance tests are reported as balanced accuracy) and the
four mixed-versus-single-parameter ablation paired contrasts with BH correction
(the q = 0.075 behind the 2026-11-24 attribution downgrade).
Writes bpfix_bh_recalc.json next to this script and checks headline values against
VERIFY-20261123-bpfix.md."""
import os, json
import numpy as np, pandas as pd
from scipy import stats

UF = os.path.expanduser("~/workspace/user/files")
D = os.path.expanduser("~/workspace/goals/2029-qs-200/files/stroke-aperiodic-paper")

F = {
 "bpfix": dict(probe="probe_results_bpfix_20261123_155344.csv", robust="robust_results_bpfix_20261123_155910.csv",
              abl="ablation_results_bpfix_20261123_161329.csv", exp3="exp3_results_bpfix_20261123_160348.csv",
              calA="exp3_svm_calA_results_bpfix_20261123_160833.csv",
              ff="xuanwu_refit_fixedfull_subject_acc_bpfix_20261123_161803.csv",
              rf="xuanwu_refit335_subject_acc_bpfix_20261123_162244.csv",
              riem="riemann_results_segfix_20261121_204511.csv",
              exp4="exp4_subject_data_segfix_20261121_205107.csv"),
 "bpfixdrop18": dict(probe="probe_results_bpfixdrop18_20261123_162716.csv", robust="robust_results_bpfixdrop18_20261123_163242.csv",
             abl="ablation_results_bpfixdrop18_20261123_164705.csv", exp3="exp3_results_bpfixdrop18_20261123_163716.csv",
             calA="exp3_svm_calA_results_bpfixdrop18_20261123_164203.csv",
             ff="xuanwu_refit_fixedfull_subject_acc_bpfixdrop18_20261123_165139.csv",
             rf="xuanwu_refit335_subject_acc_bpfixdrop18_20261123_165621.csv",
             riem="riemann_results_segfix_20261121_204511.csv",
             exp4="exp4_subject_data_segfix_20261121_205107.csv"),
}

def bh(ps):
    return stats.false_discovery_control(np.asarray(ps), method="bh")

def wc_chance(x_frac):  # x in fraction; manuscript build uses percent scale for chance
    return stats.wilcoxon(np.asarray(x_frac) * 100 - 50.0).pvalue

def wc_pair(a_frac, b_frac):  # manuscript cal_row uses fraction-scale differences
    return stats.wilcoxon(np.asarray(a_frac) - np.asarray(b_frac)).pvalue

def t_ci(d):  # mean diff (points) + t 95% CI
    d = np.asarray(d) * 100
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=m, scale=se)
    return m, lo, hi

def mean_ci(x_frac):
    x = np.asarray(x_frac) * 100
    m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x))
    lo, hi = stats.t.interval(0.95, len(x) - 1, loc=m, scale=se)
    return m, lo, hi

out = {}
for tag, files in F.items():
    R = {}
    probe = pd.read_csv(os.path.join(UF, files["probe"]))
    robust = pd.read_csv(os.path.join(UF, files["robust"]))
    abl = pd.read_csv(os.path.join(UF, files["abl"]))
    exp3 = pd.read_csv(os.path.join(UF, files["exp3"]))
    calA = pd.read_csv(os.path.join(UF, files["calA"]))
    ff = pd.read_csv(os.path.join(UF, files["ff"]))
    rf = pd.read_csv(os.path.join(UF, files["rf"]))
    riem = pd.read_csv(os.path.join(UF, files["riem"]))
    for df in (probe, robust, abl, exp3, calA, ff, rf, riem):
        assert len(df) == 47, (tag, len(df))
    # --- Exp2 chance family (6)
    cols = [("LDA full", probe.acc_full), ("LDA periodic", probe.acc_periodic), ("LDA aperiodic", probe.acc_aperiodic),
            ("SVM full", robust.svm_full), ("SVM periodic", robust.svm_periodic), ("SVM aperiodic", robust.svm_aperiodic)]
    ps = [wc_chance(c) for _, c in cols]; qs = bh(ps)
    R["exp2_chance"] = {n: dict(mean=float(c.mean()*100), p=float(p), q=float(q)) for (n, c), p, q in zip(cols, ps, qs)}
    # --- paired input family (6): full-per, full-ap, per-ap x2 decoders
    pairs = [("LDA full vs periodic", probe.acc_full, probe.acc_periodic), ("LDA full vs aperiodic", probe.acc_full, probe.acc_aperiodic),
             ("LDA periodic vs aperiodic", probe.acc_periodic, probe.acc_aperiodic),
             ("SVM full vs periodic", robust.svm_full, robust.svm_periodic), ("SVM full vs aperiodic", robust.svm_full, robust.svm_aperiodic),
             ("SVM periodic vs aperiodic", robust.svm_periodic, robust.svm_aperiodic)]
    ps = [wc_pair(a, b) for _, a, b in pairs]; qs = bh(ps)
    R["exp2_paired"] = {n: dict(diff=float((a-b).mean()*100), p=float(p), q=float(q)) for (n, a, b), p, q in zip(pairs, ps, qs)}
    # --- ablation chance family (8)
    ac = [("LDA chi", abl.lda_chi), ("SVM chi", abl.svm_chi), ("LDA offset", abl.lda_offset), ("SVM offset", abl.svm_offset),
          ("LDA ap116", abl.lda_ap116), ("SVM ap116", abl.svm_ap116), ("LDA chioff", abl.lda_chioff), ("SVM chioff", abl.svm_chioff)]
    ps = [wc_chance(c) for _, c in ac]; qs = bh(ps)
    R["abl_chance"] = {n: dict(mean=float(c.mean()*100), p=float(p), q=float(q)) for (n, c), p, q in zip(ac, ps, qs)}
    # --- ablation paired (exploratory set)
    ap = [("LDA ap116 vs chi", abl.lda_ap116, abl.lda_chi), ("SVM ap116 vs chi", abl.svm_ap116, abl.svm_chi),
          ("LDA ap116 vs offset", abl.lda_ap116, abl.lda_offset), ("SVM ap116 vs offset", abl.svm_ap116, abl.svm_offset),
          ("LDA chioff vs chi", abl.lda_chioff, abl.lda_chi), ("SVM chioff vs chi", abl.svm_chioff, abl.svm_chi),
          ("LDA chioff vs ap116", abl.lda_chioff, abl.lda_ap116), ("SVM chioff vs ap116", abl.svm_chioff, abl.svm_ap116),
          ("LDA chi vs full", abl.lda_chi, probe.acc_full), ("SVM chi vs full", abl.svm_chi, robust.svm_full)]
    R["abl_paired"] = {}
    for n, a, b in ap:
        m, lo, hi = t_ci(np.asarray(a) - np.asarray(b))
        R["abl_paired"][n] = dict(diff=m, ci=[lo, hi], p=float(wc_pair(a, b)))
    # --- ablation mixed-vs-single-parameter paired contrasts, BH over the four
    # (mixed aperiodic vs chi / offset x LDA / SVM). The SVM mixed-vs-chi
    # contrast is nominally significant (p = 0.019) but q = 0.075 here, so the
    # attribution is reported as narrowed, not settled (2026-11-24 proofread).
    ap4 = [("SVM ap116 vs chi", abl.svm_ap116, abl.svm_chi), ("LDA ap116 vs chi", abl.lda_ap116, abl.lda_chi),
           ("SVM ap116 vs offset", abl.svm_ap116, abl.svm_offset), ("LDA ap116 vs offset", abl.lda_ap116, abl.lda_offset)]
    ps = [wc_pair(a, b) for _, a, b in ap4]; qs = bh(ps)
    R["abl_mixed_vs_single_BH4"] = {}
    for (n, a, b), p, q in zip(ap4, ps, qs):
        m, lo, hi = t_ci(np.asarray(a) - np.asarray(b))
        R["abl_mixed_vs_single_BH4"][n] = dict(diff=m, ci=[lo, hi], p=float(p), q=float(q))
    # --- balanced-accuracy chance family (6 inputs). In the bpfix fix round
    # classes are balanced (20/20 per subject) so bal == ordinary accuracy;
    # in drop18, 11 subjects have unequal class counts, so the sensitivity
    # chance tests are reported on balanced accuracy.
    bc = [("LDA full", probe.bal_acc_full), ("LDA periodic", probe.bal_acc_periodic), ("LDA aperiodic", probe.bal_acc_aperiodic),
          ("SVM full", robust.bal_acc_svm_full), ("SVM periodic", robust.bal_acc_svm_periodic), ("SVM aperiodic", robust.bal_acc_svm_aperiodic)]
    ps = [wc_chance(c) for _, c in bc]; qs = bh(ps)
    R["exp2_chance_bal"] = {n: dict(mean=float(c.mean()*100), p=float(p), q=float(q)) for (n, c), p, q in zip(bc, ps, qs)}
    if tag == "bpfix":
        for (n, c), (n2, c2) in zip(bc, cols):
            assert abs(float(c.mean()) - float(c2.mean())) < 1e-9, (n, "bal != acc in balanced round")
    # --- calibration family (6), manuscript bases: LDA base=exp3.acc_base; SVM calA base=calA.acc_base_svm; SVM calB/C base=robust.svm_full
    cc = [("LDA calA", exp3.acc_base, exp3.acc_calA), ("LDA calB", exp3.acc_base, exp3.acc_calB), ("LDA calC", exp3.acc_base, exp3.acc_calC),
          ("SVM calA", calA.acc_base_svm, calA.acc_calA_svm), ("SVM calB", robust.svm_full, robust.svm_calB), ("SVM calC", robust.svm_full, robust.svm_calC)]
    ps = [wc_pair(b, a) for _, b, a in cc]; qs = bh(ps)
    R["cal"] = {}
    for (n, b, a), p, q in zip(cc, ps, qs):
        m, lo, hi = t_ci(np.asarray(a) - np.asarray(b))
        R["cal"][n] = dict(acc=float(a.mean()*100), diff=m, ci=[lo, hi], p=float(p), q=float(q))
    R["cal_base"] = dict(LDA=float(exp3.acc_base.mean()*100), SVM_full=float(robust.svm_full.mean()*100),
                         SVM_calAfile_base=float(calA.acc_base_svm.mean()*100),
                         calA_LDA_crosscheck_base=float(calA.acc_base.mean()*100), calA_LDA_crosscheck_calA=float(calA.acc_calA.mean()*100))
    # SVM calB/C vs aperiodic base (VERIFY-record convention) for discrepancy note
    for n, col in [("SVM calB vs ap base", robust.svm_calB), ("SVM calC vs ap base", robust.svm_calC)]:
        m, lo, hi = t_ci(np.asarray(col) - np.asarray(robust.svm_aperiodic))
        R["cal"][n] = dict(acc=float(col.mean()*100), diff=m, ci=[lo, hi], p=float(wc_pair(robust.svm_aperiodic, col)))
    # --- CSP
    R["csp"] = dict(mean=float(robust.csp_full.mean()*100), p=float(wc_chance(robust.csp_full)))
    # --- Riemann
    R["riem"] = {}
    for n, col in [("TS+LR", riem.ts_lr), ("TS+LDA", riem.ts_lda)]:
        m, lo, hi = mean_ci(col)
        R["riem"][n] = dict(mean=m, ci=[lo, hi], p=float(wc_chance(col)))
    R["riem"]["TS+LR vs LDA full"] = dict(diff=float((riem.ts_lr - probe.acc_full).mean()*100), p=float(wc_pair(probe.acc_full, riem.ts_lr)))
    R["riem"]["TS+LR vs CSP"] = dict(diff=float((riem.ts_lr - robust.csp_full).mean()*100), p=float(wc_pair(robust.csp_full, riem.ts_lr)))
    # --- refits
    for key, df in [("fixedfull", ff), ("refit335", rf)]:
        cc2 = [("LDA full", df.lda_full), ("LDA periodic", df.lda_periodic), ("LDA aperiodic", df.lda_aperiodic),
               ("SVM full", df.svm_full), ("SVM periodic", df.svm_periodic), ("SVM aperiodic", df.svm_aperiodic)]
        ps = [wc_chance(c) for _, c in cc2]; qs = bh(ps)
        R[key] = {n: dict(mean=float(c.mean()*100), p=float(p), q=float(q)) for (n, c), p, q in zip(cc2, ps, qs)}
    out[tag] = R

# ---------------- Exp 4 full suite ----------------
def ols_dR2(y, Xbase, xadd):
    def fit(X):
        beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1 - ss_res / ss_tot, ss_res
    Xb = np.column_stack([np.ones(len(y)), Xbase]); Xf = np.column_stack([Xb, xadd])
    r2b, ssb = fit(Xb); r2f, ssf = fit(Xf)
    df1 = 1; df2 = len(y) - Xf.shape[1]
    Fv = ((ssb - ssf) / df1) / (ssf / df2)
    return r2b, r2f - r2b, float(stats.f.sf(Fv, df1, df2))

def partial_spearman(x, y, covs):
    rx = stats.rankdata(x); ry = stats.rankdata(y)
    C = np.column_stack([np.ones(len(x))] + [stats.rankdata(c) for c in covs])
    bx, *_ = np.linalg.lstsq(C, rx, rcond=None); by, *_ = np.linalg.lstsq(C, ry, rcond=None)
    return stats.pearsonr(rx - C @ bx, ry - C @ by)

def exp4_suite(path):
    e = pd.read_csv(path); S = {}
    feats = [("chi_global", "chi_global"), ("chi_AI", "chi_AI")]
    scales = [("NIHSS", "NIHSS"), ("mRS", "mrs"), ("MBI", "mbi")]
    rows, ps = [], []
    for fn, fc in feats:
        for sn, sc in scales:
            r, p = stats.spearmanr(e[fc], e[sc]); rows.append((f"{fn} x {sn}", float(r), float(p))); ps.append(p)
    qs = bh(ps)
    S["A1"] = {n: dict(rho=r, p=p, q=float(q)) for (n, r, p), q in zip(rows, qs)}
    # TOST + 90% CI
    S["TOST"] = {}
    for (n, r, p) in rows:
        z = np.arctanh(r); se = 1 / np.sqrt(len(e) - 3); b = np.arctanh(0.30)
        p_tost = max(stats.norm.cdf((z - b) / se), 1 - stats.norm.cdf((z + b) / se)) if False else None
        # two one-sided: H01 z<=-b ; H02 z>=+b ; tost p = max(p1,p2)
        p1 = 1 - stats.norm.cdf((z + b) / se); p2 = stats.norm.cdf((z - b) / se)
        lo, hi = np.tanh(z - 1.644854 * se), np.tanh(z + 1.644854 * se)
        S["TOST"][n] = dict(tost_p=float(max(p1, p2)), equiv=bool(max(p1, p2) < 0.05), ci90=[float(lo), float(hi)])
    # A5 AUC + MWU
    g1 = e[e.mrs <= 2]; g2 = e[e.mrs >= 3]; S["A5"] = {}
    for fn, fc in feats:
        u = stats.mannwhitneyu(g2[fc], g1[fc]); S["A5"][fn] = dict(auc=float(u.statistic / (len(g1) * len(g2))), p=float(u.pvalue))
    # A3
    S["A3"] = {}
    for sn, sc in [("mRS", "mrs"), ("MBI", "mbi")]:
        r2b, dr2, pf = ols_dR2(e[sc].values.astype(float), np.column_stack([e.age.values, e.NIHSS.values]), e.chi_AI.values)
        S["A3"][sn] = dict(baseR2=float(r2b), dR2=float(dr2), p=float(pf))
    # A2 supratentorial (ternary != brainstem/cerebellum)
    print("ternary values:", e.ternary.unique(), "| cortical:", e.cortical.unique())
    sub = e[~e.ternary.astype(str).str.contains("brainstem", case=False)]
    S["A2_n"] = int(len(sub)); S["A2"] = {}
    for fn, fc in feats:
        for sn, sc in scales:
            r, p = stats.spearmanr(sub[fc], sub[sc]); S["A2"][f"{fn} x {sn}"] = dict(rho=float(r), p=float(p))
    # A4 KW + medians
    groups = [g.chi_global.values for _, g in e.groupby("ternary")]
    S["A4"] = dict(chi_global_KW_p=float(stats.kruskal(*groups).pvalue),
                   chi_AI_KW_p=float(stats.kruskal(*[g.chi_AI.values for _, g in e.groupby("ternary")]).pvalue),
                   medians={str(k): float(v) for k, v in e.groupby("ternary").chi_global.median().items()},
                   cortical_MWU_chi_global_p=float(stats.mannwhitneyu(e[e.cortical == 1].chi_global, e[e.cortical == 0].chi_global).pvalue) if set(e.cortical.unique()) <= {0, 1} else None)
    # partial correlations (duration; duration+age)
    S["partial"] = {}
    for fn, fc in feats:
        for sn, sc in scales:
            r1, p1 = partial_spearman(e[fc], e[sc], [e.duration.values])
            r2, p2 = partial_spearman(e[fc], e[sc], [e.duration.values, e.age.values])
            S["partial"][f"{fn} x {sn}"] = dict(dur=[float(r1), float(p1)], dur_age=[float(r2), float(p2)])
    # LOO influence
    minqs, aucs = [], []
    for i in range(len(e)):
        ee = e.drop(e.index[i]); pp = []
        for fn, fc in feats:
            for sn, sc in scales:
                pp.append(stats.spearmanr(ee[fc], ee[sc]).pvalue)
        minqs.append(float(bh(pp).min()))
        gg1 = ee[ee.mrs <= 2]; gg2 = ee[ee.mrs >= 3]
        aucs.append(float(stats.mannwhitneyu(gg2.chi_AI, gg1.chi_AI).statistic / (len(gg1) * len(gg2))))
    S["LOO"] = dict(minq_range=[min(minqs), max(minqs)], auc_chiAI_range=[min(aucs), max(aucs)])
    S["desc"] = dict(chi_global_mean=float(e.chi_global.mean()), chi_global_sd=float(e.chi_global.std()),
                     r2_median=float(e.r2.median()))
    return S

# Exp 4 was not rerun in the bpfix round (chi/offset bitwise identical to the
# segfix fix round, VERIFY-20261123-bpfix.md), so one suite covers both rounds.
out["exp4"] = exp4_suite(os.path.join(UF, "exp4_subject_data_segfix_20261121_205107.csv"))

# ---------------- Tianjin splice CIs ----------------
tj = pd.read_csv(os.path.join(UF, "tianjin_ablation_window2s_subject_acc_20261121_193411.csv"))
T = {}
for n, a, b in [("SVM chioff vs chi", tj.svm_chioff, tj.svm_chi), ("LDA chioff vs chi", tj.lda_chioff, tj.lda_chi),
                ("SVM chioff vs ap", tj.svm_chioff, tj.svm_ap), ("LDA chioff vs ap", tj.lda_chioff, tj.lda_ap)]:
    m, lo, hi = t_ci(np.asarray(a) - np.asarray(b))
    T[n] = dict(diff=m, ci=[lo, hi], p=float(wc_pair(a, b)))
out["tianjin_splice"] = T

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpfix_bh_recalc.json"), "w") as f:
    json.dump(out, f, indent=1)

# ---------------- checks against VERIFY-20261123-bpfix.md ----------------
def close(a, b, tol=0.002):
    return abs(a - b) <= tol

def close2(a, b):  # means / diffs are published rounded to 2 decimals
    return round(a, 2) == round(b, 2)

checks = [
    ("bpfix LDA full chance q", out["bpfix"]["exp2_chance"]["LDA full"]["q"], 0.030, close),
    ("bpfix SVM aperiodic chance q", out["bpfix"]["exp2_chance"]["SVM aperiodic"]["q"], 0.030, close),
    ("bpfix LDA calA q", out["bpfix"]["cal"]["LDA calA"]["q"], 0.0071, close),
    ("bpfix LDA calA diff", out["bpfix"]["cal"]["LDA calA"]["diff"], -4.26, close2),
    ("bpfix Riemann TS+LR vs LDA full p", out["bpfix"]["riem"]["TS+LR vs LDA full"]["p"], 0.265, close),
    ("drop18 bal LDA full q", out["bpfixdrop18"]["exp2_chance_bal"]["LDA full"]["q"], 0.036, close),
    ("drop18 bal SVM aperiodic q", out["bpfixdrop18"]["exp2_chance_bal"]["SVM aperiodic"]["q"], 0.044, close),
    ("drop18 bal LDA full mean", out["bpfixdrop18"]["exp2_chance_bal"]["LDA full"]["mean"], 52.92, close2),
    ("drop18 bal SVM aperiodic mean", out["bpfixdrop18"]["exp2_chance_bal"]["SVM aperiodic"]["mean"], 52.61, close2),
    ("abl mixed-vs-chi SVM q (BH4)", out["bpfix"]["abl_mixed_vs_single_BH4"]["SVM ap116 vs chi"]["q"], 0.075, close),
    ("exp4 LOO AUC range lo", out["exp4"]["LOO"]["auc_chiAI_range"][0], 0.464, close),
    ("exp4 LOO AUC range hi", out["exp4"]["LOO"]["auc_chiAI_range"][1], 0.512, close),
]
failed = 0
for name, got, want, fn in checks:
    ok = fn(got, want)
    failed += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got:.4f}  want {want:.4f}")
print("CHECKS:", "ALL PASS" if failed == 0 else f"{failed} FAILED")
print("DONE")
