#!/usr/bin/env python3
"""Supplementary figures S1–S5 for the stroke aperiodic paper.
All data from archived result CSVs (default: layered results/ directory next to
this script, mirroring the FINAL results pack; a directory argument or the flat
~/workspace/user/files/ directory works as fallback). Same style as make-figures.py.
Usage: python3 make-supplementary-figures.py [csv_dir]  (default results/ or ~/workspace/user/files)
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
UF = os.path.expanduser("~/workspace/user/files")
# 分层结果目录优先（与 FINAL 结果包内结构一致）；argv[1] 可指定另一结果根目录；
# 平铺目录 ~/workspace/user/files/ 作 fallback。
_BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
_LAYERED = ("xuanwu/bpfix", "xuanwu/bpfixdrop18", "xuanwu/fix", "tianjin")
# 两个分层根：工作目录的 results/；以及与脚本包并排解压时的 ./results/
# （脚本在 scripts/figures/ 时，../../results 即包旁的 results/）。
_BASES = [_BASE]
_pack_results = os.path.normpath(os.path.join(HERE, "..", "..", "results"))
if _pack_results != os.path.normpath(_BASE):
    _BASES.append(_pack_results)
def find_csv(name):
    for base in _BASES:
        for root in [os.path.join(base, s) for s in _LAYERED] + [base]:
            p = os.path.join(root, name)
            if os.path.exists(p):
                return p
    p = os.path.join(UF, name)
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"缺文件：{name}（已搜分层目录与 {UF}）")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
LDA_C, SVM_C = "#4C72B0", "#C44E52"
rng = np.random.default_rng(7)

def strip_box(ax, series, labels, colors, ylabel=None, hline=None, ylim=None):
    for i, (s, c) in enumerate(zip(series, colors), start=1):
        v = np.asarray(s, dtype=float)
        ax.boxplot(v, positions=[i], widths=0.55, patch_artist=True,
                   boxprops=dict(facecolor=c, alpha=0.25, edgecolor=c),
                   medianprops=dict(color=c), whiskerprops=dict(color=c),
                   capprops=dict(color=c), flierprops=dict(marker=".", markersize=3, markerfacecolor=c))
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.16, 0.16, len(v)), v,
                   s=10, color=c, alpha=0.75, zorder=3, linewidths=0)
        ax.plot([i], [v.mean()], marker="D", color=c, markersize=5, zorder=4)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel)
    if hline is not None: ax.axhline(hline, color="grey", lw=0.8, ls="--")
    if ylim: ax.set_ylim(*ylim)

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

# ---------- S1: Xuanwu per-subject decoding ----------
probe = pd.read_csv(find_csv("probe_results_bpfix_20261123_155344.csv"))  # bpfix fix round  # LDA folds
rob = pd.read_csv(find_csv("robust_results_bpfix_20261123_155910.csv"))  # bpfix fix round   # SVM folds
abl = pd.read_csv(find_csv("ablation_results_bpfix_20261123_161329.csv"))  # chioff, bpfix fix  # ablation folds
assert len(probe) == len(rob) == len(abl) == 47
labels = ["Full", "Periodic", "Aperiodic", "χ only", "Offset only"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
strip_box(axes[0], [probe.acc_full*100, probe.acc_periodic*100, probe.acc_aperiodic*100,
                    abl.lda_chi*100, abl.lda_offset*100], labels, [LDA_C]*5,
          ylabel="Subject accuracy (%)", hline=50, ylim=(25, 80))
axes[0].set_title("LDA — primary cohort (n = 47)", fontsize=10)
strip_box(axes[1], [rob.svm_full*100, rob.svm_periodic*100, rob.svm_aperiodic*100,
                    abl.svm_chi*100, abl.svm_offset*100], labels, [SVM_C]*5, hline=50, ylim=(25, 80))
axes[1].set_title("SVM — primary cohort (n = 47)", fontsize=10)
save(fig, "figS1_xuanwu_subject_acc")
print("S1 means LDA:", [round(x*100, 1) for x in [probe.acc_full.mean(), probe.acc_periodic.mean(), probe.acc_aperiodic.mean(), abl.lda_chi.mean(), abl.lda_offset.mean()]])
print("S1 means SVM:", [round(x*100, 1) for x in [rob.svm_full.mean(), rob.svm_periodic.mean(), rob.svm_aperiodic.mean(), abl.svm_chi.mean(), abl.svm_offset.mean()]])

# ---------- S2: calibration deltas ----------
e3 = pd.read_csv(find_csv("exp3_results_bpfix_20261123_160348.csv"))  # bpfix fix round
# SVM calA：与 rob 同折同序，base_svm 须逐位复现 svm_full
calA_svm = pd.read_csv(find_csv("exp3_svm_calA_results_bpfix_20261123_160833.csv"))  # bpfix fix round
assert len(e3) == 47 and len(rob) == 47 and len(calA_svm) == 47
assert (calA_svm["test_subj"].values == rob["test_subj"].values).all()
assert np.allclose(calA_svm["acc_base_svm"].values, rob["svm_full"].values)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
strip_box(axes[0], [(e3.acc_calA-e3.acc_base)*100, (e3.acc_calB-e3.acc_base)*100, (e3.acc_calC-e3.acc_base)*100],
          ["calA\nfingerprint", "calB\nregression", "calC\nz-score"], [LDA_C]*3,
          ylabel="Δ accuracy vs base (points)", hline=0, ylim=(-16, 16))
axes[0].set_title("LDA calibration (n = 47)", fontsize=10)
strip_box(axes[1], [(calA_svm.acc_calA_svm-calA_svm.acc_base_svm)*100, (rob.svm_calB-rob.svm_full)*100, (rob.svm_calC-rob.svm_full)*100],
          ["calA\nfingerprint", "calB\nregression", "calC\nz-score"], [SVM_C]*3, hline=0, ylim=(-16, 16))
axes[1].set_title("SVM calibration (n = 47)", fontsize=10)
save(fig, "figS2_calibration_deltas")
print("S2 mean deltas LDA:", [round(x*100, 2) for x in [(e3.acc_calA-e3.acc_base).mean(), (e3.acc_calB-e3.acc_base).mean(), (e3.acc_calC-e3.acc_base).mean()]])
print("S2 mean deltas SVM:", [round(x*100, 2) for x in [(calA_svm.acc_calA_svm-calA_svm.acc_base_svm).mean(), (rob.svm_calB-rob.svm_full).mean(), (rob.svm_calC-rob.svm_full).mean()]])

# ---------- S3: Exp 4 scatter ----------
e4 = pd.read_csv(find_csv("exp4_subject_data_segfix_20261121_205107.csv"))  # segfix fix round, n=47
assert len(e4) == 47
fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6))
for r, (col, lab) in enumerate([("chi_global", "χ global"), ("chi_AI", "χ asymmetry index")]):
    for c, (sc, slab) in enumerate([("NIHSS", "NIHSS"), ("mrs", "mRS"), ("mbi", "Barthel index")]):
        ax = axes[r][c]
        ax.scatter(e4[col], e4[sc], s=14, color="#4C72B0", alpha=0.8, linewidths=0)
        rho = spearmanr(e4[col], e4[sc]).statistic
        ax.set_title(f"{lab} × {slab}\nρ = {rho:+.2f}", fontsize=9)
        ax.set_xlabel(lab); ax.set_ylabel(slab)
fig.tight_layout()
save(fig, "figS3_exp4_scatter")

# ---------- S4: Tianjin per-subject decoding ----------
tj = pd.read_csv(find_csv("tianjin_exp2_window2s_subject_acc_20261121_181628.csv"))  # fixfloor final
ta = pd.read_csv(find_csv("tianjin_ablation_window2s_subject_acc_20261121_193411.csv"))  # chioff final
assert len(tj) == len(ta) == 26
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
strip_box(axes[0], [tj.lda_full*100, tj.lda_periodic*100, tj.lda_aperiodic*100, ta.lda_chi*100, ta.lda_offset*100],
          labels, [LDA_C]*5, ylabel="Subject accuracy (%)", hline=50, ylim=(25, 85))
axes[0].set_title("LDA — Tianjin cohort (n = 26)", fontsize=10)
strip_box(axes[1], [tj.svm_full*100, tj.svm_periodic*100, tj.svm_aperiodic*100, ta.svm_chi*100, ta.svm_offset*100],
          labels, [SVM_C]*5, hline=50, ylim=(25, 85))
axes[1].set_title("SVM — Tianjin cohort (n = 26)", fontsize=10)
save(fig, "figS4_tianjin_subject_acc")
print("S4 means LDA:", [round(x*100, 1) for x in [tj.lda_full.mean(), tj.lda_periodic.mean(), tj.lda_aperiodic.mean(), ta.lda_chi.mean(), ta.lda_offset.mean()]])
print("S4 means SVM:", [round(x*100, 1) for x in [tj.svm_full.mean(), tj.svm_periodic.mean(), tj.svm_aperiodic.mean(), ta.svm_chi.mean(), ta.svm_offset.mean()]])

# ---------- S5: chi and fit quality ----------
ref = pd.read_csv(find_csv("xuanwu_refit335_subject_chi_bpfix_20261123_162244.csv"))  # truncated, bpfix fix (chi bitwise identical to segfix)
tjt = pd.read_csv(find_csv("tianjin_exp2_window2s_trial_chi_20261121_181628.csv"))  # fixfloor final
tj_chi = tjt.groupby("subject")["chi"].mean()
fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0))
# (a) subject chi, two fitting ranges (Xuanwu, median convention)
strip_box(axes[0], [e4.chi_global, ref.chi_median], ["[1, 40] Hz", "[3, 35] Hz"], ["#4C72B0", "#DD8452"],
          ylabel="Subject-level χ")
axes[0].set_title("(a) χ by fitting range (Xuanwu)", fontsize=10)
# (b) fit R2, two ranges
strip_box(axes[1], [e4.r2, ref.specparam_r2_median], ["[1, 40] Hz", "[3, 35] Hz"], ["#4C72B0", "#DD8452"],
          ylabel="specparam R² (subject median)")
axes[1].set_title("(b) Fit quality (Xuanwu)", fontsize=10)
# (c) Tianjin subject chi
strip_box(axes[2], [tj_chi], ["Tianjin [3, 35] Hz"], ["#55A868"], ylabel="Subject-level χ (trial mean)")
axes[2].set_title("(c) χ, Tianjin cohort", fontsize=10)
save(fig, "figS5_chi_fitquality")
print("S5 chi[1,40] mean %.3f | chi[3,35] median-convention mean %.3f | R2 medians %.3f / %.3f | Tianjin chi %.3f"
      % (e4.chi_global.mean(), ref.chi_median.mean(), e4.r2.median(), ref.specparam_r2_median.median(), tj_chi.mean()))
print("ALL SUPPLEMENTARY FIGURES DONE")
