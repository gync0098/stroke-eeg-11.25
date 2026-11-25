#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make-figures.py — A 版论文图件生成（从结果 CSV 现算统计、现画图）
================================================================
输入（默认分层结果目录 results/，与 FINAL 结果包内结构一致；可传参指定另一结果根目录；
  平铺目录 ~/workspace/user/files/ 作 fallback）：
  xuanwu/bpfix/probe_results_bpfix_20261123_155344.csv    Exp 2 LDA（acc_full/periodic/aperiodic）
  xuanwu/bpfix/robust_results_bpfix_20261123_155910.csv   Exp 2/3 SVM（svm_*、csp_full）
  xuanwu/bpfix/ablation_results_bpfix_20261123_161329.csv 消融（lda/svm _chi/_offset/_ap116/_chioff）
  xuanwu/bpfix/exp3_results_bpfix_20261123_160348.csv     Exp 3 LDA（acc_base/calA/calB/calC）
  xuanwu/bpfix/exp3_svm_calA_results_bpfix_20261123_160833.csv  Exp 3 SVM calA（acc_calA_svm）
  xuanwu/fix/exp4_subject_data_segfix_20261121_205107.csv（Exp 4，segfix 轮定版，bpfix 未重跑）
输出（本目录 figures/）：fig1_overview / fig2_decoding / fig3_calibration /
  fig4_association 的 PNG(300dpi)+PDF，及 table1_n47.csv / .md。
运行后会打印全部标题统计，与 RESULTS-SUMMARY.md 对照核验。
投稿前 χ 口径统一后：换输入 CSV 重跑本脚本即可全图刷新。
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

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
rng = np.random.default_rng(20261022)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
})
C_LDA, C_SVM = "#4C72B0", "#C44E52"
CHANCE = 50.0

probe = pd.read_csv(find_csv("probe_results_bpfix_20261123_155344.csv"))  # bpfix fix round (primary)
robust = pd.read_csv(find_csv("robust_results_bpfix_20261123_155910.csv"))  # bpfix fix round
abl = pd.read_csv(find_csv("ablation_results_bpfix_20261123_161329.csv"))  # chioff, bpfix fix round
exp3 = pd.read_csv(find_csv("exp3_results_bpfix_20261123_160348.csv"))  # bpfix fix round
exp4 = pd.read_csv(find_csv("exp4_subject_data_segfix_20261121_205107.csv"))  # segfix fix round (Exp 4 not rerun in bpfix)
# SVM calA：与 robust 同折同序，base_svm 须逐位复现 svm_full
calA_svm = pd.read_csv(find_csv("exp3_svm_calA_results_bpfix_20261123_160833.csv"))  # bpfix fix round
assert len(calA_svm) == 47 and (calA_svm["test_subj"].values == robust["test_subj"].values).all()
assert np.allclose(calA_svm["acc_base_svm"].values, robust["svm_full"].values)
robust = robust.assign(svm_calA=calA_svm["acc_calA_svm"])
assert len(probe) == len(robust) == len(abl) == len(exp3) == len(exp4) == 47


def wchance(x_pct):
    """被试级准确率 vs 随机（50%）的 Wilcoxon 符号秩（双侧）。"""
    return stats.wilcoxon(np.asarray(x_pct) - CHANCE).pvalue


def strip_box(ax, x, vals_pct, color, width=0.52):
    vals = np.asarray(vals_pct, dtype=float)
    ax.boxplot(vals, positions=[x], widths=width, patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.18, edgecolor=color),
               medianprops=dict(color=color, linewidth=1.4),
               whiskerprops=dict(color=color), capprops=dict(color=color),
               flierprops=dict(marker=""))
    jit = rng.uniform(-width / 2.6, width / 2.6, size=len(vals))
    ax.scatter(x + jit, vals, s=13, color=color, alpha=0.62, zorder=3,
               edgecolor="white", linewidth=0.3)
    ax.scatter([x], [vals.mean()], marker="D", s=34, color=color, zorder=4,
               edgecolor="white", linewidth=0.7)


def ptxt(p):
    return "p<.001" if p < 0.001 else f"p={p:.3f}".replace("0.", ".")


STATLINES = []
def report(name, vals_pct):
    m = float(np.mean(vals_pct)); p = wchance(vals_pct)
    STATLINES.append(f"{name:<28} mean {m:5.1f}%  Wilcoxon vs chance {ptxt(p)}")
    return m, p


# ================================================================ Fig 2 解码
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
inputs = [("Full spectrum", "acc_full", "svm_full"),
          ("Periodic only", "acc_periodic", "svm_periodic"),
          ("Aperiodic only", "acc_aperiodic", "svm_aperiodic")]
ax = axes[0]
for i, (lab, cl, cs) in enumerate(inputs):
    strip_box(ax, i - 0.21, probe[cl] * 100, C_LDA, width=0.34)
    strip_box(ax, i + 0.21, robust[cs] * 100, C_SVM, width=0.34)
    report(f"Exp2 LDA {lab}", probe[cl] * 100)
    report(f"Exp2 SVM {lab}", robust[cs] * 100)
ax.axhline(CHANCE, color="0.35", ls="--", lw=0.9)
ax.set_xticks(range(3)); ax.set_xticklabels([x[0] for x in inputs])
ax.set_ylabel("Subject accuracy (%)")
ax.set_title("A  Cross-subject MI decoding by input", loc="left", fontsize=9.5)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="s", ls="", color=C_LDA, label="LDA"),
                   Line2D([], [], marker="s", ls="", color=C_SVM, label="SVM (RBF)")],
          frameon=False, fontsize=8, loc="upper right")
ax.set_ylim(18, 92)

ax = axes[1]
ab_inputs = [("χ only", "lda_chi", "svm_chi"),
             ("Offset only", "lda_offset", "svm_offset"),
             ("Aperiodic mix", "lda_ap116", "svm_ap116")]
for i, (lab, cl, cs) in enumerate(ab_inputs):
    strip_box(ax, i - 0.21, abl[cl] * 100, C_LDA, width=0.34)
    strip_box(ax, i + 0.21, abl[cs] * 100, C_SVM, width=0.34)
    report(f"Ablation LDA {lab}", abl[cl] * 100)
    report(f"Ablation SVM {lab}", abl[cs] * 100)
ax.axhline(CHANCE, color="0.35", ls="--", lw=0.9)
ax.set_xticks(range(3)); ax.set_xticklabels([x[0] for x in ab_inputs])
ax.set_title("B  Ablation: what carries the weak effect", loc="left", fontsize=9.5)
ax.set_ylim(18, 92)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_decoding.png"), dpi=300)
fig.savefig(os.path.join(OUT, "fig2_decoding.pdf"))
plt.close(fig)
report("CSP+LDA anchor (full)", robust["csp_full"] * 100)

# ================================================================ Fig 3 校准
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
panels = [
    (axes[0], "A  LDA", [("calA\nfingerprint", "acc_calA"), ("calB\nregression norm.", "acc_calB"),
                        ("calC\nown z-score", "acc_calC")],
     exp3, "acc_base", None),
    (axes[1], "B  SVM (RBF)", [("calA\nfingerprint", "svm_calA"), ("calB\nregression norm.", "svm_calB"), ("calC\nown z-score", "svm_calC")],
     robust, None, "svm_full"),
]
for ax, title, conds, dfc, base_col, base_from_self in panels:
    base_f = dfc[base_col] if base_col else dfc[base_from_self]
    for i, (lab, col) in enumerate(conds):
        delta_f = (dfc[col] - base_f).values          # 分数尺度差（检验用，与核验口径一致）
        delta = delta_f * 100                          # 百分数尺度（作图用）
        strip_box(ax, i, delta, "#8172B2", width=0.5)
        p = stats.wilcoxon(delta_f).pvalue
        ax.annotate(f"Δ {delta.mean():+.1f}%\n{ptxt(p)}", (i, delta.max() + 3.2),
                    ha="center", fontsize=8)
        STATLINES.append(f"Exp3 {title} {lab.splitlines()[0]:<10} Δ {delta.mean():+.2f}%  paired Wilcoxon {ptxt(p)}")
    ax.axhline(0, color="0.35", ls="--", lw=0.9)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([c[0] for c in conds])
    ax.set_title(title + "  (change vs no calibration)", loc="left", fontsize=9.5)
    ax.set_ylim(-33, 33)
axes[0].set_ylabel("Δ accuracy vs base (percentage points)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_calibration.png"), dpi=300)
fig.savefig(os.path.join(OUT, "fig3_calibration.pdf"))
plt.close(fig)

# ================================================================ Fig 4 关联
n = len(exp4)
pairs = [("chi_global", "NIHSS"), ("chi_global", "mrs"), ("chi_global", "mbi"),
         ("chi_AI", "NIHSS"), ("chi_AI", "mrs"), ("chi_AI", "mbi")]
labels = ["χ global × NIHSS", "χ global × mRS", "χ global × MBI",
          "χ asymmetry × NIHSS", "χ asymmetry × mRS", "χ asymmetry × MBI"]
rhos, los, his = [], [], []
for f, s in pairs:
    r, p = stats.spearmanr(exp4[f], exp4[s])
    z = np.arctanh(np.clip(r, -0.999, 0.999)); se = 1 / np.sqrt(n - 3)
    rhos.append(r); los.append(np.tanh(z - 1.96 * se)); his.append(np.tanh(z + 1.96 * se))
    STATLINES.append(f"Exp4 {labels[len(rhos)-1]:<22} rho {r:+.3f}  p {p:.3f}")

fig = plt.figure(figsize=(7.4, 5.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.55, wspace=0.3)
axf = fig.add_subplot(gs[0, :])
axf.axvspan(-0.3, 0.3, color="0.92", zorder=0)
axf.axvline(0, color="0.35", lw=0.9)
ypos = np.arange(len(pairs))[::-1]
axf.errorbar(rhos, ypos, xerr=[np.array(rhos) - np.array(los), np.array(his) - np.array(rhos)],
             fmt="o", color="#4C72B0", ecolor="#4C72B0", capsize=3, ms=5)
for y, r in zip(ypos, rhos):
    axf.annotate(f"{r:+.2f}", (r, y), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=8)
axf.set_yticks(ypos); axf.set_yticklabels(labels)
axf.set_xlim(-0.62, 0.62)
axf.set_xlabel("Spearman rho (95% CI); shaded band = equivalence region ±0.30")
axf.set_title("A  Cross-sectional associations with clinical state (n=47, all n.s. after BH-FDR)",
              loc="left", fontsize=9.5)

axb = fig.add_subplot(gs[1, 0])
jit = rng.uniform(-0.28, 0.28, size=n)
axb.scatter(exp4["NIHSS"] + jit, exp4["chi_AI"], s=16, color="#4C72B0", alpha=0.65,
            edgecolor="white", linewidth=0.3)
axb.axhline(0, color="0.35", ls="--", lw=0.9)
axb.set_xlabel("NIHSS"); axb.set_ylabel("χ asymmetry (ipsi − contra)")
r_ni, p_ni = stats.spearmanr(exp4["chi_AI"], exp4["NIHSS"])
axb.set_title(f"B  Largest association: rho={r_ni:+.2f}, p={p_ni:.2f}", loc="left", fontsize=9.5)

axc = fig.add_subplot(gs[1, 1])
axc.hist(exp4["chi_global"], bins=12, color="#4C72B0", alpha=0.75, edgecolor="white")
axc.axvline(exp4["chi_global"].mean(), color="#C44E52", lw=1.4)
axc.set_xlabel("χ global (subject mean)"); axc.set_ylabel("Subjects")
axc.set_title(f"C  χ spans {exp4.chi_global.min():.2f}–{exp4.chi_global.max():.2f} "
              f"({exp4.chi_global.max()/exp4.chi_global.min():.1f}×) across subjects",
              loc="left", fontsize=9.5)
fig.savefig(os.path.join(OUT, "fig4_association.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig4_association.pdf"), bbox_inches="tight")
plt.close(fig)

# ================================================================ Fig 1 示意
fig, ax = plt.subplots(figsize=(7.4, 2.95))
ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

def box(x, y, w, h, text, fc, fs=8.3, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
                                fc=fc, ec="0.4", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 color="0.35", lw=1.0))

box(0.1, 1.55, 2.0, 0.95, "Public dataset\n50 acute stroke\nMI-EEG (Liu 2024)", "#EAF0F7")
arrow(2.15, 2.02, 2.55, 2.02)
box(2.6, 1.55, 2.0, 0.95, "Dataset audit\n−3 duplicate copies\n(sub-05/15/49)", "#FDECEA")
arrow(4.65, 2.02, 5.05, 2.02)
box(5.1, 1.55, 1.75, 0.95, "Analysis cohort\nn = 47\nleak-free LOSO", "#EAF0F7")
arrow(6.9, 2.02, 7.3, 2.9); arrow(6.9, 2.02, 7.3, 2.02); arrow(6.9, 2.02, 7.3, 1.15)
box(7.35, 2.58, 2.55, 0.7, "Q1 Decoding\nχ at chance", "#F2F2F2", fs=8.5)
box(7.35, 1.67, 2.55, 0.7, "Q2 Calibration\nno benefit", "#F2F2F2", fs=8.5)
box(7.35, 0.76, 2.55, 0.7, "Q3 Association\nall null", "#F2F2F2", fs=8.5)
ax.text(5.0, 3.62, "Systematic evaluation of aperiodic EEG parameters in acute ischemic stroke",
        ha="center", fontsize=10, weight="bold")
ax.text(5.0, 0.32, "Clean-protocol benchmark on this dataset: 52.8% cross-subject accuracy "
        "(published reports on the same data: up to 97.4%)", ha="center", fontsize=8.3, color="0.3")
fig.savefig(os.path.join(OUT, "fig1_overview.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig1_overview.pdf"), bbox_inches="tight")
plt.close(fig)

# ================================================================ Table 1
def ms(x): return f"{x.mean():.2f} ± {x.std():.2f}"
def miqr(x): return f"{x.median():.0f} [{x.quantile(.25):.0f}, {x.quantile(.75):.0f}]"
rows = [
    ("Analyzed subjects, n", "47 (of 50 recorded; 3 duplicate copies excluded)"),
    ("Age, years (mean ± SD)", ms(exp4["age"])),
    ("Sex (M / F)", f"{(exp4.sex=='male').sum()} / {(exp4.sex=='female').sum()}"),
    ("NIHSS (mean ± SD; median [IQR])", f"{ms(exp4['NIHSS'])}; {miqr(exp4['NIHSS'])}"),
    ("mRS (median [IQR])", miqr(exp4["mrs"])),
    ("MBI (mean ± SD)", ms(exp4["mbi"])),
    ("Days since onset (median [IQR]; range)", f"{miqr(exp4['duration'])}; {exp4.duration.min()}–{exp4.duration.max()}"),
    ("Affected hemisphere (L / R)", f"{(exp4.aff_hemi=='left').sum()} / {(exp4.aff_hemi=='right').sum()}"),
    ("Lesion stratum (brainstem-cerebellar / deep-only / cortical-involved)",
     f"{(exp4.ternary=='brainstem/cerebellum').sum()} / {(exp4.ternary=='deep-only').sum()} / {(exp4.ternary=='cortical-involved').sum()}"),
    ("χ global (mean ± SD; range)", f"{ms(exp4['chi_global'])}; {exp4.chi_global.min():.2f}–{exp4.chi_global.max():.2f}"),
    ("specparam R² (median [IQR])", f"{exp4.r2.median():.2f} [{exp4.r2.quantile(.25):.2f}, {exp4.r2.quantile(.75):.2f}]"),
]
t1 = pd.DataFrame(rows, columns=["Characteristic", "Value"])
t1.to_csv(os.path.join(OUT, "table1_n47.csv"), index=False)
with open(os.path.join(OUT, "table1_n47.md"), "w", encoding="utf-8") as f:
    f.write("| Characteristic | Value |\n|---|---|\n")
    for _, r in t1.iterrows():
        f.write(f"| {r['Characteristic']} | {r['Value']} |\n")

print("\n".join(STATLINES))
print("\nTable 1:"); print(t1.to_string(index=False))
print(f"\n图件输出目录: {OUT}")
