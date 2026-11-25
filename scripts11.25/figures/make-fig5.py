#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make-fig5.py — Fig 5：跨队列泛化 + 敏感性三分（2026-11-21 冻结版重出）
================================================================
三格（DejaVu Sans 9pt、LDA 蓝 / SVM 红、箱线+抖动散点+均值菱形）：
  (a) 两队列被试级 χ 分布（统一 [3, 35] Hz 拟合、试次均值口径）
  (b) 两队列 aperiodic 集与 χ-only 解码（宣武 [1, 40] 主分析；天津 [3, 35]；
      两队列同为被试级 LOSO 评估单元）
  (c) 宣武敏感性三分：full / aperiodic 两路 × 两解码器，三点连线——
      [1, 40] 主分析 → 仅改拟合范围（特征固定）→ 截频（特征范围 [3, 35]）

输入 CSV（默认分层结果目录 results/，与 FINAL 结果包内结构一致；可传参指定另一结果根目录；
  平铺目录 ~/workspace/user/files/ 作 fallback。宣武为 2026-11-23 bpfix 定版，天津为 fixfloor 定版）：
  tianjin_exp2_window2s_subject_acc_20261121_181628.csv    天津 Exp 2（2 秒窗定版）
  tianjin_exp2_window2s_trial_chi_20261121_181628.csv      天津每试次 χ
  tianjin_ablation_window2s_subject_acc_20261121_193411.csv 天津消融（2 秒窗定版）
  xuanwu_refit_fixedfull_subject_acc_bpfix_20261123_161803.csv  宣武仅改拟合范围（fixed-full）
  xuanwu_refit_fixedfull_subject_chi_bpfix_20261123_161803.csv  其被试级 χ（chi_mean）
  xuanwu_refit335_subject_acc_bpfix_20261123_162244.csv         宣武截频版（频率范围敏感性）
  probe_results_bpfix_20261123_155344.csv / robust_results_bpfix_20261123_155910.csv  宣武 [1,40] 定版
  ablation_results_bpfix_20261123_161329.csv                    宣武消融定版（χ-only）

输出：figures/fig5_generalisation.png（300 dpi）+ .pdf（Type 42）。
注意：offset 跨队列不可比（单位差 ≈12 log10），本图不画 offset。
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
rng = np.random.default_rng(20261121)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
})
C_LDA, C_SVM = "#4C72B0", "#C44E52"
C_XW, C_TJ = "#8172B2", "#55A868"
CHANCE = 50.0


def load(name):
    p = find_csv(name)
    return pd.read_csv(p)


tj_acc = load("tianjin_exp2_window2s_subject_acc_20261121_181628.csv")  # fixfloor final
tj_chi = load("tianjin_exp2_window2s_trial_chi_20261121_181628.csv")
tj_abl = load("tianjin_ablation_window2s_subject_acc_20261121_193411.csv")  # chioff final
xw_fit = load("xuanwu_refit_fixedfull_subject_acc_bpfix_20261123_161803.csv")  # 仅改拟合范围（bpfix fix）
xw_chi35 = load("xuanwu_refit_fixedfull_subject_chi_bpfix_20261123_161803.csv")  # bpfix fix（χ 与 segfix 逐位相同）
xw_trunc = load("xuanwu_refit335_subject_acc_bpfix_20261123_162244.csv")    # 截频版（bpfix fix）
xw_lda140 = load("probe_results_bpfix_20261123_155344.csv")  # bpfix fix
xw_svm140 = load("robust_results_bpfix_20261123_155910.csv")  # bpfix fix
xw_abl = load("ablation_results_bpfix_20261123_161329.csv")  # bpfix fix
assert len(tj_acc) == len(tj_abl) == 26 and len(xw_fit) == len(xw_trunc) == 47

tj_chi_subj = tj_chi.groupby("subject")["chi"].mean().values
xw_chi_subj = xw_chi35["chi_mean"].values


def wchance(x_pct):
    return stats.wilcoxon(np.asarray(x_pct) - CHANCE).pvalue


def ptxt(p):
    return "p<.001" if p < 0.001 else f"p={p:.3f}".replace("0.", ".")


def strip_box(ax, x, vals, color, width=0.52):
    vals = np.asarray(vals, dtype=float)
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


STATLINES = []
def report(name, vals_pct):
    m = float(np.mean(vals_pct)); p = wchance(vals_pct)
    STATLINES.append(f"{name:<34} mean {m:5.1f}%  Wilcoxon vs chance {ptxt(p)}")
    return m, p


fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.3),
                         gridspec_kw={"width_ratios": [0.85, 1.9, 1.35], "wspace": 0.26})

# ---------------------------------------------------------------- (a) χ 分布
ax = axes[0]
strip_box(ax, 0, xw_chi_subj, C_XW)
strip_box(ax, 1, tj_chi_subj, C_TJ)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Xuanwu\nn = 47", "Tianjin\nn = 26"], fontsize=8.5)
ax.set_ylabel("Aperiodic exponent χ")
ax.set_ylim(0.2, 2.3)
ax.set_title("a", loc="left", fontweight="bold")
ax.text(0, 2.18, f"mean {xw_chi_subj.mean():.2f}", ha="center", fontsize=7.6, color=C_XW)
ax.text(1, 2.18, f"mean {tj_chi_subj.mean():.2f}", ha="center", fontsize=7.6, color=C_TJ)
STATLINES.append(f"{'chi Xuanwu [3,35]':<34} mean {xw_chi_subj.mean():.3f} ± {xw_chi_subj.std(ddof=1):.3f}")
STATLINES.append(f"{'chi Tianjin [3,35]':<34} mean {tj_chi_subj.mean():.3f} ± {tj_chi_subj.std(ddof=1):.3f}")

# ---------------------------------------------------------------- (b) 两队列 aperiodic / χ-only
ax = axes[1]
groups = [("XW ap LDA", xw_lda140["acc_aperiodic"].values * 100, C_LDA),
          ("XW ap SVM", xw_svm140["svm_aperiodic"].values * 100, C_SVM),
          ("XW chi LDA", xw_abl["lda_chi"].values * 100, C_LDA),
          ("XW chi SVM", xw_abl["svm_chi"].values * 100, C_SVM),
          ("TJ ap LDA", tj_acc["lda_aperiodic"].values * 100, C_LDA),
          ("TJ ap SVM", tj_acc["svm_aperiodic"].values * 100, C_SVM),
          ("TJ chi LDA", tj_abl["lda_chi"].values * 100, C_LDA),
          ("TJ chi SVM", tj_abl["svm_chi"].values * 100, C_SVM)]
pos = [0, 1, 2, 3, 4.7, 5.7, 6.7, 7.7]
for (name, vals, col), x in zip(groups, pos):
    strip_box(ax, x, vals, col, width=0.62)
    m, p = report(name, vals)
    ax.text(x, 73.5, ptxt(p), ha="center", fontsize=7.2, color=col, rotation=90)
ax.axhline(CHANCE, color="0.45", lw=0.9, ls="--")
ax.set_xticks(pos)
ax.set_xticklabels(["LDA", "SVM", "LDA", "SVM", "LDA", "SVM", "LDA", "SVM"], fontsize=7.6)
ax.text(0.5, -0.17, "Xuanwu\naperiodic set", transform=ax.get_xaxis_transform(), ha="center", fontsize=7.8)
ax.text(2.5, -0.17, "Xuanwu\nχ only", transform=ax.get_xaxis_transform(), ha="center", fontsize=7.8)
ax.text(5.2, -0.17, "Tianjin\naperiodic set", transform=ax.get_xaxis_transform(), ha="center", fontsize=7.8)
ax.text(7.2, -0.17, "Tianjin\nχ only", transform=ax.get_xaxis_transform(), ha="center", fontsize=7.8)
ax.set_ylabel("Decoding accuracy (%)")
ax.set_ylim(33, 78)
ax.set_title("b", loc="left", fontweight="bold")

# ---------------------------------------------------------------- (c) 敏感性三分
ax = axes[2]
routes = [("Full · LDA", xw_lda140["acc_full"].values * 100, xw_fit["lda_full"].values * 100, xw_trunc["lda_full"].values * 100, C_LDA),
          ("Full · SVM", xw_svm140["svm_full"].values * 100, xw_fit["svm_full"].values * 100, xw_trunc["svm_full"].values * 100, C_SVM),
          ("Aperiodic · LDA", xw_lda140["acc_aperiodic"].values * 100, xw_fit["lda_aperiodic"].values * 100, xw_trunc["lda_aperiodic"].values * 100, C_LDA),
          ("Aperiodic · SVM", xw_svm140["svm_aperiodic"].values * 100, xw_fit["svm_aperiodic"].values * 100, xw_trunc["svm_aperiodic"].values * 100, C_SVM)]
def pshort(p):
    return "<.001" if p < 0.001 else f"{p:.3f}"[1:]

for i, (name, v_main, v_fit, v_tr, col) in enumerate(routes):
    vs = [v_main, v_fit, v_tr]
    ms, ps = [], []
    for tag, v in zip(["[1,40]", "[fit-only 3,35]", "[truncated 3,35]"], vs):
        m, p = report(f"XW {name} {tag}", v)
        ms.append(m); ps.append(p)
    xs = [i - 0.26, i, i + 0.26]
    ax.plot(xs, ms, color=col, lw=1.2, alpha=0.75, zorder=2)
    for x, v, m in zip(xs, vs, ms):
        ax.errorbar([x], [m], yerr=[v.std(ddof=1)], color=col, capsize=2.2,
                    elinewidth=0.9, zorder=3)
        ax.scatter([x], [m], marker="D", s=32, color=col, zorder=4,
                   edgecolor="white", linewidth=0.7)
    ylab = max(m + v.std(ddof=1) for m, v in zip(ms, vs)) + (1.6 if i % 2 == 0 else 4.6)
    ax.text(i, ylab, " → ".join(pshort(p) for p in ps), ha="center", fontsize=6.6, color=col)
ax.axhline(CHANCE, color="0.45", lw=0.9, ls="--")
ax.set_xlim(-0.6, 3.6)
ax.set_xticks(range(4))
ax.set_xticklabels(["Full\nLDA", "Full\nSVM", "Aper.\nLDA", "Aper.\nSVM"], fontsize=7.5)
ax.set_ylabel("Xuanwu accuracy (%)")
ax.set_ylim(40, 74)
ax.set_title("c", loc="left", fontweight="bold")
ax.text(1.5, -0.21, "each triplet: [1, 40] → fit range only → truncated", transform=ax.get_xaxis_transform(),
        ha="center", fontsize=8)

fig.savefig(os.path.join(OUT, "fig5_generalisation.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig5_generalisation.pdf"), bbox_inches="tight")
print("Fig 5 已存:", os.path.join(OUT, "fig5_generalisation.png/.pdf"))
print("\n标题统计:")
for line in STATLINES:
    print(" ", line)
