#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp4-association.py — Exp 4：χ 与临床状态的横断面关联（C3 / 全文正结果主心骨）
================================================================================
问题：χ（aperiodic 指数）不携带试次级解码信息、也不能校准解码（Exp 2/3），
但它在被试间跨度 2.19×——它是不是一个"人水平"的状态标记？
本实验检验 χ 与同期临床状态的横断面关联（全部同期测量，只说关联、不说预测）。

每被试特征（MI 任务 EEG，specparam 口径与 Exp 1–3 一致，试次中位数）：
    chi_global ：29 导 χ 均值
    chi_AI     ：患侧 χ − 健侧 χ（半球按 aff_hemi 定向；左半球 12 导 / 右 12 导，
                 中线 5 导不入 AI）
    offset 同口径各一份（次要/探索）

预设分析（顺序即报告顺序）：
    A1 Spearman：{chi_global, chi_AI} × {NIHSS, mRS, MBI}（6 个主检验，
       BH-FDR 校正）；offset 版为探索性、单独报告
    A2 敏感性：A1 在幕上病灶子集（deep-only + cortical-involved）复跑
    A3 解释增量：OLS  mRS ~ Age + NIHSS → + chi_AI（ΔR² 与 F 检验）；
       MBI 同法；附偏 Spearman（秩回归残差）作稳健性
    A4 病灶分层：chi_global / chi_AI 在三组间 Kruskal-Wallis；
       cortical vs non-cortical Mann-Whitney
    A5 当前状态区分：mRS 0–2 vs 3–6 的 chi_AI 组间检验 + AUC
       （描述"区分当前功能状态"，非预测）

临床表内嵌（Step 0 编码，源 participants.tsv 已核对；剔除 3 名拷贝被试后 n=47）。
用法: python exp4-association.py（和 probe.py 放同一文件夹，自动定位数据）
输出: ./exp4_subject_data_*.csv（被试级特征+临床，供作图）+ 屏幕分析表
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import signal as spsig

# numpy 2.x 把 trapz 改名为 trapezoid；兼容两种版本
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# ---------------------------------------------------------------- 配置
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
SPECPARAM_KW = dict(peak_width_limits=[1, 8], max_n_peaks=6,
                    min_peak_height=0.05, aperiodic_mode="fixed", verbose=False)
FREQ_RANGE = [1, 40]
N_FFT, N_OVERLAP = 1000, 500   # 与 METHODS-draft v0.2 一致
RANDOM_SEED = 20261017


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- 1. 找 EDF
def find_edfs(data_dir):
    cands = sorted(glob.glob(os.path.join(data_dir, "edffile", "**", "*.edf"), recursive=True))
    if not cands:
        cands = sorted(glob.glob(os.path.join(data_dir, "**", "*.edf"), recursive=True))
    # SEGFIX 合规补齐：剔除资源管理器复制出的"副本 / copy"文件（原版缺此过滤，2026-11-21 死命令）
    cands = [c for c in cands
             if "副本" not in os.path.basename(c)
             and "copy" not in os.path.basename(c).lower()]
    return cands


# ---------------------------------------------------------------- 2. 试次与标签识别
# 官方核查（MOABB Liu2024）：sub-05 与 sub-04 文件逐字节相同，sub-15 / sub-49 分别是
# sub-11 / sub-47 的再处理拷贝 → 全部分析剔除这 3 个、只用 47 人，否则同一人的拷贝会
# 同时进训练/测试折，跨被试准确率虚高（泄漏）。
DERIVED_SUBJECTS = {"sub-05", "sub-15", "sub-49"}

# ================================================================ SEGFIX 补丁（2026-11-21）
# 本文件是 exp4-association.py 的补丁副本，原文件未动。补丁只做三件事：
# (1) 分段修正：MAT/EDF 对齐核查 v4（VERIFY-20261121-onset-v4.md）证明正式起点对
#     98.5% 试次逐采样精确；sub-43 第 25/26/27/31/33 试次脚本起点早 1002 采样
#     （真起点 = 脚本起点 + 1002）；另有 24 个试次两边分段不一致但未判哪边错。
# (2) 死命令合规补强：被试数核对（原始 50 / 分析集 47）对不上或有人解析失败就停；
#     输出统一写脚本自身目录。原版缺失处已在相应位置注明。
# (3) 环境变量 SEG_MODE 控制模式（PowerShell: $env:SEG_MODE="fix" 后再运行）：
#       未设置 / "orig" ：与原版逐位一致；
#       "fix"           ：仅 sub-43 上述 5 段起点 +1002（主分析）；
#       "fixdrop"       ：fix 基础上整段剔除 SEGFIX_DROP 的 29 个可疑试次（敏感性）。
#     修正唯一作用点在 load_subject：起点列表产生之后、切段使用之前。
#     试次号 = 起点列表下标 + 1（时间顺序）。fix/fixdrop 的输出文件名带模式后缀防覆盖。
SEG_MODE = os.environ.get("SEG_MODE", "orig").strip().lower()
if SEG_MODE not in ("orig", "fix", "fixdrop"):
    raise SystemExit(f"SEG_MODE={SEG_MODE!r} 不认识，只许 orig / fix / fixdrop。")
SEG_SUFFIX = {"orig": "", "fix": "_segfix", "fixdrop": "_segfixdrop"}[SEG_MODE]
SEGFIX_SHIFT = {("sub-43", 25), ("sub-43", 26), ("sub-43", 27),
                ("sub-43", 31), ("sub-43", 33)}
SEGFIX_DROP = {
    ("sub-02", 29), ("sub-04", 16),
    ("sub-08", 10), ("sub-08", 21), ("sub-08", 30), ("sub-08", 32),
    ("sub-08", 33), ("sub-08", 36), ("sub-08", 37),
    ("sub-18", 3), ("sub-18", 4), ("sub-18", 10),
    ("sub-19", 13), ("sub-22", 5), ("sub-22", 24), ("sub-29", 25),
    ("sub-32", 33), ("sub-32", 35), ("sub-35", 11), ("sub-35", 34),
    ("sub-40", 8), ("sub-41", 3), ("sub-41", 8),
    ("sub-43", 25), ("sub-43", 26), ("sub-43", 27), ("sub-43", 31), ("sub-43", 33),
    ("sub-50", 36),
}  # 共 29 个 (被试, 原始试次号)


def segfix_apply(subj, onsets, labels):
    """SEGFIX 唯一作用函数：返回 (onsets, labels, kept_trial_numbers)。
    kept_trial_numbers = 保留试次的原始试次号（1 起），供逐试次输出对账。"""
    onsets, labels = list(onsets), list(labels)
    nums = list(range(1, len(onsets) + 1))
    if SEG_MODE == "orig":
        return onsets, labels, nums
    for i, tno in enumerate(nums):  # fix 与 fixdrop 都先做 sub-43 五段 +1002
        if (subj, tno) in SEGFIX_SHIFT:
            onsets[i] = int(onsets[i]) + 1002
    if SEG_MODE == "fixdrop":
        keep = [i for i, tno in enumerate(nums) if (subj, tno) not in SEGFIX_DROP]
        onsets = [onsets[i] for i in keep]
        labels = [labels[i] for i in keep]
        nums = [nums[i] for i in keep]
    return onsets, labels, nums


log(f"[SEGFIX] SEG_MODE={SEG_MODE}（orig=原版逐位一致 / fix=sub-43 五段 +1002 / "
    f"fixdrop=再整段剔除 29 个可疑试次）；输出后缀={SEG_SUFFIX or '(无)'}")
# ================================================================ /SEGFIX 补丁

# 29 导 EEG 白名单（实测通道名）；CPz 为参考（全零）、HEOL/HEOR 为眼电、
# 最后一导无名通道为 STI 触发，均不入选。
EEG29 = ["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4",
         "FT7", "FT8", "Cz", "C3", "C4", "T3", "T4", "CP3", "CP4", "TP7",
         "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"]


def _labels_from_events_tsv(edf_path, sf):
    """优先读 EDF 同目录的 BIDS events.tsv（若存在）：返回 (onsets, labels) 或 None。
    约定列：onset（秒）、trial_type（含 left/right 字样）。onsets 指想象段起点。"""
    import csv
    cand = edf_path.replace("_eeg.edf", "_events.tsv")
    # 2026-10-22 修复：文件名不含 "_eeg.edf"（如 "…_eeg - 副本.edf"）时 replace
    # 不生效、cand 会等于 EDF 本身，旧版随即把二进制 EDF 当文本读而崩
    # （sub-50 两次失败皆源于此）。替换未生效或读文件有任何异常 → 回落 STI 解析。
    if cand == edf_path or not os.path.exists(cand):
        return None
    onsets, labels = [], []
    try:
        with open(cand, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                tt = str(row.get("trial_type", "")).lower()
                if "left" in tt:
                    lab = 0
                elif "right" in tt:
                    lab = 1
                else:
                    continue
                onsets.append(int(round(float(row["onset"]) * sf)))
                labels.append(lab)
    except Exception:
        return None
    if len(onsets) == 40 and sum(labels) == 20:
        return onsets, labels
    return None


def _mi_onsets_from_pulses(pulses):
    """从脉冲序列取想象起点。实测规律（diag6 + 失败明细全量核查）：
    码2→码3 间隔恒为 2000–2003 采样（= 4 秒想象段），在所有被试、所有
    试次（包括异常试次）中都成立；不可靠的只有码1（指令）脉冲——它会
    缺失、迟到或与码2/3 轮转。故事：想象起点 = 码2 脉冲本身。
    规则：把相邻的码2 归为一 run（回跳重复），run 后紧跟的脉冲须为码3
    且间隔 1900–2100；run 内第一个满足该间隔的码2 即起点（真脉冲在前、
    回跳在后时取真脉冲；间隔对不上前者时自动落到后者）。无码3 后继的
    码2（尾部野脉冲）丢弃。返回 (onsets, n_c2_total) 或 (None, reason)。"""
    onsets = []
    n_c2 = sum(1 for _, c in pulses if c == 2)
    i, n = 0, len(pulses)
    while i < n:
        if pulses[i][1] != 2:
            i += 1
            continue
        run = []
        while i < n and pulses[i][1] == 2:
            run.append(pulses[i][0])
            i += 1
        if i < n and pulses[i][1] == 3:
            nxt = pulses[i][0]
            for pos in run:
                if 1900 <= nxt - pos <= 2100:
                    onsets.append(pos)
                    break
    if len(onsets) != 40:
        return None, f"码2脉冲={n_c2}，通过码2→码3配对的起点={len(onsets)}（需 40）"
    gaps = np.diff(onsets)
    if not np.all((gaps >= 2000) & (gaps <= 6000)):
        return None, f"起点间距异常: min={gaps.min()}, max={gaps.max()}"
    return onsets, n_c2


def _labels_from_sti(raw):
    """STI 触发通道（最后一导，µV 单位）：码 1=指令、2=想象开始、3=休息。
    实测：码2→码3（4 秒想象段）恒定可靠；不可靠的只有码1（指令）脉冲
    （缺失/迟到/与相邻码轮转）与个别回跳、幅度抖动。故解析只认码2→码3
    配对（_mi_onsets_from_pulses），码1 完全不用。数不出 40 个试次返回
    None 并附诊断。
    左右手不编码在触发里（范式左右交替）：第 1 个试次=左手，交替赋值
    （该约定只影响"哪只手"的说法、不影响二分类解码准确率；全量前用 .mat 标签复核）。
    返回 ((onsets, labels), info) 或 (None, info)；info 为诊断字符串，
    失败时随异常一起报出。labels: 0=左手, 1=右手。"""
    sti = raw.get_data(picks=[raw.ch_names[-1]]).ravel() / 1e-6   # 换算到 µV
    act = (np.abs(sti) > 0.5).astype(int)
    starts = np.flatnonzero(np.diff(np.r_[0, act]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[act, 0]) == -1)
    pulses = []
    for s, e in zip(starts, ends):
        seg = sti[s:e + 1]
        peak = float(seg[np.argmax(np.abs(seg))])
        code = int(np.rint(peak))
        if 1 <= code <= 3 and abs(peak - code) <= 0.45:
            pulses.append((int(s), code))
    onsets, extra = _mi_onsets_from_pulses(pulses)
    n_by_code = {c: sum(1 for _, cc in pulses if cc == c) for c in (1, 2, 3)}
    if onsets is None:
        info = (f"STI活动段={len(starts)}, 脉冲计数(1/2/3码)={n_by_code}, "
                f"码2→码3配对解析失败: {extra}")
        return None, info
    info = f"码2→码3配对解析 40 试次（码2脉冲共 {extra} 个）"
    labels = [i % 2 for i in range(40)]              # 交替，第 1 个=左手
    return (onsets, labels), info


def load_subject(edf_path):
    """返回 dict(trials: (n_kept, 29, 2000), labels: (n_kept,) 0=左/1=右, ch_names, sfreq, subj)。
    试次 = 想象段 4 秒（想象触发起到 +4s，与 MOABB Liu2024 的 (0,4)s 口径一致）。"""
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sf = float(raw.info["sfreq"])
    names = raw.ch_names
    subj = os.path.basename(edf_path).split("_")[0]

    eeg_names = [n for n in EEG29 if n in names]
    if len(eeg_names) != 29:
        raise RuntimeError(f"{subj}: 29 导白名单只对上 {len(eeg_names)} 导，通道名={names}")

    got = _labels_from_events_tsv(edf_path, sf)
    how = "events.tsv"
    sti_info = ""
    if got is None:
        got, sti_info = _labels_from_sti(raw)
        how = f"STI码2→码3配对解析（{sti_info}）"
    if got is None:
        raise RuntimeError(
            f"{subj}: 仍识别不出试次（无 events.tsv；STI 诊断: {sti_info}）。"
            f"通道数={len(names)}，29导白名单对上={len(eeg_names)}。")
    onsets, labels = got
    # SEGFIX: 起点产生后、切段前的唯一修正点（orig 模式为恒等变换）
    onsets, labels, trial_idx = segfix_apply(subj, onsets, labels)
    labels = np.array(labels)

    n_samp = int(4 * sf)                              # 想象段 4 秒 = 2000 采样
    data = raw.get_data(picks=eeg_names)              # (29, 全长)
    trials = np.stack([data[:, s:s + n_samp] for s in onsets])
    assert trials.shape == (len(onsets), 29, n_samp), f"{subj}: trials shape={trials.shape}"
    log(f"  {subj}: {len(onsets)} 试次 ✓（左 {int((labels == 0).sum())} / 右 {int((labels == 1).sum())}），"
        f"识别方式={how}，EEG 29 导，想象段 4s，SEG_MODE={SEG_MODE}")
    return dict(trials=trials, labels=labels, ch_names=eeg_names, sfreq=sf, subj=subj,
                trial_idx=trial_idx)  # SEGFIX: 保留试次的原始试次号


# ---------------------------------------------------------------- 3. 特征提取
def extract_features(trials, sf):
    """trials: (n_trial, n_ch, n_times) → 三种输入的特征矩阵 + χ。(全部 log10 域外已转回线性再积分)"""
    from specparam import SpectralGroupModel

    n_trial, n_ch, n_times = trials.shape
    freqs, _ = spsig.welch(trials[0, 0], fs=sf, nperseg=N_FFT, noverlap=N_OVERLAP)
    fmask = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
    freqs = freqs[fmask]

    # 全部试次×通道的 PSD（向量化一次算完）
    spectra = np.empty((n_trial * n_ch, len(freqs)))
    for ti in range(n_trial):
        for ci in range(n_ch):
            _, p = spsig.welch(trials[ti, ci], fs=sf, nperseg=N_FFT, noverlap=N_OVERLAP)
            spectra[ti * n_ch + ci] = p[fmask]

    fg = SpectralGroupModel(**SPECPARAM_KW)
    try:
        fg.fit(freqs, spectra, FREQ_RANGE)
    except TypeError:  # 极少数版本签名差异
        fg.fit(freqs, spectra, freq_range=FREQ_RANGE)
    try:
        ap = fg.get_params("aperiodic_params")      # specparam 1.x → (n, 2): [offset, exponent]
    except Exception:
        try:
            ap = fg.get_params("aperiodic")         # specparam 2.x 改名（2026-10-20 实测 2.0.0rc7）
        except Exception as e:
            raise RuntimeError(f"specparam 取 aperiodic 参数失败: {e}")
    try:
        r2 = np.asarray(fg.get_params("r_squared"), dtype=float)   # specparam 1.x
    except Exception:
        try:
            r2 = np.asarray(fg.get_metrics("gof"), dtype=float)    # specparam 2.x
        except Exception:
            r2 = np.full(len(spectra), np.nan)

    logf = np.log10(freqs)
    ap_log = ap[:, 0:1] - ap[:, 1:2] * logf          # specparam 拟合的是 log10 功率
    ap_lin = 10.0 ** ap_log
    per_lin = np.maximum(spectra - ap_lin, 1e-12)    # periodic = 全谱 − 非周期

    feats = {}
    for name, spec in (("full", spectra), ("periodic", per_lin), ("aperiodic", ap_lin)):
        band_feats = []
        for lo, hi in BANDS.values():
            idx = (freqs >= lo) & (freqs < hi)
            bp = _trapz(spec[:, idx], freqs[idx], axis=1)
            band_feats.append(np.log10(np.maximum(bp, 1e-12)))
        fmat = np.stack(band_feats, axis=1).reshape(n_trial, n_ch, len(BANDS))
        feats[name] = fmat.reshape(n_trial, n_ch * len(BANDS))

    chi = ap[:, 1].reshape(n_trial, n_ch)            # χ（指数）
    offset = ap[:, 0].reshape(n_trial, n_ch)         # offset（截距）
    return feats, chi, offset, float(np.nanmedian(r2))


# ---------------------------------------------------------------- 4. Exp 4 分析
LEFT_CH = ["FP1", "F3", "F7", "FC3", "FT7", "C3", "T3", "CP3", "TP7", "P3", "T5", "O1"]
RIGHT_CH = ["FP2", "F4", "F8", "FC4", "FT8", "C4", "T4", "CP4", "TP8", "P4", "T6", "O2"]

# 临床表（Step 0 编码，源 participants.tsv）：aff=患侧半球，ternary=三分层，
# cortical=cortical/non-cortical，其余为 NIHSS/mRS/MBI/Age/Duration/sex
CLINICAL = {
    'sub-01': {'aff': 'left', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 11, 'mrs': 4, 'mbi': 50, 'age': 45, 'duration': 1, 'sex': 'male'},
    'sub-02': {'aff': 'right', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 3, 'mrs': 4, 'mbi': 55, 'age': 60, 'duration': 2, 'sex': 'male'},
    'sub-03': {'aff': 'right', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 3, 'mrs': 1, 'mbi': 90, 'age': 60, 'duration': 2, 'sex': 'male'},
    'sub-04': {'aff': 'left', 'ternary': 'cortical-involved', 'cortical': 'cortical', 'nihss': 6, 'mrs': 3, 'mbi': 90, 'age': 56, 'duration': 14, 'sex': 'male'},
    'sub-05': {'aff': 'right', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 4, 'mrs': 4, 'mbi': 60, 'age': 44, 'duration': 4, 'sex': 'female'},
    'sub-06': {'aff': 'right', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 3, 'mrs': 3, 'mbi': 85, 'age': 66, 'duration': 6, 'sex': 'male'},
    'sub-07': {'aff': 'left', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 2, 'mrs': 1, 'mbi': 100, 'age': 62, 'duration': 5, 'sex': 'male'},
    'sub-08': {'aff': 'right', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 3, 'mrs': 2, 'mbi': 85, 'age': 64, 'duration': 5, 'sex': 'male'},
    'sub-09': {'aff': 'left', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 6, 'mrs': 1, 'mbi': 55, 'age': 57, 'duration': 3, 'sex': 'male'},
    'sub-10': {'aff': 'right', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 3, 'mrs': 0, 'mbi': 55, 'age': 55, 'duration': 2, 'sex': 'male'},
    'sub-11': {'aff': 'right', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 5, 'mrs': 4, 'mbi': 55, 'age': 31, 'duration': 7, 'sex': 'male'},
    'sub-12': {'aff': 'left', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 1, 'mrs': 1, 'mbi': 100, 'age': 58, 'duration': 1, 'sex': 'male'},
    'sub-13': {'aff': 'left', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 9, 'mrs': 4, 'mbi': 55, 'age': 46, 'duration': 3, 'sex': 'male'},
    'sub-14': {'aff': 'left', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 2, 'mrs': 1, 'mbi': 75, 'age': 67, 'duration': 2, 'sex': 'female'},
    'sub-15': {'aff': 'right', 'ternary': 'cortical-involved', 'cortical': 'cortical', 'nihss': 7, 'mrs': 1, 'mbi': 55, 'age': 63, 'duration': 1, 'sex': 'male'},
    'sub-16': {'aff': 'left', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 4, 'mrs': 2, 'mbi': 90, 'age': 57, 'duration': 1, 'sex': 'male'},
    'sub-17': {'aff': 'right', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 3, 'mrs': 1, 'mbi': 85, 'age': 60, 'duration': 3, 'sex': 'male'},
    'sub-18': {'aff': 'right', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 10, 'mrs': 4, 'mbi': 45, 'age': 60, 'duration': 1, 'sex': 'female'},
    'sub-19': {'aff': 'right', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 8, 'mrs': 4, 'mbi': 40, 'age': 62, 'duration': 1, 'sex': 'female'},
    'sub-20': {'aff': 'left', 'ternary': 'cortical-involved', 'cortical': 'cortical', 'nihss': 2, 'mrs': 1, 'mbi': 95, 'age': 34, 'duration': 24, 'sex': 'male'},
    'sub-21': {'aff': 'right', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 2, 'mrs': 1, 'mbi': 90, 'age': 41, 'duration': 5, 'sex': 'male'},
    'sub-22': {'aff': 'right', 'ternary': 'cortical-involved', 'cortical': 'cortical', 'nihss': 11, 'mrs': 4, 'mbi': 45, 'age': 52, 'duration': 6, 'sex': 'male'},
    'sub-23': {'aff': 'right', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 4, 'mrs': 4, 'mbi': 40, 'age': 57, 'duration': 2, 'sex': 'male'},
    'sub-24': {'aff': 'left', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 1, 'mrs': 4, 'mbi': 60, 'age': 55, 'duration': 3, 'sex': 'female'},
    'sub-25': {'aff': 'right', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 5, 'mrs': 4, 'mbi': 55, 'age': 47, 'duration': 10, 'sex': 'male'},
    'sub-26': {'aff': 'right', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 4, 'mrs': 4, 'mbi': 70, 'age': 61, 'duration': 1, 'sex': 'male'},
    'sub-27': {'aff': 'left', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 3, 'mrs': 3, 'mbi': 70, 'age': 52, 'duration': 5, 'sex': 'female'},
    'sub-28': {'aff': 'left', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 1, 'mrs': 1, 'mbi': 85, 'age': 42, 'duration': 2, 'sex': 'female'},
    'sub-29': {'aff': 'left', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 1, 'mrs': 0, 'mbi': 95, 'age': 53, 'duration': 16, 'sex': 'male'},
    'sub-30': {'aff': 'right', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 4, 'mrs': 1, 'mbi': 100, 'age': 68, 'duration': 4, 'sex': 'male'},
    'sub-31': {'aff': 'left', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 3, 'mrs': 3, 'mbi': 80, 'age': 59, 'duration': 5, 'sex': 'female'},
    'sub-32': {'aff': 'left', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 3, 'mrs': 4, 'mbi': 81, 'age': 74, 'duration': 2, 'sex': 'male'},
    'sub-33': {'aff': 'left', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 3, 'mrs': 4, 'mbi': 58, 'age': 63, 'duration': 7, 'sex': 'male'},
    'sub-34': {'aff': 'right', 'ternary': 'cortical-involved', 'cortical': 'cortical', 'nihss': 1, 'mrs': 2, 'mbi': 85, 'age': 69, 'duration': 1, 'sex': 'female'},
    'sub-35': {'aff': 'right', 'ternary': 'cortical-involved', 'cortical': 'cortical', 'nihss': 1, 'mrs': 4, 'mbi': 52, 'age': 69, 'duration': 11, 'sex': 'male'},
    'sub-36': {'aff': 'right', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 6, 'mrs': 3, 'mbi': 63, 'age': 69, 'duration': 30, 'sex': 'male'},
    'sub-37': {'aff': 'right', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 3, 'mrs': 4, 'mbi': 88, 'age': 49, 'duration': 7, 'sex': 'male'},
    'sub-38': {'aff': 'left', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 3, 'mrs': 3, 'mbi': 64, 'age': 53, 'duration': 2, 'sex': 'male'},
    'sub-39': {'aff': 'left', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 11, 'mrs': 5, 'mbi': 32, 'age': 56, 'duration': 2, 'sex': 'male'},
    'sub-40': {'aff': 'right', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 6, 'mrs': 4, 'mbi': 65, 'age': 56, 'duration': 6, 'sex': 'female'},
    'sub-41': {'aff': 'left', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 7, 'mrs': 4, 'mbi': 60, 'age': 77, 'duration': 2, 'sex': 'male'},
    'sub-42': {'aff': 'left', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 7, 'mrs': 4, 'mbi': 60, 'age': 54, 'duration': 2, 'sex': 'male'},
    'sub-43': {'aff': 'right', 'ternary': 'cortical-involved', 'cortical': 'cortical', 'nihss': 2, 'mrs': 4, 'mbi': 65, 'age': 32, 'duration': 3, 'sex': 'male'},
    'sub-44': {'aff': 'right', 'ternary': 'cortical-involved', 'cortical': 'cortical', 'nihss': 1, 'mrs': 1, 'mbi': 80, 'age': 59, 'duration': 1, 'sex': 'male'},
    'sub-45': {'aff': 'left', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 6, 'mrs': 4, 'mbi': 84, 'age': 64, 'duration': 30, 'sex': 'male'},
    'sub-46': {'aff': 'left', 'ternary': 'cortical-involved', 'cortical': 'cortical', 'nihss': 1, 'mrs': 1, 'mbi': 85, 'age': 66, 'duration': 10, 'sex': 'male'},
    'sub-47': {'aff': 'right', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 7, 'mrs': 4, 'mbi': 55, 'age': 40, 'duration': 4, 'sex': 'male'},
    'sub-48': {'aff': 'right', 'ternary': 'cortical-involved', 'cortical': 'cortical', 'nihss': 2, 'mrs': 1, 'mbi': 90, 'age': 75, 'duration': 18, 'sex': 'male'},
    'sub-49': {'aff': 'right', 'ternary': 'deep-only', 'cortical': 'non-cortical', 'nihss': 1, 'mrs': 1, 'mbi': 85, 'age': 52, 'duration': 3, 'sex': 'male'},
    'sub-50': {'aff': 'right', 'ternary': 'brainstem/cerebellum', 'cortical': 'non-cortical', 'nihss': 3, 'mrs': 2, 'mbi': 85, 'age': 64, 'duration': 1, 'sex': 'female'},
}


def _bh_fdr(pvals):
    """Benjamini-Hochberg 校正，返回与输入同序的 q 值。"""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def _ols(y, X):
    """OLS：返回 (R², 系数)。X 含截距列。"""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return (1 - ss_res / ss_tot if ss_tot > 0 else 0.0), beta, ss_res


def _increment(y, X_base, x_new):
    """解释增量：base 模型 vs base+x_new 的 ΔR² 与 F 检验 p。"""
    from scipy import stats
    n = len(y)
    r2_b, _, ssr_b = _ols(y, X_base)
    X_full = np.column_stack([X_base, x_new])
    r2_f, _, ssr_f = _ols(y, X_full)
    df1, df2 = 1, n - X_full.shape[1]
    F = ((ssr_b - ssr_f) / df1) / (ssr_f / df2) if ssr_f > 0 and df2 > 0 else np.nan
    p = float(stats.f.sf(F, df1, df2)) if np.isfinite(F) else np.nan
    return r2_b, r2_f, r2_f - r2_b, p


def _partial_spearman(x, y, covars):
    """偏 Spearman：对秩做线性回归取残差后相关。covars: (n, k) 原始尺度。"""
    from scipy import stats
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    C = np.column_stack([np.ones(len(x))] + [stats.rankdata(covars[:, j])
                                             for j in range(covars.shape[1])])
    bx, *_ = np.linalg.lstsq(C, rx, rcond=None)
    by, *_ = np.linalg.lstsq(C, ry, rcond=None)
    return stats.pearsonr(rx - C @ bx, ry - C @ by)


def exp4_analyses(df):
    from scipy import stats
    log("\n" + "=" * 68)
    log(f"Exp 4 分析（n={len(df)}；横断面同期关联，全文禁用预测措辞）")
    log("=" * 68)

    def spear_block(sub, title):
        log(f"\n— {title}（n={len(sub)}）—")
        keys = [("chi_global", "NIHSS"), ("chi_global", "mrs"), ("chi_global", "mbi"),
                ("chi_AI", "NIHSS"), ("chi_AI", "mrs"), ("chi_AI", "mbi")]
        ps = []
        res = []
        for xk, yk in keys:
            r, p = stats.spearmanr(sub[xk], sub[yk])
            res.append((xk, yk, r, p))
            ps.append(p)
        qs = _bh_fdr(ps)
        for (xk, yk, r, p), q in zip(res, qs):
            log(f"  {xk:10s} × {yk:6s}: rho={r:+.3f}, p={p:.4f}, q(BH)={q:.4f}")

    spear_block(df, "A1 主分析：χ × 临床量表（Spearman，BH 校正）")
    supra = df[df["ternary"] != "brainstem/cerebellum"]
    if len(supra) >= 15:
        spear_block(supra, "A2 敏感性：幕上病灶子集")

    log("\n— 探索性：offset × 临床量表（不校正，单独看）—")
    for xk in ("offset_global", "offset_AI"):
        for yk in ("NIHSS", "mrs", "mbi"):
            r, p = stats.spearmanr(df[xk], df[yk])
            log(f"  {xk:13s} × {yk:6s}: rho={r:+.3f}, p={p:.4f}")

    log("\n— A3 解释增量（ΔR²：Age+NIHSS 之外，chi_AI 还能解释多少）—")
    for yk in ("mrs", "mbi"):
        y = df[yk].to_numpy(dtype=float)
        Xb = np.column_stack([np.ones(len(df)), df["age"], df["NIHSS"]])
        r2b, r2f, dr2, p = _increment(y, Xb, df["chi_AI"].to_numpy(dtype=float))
        rho, pp = _partial_spearman(df["chi_AI"], df[yk],
                                    df[["age", "NIHSS"]].to_numpy(dtype=float))
        log(f"  {yk}: base R²={r2b:.3f} → +chi_AI R²={r2f:.3f}（ΔR²={dr2:+.3f}, "
            f"F 检验 p={p:.4f}）；偏 Spearman rho={rho:+.3f}, p={pp:.4f}")

    log("\n— A4 病灶分层 —")
    groups = [g["chi_global"].to_numpy() for _, g in df.groupby("ternary")]
    if len(groups) == 3:
        H, p = stats.kruskal(*groups)
        log(f"  chi_global 三组 Kruskal-Wallis: H={H:.2f}, p={p:.4f}")
        groups = [g["chi_AI"].to_numpy() for _, g in df.groupby("ternary")]
        H, p = stats.kruskal(*groups)
        log(f"  chi_AI     三组 Kruskal-Wallis: H={H:.2f}, p={p:.4f}")
    a = df[df["cortical"] == "cortical"]["chi_global"]
    b = df[df["cortical"] != "cortical"]["chi_global"]
    if len(a) >= 3 and len(b) >= 3:
        U, p = stats.mannwhitneyu(a, b)
        log(f"  chi_global cortical(n={len(a)}) vs non-cortical(n={len(b)}): "
            f"中位 {a.median():.3f} vs {b.median():.3f}, p={p:.4f}")

    log("\n— A5 当前状态区分：mRS 0–2 vs 3–6 —")
    lo = df[df["mrs"] <= 2]["chi_AI"]
    hi = df[df["mrs"] >= 3]["chi_AI"]
    if len(lo) >= 5 and len(hi) >= 5:
        U, p = stats.mannwhitneyu(lo, hi)
        auc = 1 - U / (len(lo) * len(hi))   # P(随机抽的高分组被试 χ_AI 更大)
        log(f"  chi_AI：mRS 0–2（n={len(lo)}）中位 {lo.median():+.3f} vs "
            f"mRS 3–6（n={len(hi)}）中位 {hi.median():+.3f}，p={p:.4f}，AUC={auc:.3f}")
    log("=" * 68)


# ---------------------------------------------------------------- 5.  主流程


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="数据目录；不填则自动在脚本所在目录/当前目录下找 EDF")
    ap.add_argument("--subjects", type=int, default=47)
    ap.add_argument("--list", action="store_true", help="只打印第一个 EDF 的结构诊断")
    args = ap.parse_args()

    if args.data_dir:
        search_dirs = [args.data_dir]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        search_dirs = [script_dir, os.getcwd(), os.path.join(os.getcwd(), "stroke-eeg")]
    edfs = []
    for d in search_dirs:
        edfs = find_edfs(d)
        if edfs:
            args.data_dir = d
            break
    if not edfs:
        sys.exit("在脚本所在目录和当前目录下都没找到 .edf。"
                 "把脚本放到解压好的数据文件夹里（和 sub-01 等文件夹同层）再跑。")
    log(f"数据目录: {args.data_dir}")
    kept, skipped = [], []
    for ef in edfs:
        sid = os.path.basename(ef).split("_")[0]
        (skipped if sid in DERIVED_SUBJECTS else kept).append(ef)
    if skipped:
        log(f"已剔除重复被试（官方核查的拷贝）: "
            f"{[os.path.basename(e).split('_')[0] for e in skipped]}")
    probe_edfs = kept[:args.subjects]
    log(f"找到 {len(edfs)} 个 EDF，用其中 {len(probe_edfs)} 个做 Exp 4: "
        f"{[os.path.basename(e).split('_')[0] for e in probe_edfs]}")

    if args.list:
        import mne
        raw = mne.io.read_raw_edf(edfs[0], preload=True, verbose=False)
        log(f"文件: {edfs[0]}")
        log(f"时长: {raw.times[-1]:.1f}s, 采样率: {raw.info['sfreq']} Hz")
        log(f"通道({len(raw.ch_names)}): {raw.ch_names}")
        ann = raw.annotations
        log(f"annotations({len(ann)}): {sorted(set(ann.description)) if len(ann) else '无'}")
        return

    # SEGFIX 死命令补强：原始 50 人 / 分析集 47 人，对不上就停（原版只记失败、会静默少跑）
    found_ids = sorted({os.path.basename(e).split("_")[0] for e in edfs})
    if len(found_ids) != 50 or len(kept) != 47:
        sys.exit(f"被试数异常：找到 {len(found_ids)} 人（预期 50）、分析集 {len(kept)} 人"
                 "（预期 47）。先查数据目录再跑，不静默少跑。")
    log(f"[SEGFIX] 被试数核对通过：原始 {len(found_ids)} / 分析集 {len(kept)}")

    t0 = time.time()
    rows, failed = [], []
    n_trials_total = 0  # SEGFIX: 分析集总试次累计
    for ef in probe_edfs:
        sid = os.path.basename(ef).split("_")[0]
        clin = CLINICAL.get(sid)
        if clin is None:
            failed.append((sid, "临床表缺此被试"))
            log(f"  [跳过] {sid}: 临床表缺此被试")
            continue
        try:
            d = load_subject(ef)
        except Exception as e:
            failed.append((sid, str(e)))
            log(f"  [跳过] {sid}: {e}")
            continue
        n_trials_total += int(d["trials"].shape[0])  # SEGFIX: 累计该被试实际试次数
        feats, chi, offset, r2 = extract_features(d["trials"], d["sfreq"])
        chi_ch = np.median(chi, axis=0)              # 每导 χ（试次中位数）
        off_ch = np.median(offset, axis=0)
        names = list(d["ch_names"])
        li = [names.index(c) for c in LEFT_CH if c in names]
        ri = [names.index(c) for c in RIGHT_CH if c in names]
        left_chi, right_chi = float(chi_ch[li].mean()), float(chi_ch[ri].mean())
        left_off, right_off = float(off_ch[li].mean()), float(off_ch[ri].mean())
        ipsi_chi, contra_chi = (left_chi, right_chi) if clin["aff"] == "left" else (right_chi, left_chi)
        ipsi_off, contra_off = (left_off, right_off) if clin["aff"] == "left" else (right_off, left_off)
        rows.append(dict(
            subj=sid, chi_global=float(chi_ch.mean()), chi_ipsi=ipsi_chi,
            chi_contra=contra_chi, chi_AI=ipsi_chi - contra_chi,
            offset_global=float(off_ch.mean()), offset_AI=ipsi_off - contra_off,
            r2=r2, NIHSS=clin["nihss"], mrs=clin["mrs"], mbi=clin["mbi"],
            age=clin["age"], duration=clin["duration"], sex=clin["sex"],
            aff_hemi=clin["aff"], ternary=clin["ternary"], cortical=clin["cortical"]))
        log(f"  {sid}: χ_global={chi_ch.mean():.3f}, χ_AI={ipsi_chi - contra_chi:+.3f}, "
            f"NIHSS={clin['nihss']}, mRS={clin['mrs']}, MBI={clin['mbi']}")
    if failed:
        log(f"\n注意：{len(failed)} 个被试未能处理、已跳过：{[s for s, _ in failed]}")
    if failed:
        sys.exit(f"[SEGFIX] {len(failed)} 个被试未能处理：{[s for s, _ in failed]}。"
                 "按死命令停下，不静默少跑一人；先解决问题再跑。")
    if len(rows) != len(probe_edfs):
        sys.exit(f"[SEGFIX] 成功处理 {len(rows)} 人 ≠ 计划 {len(probe_edfs)} 人，停下。")
    log(f"[SEGFIX] 分析集总试次 = {n_trials_total}（orig/fix 应 1880；fixdrop 应 1851）")
    if len(rows) < 10:
        sys.exit(f"成功处理的被试只剩 {len(rows)} 个，太少，先解决问题再跑。")

    df = pd.DataFrame(rows)
    log(f"\n特征提取完成，用时 {(time.time() - t0) / 60:.1f} 分钟；共 {len(df)} 人进入分析。")
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"exp4_subject_data{SEG_SUFFIX}_{ts}.csv")  # SEGFIX: 输出改脚本目录（原版 getcwd 不合规）
    df.to_csv(out, index=False)
    log(f"被试级数据已存: {out}")
    exp4_analyses(df)


if __name__ == "__main__":
    main()
