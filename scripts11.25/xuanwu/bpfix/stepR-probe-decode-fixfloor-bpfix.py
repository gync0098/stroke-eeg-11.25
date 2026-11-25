#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stepR-probe-decode.py — 2026-10-17 pivot 探针：specparam + 三种输入解码（默认全量 47 人）
================================================================================
这是 pivot 后 Step R 的"1 天探针"脚本。跑完直接回答一个问题：
    aperiodic 到底携不携带左右手 MI 解码信息？（定 C1 还是 C2/C2 方向）

做什么（对应 METHODS-draft v0.2 的 Exp 1 + Exp 2 探针版）:
    1. 读被试的预处理 EDF（edffile.zip 解压后），自动识别 40 试次与左右手标签
    2. 每试次每通道：Welch PSD(1–40 Hz) → specparam（fixed 模式，参数与 Methods 一致）
    3. 三种输入：完整谱 / 仅 periodic / 仅 aperiodic → 4 频段功率特征
    4. 试次级 leave-one-subject-out（N 折；每折 (N−1) 人×40 试次训练、1 人 40 试次测试）
       —— full_guide 手册原伪代码是被试级 LOO（设计 bug），这里已按修正写法实现
    5. 输出三种输入的准确率对比 + 一句话方向判断（C1 / 边界 / C2·C3）

用法:
    python stepR-probe-decode.py [--data-dir stroke-eeg] [--subjects 5]
    python stepR-probe-decode.py --list        # 只打印第一个 EDF 的结构（试次/标签识别诊断），不跑

依赖: mne, specparam, sklearn, scipy, numpy, pandas（见 stroke-project-quickstart.md）
输入: <data-dir>/edffile/**/*.edf（或 <data-dir>/**/*.edf），可选 participants.tsv（用于 χ 不对称速览）
输出: ./probe_results_YYYYMMDD_HHMMSS.csv（每折三种输入准确率）+ 屏幕 verdict

已核实的数据事实（2026-10-21 实测 + MOABB Liu2024 官方口径）:
    - 每被试 40 试次、8 秒/试次，结构为 指令(2s) + 想象(4s) + 休息(2s)；本脚本只取
      想象段 4 秒（2000 采样 @500 Hz），与 MOABB 的 (0,4)s 口径一致。
    - 标签不在 annotations 里：最后一导（无名）是 STI 触发通道，值为 µV 单位的整数码
      1=指令、2=想象开始、3=休息，每试次各一个单采样脉冲（共 120 个）。
      左右手不编码在触发里（范式左右交替）：第 1 个试次=左手，交替赋值；
      若 EDF 同目录有 BIDS events.tsv 则优先读它。标签 1=左手、2=右手。
    - 通道：29 EEG + CPz 参考（全零）+ 2 眼电（HEOL/HEOR）+ 1 STI 通道。
    - **重复被试（重要）**：sub-05 与 sub-04 的 EEG 数据流逐样本相同（整文件字节不同，2026-11-21 定量核查），sub-15 / sub-49 分别是
      sub-11 / sub-47 的再处理拷贝 → 全部分析剔除这 3 人、只用 47 人（MOABB 官方做法），
      否则拷贝对跨折泄漏、跨被试准确率虚高。
    - 预处理 EDF 路径形如 sub-01/eeg/sub-01_task-motor-imagery_eeg.edf
    - 作者自己的解码基线：TWFB+DGFMDM 达到 72.21%。探针用的是更简单的
      LDA + 频段功率特征，数字会低一截——重点看三种输入的相对关系、
      以及 aperiodic 是否显著高于 50% 随机线，不要跟 72.21% 直接比。
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
    # 剔除资源管理器复制出的"副本 / copy"文件：sub-50 有一份副本 EDF，排序靠前
    # 且头部字节不可读（utf-8 解码失败，2026-11-21 18:11 首跑实测 sub-50 因此被跳过、只剩 46 人）。
    cands = [c for c in cands
             if "副本" not in os.path.basename(c)
             and "copy" not in os.path.basename(c).lower()]
    return cands


# ---------------------------------------------------------------- 2. 试次与标签识别
# 官方核查（MOABB Liu2024）：sub-05 与 sub-04 的 EEG 数据流逐样本相同（整文件字节不同，2026-11-21 定量核查），sub-15 / sub-49 分别是
# sub-11 / sub-47 的再处理拷贝 → 全部分析剔除这 3 个、只用 47 人，否则同一人的拷贝会
# 同时进训练/测试折，跨被试准确率虚高（泄漏）。
DERIVED_SUBJECTS = {"sub-05", "sub-15", "sub-49"}

# ================================================================ SEGFIX 补丁（2026-11-21）
# 本文件是 stepR-probe-decode-fixfloor.py 的补丁副本（本 -bpfix 版以同名 -segfix 版为底复制，segfix 版与原文件均未动）。补丁内容 = SEGFIX 三件事 + BPFIX 三件事：
# (1) 分段修正：MAT/EDF 对齐核查 v4（VERIFY-20261121-onset-v4.md）证明正式起点对
#     98.5% 试次逐采样精确；sub-43 第 25/26/27/31/33 试次脚本起点早 1002 采样
#     （真起点 = 脚本起点 + 1002）；另有 24 个试次两边分段不一致但未判哪边错。
# (2) 死命令合规补强：被试数核对（原始 50 / 分析集 47）对不上或有人解析失败就停；
#     输出统一写脚本自身目录。原版缺失处已在相应位置注明。
# (3) 环境变量 SEG_MODE 控制模式（PowerShell: $env:SEG_MODE="fix" 后再运行）：
#       未设置 / "orig" ：分段同原版（但带功率下限已改 1e-20、特征不同；本轮不用）；
#       "fix"           ：仅 sub-43 上述 5 段起点 +1002（主分析）；
#       "fixdrop"       ：fix 基础上整段剔除 SEGFIX_DROP 的 29 个可疑试次（敏感性；本轮不用）。
#       "drop18"        ：fix 基础上整段剔除 BPFIX_DROP18 的 18 个边界码2 试次（敏感性；本轮用）。
#     修正唯一作用点在 load_subject：起点列表产生之后、切段使用之前。
#     试次号 = 起点列表下标 + 1（时间顺序）。fix/fixdrop 的输出文件名带模式后缀防覆盖。
# BPFIX 追加层（2026-11-21）：本文件是同名 -segfix 版的 -bpfix 副本，
# segfix 版与原文件均未动。在 SEGFIX 之上只再改三类地方：
# (4) 带功率下限对照：特征提取的 log10(max(bp, 1e-12)) 经计数核查
#     （bpfloor_count_20261121_215352.csv，fix 分段口径）证实大面积截平——
#     full 6.20% / periodic 26.74% / aperiodic 16.31%（分母各 218,080），
#     47 人全部触底。故本副本带功率下限一律改 1e-20（仅拦截 <=0 的积分值）。
#     segfix 版 stepR/robust 注释中“频段功率实测从未触底”的说法系无效计数
#     （VERIFY-20261121-bpfloor-and-wording.md），在本副本中已更正、作废。
# (5) SEG_MODE 新增第四档 "drop18"：fix 基础上、切段前整段剔除 BPFIX_DROP18
#     的 18 个边界码2 试次（事件标记不确定性敏感性；保留原始标签、不重排）。
#     本轮只跑 "fix" 与 "drop18" 两档；"orig" / "fixdrop" 保留但不用，
#     其输出后缀已加 bpfix 标记（_bpfixorig / _bpfixdrop）防与旧结果混淆。
# (6) 解码脚本（stepR / robust / ablation）结果 CSV 新增 bal_acc_* 平衡
#     准确率列（左/右两类各自准确率的均值）；原 acc 列与逻辑一字未动。
SEG_MODE = os.environ.get("SEG_MODE", "orig").strip().lower()
if SEG_MODE not in ("orig", "fix", "fixdrop", "drop18"):
    raise SystemExit(f"SEG_MODE={SEG_MODE!r} 不认识，只许 orig / fix / fixdrop / drop18。")
SEG_SUFFIX = {"orig": "_bpfixorig", "fix": "_bpfix", "fixdrop": "_bpfixdrop", "drop18": "_bpfixdrop18"}[SEG_MODE]
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

BPFIX_DROP18 = {
    ("sub-08", 22), ("sub-13", 25),
    ("sub-14", 21), ("sub-14", 35),
    ("sub-20", 21), ("sub-22", 24), ("sub-22", 28), ("sub-24", 39),
    ("sub-25", 20), ("sub-29", 6), ("sub-29", 9), ("sub-29", 16),
    ("sub-30", 37), ("sub-40", 21), ("sub-40", 23), ("sub-40", 27),
    ("sub-48", 18), ("sub-48", 26),
}  # 共 18 个 (被试, 原始试次号)：边界码2 试次（清单 = xuanwu-boundary-code2-trials.csv，2026-11-21 逐行核对后写死）



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
    drop_set = {"fixdrop": SEGFIX_DROP, "drop18": BPFIX_DROP18}.get(SEG_MODE)  # BPFIX: 新增 drop18 档
    if drop_set is not None:
        keep = [i for i, tno in enumerate(nums) if (subj, tno) not in drop_set]
        onsets = [onsets[i] for i in keep]
        labels = [labels[i] for i in keep]
        nums = [nums[i] for i in keep]
    return onsets, labels, nums


log(f"[BPFIX] SEG_MODE={SEG_MODE}（fix=sub-43 五段 +1002 / drop18=再整段剔除 18 个边界码2 试次 / "
    f"fixdrop=再剔 29、orig=分段原样（后两档本轮不用）；带功率下限=1e-20）；输出后缀={SEG_SUFFIX}")
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
    if not os.path.exists(cand):
        return None
    onsets, labels = [], []
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
    per_lin = np.maximum(spectra - ap_lin, 1e-20)    # 2026-11-21 下限修正：1e-12 在 V^2/Hz 尺度上约等于 PSD 中位量级、会把 periodic 残差大面积截平；改 1e-20 后仅拦截 <=0 的残差。频段功率下限在本 -bpfix 版亦已改 1e-20（segfix 版原注释称“实测从未触底”系无效计数、已作废，见文件头 (4)）    # periodic = 全谱 − 非周期

    feats = {}
    for name, spec in (("full", spectra), ("periodic", per_lin), ("aperiodic", ap_lin)):
        band_feats = []
        for lo, hi in BANDS.values():
            idx = (freqs >= lo) & (freqs < hi)
            bp = _trapz(spec[:, idx], freqs[idx], axis=1)
            band_feats.append(np.log10(np.maximum(bp, 1e-20)))    # BPFIX: 带功率下限 1e-12→1e-20（计数核查证实截平，见文件头 (4)）
        fmat = np.stack(band_feats, axis=1).reshape(n_trial, n_ch, len(BANDS))
        feats[name] = fmat.reshape(n_trial, n_ch * len(BANDS))

    chi = ap[:, 1].reshape(n_trial, n_ch)            # χ（指数）
    return feats, chi, float(np.nanmedian(r2))


# ---------------------------------------------------------------- 4. 试次级 LOSO
def _bal_acc(yte, pred):
    """BPFIX 新增：平衡准确率 = 左/右两类各自准确率的均值（代码标签 0=左、1=右，
    对应论文标签 1/2）。某类在该被试无试次时返回 NaN（调用处打印提示）。"""
    accs = []
    for c in (0, 1):
        m = yte == c
        if int(m.sum()) == 0:
            return float("nan")
        accs.append(float((pred[m] == c).mean()))
    return float(np.mean(accs))


def loso_decode(all_feats, all_labels, subj_ids):
    """all_feats: dict名 → list[每被试 (40, n_feat)]；返回 DataFrame(每折×三种输入准确率)。"""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    n_subj = len(all_labels)
    rows = []
    for te in range(n_subj):
        tr = [i for i in range(n_subj) if i != te]
        row = {"fold": te + 1, "test_subj": subj_ids[te]}
        for name in ("full", "periodic", "aperiodic"):
            Xtr = np.concatenate([all_feats[name][i] for i in tr])
            ytr = np.concatenate([all_labels[i] for i in tr])
            Xte, yte = all_feats[name][te], all_labels[te]
            sc = StandardScaler().fit(Xtr)          # 只 fit 训练集：防泄漏
            clf = LinearDiscriminantAnalysis().fit(sc.transform(Xtr), ytr)
            row[f"acc_{name}"] = float(clf.score(sc.transform(Xte), yte))
            pred = clf.predict(sc.transform(Xte))     # BPFIX: 同一已拟合分类器的预测，供平衡准确率（acc 行未动）
            row[f"bal_acc_{name}"] = _bal_acc(yte, pred)
            if np.isnan(row[f"bal_acc_{name}"]):
                log(f"  [BPFIX] {subj_ids[te]} {name}: 某类试次为 0，bal_acc 置 NaN")
        rows.append(row)
        log(f"  fold {te + 1}/{n_subj}（测 {subj_ids[te]}）: "
            f"full={row['acc_full']:.3f} periodic={row['acc_periodic']:.3f} "
            f"aperiodic={row['acc_aperiodic']:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5.  verdict
def verdict(df):
    m = df[[f"acc_{n}" for n in ("full", "periodic", "aperiodic")]].mean()
    s = df[[f"acc_{n}" for n in ("full", "periodic", "aperiodic")]].std(ddof=1)
    log("\n" + "=" * 64)
    log(f"探针结果（{len(df)} 被试、试次级 LOSO、LDA + 频段功率特征）")
    log("=" * 64)
    for n in ("full", "periodic", "aperiodic"):
        log(f"  {n:9s}: {m[f'acc_{n}'] * 100:5.1f}% ± {s[f'acc_{n}'] * 100:4.1f}%")
    log("  随机线: 50.0%（二分类）；作者 TWFB+DGFMDM 基线 72.21%（复杂方法，仅作天花板参考）")

    ap = m["acc_aperiodic"]
    log("\n方向判断（探针级启发式，最终由全量 47 被试 + 统计检验定）:")
    if ap >= 0.60:
        log("  → C1 方向值得深入：aperiodic 携带左右手解码信息（≥60%）")
    elif ap >= 0.53:
        log("  → 边界信号（53–60%）：等全量 47 被试 + Wilcoxon 检验再定 C1 还是 C2")
    else:
        log("  → C1 先验不支持（≈随机）：aperiodic 基本不携带试次级左右手信息；")
        log("    重点转向 C2（aperiodic 作个体化校准）/ C3（关联保底）")
    if m["acc_periodic"] >= m["acc_full"] - 0.02:
        log("  另：periodic ≈ full，信息主要在周期振荡（符合 ERD/ERS 预期）")
    elif m["acc_full"] > m["acc_periodic"] + 0.03:
        log("  另：full 明显 > periodic，aperiodic 可能贡献了互补信息（值得细看）")
    log("=" * 64)


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
    log(f"找到 {len(edfs)} 个 EDF，用其中 {len(probe_edfs)} 个做探针: "
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
    all_feats = {"full": [], "periodic": [], "aperiodic": []}
    all_labels, chi_means, r2s, subj_ids, failed = [], [], [], [], []
    for ef in probe_edfs:
        sid = os.path.basename(ef).split("_")[0]
        try:
            d = load_subject(ef)
        except Exception as e:
            failed.append((sid, str(e)))
            log(f"  [跳过] {sid}: {e}")
            continue
        feats, chi, r2 = extract_features(d["trials"], d["sfreq"])
        for k in all_feats:
            all_feats[k].append(feats[k])
        all_labels.append(d["labels"])
        chi_means.append(float(chi.mean()))
        r2s.append(r2)
        subj_ids.append(d["subj"])
        log(f"  {d['subj']}: specparam 拟合完成，χ 均值={chi.mean():.3f}，R² 中位数={r2:.3f}")
    if failed:
        log(f"\n注意：{len(failed)} 个被试未能解析、已跳过：{[s for s, _ in failed]}")
    if failed:
        sys.exit(f"[SEGFIX] {len(failed)} 个被试解析失败：{[s for s, _ in failed]}。"
                 "按死命令停下，不静默少跑一人；先解决问题再跑。")
    if len(subj_ids) != len(probe_edfs):
        sys.exit(f"[SEGFIX] 成功解析 {len(subj_ids)} 人 ≠ 计划 {len(probe_edfs)} 人，停下。")
    log(f"[SEGFIX] 分析集总试次 = {sum(len(x) for x in all_labels)}（orig/fix 应 1880；fixdrop 应 1851；drop18 应 1862）")
    if len(subj_ids) < 3:
        sys.exit(f"成功解析的被试只剩 {len(subj_ids)} 个，太少，先解决问题再跑。")

    log(f"\n特征提取完成，用时 {(time.time() - t0) / 60:.1f} 分钟；开始试次级 LOSO…")
    df = loso_decode(all_feats, all_labels, subj_ids)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"probe_results{SEG_SUFFIX}_{ts}.csv")  # SEGFIX: 输出改脚本目录（原版 getcwd 不合规）
    df.to_csv(out, index=False)
    log(f"\n结果已存: {out}")

    log("\nχ 速览（每被试全脑平均，供 Exp 4 参考）:")
    for sid, (c, r) in zip(subj_ids, zip(chi_means, r2s)):
        log(f"  {sid}: χ={c:.3f}, specparam R²={r:.3f}")

    if failed:
        fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"probe_failures{SEG_SUFFIX}_{ts}.txt")  # SEGFIX: 同上
        with open(fpath, "w", encoding="utf-8") as f:
            for sid, why in failed:
                f.write(f"{sid}: {why}\n")
        log("\n" + "=" * 64)
        log(f"失败被试明细（共 {len(failed)} 个，已存 {fpath}）:")
        for sid, why in failed:
            log(f"  {sid}: {why}")
        log("=" * 64)

    verdict(df)


if __name__ == "__main__":
    main()
