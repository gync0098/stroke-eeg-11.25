#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xuanwu-bpfloor-count.py — 频带功率第二道下限（log10(max(bp, 1e-12))）触底计数核查
================================================================================
背景（第六份外审 21pm 指控，我方旧结论存疑）：
  正式管线对三种表示（full / periodic / aperiodic）都执行
      band_feats = log10(max(bp, 1e-12))
  旧诊断 xuanwu-clip-check.py 只数了 periodic 一支的 bp<=1e-12，且那支的 bp 是
  对已经逐点截平（per_lin = max(resid, 1e-12)）的残差积分——积分值结构性地
  >= 1e-12 × 频段宽度 > 1e--12，计数注定为 0，证据无效；且只跑了 8/47 人。
  本脚本重数：计数发生在任何截平之前，逐被试统计三表示 × 4 频段的
  bp<=1e-12 与 bp<1e-12 比例及 bp 分位数，全 47 人、fix 分段口径。

口径：
  - 分段 = 主分析 fix 口径：正式规则起点 + sub-43 第 25/26/27/31/33 试次 +1002；
    不剔除任何试次（47 人 × 40 试次；每表示分母 = 47*40*29*4 = 218,080）。
  - periodic 用未截平的原始残差 spectra - ap_lin 积分；另附 periodic_fixfloor
    一支 = 残差先逐点 max(·, 1e-20)（现行 fixfloor 管线实际做法）再积分。
  - 积分 = np.trapezoid，与正式管线同一函数、同一频段定义（含上界排他）。
  - 只计数、不训练、不解码。

死命令合规：枚举剔"副本/copy"；原始 50 / 分析集 47 断言，对不上停；
输出 CSV 写脚本自身目录（__file__ 解析）。

用法：放进宣武数据文件夹（与正式脚本同层）直接运行；回传 CSV + 控制台总表。
冒烟：python xuanwu-bpfloor-count.py --smoke（合成数据验计数逻辑，不读数据）
"""

import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import signal as spsig

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
SPECPARAM_KW = dict(peak_width_limits=[1, 8], max_n_peaks=6,
                    min_peak_height=0.05, aperiodic_mode="fixed", verbose=False)
FREQ_RANGE = [1, 40]
N_FFT, N_OVERLAP = 1000, 500
FLOOR = 1e-12
DERIVED_SUBJECTS = {"sub-05", "sub-15", "sub-49"}
SEGFIX_SHIFT = {("sub-43", 25), ("sub-43", 26), ("sub-43", 27),
                ("sub-43", 31), ("sub-43", 33)}
EEG29 = ["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4",
         "FT7", "FT8", "Cz", "C3", "C4", "T3", "T4", "CP3", "CP4", "TP7",
         "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"]
REPRESENTATIONS = ("full", "periodic_raw", "periodic_fixfloor", "aperiodic")
QUANTILES = (1, 5, 25, 50, 75, 95, 99)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- 找 EDF
def find_edfs(data_dir):
    cands = sorted(glob.glob(os.path.join(data_dir, "edffile", "**", "*.edf"),
                             recursive=True))
    if not cands:
        cands = sorted(glob.glob(os.path.join(data_dir, "**", "*.edf"),
                                 recursive=True))
    # 死命令：先剔除文件名含"副本"/"copy"的拷贝，再取文件
    cands = [c for c in cands
             if "副本" not in os.path.basename(c)
             and "copy" not in os.path.basename(c).lower()]
    return cands


# ---------------------------------------------------------------- 试次识别
# 以下脉冲解析与正式管线（stepR 系列）逐字一致：想象起点 = 码2 本身，
# 其后须有码3 且间隔 1900-2100；码1 不用。
def _labels_from_events_tsv(edf_path, sf):
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
        return None, f"码2脉冲={n_c2}，通过配对的起点={len(onsets)}（需 40）"
    gaps = np.diff(onsets)
    if not np.all((gaps >= 2000) & (gaps <= 6000)):
        return None, f"起点间距异常: min={gaps.min()}, max={gaps.max()}"
    return onsets, n_c2


def _labels_from_sti(raw):
    sti = raw.get_data(picks=[raw.ch_names[-1]]).ravel() / 1e-6
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
    if onsets is None:
        return None, f"STI 配对解析失败: {extra}"
    labels = [i % 2 for i in range(40)]
    return (onsets, labels), "STI码2→码3配对"


def load_subject(edf_path):
    """fix 口径切段：正式规则起点 + sub-43 五段 +1002；返回 (40, 29, 2000) 与 sf。"""
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sf = float(raw.info["sfreq"])
    names = raw.ch_names
    subj = os.path.basename(edf_path).split("_")[0]
    eeg_names = [n for n in EEG29 if n in names]
    if len(eeg_names) != 29:
        raise RuntimeError(f"{subj}: 29 导白名单只对上 {len(eeg_names)} 导")
    got = _labels_from_events_tsv(edf_path, sf)
    if got is None:
        got, _ = _labels_from_sti(raw)
    if got is None:
        raise RuntimeError(f"{subj}: 识别不出 40 个试次")
    onsets, _labels = got
    onsets = list(onsets)
    for tno in range(1, 41):  # fix：sub-43 五段 +1002（试次号 = 下标 + 1）
        if (subj, tno) in SEGFIX_SHIFT:
            onsets[tno - 1] = int(onsets[tno - 1]) + 1002
    n_samp = int(4 * sf)
    data = raw.get_data(picks=eeg_names)
    trials = np.stack([data[:, s:s + n_samp] for s in onsets])
    assert trials.shape == (40, 29, n_samp), f"{subj}: {trials.shape}"
    return trials, sf, subj


# ---------------------------------------------------------------- 计数核心
def band_powers(spectra, ap_lin, freqs):
    """返回 dict 表示 -> bp 数组 (n_spec, 4)。计数前不做任何截平。
    periodic_raw 用原始残差（可为负，trapz 照常积分）；
    periodic_fixfloor 用逐点 max(resid, 1e-20) 后的残差（现行管线做法）。"""
    resid = spectra - ap_lin
    specs = {"full": spectra,
             "periodic_raw": resid,
             "periodic_fixfloor": np.maximum(resid, 1e-20),
             "aperiodic": ap_lin}
    out = {}
    for name, spec in specs.items():
        cols = []
        for lo, hi in BANDS.values():
            idx = (freqs >= lo) & (freqs < hi)
            cols.append(_trapz(spec[:, idx], freqs[idx], axis=1))
        out[name] = np.stack(cols, axis=1)
    return out


def summarize_bp(bp, subj, rep, band):
    """bp: 一维数组 -> 一行统计（计数 + 分位数）。"""
    row = {"subj": subj, "representation": rep, "band": band, "n": int(bp.size),
           "n_le_floor": int((bp <= FLOOR).sum()),
           "n_lt_floor": int((bp < FLOOR).sum())}
    row["pct_le_floor"] = 100.0 * row["n_le_floor"] / max(bp.size, 1)
    for q in QUANTILES:
        row[f"bp_p{q:02d}"] = float(np.percentile(bp, q)) if bp.size else np.nan
    row["bp_min"] = float(bp.min()) if bp.size else np.nan
    return row


def extract_bp(trials, sf):
    """与正式管线同一特征路径（Welch + specparam），只多算未截平的 bp。"""
    from specparam import SpectralGroupModel
    n_trial, n_ch, _ = trials.shape
    freqs, _ = spsig.welch(trials[0, 0], fs=sf, nperseg=N_FFT, noverlap=N_OVERLAP)
    fmask = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
    freqs = freqs[fmask]
    spectra = np.empty((n_trial * n_ch, len(freqs)))
    for ti in range(n_trial):
        for ci in range(n_ch):
            _, p = spsig.welch(trials[ti, ci], fs=sf,
                               nperseg=N_FFT, noverlap=N_OVERLAP)
            spectra[ti * n_ch + ci] = p[fmask]
    fg = SpectralGroupModel(**SPECPARAM_KW)
    try:
        fg.fit(freqs, spectra, FREQ_RANGE)
    except TypeError:
        fg.fit(freqs, spectra, freq_range=FREQ_RANGE)
    try:
        ap = fg.get_params("aperiodic_params")
    except Exception:
        ap = fg.get_params("aperiodic")
    logf = np.log10(freqs)
    ap_lin = 10.0 ** (ap[:, 0:1] - ap[:, 1:2] * logf)
    return band_powers(spectra, ap_lin, freqs)


# ---------------------------------------------------------------- 冒烟
def smoke():
    freqs = np.arange(1.0, 40.5, 0.5)
    rng = np.random.default_rng(0)
    spectra = np.full((4, len(freqs)), 4e-13)          # bp 量级 ~1e-12 附近
    spectra[0] = 1e-14                                  # 全频段远低于下限
    spectra[1] = 1e-11                                  # 全频段远高于下限
    ap_lin = spectra * 0.5
    bp = band_powers(spectra, ap_lin, freqs)
    assert set(bp) == set(REPRESENTATIONS)
    r = summarize_bp(bp["full"][:, 0], "synthetic", "full", "delta")
    # 第 0 行谱恒 1e-14：delta bp = 1e-14 * 3Hz = 3e-14 < 1e-12 必触底；
    # 第 1 行 1e-11：bp = 3e-11 > 1e-12 必不触底
    assert r["n"] == 4 and 1 <= r["n_le_floor"] <= 3, r
    r2 = summarize_bp(bp["aperiodic"][:, 3], "synthetic", "aperiodic", "beta")
    assert r2["n"] == 4
    # 常量谱下 full 的 delta bp 应恰为 4e-13 * 2.5 = 1.0e-12（第 2/3 行；
    # delta 频点为 1.0..3.5 共 6 点，trapz 跨度 2.5 Hz）
    assert abs(bp["full"][2, 0] - 1.0e-12) < 1e-24, bp["full"][2, 0]
    print("SMOKE OK: band_powers / summarize_bp 计数与解析值一致")
    return 0


# ---------------------------------------------------------------- 主程序
def main():
    if "--smoke" in sys.argv:
        return smoke()
    t0 = time.time()
    edfs = find_edfs(SCRIPT_DIR)
    by_subj = {}
    for ef in edfs:
        by_subj.setdefault(os.path.basename(ef).split("_")[0], ef)
    # 死命令：被试数核对（原始 50），对不上停
    assert len(by_subj) == 50, f"原始被试数 {len(by_subj)} != 50，停。"
    kept = {s: p for s, p in sorted(by_subj.items()) if s not in DERIVED_SUBJECTS}
    assert len(kept) == 47, f"分析集被试数 {len(kept)} != 47，停。"
    log(f"[BPFLOOR] 原始 50 / 分析集 47 核对通过；fix 分段口径；只计数不训练")

    rows, pooled = [], {rep: {b: [] for b in BANDS} for rep in REPRESENTATIONS}
    for subj, ef in kept.items():
        trials, sf, _ = load_subject(ef)
        bp = extract_bp(trials, sf)
        for rep in REPRESENTATIONS:
            for bi, band in enumerate(BANDS):
                vals = bp[rep][:, bi]
                rows.append(summarize_bp(vals, subj, rep, band))
                pooled[rep][band].append(vals)
        log(f"  {subj}: 已计入（{time.time() - t0:.0f}s）")

    for rep in REPRESENTATIONS:
        for band in BANDS:
            rows.append(summarize_bp(np.concatenate(pooled[rep][band]),
                                     "ALL47", rep, band))
    df = pd.DataFrame(rows)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(SCRIPT_DIR, f"bpfloor_count_{ts}.csv")
    df.to_csv(out, index=False)

    log("\n================ 全队列汇总（ALL47，每格分母 = 47*40*29 = 54,520）"
        "================")
    allr = df[df.subj == "ALL47"]
    for rep in REPRESENTATIONS:
        for band in BANDS:
            r = allr[(allr.representation == rep) & (allr.band == band)].iloc[0]
            log(f"  {rep:18s} {band:6s} bp<=1e-12: {int(r.n_le_floor):6d} / "
                f"{int(r.n)} = {r.pct_le_floor:6.2f}%   "
                f"bp 中位 {r.bp_p50:.3e} | p05 {r.bp_p05:.3e} | p95 {r.bp_p95:.3e}")
    tot = allr.groupby("representation")[["n", "n_le_floor"]].sum()
    log("---------------- 每表示四频段合计（分母 218,080）----------------")
    for rep in REPRESENTATIONS:
        log(f"  {rep:18s} {int(tot.loc[rep, 'n_le_floor']):6d} / "
            f"{int(tot.loc[rep, 'n'])} = "
            f"{100.0 * tot.loc[rep, 'n_le_floor'] / tot.loc[rep, 'n']:.2f}%")
    n_subj_touch = (df[(df.subj != "ALL47")]
                    .groupby(["representation", "subj"])["n_le_floor"]
                    .sum().gt(0).groupby("representation").sum())
    log("---------------- 至少触底一次的被试数（/47）----------------")
    for rep in REPRESENTATIONS:
        log(f"  {rep:18s} {int(n_subj_touch.get(rep, 0))} / 47")
    log(f"\nCSV 已保存：{out}\n完成。把 CSV + 本控制台总表贴回 Muse。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
