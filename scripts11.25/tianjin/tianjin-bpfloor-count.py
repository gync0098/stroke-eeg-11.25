#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tianjin-bpfloor-count.py -- 天津带功率第二道下限触底计数核查
================================================================
背景（2026-11-24 Orin 校对指出，Muse 独立复算确认）：
  天津主解码脚本 tianjin-exp2-window2s-fixfloor.py 只在 PSD 残差处改了
  1e-20（第 270 行），积分后的带功率仍用 log10(max(bp, 1e-12))
  （第 277 行），行尾注释还写着“实测从未触底”——但天津从来没有数过
  触底率。宣武的同一句注释已被 2026-11-21 的计数证伪过一次。
  本脚本补这一项计数，用它决定天津是否需要重跑：
  计数为 0（或可忽略）-> 天津结果不动，只改 Methods 的下限措辞；
  计数不可忽略 -> 天津按 1e-20 重跑全部谱依赖结果。

口径（与天津正式管线逐字一致，只计数、不训练、不解码）：
  - 只用文件名 token == pre 的文件（基线纯 MI；post/follow/MI-IES/MI-SES 不用）。
  - 切段 = 正式规则：MI 窗 [码 3, +1250)，idle 窗 [码 9, +1250)，
    窗内不得有 boundary、不得越界；每 run 10 MI + 10 idle。
  - 被试纳入线 = 正式规则：两类各 >= 40 且相差 <= 2，否则整人跳过。
    预期：原始 27 人 / 分析集 26 人（sub-06 因 run-3 缺 1 试次被排除）。
  - Welch nperseg=500 / noverlap=250（2 秒窗），拟合区间 [3, 35] Hz，
    specparam 口径与正式管线相同；delta 在拟合域内只剩 3-4 Hz。
  - 计数发生在任何截平之前，逐被试统计四表示 x 四频段的
    bp <= 1e-12 与 bp < 1e-12 比例及 bp 分位数。
    periodic_raw 用未截平的原始残差积分（可为负）；
    periodic_fixfloor 用逐点 max(resid, 1e-20) 后的残差（现行管线做法）。
  - 积分 = np.trapezoid，与正式管线同一函数、同一频段定义（含上界排他）。
  - 另报 PSD 中位数量级：下限是否安全，首先看它与数据尺度的距离。

死命令合规：枚举剔“副本/copy”；打印并核对被试数（原始 27 / 分析集 26），
对不上停；输出 CSV 写脚本自身目录（__file__ 解析）。日志含中文，直接在
VS Code 终端运行（与宣武计数脚本同款做法）；不要经无 GBK 防护的 runner
管道捕获输出——非 UTF-8 管道下中文日志会编码失败。

用法：把本脚本放进天津解压根目录的上一层、或直接带数据目录参数：
    python tianjin-bpfloor-count.py [解压根目录]
    # 不给目录时，依次试 ./BIDS/PEEG、./PEEG、脚本所在目录、当前目录
    # 只读 .set 文件；输出写到脚本所在目录：
    #   tianjin_bpfloor_count_YYYYMMDD_HHMMSS.csv
    # 回传 CSV + 控制台总表。
冒烟：python tianjin-bpfloor-count.py --smoke（合成数据验计数逻辑，不读数据）
"""

import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import signal as spsig
from scipy.io import loadmat

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
SPECPARAM_KW = dict(peak_width_limits=[1, 8], max_n_peaks=6,
                    min_peak_height=0.05, aperiodic_mode="fixed", verbose=False)
FREQ_RANGE = [3, 35]
N_FFT, N_OVERLAP = 500, 250
SEG_SEC = 5
FLOOR = 1e-12
FOOT_CLUSTER = ["Cz", "C1", "C2", "FCz", "CPz"]
FNAME_RE = re.compile(r"^sub-(\d+)_([A-Za-z\-]+)_run-(\d+)_eeg\.set$",
                      re.IGNORECASE)
REPRESENTATIONS = ("full", "periodic_raw", "periodic_fixfloor", "aperiodic")
QUANTILES = (1, 5, 25, 50, 75, 95, 99)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- 0. 定位
def find_root(arg):
    if arg:
        return arg
    here = SCRIPT_DIR
    for cand in ("./BIDS/PEEG", "./PEEG", here, "."):
        if os.path.isdir(cand):
            return cand
    return "."


def as_str(x):
    if isinstance(x, np.ndarray):
        x = x.flat[0] if x.size else ""
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    return str(x)


def get_field(eeg, name):
    if isinstance(eeg, dict):
        return eeg.get(name)
    if isinstance(eeg, np.ndarray):
        eeg = eeg.flat[0] if eeg.size else None
    if eeg is None:
        return None
    if hasattr(eeg, name):
        return getattr(eeg, name)
    try:
        return eeg[name]
    except Exception:
        return None


# ---------------------------------------------------------------- 1. 读一个 .set 文件
# 与 tianjin-exp2-window2s-fixfloor.py 的 parse_set_file 逐字一致
# （切段规则见该脚本 docstring：2026-11-20 探针实测后定版）。
def parse_set_file(path):
    """返回 (trials (n_trial, n_ch, 1250), labels, ch_labels, srate), note
    或 (None, reason)。"""
    try:
        mat = loadmat(path, struct_as_record=False, squeeze_me=True)
    except Exception as exc:
        return None, "loadmat 打不开（疑似 v7.3 HDF5）：%s" % exc
    eeg = mat.get("EEG")
    if eeg is None and "data" in mat:
        eeg = mat  # 字段平铺存法（天津实测）：整个 mat 字典就是 EEG
    if eeg is None:
        return None, "文件里没有 EEG 变量、顶层也没有 data"
    data = get_field(eeg, "data")
    if isinstance(data, str):
        return None, "EEG.data 是 .fdt 文件引用（%s），需同目录 .fdt" % data
    if data is None:
        return None, "EEG.data 缺失"
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        return None, "EEG.data 不是二维（shape=%s）" % (data.shape,)
    srate = get_field(eeg, "srate")
    srate = float(np.asarray(srate).flat[0]) if srate is not None else None
    if srate is None or int(srate) != 250:
        return None, "采样率不是 250 Hz（实测 %s）" % srate
    n_ch, n_times = data.shape
    if n_ch != 40:
        return None, "通道数不是 40（实测 %d）" % n_ch
    seg_len = int(SEG_SEC * srate)  # 1250

    cl = get_field(eeg, "chanlocs")
    labels_ch = None
    if cl is not None:
        labels_ch = [as_str(get_field(c, "labels")) for c in np.atleast_1d(cl).flat]

    ev = get_field(eeg, "event")
    if ev is None:
        return None, "EEG.event 缺失，无法定段"
    coded, boundaries = [], []
    for e in np.atleast_1d(ev).flat:
        t = get_field(e, "type")
        lat = get_field(e, "latency")
        if t is None or lat is None:
            continue
        lat = float(np.asarray(lat).flat[0])
        ts = as_str(t)
        if ts == "boundary":
            boundaries.append(lat)
            continue
        try:
            coded.append((lat, int(float(ts))))
        except ValueError:
            continue
    coded.sort()
    codes = Counter(c for _, c in coded)
    if codes.get(3, 0) == 0 or codes.get(9, 0) == 0:
        return None, "码 3 / 码 9 缺失（计数 %s），无法定 MI/idle 试次" % dict(codes)
    trials, labs, dropped = [], [], 0
    for lat, code in coded:
        if code not in (3, 9):
            continue
        start = int(lat) - 1  # 1-based latency -> 0-based 下标
        if start < 0 or start + seg_len > n_times:
            dropped += 1
            continue
        if any(lat < b < lat + seg_len for b in boundaries):
            dropped += 1
            continue
        trials.append(data[:, start:start + seg_len])
        labs.append(1 if code == 3 else 0)
    if not trials:
        return None, "所有试次窗均不合格（越界或跨 boundary）"
    labs = np.array(labs, dtype=int)
    n_mi, n_idle = int((labs == 1).sum()), int((labs == 0).sum())
    note = "ok (%d MI + %d idle 试次" % (n_mi, n_idle)
    marks = (codes.get(4, 0), codes.get(5, 0), codes.get(6, 0),
             codes.get(10, 0), codes.get(11, 0), codes.get(12, 0))
    if marks != (codes.get(3, 0),) * 3 + (codes.get(9, 0),) * 3:
        note += "；注意：试次内标记码计数 %s 与码 3/9 计数不齐" % (marks,)
    if dropped:
        note += "；丢弃不合格窗 %d 个" % dropped
    note += ")"
    return (np.stack(trials), labs, labels_ch, srate), note


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
    """与正式管线同一特征路径（Welch + specparam），只多算未截平的 bp。
    返回 (band_powers dict, spectra 中位数)。"""
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
    return band_powers(spectra, ap_lin, freqs), float(np.median(spectra))


# ---------------------------------------------------------------- 冒烟
def smoke():
    # 天津拟合域 [3, 35] Hz：delta 频点为 3.0 / 3.5，trapz 跨度 0.5 Hz。
    freqs = np.arange(3.0, 35.5, 0.5)
    spectra = np.full((4, len(freqs)), 2e-12)   # delta bp 恰为 1e-12（触底边界）
    spectra[0] = 1e-14                           # 全频段远低于下限
    spectra[1] = 1e-11                           # 全频段远高于下限
    ap_lin = spectra * 0.5
    ap_lin[3] = 3e-12                            # 第 3 行残差为负（周期残差可为负）
    bp = band_powers(spectra, ap_lin, freqs)
    assert set(bp) == set(REPRESENTATIONS)
    r = summarize_bp(bp["full"][:, 0], "synthetic", "full", "delta")
    # full delta bp：5e-15 / 5e-12 / 1e-12 / 1e-12 -> 触底 3 个
    assert r["n"] == 4 and r["n_le_floor"] == 3, r
    assert abs(bp["full"][2, 0] - 1e-12) < 1e-24, bp["full"][2, 0]
    r2 = summarize_bp(bp["periodic_raw"][:, 0], "synthetic",
                      "periodic_raw", "delta")
    # periodic_raw delta bp：2.5e-15 / 2.5e-12 / 5e-13 / -5e-13 -> 触底 3 个
    assert r2["n_le_floor"] == 3, r2
    r3 = summarize_bp(bp["aperiodic"][:, 3], "synthetic", "aperiodic", "beta")
    assert r3["n"] == 4
    print("SMOKE OK: band_powers / summarize_bp 计数与解析值一致")
    return 0


# ---------------------------------------------------------------- 主程序
def main():
    if "--smoke" in sys.argv:
        return smoke()
    t0 = time.time()
    root = find_root(sys.argv[1] if len(sys.argv) > 1 else None)
    log("天津带功率下限触底计数：根目录 %s" % os.path.abspath(root))
    log("提醒：delta 频段在 [3, 35] Hz 拟合域内只剩 3-4 Hz；只计数不训练。")
    if not os.path.isdir(root):
        log("目录不存在：%s" % root)
        return 1
    set_files = []
    for dirpath, _dirs, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".set"):
                set_files.append(os.path.join(dirpath, f))
    set_files.sort()
    # 死命令：先剔除文件名含“副本”/“copy”的拷贝
    set_files = [p for p in set_files
                 if "副本" not in os.path.basename(p)
                 and "copy" not in os.path.basename(p).lower()]
    pre_files = []
    for p in set_files:
        m = FNAME_RE.match(os.path.basename(p))
        if m and m.group(2).lower() == "pre":
            pre_files.append(p)
    log("共 .set 文件 %d 个；其中 pre（基线纯 MI）%d 个（其余 token 一律不用）"
        % (len(set_files), len(pre_files)))
    if not pre_files:
        log("没找到 pre 文件——先跑 tianjin-first-check.py 看实际目录结构。")
        return 1

    by_subj = defaultdict(list)
    skipped_files = []
    for p in pre_files:
        m = FNAME_RE.match(os.path.basename(p))
        out, note = parse_set_file(p)
        if out is None:
            skipped_files.append((p, note))
            log("  跳过文件 %s：%s" % (os.path.basename(p), note))
            continue
        trials, labs, ch_labels, _sf = out
        by_subj["sub-%s" % m.group(1)].append((p, trials, labs, ch_labels))

    # 死命令：被试数核对（原始 27），对不上停
    assert len(by_subj) == 27, "原始被试数 %d != 27，停。" % len(by_subj)
    log("[BPFLOOR] 原始 27 人核对通过；按正式纳入线筛分析集")

    rows, pooled = [], {rep: {b: [] for b in BANDS} for rep in REPRESENTATIONS}
    psd_medians, kept, skipped_subj = [], [], []
    for subj in sorted(by_subj):
        files = sorted(by_subj[subj])
        trials = np.concatenate([t for _, t, _, _ in files])
        labs = np.concatenate([l for _, _, l, _ in files])
        n_mi, n_idle = int((labs == 1).sum()), int((labs == 0).sum())
        if min(n_mi, n_idle) < 40 or abs(n_mi - n_idle) > 2:
            skipped_subj.append((subj, len(labs), n_mi, n_idle))
            log("  %s：试次不足或不平衡（MI %d / idle %d），整人跳过"
                % (subj, n_mi, n_idle))
            continue
        bp, psd_med = extract_bp(trials, 250)
        psd_medians.append(psd_med)
        kept.append(subj)
        for rep in REPRESENTATIONS:
            for bi, band in enumerate(BANDS):
                vals = bp[rep][:, bi]
                rows.append(summarize_bp(vals, subj, rep, band))
                pooled[rep][band].append(vals)
        log("  %s：已计入（%d 试次，%d s）" % (subj, len(labs), time.time() - t0))

    # 死命令：分析集被试数核对（预期 26，sub-06 被排除），对不上停
    assert len(kept) == 26, "分析集被试数 %d != 26，停。跳过：%s" % (
        len(kept), skipped_subj)
    log("[BPFLOOR] 分析集 26 人核对通过；跳过文件 %d 个，跳过被试 %s"
        % (len(skipped_files), skipped_subj if skipped_subj else "无"))

    n_trials_total = sum(len(np.concatenate(
        [l for _, _, l, _ in sorted(by_subj[s])])) for s in kept)
    for rep in REPRESENTATIONS:
        for band in BANDS:
            rows.append(summarize_bp(np.concatenate(pooled[rep][band]),
                                     "ALL26", rep, band))
    df = pd.DataFrame(rows)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(SCRIPT_DIR, f"tianjin_bpfloor_count_{ts}.csv")
    df.to_csv(out, index=False)

    denom = n_trials_total * 40
    log("\n================ 全队列汇总（ALL26，每格分母 = 试次数 x 40 导）================")
    allr = df[df.subj == "ALL26"]
    for rep in REPRESENTATIONS:
        for band in BANDS:
            r = allr[(allr.representation == rep) & (allr.band == band)].iloc[0]
            log("  %s %s bp<=1e-12: %d / %d = %.2f%%   "
                "bp 中位 %.3e | p05 %.3e | p95 %.3e"
                % (rep, band, int(r.n_le_floor), int(r.n),
                   r.pct_le_floor, r.bp_p50, r.bp_p05, r.bp_p95))
    tot = allr.groupby("representation")[["n", "n_le_floor"]].sum()
    log("---------------- 每表示四频段合计（分母 %d）----------------" % int(tot["n"].iloc[0]))
    for rep in REPRESENTATIONS:
        log("  %s %d / %d = %.2f%%"
            % (rep, int(tot.loc[rep, "n_le_floor"]), int(tot.loc[rep, "n"]),
               100.0 * tot.loc[rep, "n_le_floor"] / tot.loc[rep, "n"]))
    n_subj_touch = (df[(df.subj != "ALL26")]
                    .groupby(["representation", "subj"])["n_le_floor"]
                    .sum().gt(0).groupby("representation").sum())
    log("---------------- 至少触底一次的被试数（/26）----------------")
    for rep in REPRESENTATIONS:
        log("  %s %d / 26" % (rep, int(n_subj_touch.get(rep, 0))))
    log("PSD 中位数（全部试次 x 通道谱的中位）：%.3e（下限 1e-12 与它的比值约 %.1e）"
        % (float(np.median(psd_medians)),
           1e-12 / float(np.median(psd_medians))))
    log("\n判读：触底合计为 0 -> 天津结果不动，只改 Methods 下限措辞为仅限宣武；"
        "\n触底不可忽略 -> 天津按 1e-20 重跑全部谱依赖结果。")
    log("\nCSV 已保存：%s\n完成。把 CSV + 本控制台总表贴回 Muse。" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
