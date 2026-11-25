#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tianjin-ablation-window2s.py — WINDOW-ALIGNED RERUN (2 s Welch, JNE.md item 02) — 天津第二队列 aperiodic 消融：信号到底来自 χ（斜率）还是 offset（高度）
================================================================
背景：天津 Exp 2 已于 2026-11-20 跑完（n=26，被试级 LOSO，MI vs idle）：
三路均显著高于随机，其中 aperiodic-only 为 LDA 53.8%（p=0.0026）、
SVM 56.7%（p=4.8e-05）。但 aperiodic 特征是背景谱的频段功率，χ（斜率）
与 offset（高度）的信息搅在一起。试次级探针（Exp 2 的 trial_chi CSV）
显示 χ 在 MI/idle 间无差异（p=0.65）、offset 有差异（p=0.024）——
本脚本按宣武消融（exp2-ablation.py）的同一口径，在解码层面坐实归因：
    只给 χ     ：每试次 40 个数（40 导 × 每导的 χ）
    只给 offset：每试次 40 个数（40 导 × 每导的 offset）
    对照 ap    ：Exp 2 的 aperiodic 频段功率特征（160 维），同批被试同折，
                应复现 window2s 版 Exp 2 的 aperiodic 数（不再是 4 秒窗的 LDA 53.8% / SVM 56.7%），差 >1pp 先查管道。
两组各跑 LDA 与 RBF-SVM（固定 C=1.0，不调参），被试级 LOSO，与 Exp 2 同折。

判读：χ-only ≈ 随机、offset-only ≈ 对照 ap → 信息在宽带水平（offset），
     χ 本人不带信息，天津 aperiodic 信号归因关闭，与宣武消融同一结论；
     χ-only 也显著 → χ 以弱形式携带信息，结论改写为弱阳性。

前置条件（已完成）：tianjin-first-check.py 与 tianjin-exp2.py 均已于
2026-11-20 在真实数据上跑完（结构与计数见下方解析口径）。本脚本与
Exp 2 共用同一解析器与被试纳入线，被试集合应同为 26 人（sub-06 的
排除原因看任一脚本日志的文件段）；若本脚本日志里被试数 ≠ 26，
先停下把日志贴回，不要继续。

用法：
    python tianjin-ablation.py [解压根目录]
    # 不给目录时，依次试 ./BIDS/PEEG、./PEEG、脚本所在目录、当前目录
    # （VS Code 点运行的工作目录可能是 VS Code 安装目录，尽量带参数）
    # 只读 .set 文件；输出写到脚本所在目录：
    #   tianjin_ablation_window2s_subject_acc_YYYYMMDD_HHMMSS.csv   （交付 1）
    #   tianjin_ablation_window2s_trial_chi_YYYYMMDD_HHMMSS.csv     （交付 2，与 Exp 2 的
    #     同名文件应逐值一致——管道自检）
    # 屏幕打印解析日志 = 交付 3（每类试次数、剔除说明）。

与宣武管线照搬 / 改三处（PIPELINE-SOP.md §6 P41）：
  改：Welch 按秒对齐，nperseg=500 / noverlap=250（2 秒窗）；其余照搬 specparam 口径
        peak_width_limits=[1,8]、max_n_peaks=6、min_peak_height=0.05、
        aperiodic_mode="fixed"；三输入频段功率特征（delta/theta/alpha/beta
        × 全部通道，log10）；StandardScaler 只 fit 训练集；SVM 固定
        C=1.0、gamma='scale'，不做超参搜索；统计 Wilcoxon vs chance 0.5。
  改 1：任务是步态 MI vs idle（label：MI=1，idle=0），不是左 vs 右。
  改 2：试次窗 5 秒（1250 采样 @250 Hz），不是 4 秒；评估是被试级 LOSO
        （27 人，每折 26 训 1 测）；与宣武一样是被试级 LOSO（试次级预测），
        不是更严——JNE.md 03 已证伪“更严”说法，方法节须删此表述。
  改 3：χ 拟合区间统一为 [3, 35] Hz（P39）：天津官方预处理为 3–35 Hz
        带通，拟合 1–40 Hz 会拟合到滤波器滚降区，χ 不可比。注意这意味着
        与宣武的 χ 直接并排比较前，宣武侧也须用 3–35 Hz 重拟合一次
        （本脚本不管宣武侧，只在日志里提醒）。
        副作用：delta 频段（1–4 Hz）在拟合频域内只剩 3–4 Hz，特征里
        delta 列即 3–4 Hz 功率，脚本启动时打印此提醒。

解析口径（2026-11-20 核查 + 事件探针实测后定版，推翻开工包先验）：
  只用文件名 token == pre 的文件（基线纯 MI 范式）；post / follow 是
  训练后 / 随访时点的纯 MI，本研究不用；MI-IES / MI-SES 是电刺激范式，
  电刺激会污染非周期成分，一律不用。
  文件结构：每 run 一个文件，40 通道 × 约 50,020 采样（250 Hz），内含
  20 个约 10 秒试次拼接，每试次起点有一个 'boundary' 事件。试次内
  时间线（相对试次起点）：准备码 1/7 ≈0 秒 → 提示码 2/8 ≈3.0 秒 →
  3/9 ≈4.0 秒 → 4/10 ≈5.0 秒 → 5/11 ≈5.6 秒 → 6/12 ≈7.4 秒 →
  试次结束 ≈10 秒（MI 试次用码组 1–6，idle 试次用码组 7–12）。
  切段：MI 窗 = [码 3 的 latency, +1250 采样 = 5 秒)，idle 窗 =
  [码 9, +1250)；窗位于试次内部（约 4.0–9.0 秒），不跨 boundary，
  解析时逐窗校验（越界或窗内有 boundary 的窗丢弃并计数）。
  每 run 10 MI + 10 idle 试次；每被试 4–5 个 run → 80 或 100 试次
  （40/40 或 50/50）。被试纳入线：两类各 ≥40 且相差 ≤2，否则整人跳过
  并记入剔除日志。单文件异常（读不开、通道 / 采样率不对、无码 3/9）
  跳过该文件并报原因，不崩溃。
  存储：.set 字段平铺在 .mat 顶层（无 EEG 外层变量）、数据内嵌无 .fdt；
  解析器两种存法都兼容。若为 v7.3 HDF5，scipy 读不了，报错并提示
  改用 MNE/h5py（与核查脚本同一处理）。

输出口径：
  每被试准确率 CSV 列：subject, n_trials, n_mi, n_idle,
    lda_chi, lda_offset, lda_ap, svm_chi, svm_offset, svm_ap
    （ap = 对照，Exp 2 的 aperiodic 160 维频段功率特征）。
  每试次 χ/offset CSV 列与 Exp 2 版相同：subject, trial, label, chi,
    offset, chi_foot, offset_foot——chi/offset 为全脑 40 导均值，
    chi_foot/offset_foot 为足区簇 Cz/C1/C2/FCz/CPz 均值；数值应与
    Exp 2 版逐值一致。trial 为该被试内 pooled 试次序号（按文件名、段顺序）。
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

# ---------------------------------------------------------------- 配置
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
SPECPARAM_KW = dict(peak_width_limits=[1, 8], max_n_peaks=6,
                    min_peak_height=0.05, aperiodic_mode="fixed", verbose=False)
FREQ_RANGE = [3, 35]          # P39：与天津 3–35 Hz 预处理统一（宣武侧须另行重拟合）
N_FFT, N_OVERLAP = 500, 250  # 按秒对齐宣武：250 Hz 下 500 采样 = 2 秒窗，50% 重叠； 5 秒试次得 4 个窗（宣武 500 Hz/1000 采样/4 秒试次得 3 个窗）。 原 1000/500 在 250 Hz 下是 4 秒单窗，见 JNE.md 02
SEG_SEC = 5
MI_CODES = {3, 4, 5, 6}
IDLE_CODES = {9, 10, 11, 12}
FOOT_CLUSTER = ["Cz", "C1", "C2", "FCz", "CPz"]
FNAME_RE = re.compile(r"^sub-(\d+)_([A-Za-z\-]+)_run-(\d+)_eeg\.set$",
                      re.IGNORECASE)
RANDOM_SEED = 20261017


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- 0. 定位
def find_root(arg):
    if arg:
        return arg
    here = os.path.dirname(os.path.abspath(__file__))
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


# ---------------------------------------------------------------- 1. 读一个 .set 文件 → (trials, labels)
def parse_set_file(path):
    """返回 (trials (n_trial, n_ch, 1250), labels (n_trial,), ch_labels, srate)
    或 (None, reason)。

    切段规则（2026-11-20 探针实测后定版）：文件 = 20 个约 10 秒试次拼接，
    每试次起点有一个 'boundary' 事件；试次内时间线（相对试次起点）：
    准备码 1/7 ≈0 秒 → 提示码 2/8 ≈3.0 秒 → 3/9 ≈4.0 秒 → 4/10 ≈5.0 秒
    → 5/11 ≈5.6 秒 → 6/12 ≈7.4 秒 → 试次结束 ≈10 秒。
    MI 试次窗 = [码 3 的 latency, +1250 采样)；idle 试次窗 = [码 9, +1250)。
    窗位于试次内部（约 4.0–9.0 秒），不跨 boundary；逐窗校验：窗内不得有
    boundary 事件、窗不得越出数据范围，不合格的窗丢弃并计数。
    码 4/5/6 与 10/11/12 是试次内标记，不作窗起点，只计数核对。
    存储兼容两种形态：EEG 外层变量，或字段平铺在 .mat 顶层（天津实测）。"""
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

    # 通道名
    cl = get_field(eeg, "chanlocs")
    labels_ch = None
    if cl is not None:
        labels_ch = [as_str(get_field(c, "labels")) for c in np.atleast_1d(cl).flat]

    # 事件：数字码与 'boundary' 分开收集（latency 均为 EEGLAB 1-based 采样序号）
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
        start = int(lat) - 1  # 1-based latency → 0-based 下标
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


# ---------------------------------------------------------------- 2. 特征提取（照搬 exp23-robust-svm.py extract_features）
def extract_features(trials, sf):
    from specparam import SpectralGroupModel
    n_trial, n_ch, _ = trials.shape
    freqs, _ = spsig.welch(trials[0, 0], fs=sf, nperseg=N_FFT, noverlap=N_OVERLAP)
    fmask = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
    freqs = freqs[fmask]
    spectra = np.empty((n_trial * n_ch, len(freqs)))
    for ti in range(n_trial):
        for ci in range(n_ch):
            _, p = spsig.welch(trials[ti, ci], fs=sf, nperseg=N_FFT, noverlap=N_OVERLAP)
            spectra[ti * n_ch + ci] = p[fmask]
    fg = SpectralGroupModel(**SPECPARAM_KW)
    try:
        fg.fit(freqs, spectra, FREQ_RANGE)
    except TypeError:
        fg.fit(freqs, spectra, freq_range=FREQ_RANGE)
    try:
        ap = fg.get_params("aperiodic_params")
    except Exception:
        try:
            ap = fg.get_params("aperiodic")
        except Exception as e:
            raise RuntimeError(f"specparam 取 aperiodic 参数失败: {e}")
    try:
        r2 = np.asarray(fg.get_params("r_squared"), dtype=float)
    except Exception:
        try:
            r2 = np.asarray(fg.get_metrics("gof"), dtype=float)
        except Exception:
            r2 = np.full(len(spectra), np.nan)
    logf = np.log10(freqs)
    ap_log = ap[:, 0:1] - ap[:, 1:2] * logf
    ap_lin = 10.0 ** ap_log
    per_lin = np.maximum(spectra - ap_lin, 1e-12)
    feats = {}
    for name, spec in (("full", spectra), ("periodic", per_lin), ("aperiodic", ap_lin)):
        band_feats = []
        for lo, hi in BANDS.values():
            idx = (freqs >= lo) & (freqs < hi)
            bp = _trapz(spec[:, idx], freqs[idx], axis=1)
            band_feats.append(np.log10(np.maximum(bp, 1e-12)))
        fmat = np.stack(band_feats, axis=1).reshape(n_trial, n_ch, len(BANDS))
        feats[name] = fmat.reshape(n_trial, n_ch * len(BANDS))
    chi = ap[:, 1].reshape(n_trial, n_ch)
    offset = ap[:, 0].reshape(n_trial, n_ch)
    return feats, chi, offset, float(np.nanmedian(r2))


# ---------------------------------------------------------------- 3. 解码器（照搬 robust 脚本，StandardScaler 只 fit 训练集）
def _lda_acc(Xtr, ytr, Xte, yte):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LinearDiscriminantAnalysis().fit(sc.transform(Xtr), ytr)
    return float(clf.score(sc.transform(Xte), yte))


def _svm_acc(Xtr, ytr, Xte, yte):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    sc = StandardScaler().fit(Xtr)
    clf = SVC(kernel="rbf", C=1.0, gamma="scale").fit(sc.transform(Xtr), ytr)
    return float(clf.score(sc.transform(Xte), yte))


# ---------------------------------------------------------------- 4. 主流程
def main():
    t0 = time.time()
    root = find_root(sys.argv[1] if len(sys.argv) > 1 else None)
    log("天津消融（χ / offset 归因）：根目录 %s" % os.path.abspath(root))
    log("提醒：delta 频段在 [3, 35] Hz 拟合域内只剩 3–4 Hz；"
        "与宣武 χ 并排比较前，宣武侧须用同一 [3, 35] Hz 重拟合。")
    if not os.path.isdir(root):
        log("目录不存在：%s" % root)
        return 1
    set_files = []
    for dirpath, _dirs, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".set"):
                set_files.append(os.path.join(dirpath, f))
    set_files.sort()
    # 只取文件名 token == pre 的文件（基线纯 MI；post/follow/MI-IES/MI-SES 不用）
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
        log("  %s：%s，%d 段" % (os.path.basename(p), note, len(labs)))

    # 按被试 pooled：预期每人 80 或 100 试次（4–5 个 run × 每 run 10 MI + 10 idle）
    subj_ids, all_labels = [], []
    all_chi, all_offset, all_ap = [], [], []
    trial_rows = []
    skipped_subj = []
    foot_idx = None
    for subj in sorted(by_subj):
        files = sorted(by_subj[subj])
        trials = np.concatenate([t for _, t, _, _ in files])
        labs = np.concatenate([l for _, _, l, _ in files])
        ch_labels = files[0][3]
        n_mi, n_idle = int((labs == 1).sum()), int((labs == 0).sum())
        log(f"  {subj}：{len(files)} 个文件，{len(labs)} 试次（MI {n_mi} / idle {n_idle}）")
        if min(n_mi, n_idle) < 40 or abs(n_mi - n_idle) > 2:
            skipped_subj.append((subj, len(labs), n_mi, n_idle))
            log(f"    WARNING：{subj} 试次不足或不平衡（MI {n_mi} / idle {n_idle}），整人跳过")
            continue
        if foot_idx is None and ch_labels:
            missing = [c for c in FOOT_CLUSTER if c not in ch_labels]
            if missing:
                log(f"    WARNING：足区簇缺导联 {missing}（实测导联见首文件），"
                    "chi_foot 列将为空，chi 全脑口径不受影响")
                foot_idx = []
            else:
                foot_idx = [ch_labels.index(c) for c in FOOT_CLUSTER]
        feats, chi, offset, r2 = extract_features(trials, 250)
        log(f"    特征完成，χ 全脑均值={chi.mean():.3f}，specparam R² 中位数={r2:.3f}")
        subj_ids.append(subj)
        all_chi.append(chi)
        all_offset.append(offset)
        all_ap.append(feats["aperiodic"])
        all_labels.append(labs)
        chi_g = chi.mean(axis=1)
        off_g = offset.mean(axis=1)
        if foot_idx:
            chi_f = chi[:, foot_idx].mean(axis=1)
            off_f = offset[:, foot_idx].mean(axis=1)
        else:
            chi_f = np.full(len(labs), np.nan)
            off_f = np.full(len(labs), np.nan)
        for i in range(len(labs)):
            trial_rows.append(dict(subject=subj, trial=i + 1, label=int(labs[i]),
                                  chi=float(chi_g[i]), offset=float(off_g[i]),
                                  chi_foot=float(chi_f[i]),
                                  offset_foot=float(off_f[i])))
    if len(subj_ids) < 2:
        log("可用被试不足 2 人（%d），无法 LOSO。" % len(subj_ids))
        return 1
    log(f"\n可用被试 {len(subj_ids)} 人（预期 26，与 Exp 2 同批）；跳过文件 {len(skipped_files)} 个，"
        f"跳过被试 {len(skipped_subj)} 人：{skipped_subj if skipped_subj else '无'}")

    # 被试级 LOSO：每折 N-1 训 1 测（与 Exp 2 同折）
    feats3 = {"chi": all_chi, "offset": all_offset, "ap": all_ap,
              "chioff": [np.concatenate([c, o], axis=1) for c, o in zip(all_chi, all_offset)]}  # [χ,offset] 直接拼接（2026-11-21 补）
    rows = []
    for te in range(len(subj_ids)):
        tr = [i for i in range(len(subj_ids)) if i != te]
        ytr = np.concatenate([all_labels[i] for i in tr])
        yte = all_labels[te]
        row = {"subject": subj_ids[te], "n_trials": len(yte),
               "n_mi": int((yte == 1).sum()), "n_idle": int((yte == 0).sum())}
        for name in ("chi", "offset", "ap", "chioff"):
            Xtr = np.concatenate([feats3[name][i] for i in tr])
            row[f"lda_{name}"] = _lda_acc(Xtr, ytr, feats3[name][te], yte)
            row[f"svm_{name}"] = _svm_acc(Xtr, ytr, feats3[name][te], yte)
        rows.append(row)
        log(f"  fold {te + 1}/{len(subj_ids)}（测 {subj_ids[te]}）: "
            f"lda_chi={row['lda_chi']:.3f} lda_off={row['lda_offset']:.3f} lda_ap={row['lda_ap']:.3f} "
            f"svm_chi={row['svm_chi']:.3f} svm_off={row['svm_offset']:.3f} svm_ap={row['svm_ap']:.3f} "
            f"lda_chioff={row['lda_chioff']:.3f} svm_chioff={row['svm_chioff']:.3f}")
    df = pd.DataFrame(rows)
    # 列序：subject, n_trials, n_mi, n_idle,
    # lda_chi, lda_offset, lda_ap, svm_chi, svm_offset, svm_ap
    df = df[["subject", "n_trials", "n_mi", "n_idle",
             "lda_chi", "lda_offset", "lda_ap", "lda_chioff",
             "svm_chi", "svm_offset", "svm_ap", "svm_chioff"]]

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.dirname(os.path.abspath(__file__))
    acc_path = os.path.join(outdir, f"tianjin_ablation_window2s_subject_acc_{ts}.csv")
    chi_path = os.path.join(outdir, f"tianjin_ablation_window2s_trial_chi_{ts}.csv")
    df.to_csv(acc_path, index=False)
    pd.DataFrame(trial_rows).to_csv(chi_path, index=False)

    from scipy import stats
    log("\n" + "=" * 64)
    log(f"天津消融结果（{len(df)} 被试、被试级 LOSO，MI vs idle，chance=0.5）")
    log("=" * 64)
    for c in ("lda_chi", "lda_offset", "lda_ap", "lda_chioff",
              "svm_chi", "svm_offset", "svm_ap", "svm_chioff"):
        a = df[c].values
        p = stats.wilcoxon((a - 0.5)).pvalue if np.any(a != 0.5) else float("nan")
        log(f"  {c:14s} 均值={a.mean():.3f}  中位数={np.median(a):.3f}  "
            f"Wilcoxon vs 0.5 p={p:.4g}")
    log("\n关键配对（Wilcoxon）:")
    for a_, b_, lab in [("lda_offset", "lda_chi", "LDA offset − χ"),
                        ("svm_offset", "svm_chi", "SVM offset − χ"),
                        ("lda_ap", "lda_offset", "LDA 对照ap − offset"),
                        ("svm_ap", "svm_offset", "SVM 对照ap − offset"),
                        ("lda_ap", "lda_chi", "LDA 对照ap − χ"),
                        ("svm_ap", "svm_chi", "SVM 对照ap − χ"),
                        ("lda_chioff", "lda_chi", "LDA [χ,off]拼接 − χ"),
                        ("svm_chioff", "svm_chi", "SVM [χ,off]拼接 − χ"),
                        ("lda_chioff", "lda_ap", "LDA [χ,off]拼接 − 对照ap"),
                        ("svm_chioff", "svm_ap", "SVM [χ,off]拼接 − 对照ap")]:
        d = df[a_].values - df[b_].values
        p = stats.wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
        log(f"  {lab}: Δ均值={d.mean():+.3f}  p={p:.4g}")
    log("\n管道自检：对照 ap 应复现 Exp 2 的 lda_aperiodic 0.538 / "
        "svm_aperiodic 0.567（同 26 人同折），差 >0.01 先查管道再谈判读。")
    log("\n判读（按本批数字现算，不预设结论；是否显著看上面 Wilcoxon p 值）:")
    log("  χ-only ≈ 随机 且 offset-only ≈ 对照 ap（配对不显著）且 offset > χ"
        " → 天津 aperiodic 信号由 offset（宽带水平）携带、χ 不带信息，"
        "归因关闭，与宣武消融同一结论；")
    log("  χ-only 也显著高于随机 → χ 以弱形式携带信息，天津结论改写为弱阳性，"
        "讨论节如实改写，不得藏。")
    log(f"\n输出：\n  {acc_path}\n  {chi_path}")
    log(f"用时 {(time.time() - t0) / 60:.1f} 分钟；"
        "把两个 CSV + 上面日志贴回 Muse 做独立复算（SOP P34：新数字必走复算）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
