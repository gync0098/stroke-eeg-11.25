# -*- coding: utf-8 -*-
"""
xuanwu-riemann-ts.py
标准 BCI 基准补强（2026-11-21 外审建议采纳项）：黎曼切空间 + 逻辑回归（附 TS+LDA）。
与主分析完全同折：宣武 47 人（剔除 sub-05/15/49）、被试级 LOSO、想象段 4 秒、
29 导原始试次 → 每试次协方差矩阵 → 以训练折协方差的 log-Euclidean 均值为参考点
做切空间映射（上三角向量化、非对角 ×√2）→ StandardScaler（仅训练集 fit）→ 分类。
标签约定已于 2026-11-21 用 .mat 真值逐人逐试次核实（50/50 一致）。
硬规则（2026-11-21 死命令）：枚举 EDF 先剔"副本/copy"；找到 50 人、分析 47 人，
数目对不上或任一被试解析失败 → 报错停下，不静默少跑；CSV 写脚本所在目录。
用法：放进宣武数据文件夹（与 EDF 同层），直接运行，输出贴回 Muse。
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
# 本文件是 xuanwu-riemann-ts.py 的补丁副本，原文件未动。补丁只做三件事：
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



# ---------------------------------------------------------------- 3. 协方差与切空间
def trial_covariances(trials):
    """trials: (n, 29, T) → 每试次样本协方差 (n, 29, 29)，轻收缩保正定。"""
    n, d, T = trials.shape
    X = trials - trials.mean(axis=2, keepdims=True)
    C = np.einsum("nit,njt->nij", X, X) / (T - 1)
    tr = np.trace(C, axis1=1, axis2=2) / d
    C = C + (1e-6 * tr)[:, None, None] * np.eye(d)[None]
    return C


def _sym_logm(A):
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 1e-12, None)
    return (V * np.log(w)) @ V.T


def _sym_expm(A):
    w, V = np.linalg.eigh(A)
    return (V * np.exp(w)) @ V.T


def _sym_inv_sqrt(A):
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 1e-12, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def log_euclidean_mean(covs):
    return _sym_expm(np.mean([_sym_logm(C) for C in covs], axis=0))


def tangent_vectors(covs, ref):
    W = _sym_inv_sqrt(ref)
    d = ref.shape[0]
    iu = np.triu_indices(d)
    offdiag = iu[0] != iu[1]
    out = []
    for C in covs:
        S = _sym_logm(W @ C @ W)
        v = S[iu]
        v[offdiag] *= np.sqrt(2.0)
        out.append(v)
    return np.array(out)


# ---------------------------------------------------------------- 4. 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_dir:
        search_dirs = [args.data_dir]
    else:
        search_dirs = [script_dir, os.getcwd(), os.path.join(os.getcwd(), "stroke-eeg")]
    edfs = []
    for d in search_dirs:
        edfs = find_edfs(d)
        if edfs:
            args.data_dir = d
            break
    if not edfs:
        sys.exit("没找到 .edf。把脚本放到数据文件夹里（和 sub-01 等同层）再跑。")
    log(f"数据目录: {args.data_dir}")
    kept, skipped = [], []
    for ef in edfs:
        sid = os.path.basename(ef).split("_")[0]
        (skipped if sid in DERIVED_SUBJECTS else kept).append(ef)
    found_ids = sorted({os.path.basename(e).split("_")[0] for e in edfs})
    log(f"找到 EDF 被试 {len(found_ids)} 人；剔除重复 {sorted(DERIVED_SUBJECTS & set(found_ids))}；"
        f"分析集 {len(kept)} 人")
    # 死命令：数目对不上就停（2026-11-21）
    if len(found_ids) != 50 or len(kept) != 47:
        sys.exit(f"被试数异常：找到 {len(found_ids)}（预期 50）、分析集 {len(kept)}（预期 47）。"
                 "先查数据目录再跑，不静默少跑。")

    t0 = time.time()
    all_covs, all_labels, subj_ids, failed = [], [], [], []
    for ef in kept:
        sid = os.path.basename(ef).split("_")[0]
        try:
            d = load_subject(ef)
        except Exception as e:
            failed.append((sid, str(e)))
            log(f"  [失败] {sid}: {e}")
            continue
        all_covs.append(trial_covariances(d["trials"]))
        all_labels.append(d["labels"])
        subj_ids.append(d["subj"])
    if failed:
        fpath = os.path.join(script_dir, f"riemann_failures{SEG_SUFFIX}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        with open(fpath, "w", encoding="utf-8") as fh:
            for s, r in failed:
                fh.write(f"{s}: {r}\n")
        sys.exit(f"{len(failed)} 个被试解析失败：{[s for s, _ in failed]}（明细 {fpath}）。"
                 "按死命令停下，不带缺口继续跑。")
    if len(subj_ids) != 47:
        sys.exit(f"成功解析 {len(subj_ids)} 人 ≠ 47，停下。")
    log(f"[SEGFIX] 分析集总试次 = {sum(len(x) for x in all_labels)}（orig/fix 应 1880；fixdrop 应 1851）")
    log(f"\n协方差提取完成（47 人），用时 {(time.time() - t0) / 60:.1f} 分钟；开始 LOSO…")

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rows = []
    for te in range(len(subj_ids)):
        tr = [i for i in range(len(subj_ids)) if i != te]
        Ctr = np.concatenate([all_covs[i] for i in tr])
        ytr = np.concatenate([all_labels[i] for i in tr])
        ref = log_euclidean_mean(Ctr)
        Xtr = tangent_vectors(Ctr, ref)
        Xte = tangent_vectors(all_covs[te], ref)
        yte = all_labels[te]
        sc = StandardScaler().fit(Xtr)
        row = {"fold": te + 1, "test_subj": subj_ids[te]}
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
        row["ts_lr"] = float((lr.predict(sc.transform(Xte)) == yte).mean())
        lda = LinearDiscriminantAnalysis().fit(sc.transform(Xtr), ytr)
        row["ts_lda"] = float((lda.predict(sc.transform(Xte)) == yte).mean())
        rows.append(row)
        log(f"  fold {te + 1}/47（测 {subj_ids[te]}）: ts_lr={row['ts_lr']:.3f} ts_lda={row['ts_lda']:.3f}")
    df = pd.DataFrame(rows)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(script_dir, f"riemann_results{SEG_SUFFIX}_{ts}.csv")
    df.to_csv(out, index=False)

    from scipy import stats
    log("\n" + "=" * 64)
    log("黎曼切空间基准（47 被试、被试级 LOSO，chance=0.5）")
    log("=" * 64)
    for c, lab in [("ts_lr", "切空间 + 逻辑回归"), ("ts_lda", "切空间 + LDA")]:
        a = df[c].values
        p = stats.wilcoxon(a - 0.5).pvalue if np.any(a != 0.5) else float("nan")
        log(f"  {lab}: {a.mean() * 100:.1f}% ± {a.std() * 100:.1f}%（vs 随机 p={p:.4g}）")
    log("参照（不同特征/协议，只作量级对照）：主分析 LDA full 52.3%、CSP+LDA 跨被试 47.4%；")
    log("数据论文 55.57% 为被试内协议，不可直接比。")
    log(f"\n输出: {out}")
    log(f"用时 {(time.time() - t0) / 60:.1f} 分钟；把 CSV + 日志贴回 Muse 独立复算。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
