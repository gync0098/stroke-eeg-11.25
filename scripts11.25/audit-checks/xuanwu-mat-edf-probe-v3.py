# -*- coding: utf-8 -*-
"""
xuanwu-mat-edf-probe-v3.py — MAT 与 EDF 的真实对应关系探针（只探 3 个试次）

背景：v1（按下标配通道）与 v2（包络对齐）的相关都落在噪声水平（r_env 中位
0.087），说明 MAT 与 EDF 的波形对应关系不是此前假设的任何一种。本探针不再
假设任何对应：对选定试次做 MAT 通道 × EDF 通道 × 滞后的全扫描，报告全局
最佳匹配，用数据本身回答：通道映射是什么、滞后是多少、最佳相关有多高。
另打印两边的通道统计（全零通道、通道均值波形幅度），检验"MAT 被平均参考"
等结构假设。
死命令合规：枚举剔"副本/copy"；输出只打印（不写文件也可，CSV 写脚本目录）。
用法：放进宣武数据文件夹直接运行，输出全部贴回。
"""
import os
import glob

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR
CASES = [("sub-01", 1), ("sub-43", 1), ("sub-43", 25)]  # (被试, 试次号 1-based)


def find_edf(data_dir, sid):
    cands = sorted(glob.glob(os.path.join(data_dir, "edffile", "**", "*.edf"), recursive=True))
    if not cands:
        cands = sorted(glob.glob(os.path.join(data_dir, "**", "*.edf"), recursive=True))
    cands = [c for c in cands
             if "副本" not in os.path.basename(c)
             and "copy" not in os.path.basename(c).lower()]
    hits = [c for c in cands if sid in c.replace("\\", "/").split("/")]
    return hits[0] if hits else None


def extract_pulses(trig):
    idx = np.where(trig > 0.5)[0]
    pulses = []
    if len(idx) == 0:
        return pulses
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    for run in runs:
        code = int(np.rint(np.median(trig[run])))
        if 1 <= code <= 3:
            pulses.append((int(run[0]), code))
    return pulses


def mi_onsets_from_pulses(pulses):
    onsets = []
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
    return onsets


def extract_rawdata(mat):
    hits = []

    def walk(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if not k.startswith("__"):
                    walk(v, f"{path}.{k}" if path else k)
        elif hasattr(obj, "_fieldnames"):
            for f in obj._fieldnames:
                walk(getattr(obj, f), f"{path}.{f}" if path else f)
        elif isinstance(obj, np.ndarray):
            if obj.dtype == object:
                for el in obj.ravel():
                    walk(el, path)
            elif path.endswith("rawdata") and obj.ndim == 3:
                hits.append(np.asarray(obj, dtype=float))

    walk(mat, "")
    return hits[0] if hits else None


def slide_best(long_sig, seg):
    """seg 在 long_sig 内的最佳滞后与相关（去均值归一化）。"""
    a = long_sig - long_sig.mean()
    b = seg - seg.mean()
    if b.std() == 0 or a.std() == 0:
        return -1, 0.0
    corr = np.correlate(a, b, mode="valid")
    denom = np.sqrt(np.convolve(a * a, np.ones(len(b)), mode="valid") * (b * b).sum())
    r = corr / np.maximum(denom, 1e-30)
    k = int(np.argmax(np.abs(r)))
    return k, float(r[k])


def main():
    import mne
    from scipy.io import loadmat
    mne.set_log_level("error")

    cache = {}
    for sid, trial in CASES:
        if sid not in cache:
            ep = find_edf(DATA_DIR, sid)
            raw = mne.io.read_raw_edf(ep, preload=True, verbose=False)
            data = raw.get_data()
            trig = np.rint(data[-1] * 1e6).astype(float)
            onsets = mi_onsets_from_pulses(extract_pulses(trig))
            mp = sorted(glob.glob(os.path.join(
                DATA_DIR, "**", f"{sid}_task-motor-imagery_eeg.mat"), recursive=True))[0]
            mat = loadmat(mp, struct_as_record=False, squeeze_me=True)
            rd = extract_rawdata(mat)
            cache[sid] = (raw.ch_names, data, onsets, rd)
        ch_names, data, onsets, rd = cache[sid]
        onset = onsets[trial - 1]
        lo = max(0, onset - 3600)
        hi = onset + 4600
        mat_trial = rd[trial - 1]                       # (33, 4000)
        print(f"\n================ {sid} trial {trial}（脚本起点 {onset}）================")
        if trial == 1:
            print("EDF 通道名:", list(ch_names))
            print("MAT trial1 逐通道 std:", [round(float(x), 2) for x in mat_trial.std(axis=1)])
            print("EDF 同段逐通道 std:", [round(float(x), 6) for x in data[:, lo:lo + 4000].std(axis=1)])
            print(f"MAT 通道均值波形 std: {mat_trial.mean(axis=0).std():.4f} | "
                  f"EDF 通道均值波形 std: {data[:, lo:lo + 4000].mean(axis=0).std():.8f}")
        results = []
        for j in range(mat_trial.shape[0]):
            seg = mat_trial[j]
            if seg.std() == 0:
                continue
            for i in range(data.shape[0] - 1):          # 跳过 EDF 触发通道
                off, r = slide_best(data[i, lo:hi], seg)
                results.append((abs(r), r, j, i, lo + off))
        results.sort(reverse=True)
        print("全局最佳 10 组匹配（|r|, r, MAT通道, EDF通道, MAT段在EDF中的起点）:")
        for a, r, j, i, start in results[:10]:
            print(f"  r={r:+.4f}  MAT ch{j} <-> EDF ch{i} ({ch_names[i]})  start={start}  "
                  f"相对脚本起点 {start - onset:+d}")
        # 在全局最佳滞后处，逐 MAT 通道报最佳 EDF 伙伴
        best_start = results[0][4]
        print(f"在最佳起点 {best_start} 处，逐 MAT 通道的最佳 EDF 伙伴:")
        pairs = []
        for j in range(mat_trial.shape[0]):
            seg = mat_trial[j]
            if seg.std() == 0:
                pairs.append((j, -1, 0.0))
                continue
            best = (-1, 0.0)
            for i in range(data.shape[0] - 1):
                a = data[i, best_start:best_start + 4000]
                if a.std() == 0:
                    continue
                r = float(np.corrcoef(a, seg)[0, 1])
                if abs(r) > abs(best[1]):
                    best = (i, r)
            pairs.append((j, best[0], best[1]))
        print("  " + " | ".join(f"M{j}->E{i}({r:+.2f})" for j, i, r in pairs))


if __name__ == "__main__":
    main()
