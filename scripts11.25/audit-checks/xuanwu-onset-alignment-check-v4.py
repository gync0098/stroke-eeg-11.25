# -*- coding: utf-8 -*-
"""
xuanwu-onset-alignment-check-v4.py — 宣武 MAT/EDF 分段对齐全量核查（定版方法）

方法依据（v3 探针已实测确立）：
  - MAT 与 EDF 通道顺序相同（对角匹配主导；CPz 同为 17 号全零、触发同为 32 号）；
  - 两边数值经不同处理（EDF 为 0.5–40Hz 发布版、伏特尺度；MAT 为 µV 尺度），
    逐样本相关上限约 0.85——单通道定滞后会被噪声峰骗（v1 之败），须多通道求和；
  - MAT 试次切在指令时刻，试次内码 2 位置 p_k ≈ 1003（MAT 触发通道直读）。
判定量：d_k = (A_k + p_k) − S_k，其中 A_k 为 MAT 试次在 EDF 中的绝对起点
（12 个高信号通道相关求和取峰），S_k 为正式规则的脚本起点。正确对齐时
d_k ≈ 0；被多余脉冲对骗早切时 d_k ≈ +1002。|d_k − 本人中位| > 50 判错位。
脚本先打印有效性证据（峰值相关分布、每人 d 的离散度），证据不过关判定作废。
死命令合规：枚举剔"副本/copy"、EDF/MAT 各 50 断言、CSV 写脚本自身目录。
用法：放进宣武数据文件夹直接运行（约数分钟），输出全部贴回。
"""
import os
import glob
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR
EXCLUDE_SUBJ = {"sub-05", "sub-15", "sub-49"}
N_TOP_CH = 12
FLAG_TOL = 50


def find_files(data_dir):
    edfs = sorted(glob.glob(os.path.join(data_dir, "edffile", "**", "*.edf"), recursive=True))
    if not edfs:
        edfs = sorted(glob.glob(os.path.join(data_dir, "**", "*.edf"), recursive=True))
    mats = sorted(glob.glob(os.path.join(
        data_dir, "**", "sub-*_task-motor-imagery_eeg.mat"), recursive=True))
    keep = lambda p: ("副本" not in os.path.basename(p)
                      and "copy" not in os.path.basename(p).lower())
    edfs, mats = [p for p in edfs if keep(p)], [p for p in mats if keep(p)]
    edf_by, mat_by = {}, {}
    for p in edfs:
        for part in p.replace("\\", "/").split("/"):
            if part.startswith("sub-"):
                edf_by[part.split("_")[0]] = p
                break
    for p in mats:
        mat_by[os.path.basename(p).split("_")[0]] = p
    return edf_by, mat_by


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


def detect_mat_channels(rd):
    trig_ch, zero_ch = None, set()
    for c in range(rd.shape[1]):
        col = rd[:, c, :]
        if col.std() == 0:
            zero_ch.add(c)
            continue
        vals = np.unique(np.rint(col).astype(int))
        if len(vals) <= 5 and set(vals.tolist()) <= {0, 1, 2, 3}:
            frac0 = float((np.rint(col) == 0).mean())
            if frac0 > 0.9:
                trig_ch = c
    return trig_ch, zero_ch


def summed_r_curve(edf_win, mat_seg):
    """多通道相关求和曲线：edf_win (C, L)、mat_seg (C, 4000) -> (lags, r_sum)。"""
    total = None
    for c in range(mat_seg.shape[0]):
        a = edf_win[c] - edf_win[c].mean()
        b = mat_seg[c] - mat_seg[c].mean()
        if b.std() == 0 or a.std() == 0:
            continue
        corr = np.correlate(a, b, mode="valid")
        denom = np.sqrt(np.convolve(a * a, np.ones(len(b)), mode="valid") * (b * b).sum())
        r = corr / np.maximum(denom, 1e-30)
        total = r if total is None else total + r
    return total


def main():
    import mne
    from scipy.io import loadmat
    mne.set_log_level("error")
    t0 = time.time()

    edf_by, mat_by = find_files(DATA_DIR)
    print(f"EDF {len(edf_by)} 个 | MAT {len(mat_by)} 个")
    assert len(edf_by) == 50 and len(mat_by) == 50, "文件数与预期 50 不符，停。"

    rows = []
    for sid in sorted(edf_by):
        raw = mne.io.read_raw_edf(edf_by[sid], preload=True, verbose=False)
        data = raw.get_data().astype(np.float64)
        trig = np.rint(data[-1] * 1e6)
        pulses = extract_pulses(trig)
        onsets = mi_onsets_from_pulses(pulses)
        assert len(onsets) == 40, f"{sid}: 脚本起点 {len(onsets)} != 40，停。"

        mat = loadmat(mat_by[sid], struct_as_record=False, squeeze_me=True)
        rd = extract_rawdata(mat)
        trig_ch, zero_ch = detect_mat_channels(rd)
        assert trig_ch is not None, f"{sid}: MAT 触发通道未检出，停。"
        sig_ch = [c for c in range(rd.shape[1]) if c != trig_ch and c not in zero_ch]
        # 通道选择：以 trial 1 的 std 排序取前 N_TOP_CH 个信号通道（全程固定）
        stds = rd[0, sig_ch, :].std(axis=1)
        top = [sig_ch[i] for i in np.argsort(stds)[::-1][:N_TOP_CH]]

        for k in range(40):
            onset = onsets[k]
            lo = max(0, onset - 3600)
            hi = onset + 4600
            win = data[top, lo:hi]
            seg = rd[k, top, :]
            curve = summed_r_curve(win, seg)
            j = int(np.argmax(curve))
            A = lo + j
            peak = float(curve[j]) / len(top)
            seg_trig = np.rint(rd[k, trig_ch, :])
            pk_list = mi_onsets_from_pulses(extract_pulses(seg_trig))
            p_k = pk_list[0] if len(pk_list) == 1 else np.nan
            d = (A + p_k) - onset if np.isfinite(p_k) else np.nan
            rows.append(dict(subj=sid, trial=k + 1, script_onset=int(onset),
                             mat_start=int(A), mat_code2_within=p_k,
                             diff=d, peak_r_mean=peak))
        print(f"  {sid} done ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df["dev"] = df["diff"] - df.groupby("subj")["diff"].transform("median")
    print("\n================ 有效性证据 ================")
    print(f"峰值平均相关：中位 {df.peak_r_mean.median():.3f} | 5% 分位 {df.peak_r_mean.quantile(0.05):.3f}"
          f"（须整体接近 0.8 量级，落在 0.1 附近说明对齐失败、判定作废）")
    med_iqr = df.groupby("subj")["diff"].agg(lambda x: x.quantile(.75) - x.quantile(.25)).median()
    print(f"每人 diff 的 IQR 中位：{med_iqr:.1f} 采样（应为个位数到几十）")
    t1 = df[df.trial == 1]["diff"]
    print(f"trial 1 的 diff：中位 {t1.median():.0f}（对照组，应 ≈0）")

    flag = df[df.dev.abs() > FLAG_TOL]
    print(f"\n================ 判定：错位试次 {len(flag)} 个 ================")
    for _, r in flag.iterrows():
        print(f"  {r.subj} trial {int(r.trial)}: diff={r['diff']:.0f} dev={r.dev:+.0f} "
              f"peak_r={r.peak_r_mean:.2f} in_analysis={'NO' if r.subj in EXCLUDE_SUBJ else 'YES'}")
    print("\nsub-43 五段专项：")
    for t in (25, 26, 27, 31, 33):
        r = df[(df.subj == "sub-43") & (df.trial == t)].iloc[0]
        print(f"  trial {t}: diff={r['diff']:.0f} dev={r.dev:+.0f} peak_r={r.peak_r_mean:.2f}")
    r = df[(df.subj == "sub-13") & (df.trial == 10)].iloc[0]
    print(f"sub-13 trial 10: diff={r['diff']:.0f} dev={r.dev:+.0f} peak_r={r.peak_r_mean:.2f}")

    out = os.path.join(SCRIPT_DIR, "onset_alignment_v4_20261121.csv")
    df.to_csv(out, index=False)
    print(f"\nCSV 已保存：{out}")


if __name__ == "__main__":
    main()
