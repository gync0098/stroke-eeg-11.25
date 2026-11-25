# -*- coding: utf-8 -*-
"""
xuanwu-duplicate-quantify.py
对三对重复记录（sub-04/05、sub-11/15、sub-47/49）做定量刻画：
  - 逐样本完全一致的比例
  - 全部样本合起来的 Pearson 相关
  - 差值的 RMS 相对信号 RMS 的比例
目的：把"近乎一致"写成可引用的数字（稿件里 sub-15/sub-49 的
"reprocessed copies"表述需要这个量级证据）。
用法：把本脚本放进宣武数据文件夹（与 EDF 同层），直接运行。
输出打印在屏幕上，全部贴回 Muse 即可。
"""
import os
import sys
import glob

import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PAIRS = [("sub-04", "sub-05"), ("sub-11", "sub-15"), ("sub-47", "sub-49")]
N_EEG = 29  # 前 29 导为 EEG（与主分析口径一致）


def find_edf(sub):
    pats = sorted(glob.glob(os.path.join(DATA_DIR, "**", f"{sub}*.edf"), recursive=True))
    pats = [p for p in pats if "副本" not in p and "copy" not in os.path.basename(p).lower()]
    if not pats:
        pats = sorted(glob.glob(os.path.join(DATA_DIR, "**", f"{sub}*.edf"), recursive=True))
    return pats[0] if pats else None


def main():
    import mne

    print(f"数据目录：{DATA_DIR}")
    print("=" * 64)
    for a, b in PAIRS:
        pa, pb = find_edf(a), find_edf(b)
        if pa is None or pb is None:
            print(f"{a} vs {b}: 文件缺失（{pa} / {pb}），跳过")
            continue
        ra = mne.io.read_raw_edf(pa, preload=True, verbose=False)
        rb = mne.io.read_raw_edf(pb, preload=True, verbose=False)
        da = ra.get_data()[:N_EEG]
        db = rb.get_data()[:N_EEG]
        if da.shape != db.shape:
            print(f"{a} vs {b}: 形状不同 {da.shape} vs {db.shape}，跳过")
            continue
        x = da.ravel()
        y = db.ravel()
        diff = x - y
        pct_exact = 100.0 * np.mean(diff == 0)
        r = np.corrcoef(x, y)[0, 1]
        rms_sig = np.sqrt(np.mean(x ** 2))
        rms_diff = np.sqrt(np.mean(diff ** 2))
        print(f"{a} vs {b}:  {da.shape[0]} 导 × {da.shape[1]} 采样")
        print(f"  逐样本完全一致比例 : {pct_exact:.4f}%")
        print(f"  全样本 Pearson r    : {r:.8f}")
        print(f"  差值 RMS / 信号 RMS : {rms_diff / rms_sig:.3e}"
              f"（差值 RMS = {rms_diff:.3e} V，信号 RMS = {rms_sig:.3e} V）")
        print(f"  最大绝对差          : {np.max(np.abs(diff)):.3e} V")
        print()
    print("完成。把全部输出贴回 Muse。")


if __name__ == "__main__":
    main()
