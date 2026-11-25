# -*- coding: utf-8 -*-
"""
xuanwu-label-check-v2.py
.mat 结构已探明（2026-11-21）：struct `eeg`，字段 rawdata (40,33,4000) 与 label (40,) uint8。
本脚本逐人提取 eeg.label，与分析管线的交替约定（第 1 试次=左手=1，左右交替）逐试次比对：
  A = [1,2,1,2,...] 完全一致 → 约定对该被试成立
  B = [2,1,2,1,...] 完全一致 → 该被试整体翻转（管线赋的标签与真标签左右互换）
  其他 = 不规则（打印完整序列与不一致位置）
用法：放进宣武数据文件夹（与 sourcedata 同层），直接运行，输出全部贴回 Muse。
"""
import os
import glob

import numpy as np
from scipy.io import loadmat

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_label(mat):
    """从 loadmat 结果里递归找名为 label 的 (40,) 数组（值域 {1,2}）。"""
    hits = []

    def walk(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.startswith("__"):
                    continue
                walk(v, f"{path}.{k}" if path else k)
        elif hasattr(obj, "_fieldnames"):
            for f in obj._fieldnames:
                walk(getattr(obj, f), f"{path}.{f}" if path else f)
        elif isinstance(obj, np.ndarray):
            if obj.dtype == object:
                for i, el in enumerate(obj.ravel()):
                    walk(el, f"{path}[{i}]")
            elif path.endswith("label") and obj.size == 40:
                vals = obj.ravel().astype(int)
                if set(np.unique(vals)) <= {1, 2}:
                    hits.append(vals)

    walk(mat, "")
    return hits[0] if hits else None


def main():
    mats = sorted(glob.glob(os.path.join(DATA_DIR, "**", "sub-*_task-motor-imagery_eeg.mat"),
                            recursive=True))
    if not mats:
        mats = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.mat"), recursive=True))
    print(f"找到 .mat 文件 {len(mats)} 个")
    alt_a = np.array([1 if i % 2 == 0 else 2 for i in range(40)])  # 第1试次=1(左)
    alt_b = 3 - alt_a                                              # 第1试次=2(右)
    n_a = n_b = n_irr = n_none = 0
    flipped, irregular = [], []
    for mp in mats:
        base = os.path.basename(mp)
        sid = base.split("_")[0]
        try:
            mat = loadmat(mp, struct_as_record=False, squeeze_me=True)
            lab = extract_label(mat)
        except Exception as e:
            lab = None
            print(f"  {sid}: 读取失败 {e}")
        if lab is None:
            n_none += 1
            print(f"  {sid}: 未提取到 label")
            continue
        if np.array_equal(lab, alt_a):
            n_a += 1
            print(f"  {sid}: 与约定完全一致（首试次=1，交替）")
        elif np.array_equal(lab, alt_b):
            n_b += 1
            flipped.append(sid)
            print(f"  {sid}: 整体翻转（首试次=2，交替；管线标签与真标签左右互换）")
        else:
            n_irr += 1
            bad = np.where(lab != alt_a)[0]
            irregular.append(sid)
            print(f"  {sid}: 不规则。与约定不一致的试次序号(0起): {bad.tolist()}")
            print(f"        完整标签序列: {lab.tolist()}")
    print("\n" + "=" * 64)
    print(f"汇总：约定一致 {n_a} 人；整体翻转 {n_b} 人 {flipped}；"
          f"不规则 {n_irr} 人 {irregular}；未提取 {n_none} 人")
    print("完成。把全部输出贴回 Muse。")


if __name__ == "__main__":
    main()
