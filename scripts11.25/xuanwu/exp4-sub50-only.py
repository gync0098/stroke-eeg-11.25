#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp4-sub50-only.py — 只补跑 sub-50 的 Exp 4 特征（分阶段诊断版）
================================================================
背景：2026-10-21 晚 exp4-association.py 全量运行时 46 人成功、仅 sub-50
跳过。2026-10-22 本脚本首跑回传报错，根因查实：sub-50 文件夹多了一个
"…_eeg - 副本.edf" 拷贝，脚本按名排序先抓中它；而 _labels_from_events_tsv
对该文件名做 "_eeg.edf"→"_events.tsv" 替换不生效，把 EDF 二进制本身
当文本读而崩（UnicodeDecodeError）。exp4-association.py 已修（替换未
生效或读取异常一律回落 STI 解析）；本脚本亦改为优先正式文件名。

本脚本只做一件事：对 sub-50 一个人跑完 exp4 的全部三个阶段，
每阶段独立 try/except 并打印诊断——成功则输出一行可与
exp4_subject_data_*.csv 直接拼接的 CSV；失败则精确指出卡在哪一阶段，
把那一段报错贴回即可定位。

用法（和 exp4-association.py 放同一文件夹）:
    python exp4-sub50-only.py [--data-dir <数据目录>]
输出:
    ./exp4_sub50补齐_YYYYMMDD_HHMMSS.csv（单行，列名与 exp4_subject_data 完全一致）
"""
import argparse
import importlib.util
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(msg, flush=True)


def load_pipeline():
    """从同目录的 exp4-association.py 导入解析/特征函数（不执行其 main）。"""
    path = os.path.join(HERE, "exp4-association.py")
    spec = importlib.util.spec_from_file_location("exp4mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    E = load_pipeline()
    log("阶段 0：定位 sub-50 的 EDF")
    if args.data_dir:
        search_dirs = [args.data_dir]
    else:
        search_dirs = [HERE, os.getcwd(), os.path.join(os.getcwd(), "stroke-eeg")]
    edfs = []
    for d in search_dirs:
        edfs = E.find_edfs(d)
        if edfs:
            break
    cands = [e for e in edfs
             if os.path.basename(e).split("_")[0] == "sub-50"
             and os.path.basename(e).split("_")[0] not in E.DERIVED_SUBJECTS]
    log(f"  数据目录候选: {search_dirs[0] if edfs else '未找到 EDF'}")
    log(f"  找到 EDF 总数: {len(edfs)}；sub-50 候选: {cands}")
    if not cands:
        sys.exit("  ✗ 没找到 sub-50 的 EDF——先确认数据文件夹里有 sub-50 的文件。")

    def _pref(p):
        b = os.path.basename(p)
        bad = ("副本" in b) or ("copy" in b.lower()) or (" - " in b)
        return (bad, len(b), b)

    ef = sorted(cands, key=_pref)[0]
    if len(cands) > 1:
        log(f"  （多个候选，已按正式文件名优先选定；副本类文件跳过）")
    log(f"  → 用: {ef}")

    clin = E.CLINICAL.get("sub-50")
    log(f"  临床表: {clin}")

    log("\n阶段 1：试次解析 load_subject（STI 码2→码3 配对）")
    t0 = time.time()
    try:
        d = E.load_subject(ef)
    except Exception:
        log("  ✗ 阶段 1 失败，完整报错：")
        traceback.print_exc()
        sys.exit(1)
    log(f"  ✓ 40 试次解析成功，用时 {time.time()-t0:.1f}s")

    log("\n阶段 2：特征提取 extract_features（Welch + specparam，约 1–2 分钟）")
    t0 = time.time()
    try:
        feats, chi, offset, r2 = E.extract_features(d["trials"], d["sfreq"])
    except Exception:
        log("  ✗ 阶段 2 失败，完整报错：")
        traceback.print_exc()
        sys.exit(1)
    log(f"  ✓ chi shape={chi.shape}, offset shape={offset.shape}, "
        f"specparam R²中位数={r2:.3f}, 用时 {time.time()-t0:.1f}s")

    log("\n阶段 3：被试级聚合（与 exp4-association.py 主流程逐行一致）")
    try:
        chi_ch = np.median(chi, axis=0)
        off_ch = np.median(offset, axis=0)
        names = list(d["ch_names"])
        li = [names.index(c) for c in E.LEFT_CH if c in names]
        ri = [names.index(c) for c in E.RIGHT_CH if c in names]
        log(f"  左半球对上 {len(li)}/12 导，右半球对上 {len(ri)}/12 导")
        left_chi, right_chi = float(chi_ch[li].mean()), float(chi_ch[ri].mean())
        left_off, right_off = float(off_ch[li].mean()), float(off_ch[ri].mean())
        aff = clin["aff"]
        ipsi_chi, contra_chi = (left_chi, right_chi) if aff == "left" else (right_chi, left_chi)
        ipsi_off, contra_off = (left_off, right_off) if aff == "left" else (right_off, left_off)
        row = dict(
            subj="sub-50", chi_global=float(chi_ch.mean()), chi_ipsi=ipsi_chi,
            chi_contra=contra_chi, chi_AI=ipsi_chi - contra_chi,
            offset_global=float(off_ch.mean()), offset_AI=ipsi_off - contra_off,
            r2=r2, NIHSS=clin["nihss"], mrs=clin["mrs"], mbi=clin["mbi"],
            age=clin["age"], duration=clin["duration"], sex=clin["sex"],
            aff_hemi=aff, ternary=clin["ternary"], cortical=clin["cortical"])
    except Exception:
        log("  ✗ 阶段 3 失败，完整报错：")
        traceback.print_exc()
        sys.exit(1)
    log(f"  ✓ sub-50: χ_global={row['chi_global']:.3f}, χ_AI={row['chi_AI']:+.3f}, "
        f"NIHSS={row['NIHSS']}, mRS={row['mrs']}, MBI={row['mbi']}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(os.getcwd(), f"exp4_sub50补齐_{ts}.csv")
    pd.DataFrame([row]).to_csv(out, index=False)
    log(f"\n全部三阶段通过。单行 CSV 已存: {out}")
    log("把它与 exp4_subject_data_20261021_200824.csv 上下拼接即得 n=47 全量表。")


if __name__ == "__main__":
    main()
