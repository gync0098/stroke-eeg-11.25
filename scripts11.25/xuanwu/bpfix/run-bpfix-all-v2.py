# -*- coding: utf-8 -*-
"""
run-bpfix-all-v2.py — 一键串行跑完 bpfix 全部 14 个任务（7 脚本 × fix / drop18 两轮）

v2 相对 2026-11-23 07:22 版只改 runner 自身的编码防护，其余逻辑（脚本清单、
两轮顺序、CSV 核对、汇总）逐字未动。

为什么需要 v2：07:22 版只给子进程设了 PYTHONUTF8，但 runner 自己往 PowerShell
控制台 print 子脚本的尾部 15 行时，用的还是本进程的 stdout 编码。成功跑完的
stepR 尾部一定含 "specparam R²=..."（χ 速览 47 行在最后），exp3 尾部含 "− base"，
² / − / ✓ 都编不出 GBK——如果该终端的 stdout 是 GBK，runner 会在第 1 个任务
结束后自己抛 UnicodeEncodeError 而退出，剩下 13 个任务不再跑（与今早 0/14
不同的死法，同样的白跑一夜）。v2 用三层防护堵住这一层，子脚本一个字不用改。

用法（推荐，先设环境变量再启动，这样 runner 本进程也进 UTF-8 模式）：
  $env:PYTHONUTF8="1"
  & 'C:\\Users\\gync3\\.conda\\envs\\spike_processing\\python.exe' 'C:\\Users\\gync3\\Downloads\\xuanwuyiyuan\\run-bpfix-all-v2.py'

前提：7 个 *-bpfix.py 已与本脚本同放在 xuanwuyiyuan 数据文件夹。
行为：
  - 先 fix 轮 7 个、再 drop18 轮 7 个，逐个用子进程跑（环境变量 SEG_MODE 按轮设置，
    只作用于子进程，不污染你的终端）；
  - 单个脚本失败不中断全程：记入日志、继续下一个，天亮看汇总；
  - 全程日志写本脚本同目录 run-bpfix-all-log_<时间戳>.txt，控制台同步打印；
  - 每个任务跑完核对新生成的 CSV 并记入日志；结束打印总汇总（成功/失败、产出清单）。
注意：不要中途关这个终端；电脑别休眠（插电、关睡眠）。
"""
import glob
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "stepR-probe-decode-fixfloor-bpfix.py",
    "exp23-robust-svm-fixfloor-bpfix.py",
    "exp3-calibration-bpfix.py",
    "exp3-svm-calA-bpfix.py",
    "exp2-ablation-chioff-bpfix.py",
    "xuanwu-refit-fixedfull-fixfloor-bpfix.py",
    "xuanwu-refit-3-35-fixfloor-bpfix.py",
]
MODES = ["fix", "drop18"]


def main():
    # v2 防护 1：先把 runner 自身的 stdout/stderr 改成 UTF-8（errors=replace），
    # 再 print 任何含 ✓/²/− 的子脚本尾行。reconfigure 失败也不停，靠下面的 log() 兜底。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(SCRIPT_DIR, f"run-bpfix-all-log_{ts}.txt")
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        # v2 防护 2：控制台 print 失败（例如终端仍是 GBK）时不让 runner 崩掉：
        # 先按当前 stdout 编码安全转写（编不出的字符变 ?），日志文件里仍存完整原文。
        try:
            print(line, flush=True)
        except Exception:
            try:
                _enc = (getattr(sys.stdout, "encoding", None) or "gbk")
                _safe = line.encode(_enc, errors="replace").decode(_enc, errors="replace")
                print(_safe, flush=True)
            except Exception:
                pass
        logf.write(line + "\n")
        logf.flush()

    missing = [s for s in SCRIPTS if not os.path.exists(os.path.join(SCRIPT_DIR, s))]
    if missing:
        log(f"停：以下 bpfix 脚本不在本目录，先把 bpfix-rerun-kit.zip 解压到数据文件夹：{missing}")
        logf.close()
        sys.exit(1)

    log(f"run-bpfix-all 启动：{len(SCRIPTS)} 脚本 × {len(MODES)} 轮 = {len(SCRIPTS) * len(MODES)} 个任务")
    log(f"日志文件：{log_path}")
    results = []  # (mode, script, ok, seconds, [new_csvs], returncode)

    for mode in MODES:
        for name in SCRIPTS:
            path = os.path.join(SCRIPT_DIR, name)
            before = set(glob.glob(os.path.join(SCRIPT_DIR, "*.csv")))
            log(f"=== 开始 SEG_MODE={mode} :: {name} ===")
            env = dict(os.environ)
            env["SEG_MODE"] = mode
            # 关键修复（2026-11-23）：子脚本日志里有 "✓" 等非 ASCII 字符，经管道捕获时
            # Windows 按 GBK 编码 stdout 会抛 UnicodeEncodeError、被子脚本的逐被试
            # try 吞掉造成全员"解析失败"。强制子进程 UTF-8 模式后与 VS Code 终端一致。
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            t0 = time.time()
            try:
                proc = subprocess.run(
                    [sys.executable, path],
                    cwd=SCRIPT_DIR, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                )
                out, rc = proc.stdout or "", proc.returncode
            except Exception as e:  # 子进程起不来也要继续下一个
                out, rc = f"[runner] 子进程启动失败：{e}", -999
            dt = time.time() - t0
            # 子进程输出只留尾部 15 行进日志（全文太长；关键行含 [BPFIX]/总试次/产出路径）
            tail = "\n".join(out.strip().splitlines()[-15:])
            for ln in tail.splitlines():
                log(f"    {ln}")
            after = set(glob.glob(os.path.join(SCRIPT_DIR, "*.csv")))
            new_csvs = sorted(os.path.basename(p) for p in (after - before))
            ok = (rc == 0)
            results.append((mode, name, ok, dt, new_csvs, rc))
            log(f"=== {'完成' if ok else '失败 rc=' + str(rc)} SEG_MODE={mode} :: {name}"
                f"（{dt / 60:.1f} 分钟）新产出 CSV {len(new_csvs)} 个：{new_csvs} ===")

    n_ok = sum(1 for r in results if r[2])
    total_csv = sum(len(r[4]) for r in results)
    log("")
    log(f"######## 总汇总：{n_ok}/{len(results)} 个任务成功；共新产出 CSV {total_csv} 个（预期 22） ########")
    for mode, name, ok, dt, csvs, rc in results:
        log(f"  {'OK ' if ok else 'FAIL'} [{mode}] {name}（{dt / 60:.1f} 分钟，CSV {len(csvs)}）")
    fails = [r for r in results if not r[2]]
    if fails:
        # v2：直接单跑某个脚本时也必须先设 UTF-8 环境变量，否则它在 GBK 终端里
        # 会重蹈今早覆辙（逐被试 try 吞掉 ✓/² 的编码错，全员“解析失败”）。
        log("失败任务重跑方法：单独开终端，先设 $env:PYTHONUTF8=\"1\" 和 "
            "$env:PYTHONIOENCODING=\"utf-8\"，再按 RUN-SHEET-bpfix.md 对应命令重跑那几个。")
    else:
        log("全部成功。回传：两轮 22 个 CSV + 本日志文件。")
    logf.close()


if __name__ == "__main__":
    main()
