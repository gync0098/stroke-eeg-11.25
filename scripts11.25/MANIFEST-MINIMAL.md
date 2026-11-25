# 卒中非周期论文 — 最小最新有效脚本包（2026-11-24）

只留能产出正文当前数字、或为正文结论提供直接证据的文件。入选规则三条，缺一不可：
最新——是产出正文当前数字的那一版脚本（bpfix 轮；黎曼与 Exp 4 未在 bpfix 轮重跑，留其 segfix 版）；
有效——脚本确有输出进入正文、图表或核查结论；
最小——历史轮次（orig / segfix 其余 / fixfloor 天津 Exp 2）与已被取代的脚本一律不在此包。
完整版见 stroke-paper-FINAL-scripts.zip（41 py + 1 JSON），历史审计链在那里。

## 入选清单（24 个 py + 1 个 JSON）
- xuanwu/bpfix/：7 个 bpfix 分析脚本 + run-bpfix-all-v2.py（一键串行 runner，含 GBK 防护与 UTF-8 自举说明）
- figures/：make-figures.py、make-fig5.py、make-supplementary-figures.py（默认读分层结果目录）
- tianjin/：tianjin-exp2-window2s-bpfix.py（Exp 2 定版，带功率下限 1e-20）、tianjin-ablation-chioff.py（消融定版）、tianjin-bpfloor-count.py（天津下限计数）
- audit-checks/：xuanwu-bpfloor-count.py（宣武下限计数，6.20%/26.74%/16.31% 出处）、bpfix-bh-recalc.py + bpfix_bh_recalc.json（11.23 版 p/q 复算，12/12 通过）、xuanwu-label-check-v2.py（标签核查）、xuanwu-duplicate-quantify.py（重复被试量化）、xuanwu-mat-edf-probe-v3.py（MAT/EDF 通道对应探针）、xuanwu-onset-alignment-check-v4.py（分段对齐全量核查）
- xuanwu/segfix/：xuanwu-riemann-ts-segfix.py、exp4-association-segfix.py（黎曼与 Exp 4 的当前版，未在 bpfix 轮重跑）；xuanwu/exp4-sub50-only.py（Exp 4 sub-50 补跑）
