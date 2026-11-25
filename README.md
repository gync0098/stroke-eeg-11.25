# Aperiodic spectral parameters in acute stroke EEG — analysis code

Analysis code for: Gao Y. *Aperiodic Spectral Parameters in Acute Stroke EEG:
Cross-Subject Decoding, Calibration, and Clinical Association* (manuscript in
preparation, 2026).

All headline results in the paper are produced by the scripts in this
repository. The current pipeline is the `bpfix` round (band-power floor
1e-20); earlier rounds (`scripts/xuanwu/`, `scripts/xuanwu/segfix/`) are kept
for audit only and must not be cited as the paper's numbers.

## Data (not included; publicly available, CC BY 4.0)

1. **Primary cohort (Xuanwu)** — "EEG datasets of stroke patients",
   figshare doi:10.6084/m9.figshare.21679035 (download `edffile.zip`).
   Data paper: Liu H, et al. Sci Data 2024;11:131.
2. **Second cohort (Tianjin)** — lower-limb motor-imagery dataset,
   figshare doi:10.6084/m9.figshare.27130299 (download `PEEG.rar`,
   `participants.tsv`, `README.md`).
   Data paper: Liu Y, et al. Sci Data 2025;12:314.

Analysis sample: 47 of 50 primary-cohort recordings (sub-05, sub-15 and
sub-49 are duplicated recordings and are excluded by every script) and 26 of
27 Tianjin recordings (sub-06 excluded under a pre-specified trial-count
rule: 39 MI vs 40 idle trials).

## Environment

Python 3.10.20, MNE-Python 1.12.1, specparam 2.0.0rc6, scikit-learn 1.7.2,
NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, Matplotlib 3.10.9.
Random seed fixed at 20261017 where applicable.
On Windows, run with UTF-8 enabled; the runner below forces it, and the
single Tianjin script contains a UTF-8 bootstrap.

## Quick start (primary cohort)

```text
python scripts/xuanwu/bpfix/run-bpfix-all-v2.py
```

Runs the full primary-cohort pipeline (7 scripts x fix and drop18 rounds,
14 runs) and writes result CSVs next to the scripts. Expected outcome:
full-spectrum LDA 52.8% (BH q = 0.030), mixed-aperiodic SVM 52.9%
(q = 0.030); drop18 sensitivity (balanced accuracy): LDA full 52.9%
(q = 0.036), SVM aperiodic 52.6% (q = 0.044).

## What each current script produces

### Primary cohort (Xuanwu) — `scripts/xuanwu/bpfix/`

| Script | Produces |
|---|---|
| `stepR-probe-decode-fixfloor-bpfix.py` | Exp 2 decoding, LDA: full / periodic / aperiodic band-power inputs, subject-level LOSO (paper Table 2, upper block) |
| `exp23-robust-svm-fixfloor-bpfix.py` | Exp 2 decoding under RBF-SVM, plus Exp 3 calibration schemes calB / calC under SVM |
| `exp2-ablation-chioff-bpfix.py` | χ-only / offset-only / mixed-aperiodic ablation under both decoders |
| `exp3-calibration-bpfix.py` | Exp 3 calibration, LDA: base / calA / calB / calC |
| `exp3-svm-calA-bpfix.py` | Exp 3 calibration, SVM run of calA |
| `xuanwu-refit-fixedfull-fixfloor-bpfix.py` | Fitting-range sensitivity: full-spectrum features held fixed at [1, 40] Hz, specparam fit range narrowed to [3, 35] Hz |
| `xuanwu-refit-3-35-fixfloor-bpfix.py` | Frequency-range sensitivity: spectra truncated to [3, 35] Hz before all feature computation |

Exp 4 (clinical association) and the Riemannian benchmark were not rerun in
the bpfix round — they do not pass through the band-power floor. Their
current versions are `scripts/xuanwu/segfix/exp4-association-segfix.py`
(+ `scripts/xuanwu/exp4-sub50-only.py`, the sub-50 completion pass) and
`scripts/xuanwu/segfix/xuanwu-riemann-ts-segfix.py`.

### Second cohort (Tianjin) — `scripts/tianjin/`, final pipeline (2-s Welch windows)

| Script | Produces |
|---|---|
| `tianjin-exp2-window2s-bpfix.py` | Exp 2 repetition: full / periodic / aperiodic decoding, subject-level LOSO (paper Table 4; periodic 57.4% LDA / 59.1% SVM) |
| `tianjin-ablation-chioff.py` | χ-only / offset-only ablation (χ-only 53.7% LDA / 54.0% SVM); its control column must reproduce the aperiodic accuracies of the previous script subject-by-subject |

### Figures and statistics

| Script | Produces |
|---|---|
| `scripts/figures/make-figures.py` | Figures 1–4 and Table 1 |
| `scripts/figures/make-fig5.py` | Figure 5 (cross-cohort χ, decoding by cohort, sensitivity triplets) |
| `scripts/figures/make-supplementary-figures.py` | Supplementary Figures S1–S5 |
| `scripts/audit-checks/bpfix-bh-recalc.py` | Recomputes all p/q values in the paper from the result CSVs (12/12 checks; `bpfix_bh_recalc.json` is its output). Takes a results root as its argument; defaults to the layered `results/` directory. |

## Audit checks — `scripts/audit-checks/`

These scripts produce verification evidence, not paper numbers:

| Script | Checks |
|---|---|
| `xuanwu-label-check-v2.py` | .mat trial labels vs the "first trial = left hand, strictly alternating" rule (50 subjects, 0 mismatches) |
| `xuanwu-duplicate-quantify.py` | Quantifies the three duplicated recordings (sub-04/sub-05 identical sample-by-sample; sub-11/sub-15 r = 0.99996; sub-47/sub-49 r = 0.99998) |
| `xuanwu-mat-edf-probe-v3.py` | MAT vs EDF channel order is identical (channel 17 = zero CPz, channel 32 = trigger in both) |
| `xuanwu-onset-alignment-check-v4.py` | Segmentation check over all 2000 trials; 30 flagged, 5 genuinely mis-segmented (sub-43, trials 25–27, 31, 33), 24 false alarms |
| `xuanwu-bpfloor-count.py`, `tianjin-bpfloor-count.py` | Band-power floor audit: primary cohort 6.20% (full) / 26.74% (periodic) / 16.31% (aperiodic) of values clamped at 1e-12; Tianjin periodic 4.40% (driven by delta, 17.50%) |

## Notes

- Equivalence tests (TOST), covariate partial correlations and leave-one-out
  influence checks reported in the paper were computed from the Exp 4 result
  files; the archived outputs accompany the manuscript records.
- Earlier exploratory and diagnostic scripts (dataset structure probes, an
  earlier 4-s-window Tianjin pipeline, and the initial calibration script
  whose LDA results are reproduced exactly by `exp3-svm-calA.py`) are not
  part of the final pipeline and are not included.
- No generative-AI system is an author of this code repository; see the
  manuscript's declarations for the AI-use statement.
