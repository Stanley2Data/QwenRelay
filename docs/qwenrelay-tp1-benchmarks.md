# Single RTX 3090 H/F comparison

[README](../README.md) · [中文入口](../README.zh-CN.md) · [TP2 comparison](qwenrelay-benchmarks.md) · [Source merge](qwenrelay-tp-profiles.md)

This is a public summary of the historical GPU0 30-case retest completed on
2026-10-19/20. H is the recorded HyperQwen reference; F is the single-GPU
QwenRelay candidate reviewed before the unified TP1/TP2/TP4 merge. These results
are not a GPU qualification of the current merged source or a comparison against
the latest HyperQwen branch.

## Conditions and method

- Same RTX 3090 GPU for both arms; AutoRound-fast target and DFlash2 draft.
- TP1, max context 132,096, max running sequences 4, DFlash depth 4,
  GPU KV budget 6 GiB and CPU KV budget 32 GiB; BF16 activations, INT8_ACT off.
- Both arms included the five inherited launch exports and the common
  model-load scratch-capacity correction. The shared fixture correction did
  not change attention kernel math. Main/draft head sharing and the actual
  four-request / scratch-QMAX-10 allocation were checked.
- Each case had one unscored warm round and three formal rounds per arm.
  The original 22 workloads plus eight context/decode cases produced 180 formal
  primary values, 124 checked actual-input pairings and 2,448 inference requests.
- 64K and 128K C4 context/decode cases were excluded from this retest. They are
  not missing values to be filled with earlier runs.
- H-main, F-main, H-context and F-context ran sequentially in a 200 GiB / 4 CPU
  container. Other services used different GPUs but shared host CPU/memory.
  Resource contention was sampled; the experiment does not establish exclusive
  host resources or isolate the causal effect of one optimization.

For cohort/context rows, the metric is **client concurrency divided by mean
TPOT**, not total output throughput. TPOT excludes initial TTFT but can include
later scheduling delays. C8/C16 can queue under the four-sequence scheduler.
Prefill is total actual input tokens divided by complete client elapsed time,
with one generated token. The long-100K row uses mean TPOT (lower is better);
the 4×60K row uses aggregate output tokens divided by full elapsed time.

## All 30 cases

Each cell is the median of three formal rounds, followed by their minimum and
maximum in brackets. Values and percentage changes retain the recorded report's
rounding. In particular, changes should not be recomputed as if the displayed
two-decimal values were the unrounded measurements.

| Case | Metric | H median [min, max] | F median [min, max] | F change |
|---|---|---:|---:|---:|
| Real questions, default sampling, C1 | C / mean TPOT, tok/s | 124.64 [123.69, 127.98] | 128.72 [126.63, 128.80] | +3.27% |
| Real questions, default sampling, C2 | C / mean TPOT, tok/s | 228.77 [218.25, 234.22] | 229.63 [228.19, 238.46] | +0.38% |
| Real questions, default sampling, C4 | C / mean TPOT, tok/s | 162.09 [152.01, 170.34] | 324.65 [324.38, 325.51] | +100.29% |
| Real questions, default sampling, C8 | C / mean TPOT, tok/s | 319.46 [307.25, 341.81] | 637.33 [629.25, 653.41] | +99.51% |
| Real questions, temperature 0, C1 | C / mean TPOT, tok/s | 127.27 [125.19, 132.06] | 130.21 [129.12, 131.19] | +2.31% |
| Real questions, temperature 0, C2 | C / mean TPOT, tok/s | 239.96 [239.86, 240.39] | 236.46 [235.86, 239.79] | -1.46% |
| Real questions, temperature 0, C4 | C / mean TPOT, tok/s | 164.17 [156.32, 166.20] | 330.76 [324.17, 331.31] | +101.48% |
| Real questions, temperature 0, C8 | C / mean TPOT, tok/s | 331.08 [314.03, 342.54] | 666.29 [658.25, 682.41] | +101.25% |
| Prefill 1,024, C1 | input tok/s | 1294.18 [1287.49, 1299.67] | 1316.97 [1314.79, 1317.08] | +1.76% |
| Prefill 1,024, C4 | input tok/s | 1334.36 [1328.77, 1337.53] | 1348.52 [1348.23, 1349.90] | +1.06% |
| Prefill 1,024, C16 | input tok/s | 1331.07 [1316.60, 1334.48] | 1342.50 [1342.21, 1343.10] | +0.86% |
| Prefill 4,096, C1 | input tok/s | 1232.31 [1231.85, 1235.16] | 1311.59 [1309.62, 1311.89] | +6.43% |
| Prefill 4,096, C4 | input tok/s | 1255.75 [1241.54, 1256.26] | 1325.85 [1325.76, 1328.78] | +5.58% |
| Prefill 4,096, C16 | input tok/s | 1249.21 [1222.95, 1249.41] | 1321.03 [1319.54, 1323.30] | +5.75% |
| Prefill 16,384, C1 | input tok/s | 1002.12 [999.30, 1002.77] | 1231.82 [1231.39, 1232.26] | +22.92% |
| Prefill 16,384, C4 | input tok/s | 1005.76 [1001.10, 1006.06] | 1241.79 [1241.46, 1243.11] | +23.47% |
| Prefill 16,384, C8 | input tok/s | 1003.98 [996.97, 1004.06] | 1239.78 [1239.11, 1240.02] | +23.49% |
| Prefill 65,536, C1 | input tok/s | 562.93 [562.61, 564.31] | 1041.40 [1039.47, 1044.38] | +85.00% |
| Prefill 65,536, C2 | input tok/s | 564.42 [561.41, 566.23] | 1045.12 [1044.81, 1045.54] | +85.17% |
| Prefill 102,400, C1 | input tok/s | 423.01 [421.48, 425.29] | 925.88 [925.05, 926.40] | +118.88% |
| Long 100K request | mean TPOT, ms ↓ | 11.90 [10.07, 16.52] | 8.37 [7.05, 10.27] | -29.69% |
| Long 4 × 60K requests | aggregate output tok/s | 9.44 [9.43, 9.54] | 16.00 [15.94, 16.04] | +69.57% |
| Context decode 1,024, C1 | C / mean TPOT, tok/s | 210.07 [207.83, 210.28] | 213.13 [211.07, 213.40] | +1.46% |
| Context decode 1,024, C4 | C / mean TPOT, tok/s | 237.42 [221.46, 238.13] | 461.38 [413.01, 502.19] | +94.33% |
| Context decode 4,096, C1 | C / mean TPOT, tok/s | 185.68 [179.93, 196.65] | 202.95 [191.57, 203.48] | +9.30% |
| Context decode 4,096, C4 | C / mean TPOT, tok/s | 199.14 [186.96, 202.68] | 331.81 [330.95, 357.06] | +66.63% |
| Context decode 16,384, C1 | C / mean TPOT, tok/s | 179.29 [176.92, 180.75] | 196.24 [193.74, 197.41] | +9.45% |
| Context decode 16,384, C4 | C / mean TPOT, tok/s | 102.82 [101.48, 106.60] | 152.26 [147.82, 153.49] | +48.08% |
| Context decode 65,536, C1 | C / mean TPOT, tok/s | 129.61 [129.36, 129.80] | 170.20 [168.71, 170.20] | +31.31% |
| Context decode 131,072, C1 | C / mean TPOT, tok/s | 91.88 [91.27, 92.51] | 138.68 [136.97, 139.58] | +50.95% |

F had better numerical medians in **29/30** cases; T0 C2 was **−1.46%**.
This is not a significance test. The predeclared investigation gate required
both a worse median by more than 10% and non-overlapping three-round ranges
in the worse direction; no case met that gate.

## What the observations establish

The recorded run checked actual HTTP payload pairings, SSE completion,
finish reasons and server token usage. It did not count SSE chunks as tokens.
Context cases retained native split-KV graph-node/replay observations; the F
main/draft C4 FULL20 capture and replay were checked against their actual process
and manager identities.

Earlier single-card runs omitted inherited launch settings and used different
GPUs/resource limits. Their low long-context values are not the baseline here.
For example, an earlier corrected F run already measured 138.53 tok/s at 128K C1;
this run measured 138.68. That large improvement over an older misconfigured
32.64 tok/s result must not be described as a new algorithmic gain.

The historical TP2 comparison used EfficientThink and different hardware/topology.
Its values are a separate experiment, not samples in this matrix or a controlled
measurement of single-card versus dual-card scaling.

## Evidence and limits

This page exports the numeric summary and test conditions from `REPORT-v2.md`,
whose analysis identifier is SHA256
`c441fe5b4f8f3eec2a9b708ded54739a346ae9fadf00b89518b3c57f5f043c0e`.
It preserves all 30 medians/ranges, including the regression. Private service
configuration, operational logs, prompts and generated text are not included.
The original analysis and per-request evidence are not distributed by this
summary; the identifier records provenance, not an independent reproduction.

Quality, the standalone cache-restore battery, C64 throughput, vision and a
long-running soak were not rerun. Performance does not establish output quality
or prove CPU-cache restoration. Review of this implementation informed the
[current source merge](qwenrelay-tp-profiles.md); that merge still needs fresh
GPU validation under the [release validation boundary](qwenrelay-validation.md).

The public `scripts/benchmark-qwenrelay.py` driver runs 32 workloads, including
the two long C4 cases skipped here. It is useful for new measurements but does
not reproduce the original private observers or this exact 30-case matrix.
