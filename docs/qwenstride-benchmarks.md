# Dual RTX 3090 H/F comparison

[README](../README.md) · [中文入口](../README.zh-CN.md) · [Machine-readable results](../bench/results-published/tp2-hf-20261017.json)

These are the recorded 2026-10-17/18 TP2 results for the final Graph + cache-repair F candidate. Packaging did not rerun performance or quality. H is pinned HyperQwen 95ee3b6 + its 54 applicable patches on vLLM 0.31.0; F adds the local source deltas. H had the configured capture cap 20 but lacked full fixed four-request coverage; F's actual main/draft FULL(4 requests,20 tokens) capture and replay were observed on both ranks in all three formal rounds. This compares the complete candidates, not an isolated causal effect of O5 or any one patch.

## Conditions

Two RTX 3090 24 GB GPUs, 350 W limits, NVIDIA 580.126.20; vLLM 0.31.0, torch 2.13/CUDA13.0; identical 18 native extension files across H/F. Same EfficientThink mixed AWQ main, DFlash2 W4A16 draft, tokenizer, tested chat template and full model-file hashes. BF16 activations; INT8_ACT disabled. TP2, max sequences4, max model length262144, batch2048, DFlash depth4, prefix retention20800, per-GPU KV11,184,101,204 bytes, total CPU budget32 GiB. E2's physical padding changes token capacity: H616057/F600575 tokens under the same KV byte budget.

The recorded comparison disclosed a common filesystem-path adapter for shared CPU backing and external non-numerical observation hooks. This source package preserves the original native shared-memory default and omits private test-orchestration hooks. It has not repeated the benchmark in a new container. Changing the model or profile invalidates direct comparison with these numbers.

Each arm completed one full unscored warm round and three formal rounds. 264 total benchmark cases including original warmups, 2,512 inference HTTP requests, 132 complete H/F actual-input pairings and 192 formal primary values were independently checked in the original evidence. This export contains sanitized numeric aggregates and each of the three primary measurements; private infrastructure paths, raw requests and operational logs are excluded.

## Metric definitions

- Decode ROW = client concurrency **C / mean TPOT**, in tok/s. TPOT omits that request's initial TTFT, but can contain later pauses caused by another prefill or queue. This differs from C / median TPOT and from total output throughput.
- Aggregate output throughput = actual generated output tokens / complete elapsed benchmark time, including prefill and queue effects. For the original T0 C4 cohort it was H106.49/F316.10 tok/s; default C4 H101.78/F300.05 tok/s.
- Prefill input throughput = total actual input tokens / full benchmark duration, with one output token. Long100K reports mean TPOT in ms; lower is better. Long4×60K reports aggregate output throughput.
- A table value below is the **median of three formal rounds**. JSON also includes every primary measurement, min/max and arithmetic means. The latter are calculated from the same three values, without choosing the more flattering statistic.

The real-question cohorts reuse HyperQwen's eight questions in `bench/prompts_real.jsonl`, up to1024 output tokens, default sampling or temperature0. Prefill and long/context cases use seeded synthetic tokens. C8/C16 are client queue depths; the server runs at most4 sequences. Do not call the C8 ROW value eight-way resident decode throughput.

## Original 22 primary cases

| Case | Metric | H median | F median | F change |
|---|---|---:|---:|---:|
| cohort_Tdefault_c1 | C/mean TPOT, tok/s | 134.72 | 132.23 | -1.85% |
| cohort_Tdefault_c2 | C/mean TPOT, tok/s | 235.54 | 236.71 | +0.50% |
| cohort_Tdefault_c4 | C/mean TPOT, tok/s | 107.50 | 360.11 | +235.00% |
| cohort_Tdefault_c8 | C/mean TPOT, tok/s | 226.94 | 722.56 | +218.39% |
| cohort_T0_c1 | C/mean TPOT, tok/s | 138.63 | 140.95 | +1.67% |
| cohort_T0_c2 | C/mean TPOT, tok/s | 253.66 | 246.98 | -2.63% |
| cohort_T0_c4 | C/mean TPOT, tok/s | 113.19 | 374.92 | +231.22% |
| cohort_T0_c8 | C/mean TPOT, tok/s | 244.10 | 775.09 | +217.53% |
| prefill_1024_c1 | input tok/s | 1585.32 | 1603.01 | +1.12% |
| prefill_1024_c4 | input tok/s | 1663.81 | 1672.91 | +0.55% |
| prefill_1024_c16 | input tok/s | 1668.41 | 1674.32 | +0.35% |
| prefill_4096_c1 | input tok/s | 1571.60 | 1632.35 | +3.87% |
| prefill_4096_c4 | input tok/s | 1609.08 | 1663.17 | +3.36% |
| prefill_4096_c16 | input tok/s | 1607.79 | 1659.55 | +3.22% |
| prefill_16384_c1 | input tok/s | 1376.70 | 1582.09 | +14.92% |
| prefill_16384_c4 | input tok/s | 1392.75 | 1604.91 | +15.23% |
| prefill_16384_c8 | input tok/s | 1391.75 | 1607.34 | +15.49% |
| prefill_65536_c1 | input tok/s | 887.74 | 1418.95 | +59.84% |
| prefill_65536_c2 | input tok/s | 891.79 | 1431.01 | +60.47% |
| prefill_102400_c1 | input tok/s | 698.48 | 1308.30 | +87.31% |
| long_100k | TPOT ms ↓ | 9.80 | 7.30 | -25.58% |
| long_4x60k | aggregate output tok/s | 14.56 | 21.81 | +49.82% |

F has better numerical medians in20/22 original primary cases. T0 C2 (−2.63%) and default C1 (−1.85%) are regressions in this statistic. With arithmetic means, default C1 changes to+0.53% and the count becomes21/22. These are small three-round observations, not significance or quality claims. The T0 C4 means are H114.57/F374.40 tok/s; medians H113.19/F374.92.

## Five context lengths × C1/C4 decode

Exactly1,024 /4,096 /16,384 /65,536 /131,072 input tokens, fixed1,024 output tokens, temperature0, ignore-EOS, same three formal repeats after a complete warm round.

| Actual input | C | H median C/meanTPOT | F median C/meanTPOT | F change |
|---|---:|---:|---:|---:|
| 1,024 | 1 | 206.70 | 173.70 | -15.96% |
| 1,024 | 4 | 167.46 | 538.45 | +221.54% |
| 4,096 | 1 | 188.95 | 211.36 | +11.86% |
| 4,096 | 4 | 151.02 | 371.12 | +145.75% |
| 16,384 | 1 | 185.38 | 204.28 | +10.19% |
| 16,384 | 4 | 100.99 | 178.27 | +76.52% |
| 65,536 | 1 | 139.61 | 179.23 | +28.38% |
| 65,536 | 4 | 32.55 | 50.49 | +55.13% |
| 131,072 | 1 | 100.47 | 147.34 | +46.65% |
| 131,072 | 4 | 13.22 | 24.06 | +82.01% |

At1K C1 the H rounds were220.32/206.70/125.50 and F221.49/173.70/149.42 tok/s. Median F change is−15.96%; arithmetic-mean change is−1.43%. The spread prevents a stable16% regression claim.

At128K C4, **neither arm had an interval in which all four requests had emitted content and none had finished**, and neither supplied qualifying ≥2-second four-resident/no-prefill samples. The ROW still measures request-level generation times, not pure four-way steady decode. For illustration, auxiliary sampled C4 four-resident/no-prefill rates at1K/4K/16K/64K were H172.15/154.64/154.16/168.50 and F595.90/495.80/509.54/444.36 tok/s. These are sampled output rates, not kernel throughput;128K is unavailable, not zero.

## CPU cache and quality boundary

The repaired F had a separate GPU cache-restore acceptance using the original30 requests. The later32-case performance matrix measured **zero CPU→GPU restore bytes**, so it does not revalidate or quantify CPU restoration speed. Its retained cache-fix code and separate cache acceptance are different pieces of evidence.

No GSM/PPL, other quality battery, standalone cache30 or C64 throughput suite was rerun in that performance matrix or during packaging. No quality advantage or no-regression guarantee is inferred from speed. The model checkpoint and its W4/W8/A16 details are in `locks/models.json`.

## Running benchmarks yourself

`bench/prompts_real.jsonl` and the upstream benchmark scripts remain in this repository. `scripts/benchmark-qwenstride.py` provides a standalone32-case driver against an already running endpoint, with no server lifecycle or private infrastructure dependencies. Use `--plan` to inspect its commands before running it. Its future results are a new measurement period; they must not be relabelled as the historical paired test. In particular, the historical per-case HTTP/SSE and resident-interval observers are not part of this simple public driver.
