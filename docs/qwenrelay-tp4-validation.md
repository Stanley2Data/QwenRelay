# TP4 / DP1 validation

Experimental four RTX 3090 profile. Numerical checks and the requested single-pass
serving benchmark completed on 2026-10-18. Broader production qualification is
still pending.

This records the 2026-10-18 source. The later [TP1/TP2/TP4 source merge](qwenrelay-tp-profiles.md)
retains the TP4 route but requires fresh GPU regression; these results do not certify
that merged tree. The numerical checker now also accepts TP1 geometry.

- Main model and DFlash2 TP=4; DP=PP=PCP=DCP=1.
- GPU KV budget: 14 GiB per card. CPU history budget: 32 GiB per TP group.
- DFlash depth 4; maximum 4 active sequences; prefill chunk budget 2048;
  graph capture ceiling 20; original E2 and O4 settings retained.
- PF2/O2/O5 accept only the original 12Q/2KV or new 6Q/1KV main-model geometry.
  Unsupported geometry, dtypes, layouts and features retain native fallback.
- Scratch allocation, launch grids and O5 gather indexing use the selected head
  geometry. Native fallback still owns decode and unsupported mixed-batch slices.

`tests/gpu_tp_prefill.py` compares selected rows against an independent NumPy
FP64 reference, checks exact INT8 query codes/scales, verifies untouched padding,
and compares O5 fallback slices bit for bit with native vLLM. It includes TP2
regression, TP4 on each GPU, mixed batches and contexts above 128K.

The GPU test runs separately from model serving. The requested serving benchmark
uses one unscored warmup and one formal pass with archived company requests.

## Numerical checks (2026-10-18)

All 23 GPU cases passed using the pinned vLLM/PyTorch/Triton runtime. The largest
absolute error was 0.00068882 and the largest relative L2 error was 0.00241921.
Exact INT8 query codes/scales and bitwise native-fallback checks also passed.
The cases cover TP2 regression, TP4 on all four GPUs, O5 mixed batches with INT8
queries enabled/disabled, and a context longer than 128K. This is a selected-row
independent-reference check, not a model-level quality or full-context accuracy
evaluation. Seven CPU source/profile contract tests passed.

## O4 allocation correction

TP4 reduces each rank's cache page size. The original pinned allocator rounded
39,168,000-byte recurrent-state tensors up to 67,108,864 bytes, causing startup to
correctly reject the measured allocation against the 32 GiB TP-group budget.
The allocator's rounding threshold now follows the smallest planned tensor,
retaining the original 64 MiB threshold for TP2 and choosing 32 MiB for TP4.
Two fresh-process pinned-allocation probes reproduced the rounding and verified
its correction. On the successful TP4 startup, each rank requested and actually
allocated 8,574,745,600 bytes: 34,298,982,400 bytes across four ranks, below the
34,359,738,368-byte limit. The measured-budget guard remains mandatory.

## Serving benchmark (one formal pass)

One unscored pass preceded the formal pass. The warmup additionally replayed one
archived 64K request to cover O2; this request is excluded from scores. Formal
requests retained the company input, ordering, sampling parameters and output
limits. Only the model alias and a unique `cache_salt` changed. Actual HTTP
payload checks passed, with zero cached prompt tokens and zero failed requests.
Prefill requests generate one token; the two real-prompt Decode cohorts retain
their original 1024-token cap and stop behavior.

Units are token/s. Prefill is total input tokens divided by complete case time;
Decode is concurrency divided by mean TPOT. Company values are the first formal
pass. Rental TP2 values are the preceding warmed, fan-target-100% pass.

| Case | Company TP2 | Rental TP2 | Rental TP4 | TP4 vs rental TP2 |
|---|---:|---:|---:|---:|
| 4K Prefill C1 | 1635.1 | 1540.1 | 1754.3 | +13.9% |
| 4K Prefill C4 | 1663.2 | 1544.8 | 1758.9 | +13.9% |
| Default sampling Decode C4 | 363.3 | 325.4 | 332.6 | +2.2% |
| T0 Decode C4 | 374.9 | 334.4 | 357.9 | +7.0% |

All four ranks logged PF2/O2/O5 hits with 6Q/1KV, E2 attention blocks of 800
tokens, and completed CUDA Graph capture. There were nine inference JIT warnings
during warmup and none during the formal pass. Capture used
`FULL_AND_PIECEWISE` with sizes `[1, 2, 4, 5, 8, 10, 15, 16, 20]`.

These results remain thermally constrained. GPU2 reached 89 degrees C and had SW
Thermal Slowdown in every sampled busy interval of the formal cases; its median
SM clock ranged from 1455 to 1552.5 MHz. GPU3 also had thermal slowdown samples.
Fan targets were 100% and the original 350 W power limits were retained. This
machine has no GPU peer-to-peer read/write support across these pairs and uses
PYNCCL communication. TP4 also changes GPU count, shard shapes and the aggregate
KV budget, so these values do not isolate a single optimization's speedup.

The archived run is `qwenrelay-tp4-deployment-20261018T092208Z`. Runtime source-lock
SHA-256 is `e0ba731a6435064fac00ea8cd4384b236f70e8a3601aef855e92a3b29d4496ea`.
The four prefill files are unchanged from the separately tested numerical suite;
the fifth runtime change is the O4 allocator correction described above. The
runtime source and all 18 native-library checks passed. The source lock's pending
GPU-validation status is retained as the pre-qualification metadata; this document
records the completed, limited scope.

## Scope still pending

The full multi-length/concurrency matrix, TP4 O4 cold-restore acceptance, and
per-descriptor CUDA Graph replay observations are outside this serving smoke
test. The original TP2 measurements remain separate from these TP4 results.

## Communication tuning follow-up

A separate matched-temperature TP4 comparison and optional machine-specific NCCL
setting are recorded in [the communication tuning note](qwenrelay-tp4-communication-tuning.md).

A separate [T=0 greedy-output investigation](qwenrelay-tp4-greedy-correctness.md)
records channel/batch numerical effects, ordinary-decoding controls and sampled
worker/API checks. It does not expand the long-context or cold-restore scope above.
