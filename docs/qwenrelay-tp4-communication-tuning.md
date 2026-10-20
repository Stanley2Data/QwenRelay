# TP4 communication tuning on a PCIe / SHM host

Measured on 2026-10-18 with four RTX 3090 GPUs, PCIe Gen3 x16, no usable GPU
peer-to-peer path, NCCL 2.29.7 and the pinned QwenRelay runtime. All GPU pairs
report PHB; communication uses NCCL shared memory through host memory. The
portable `quad-3090` profile keeps automatic NCCL defaults. The settings below
are an optional, measured override for this topology.

## Selected setting

```bash
NCCL_MIN_CTAS=4 NCCL_MAX_CTAS=4 \
  bash scripts/start-qwenrelay.sh --profile quad-3090 --devices 0,1,2,3
```

This requests four communication CTAs/channels. Very small messages can still
use fewer channels according to NCCL's implementation. Algorithm, protocol and
thread selection remain automatic. The main model and DFlash2 remain TP4,
DP/PP/PCP/DCP remain 1, GPU KV remains 14 GiB/card, and CPU history remains
32 GiB/TP group. PF2/O2/O5, E2/O4, W4A16/BF16, DFlash4 and CUDA Graph settings
are unchanged. No runtime source or native library changed for this tuning.

## Selection and serving result

Eighteen communication configurations were checked using the installed vLLM
PyNcclCommunicator, including Ring/Tree, Simple, 4/8/16 CTAs, thread counts and
SHM copy-engine/locality variants. Every tested configuration passed exact
integer-pattern AllReduce and AllGather checks in eager mode and CUDA Graph
replay. The selected setting balanced small Decode messages and large Prefill
messages; more channels and SHM copy-engine variants did not improve that
balance on this machine.

Both serving configurations used one unscored warmup pass and one formal pass
over the same archived company requests. Each formal case started with all
GPUs at or below 45°C and no active SW thermal flag. Power limits stayed at
350 W/card and fan targets at 100%. Prompts, request ordering, sampling, output
caps and EOS handling were preserved; only the model alias and unique cache
salt changed. Prefill emits one token/request; Decode keeps the 1024-token cap.
Actual HTTP payloads were checked, with zero cached prompt tokens.

Prefill is total input tokens / full case duration; Decode is concurrency /
mean TPOT. Units: token/s.

| Case | Stock TP4 | Tuned TP4 | Change |
|---|---:|---:|---:|
| 4K Prefill C1 | 1858.6 | 1975.6 | +6.30% |
| 4K Prefill C4 | 1895.5 | 2013.4 | +6.22% |
| Decode C4, original sampling | 380.5 | 394.2 | +3.60% |
| Decode C4, T=0 | 402.2 | 394.1 | -2.00% |

These are single-pass observations without confidence intervals. The baseline
is a new TP4 measurement under the same cooling gate; the earlier hot TP4 and
TP2 scores are not used to attribute this gain. GPU temperatures and boost can
still diverge during a case, and generation trajectories can vary.

The setting benefits Prefill and original-sampling Decode in this pass, but
T=0 Decode regresses by 2.0%. T=0 output tokens per draft fall from
3.0225 to 2.9206; drafts increase from
2578 to 2670. Both T=0 runs have zero sampled SW thermal
slowdown, with GPU2 median SM clock of 1920 MHz. All eight greedy outputs diverge
from baseline, so these runs do not establish bitwise model-output equivalence.
The subsequent [greedy-output investigation](qwenrelay-tp4-greedy-correctness.md)
directly reproduced channel-dependent BF16 rounding and nearly tied logit flips,
including ordinary-decoding controls. Sampler/API checks found no correctness
error in that input set. Cross-configuration bitwise model-output equivalence is
not claimed. The observed T=0 performance regression remains; its changed
generation and acceptance trajectories preclude attributing the whole difference
to communication speed.

T=0 generated-text matches: 0/8; total output tokens: 7792 / 7798.

Formal requests passed: baseline 40/40, tuned 40/40. Formal JIT warning counts: baseline 0, tuned 0.

The no-P2P transport remains a limitation. This setting neither enables NVLink
nor removes the model's collectives. A different topology, NCCL version or
workload requires its own comparison. Full TP4 qualification remains scoped by
[the TP4 validation record](qwenrelay-tp4-validation.md).

The full evidence run is `qwenrelay-tp4-communication-opt-20261018T104232Z`: per-rank communication samples,
request/result records, telemetry, source identity, output hashes and selected
service variables. Runtime source-lock SHA-256 remains
`e0ba731a6435064fac00ea8cd4384b236f70e8a3601aef855e92a3b29d4496ea`.

References: [NCCL environment controls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
and [NCCL 2.29.7 SHM transport source](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/transport/shm.cc).
