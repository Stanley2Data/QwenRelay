# TP4 greedy-output investigation

Targeted investigation completed on 2026-10-18, using the pinned runtime,
four RTX 3090 GPUs and TP4 / DP1 throughout. The observed T=0 output changes
are consistent with channel-dependent BF16 reduction rounding and batch-dependent
numerical paths, amplified when nearly tied logits change order. The checks below
found no corrupt NCCL result, non-argmax greedy selection, request mix-up or
incorrectly accepted draft token in this input set.

## Matched inputs and channel controls

The original eight company T=0 requests were replayed with their prompts,
repetition penalty 1, 1024-token output cap and EOS behavior preserved. The
served alias and unique cache salt follow the earlier benchmark; response-only
token-ID reporting was added. C4 preserves the original concurrency and C1 is
an auxiliary serial control. Every replay passed with zero cached prompt tokens.

The new two-CTA C4 outputs match the archived automatic-setting outputs 8/8
as text; new four-CTA outputs match the archived tuned outputs 8/8 as text.
Explicit two CTAs reproduced the earlier automatic baseline for these requests.
Algorithm, protocol and thread selection stayed automatic. Very small NCCL
messages can use fewer channels than the configured CTA bounds.

With four CTAs unchanged, serial repeats match 8/8 token for token; C4 repeats
match 5/8. C1 versus C4 matches 0/8. Thus batch composition is an independent
source of output variation; a whole-text C4 comparison alone cannot isolate its
contribution from channel tuning.

| Two vs four CTA controls | Identical full token sequences | Top-1/top-2 gap at first divergence |
|---|---:|---:|
| DFlash on, C1 | 0/8 | 0–0.25 |
| DFlash on, C4 | 0/8 | 0–0.25 |
| DFlash off, C1 | 0/8 | 0–0.125 |
| DFlash off, C4 | 1/8 | 0–0.25 |

At every compared first divergence, prompt and generated prefix are identical.
Both selected tokens are within the other's top three, no more than 0.25 below
its maximum; the few cases outside top two involve tied scores. Each path
selects its own maximum. T=0 removes sampling noise; it does not make different
floating-point computation paths bitwise identical. The main and draft T=0
branches do not add Gumbel noise despite the profile's probabilistic draft option.

## Fixed-context and ordinary-decoding controls

Thirty-two actual generated prefixes were frozen as token IDs, verified by
SHA-256 and used for one-token predictions. This separates next-token changes
from later autoregressive branching. `spec` enables DFlash; `plain` disables it;
the suffix denotes the requested CTA count.

| Fixed-context comparison | Different next tokens |
|---|---:|
| spec2 vs spec4 | 4/32 |
| plain2 vs plain4 | 4/32 |
| spec2 vs plain2 | 4/32 |
| spec4 vs plain4 | 4/32 |

The no-draft controls removed speculative configuration and used graph ceiling
4 instead of 20. They explicitly retained the original 800-token cache span,
20,800-token retention, 14 GiB/card GPU KV, 32 GiB/TP-group CPU history and
PF2/O2/O5/E2/O4 settings. Without the explicit span, automatic no-draft sizing
selected 752 tokens and the O4 alignment guard correctly rejected startup;
the guard and budget checks were retained. No-draft state and zero draft-counter
increments were checked on all four workers. These are correctness controls,
with observation overhead and no performance score; ordinary-decoding stages
did not wait for cooling.

## Numerical and sampler evidence

The installed vLLM PyNcclCommunicator reduced identical BF16 inputs at shapes
20×5120 and 1600×5120. Each setting was bitwise repeatable across three eager
calls and three CUDA Graph replays; all ranks agreed. Changing automatic to four
CTAs changed 27.8496% / 32.0401% of elements, with maximum absolute difference
0.0625. Relative L2 errors against an independent CPU/Gloo FP64 sum were about
0.00268 in both settings. At 32,768 sampled positions, both outputs were exactly
explained by enumerating all 24 serial BF16 addition orders with rounding after
each addition. This supports ordinary reduction rounding rather than arbitrary
transport corruption.

A temporary worker extension observed logits and emitted IDs on GPU, copying
records to CPU only at idle. Its serial four-CTA run matched the unobserved run
8/8. Across the eight audited stages:

- All 62,327 API tokens matched worker emissions and positions.
- All 62,453 raw greedy emissions per TP replica were argmax choices;
  no NaN logit rows were found.
- All four ranks agreed bit for bit on IDs, positions, top-four logits, counts
  and request mappings. API tie choices followed the lowest maximum token ID.

These checks establish the stated scope, not a full forward-model FP64 reference
or model-quality certification. Bitwise equality across channel, batch and
ordinary/speculative paths is not promised. Full long-context, cold-restore and
other pending TP4 acceptance items remain in the
[validation record](qwenrelay-tp4-validation.md).

The original four-CTA TP4/DP1 DFlash4 service was restored and checked healthy;
the temporary worker extension, development RPC environment and systemd override
were removed. Runtime sources and the source lock were unchanged:
`e0ba731a6435064fac00ea8cd4384b236f70e8a3601aef855e92a3b29d4496ea`.
Evidence batch: `qwenrelay-t0-correctness-20261018T120402Z`. No private prompts or generated text are included
in this public note.

References: [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html),
[NCCL CTA controls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).
