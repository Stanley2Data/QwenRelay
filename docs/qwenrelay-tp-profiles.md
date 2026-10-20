# TP selection and the single-GPU source merge

`scripts/start-qwenrelay.sh` forwards its arguments to `scripts/qwenrelay.py serve`.
The default is **TP1 / AutoRound-fast**. `--tp` selects the starting RTX 3090 defaults,
including checkpoint identities, memory budgets, attention routes and draft settings.
Explicit tuning arguments override the numerical defaults without changing the
selected checkpoints or writing to the profile files.

| Argument | Named profile | Primary checkpoint | Default context ceiling | Default GPU KV per rank | Default CPU KV per TP group |
|---|---|---|---:|---:|---:|
| omitted or `--tp 1` | `single-3090` | AutoRound-fast | 132,096 | 6 GiB | 32 GiB |
| `--tp 2` | `dual-3090` | EfficientThink SimPO AWQ | 262,144 | 11,184,101,204 bytes | 32 GiB |
| `--tp 4` | `quad-3090` | EfficientThink SimPO AWQ | 262,144 | 14 GiB | 32 GiB |

All three default to the pinned syvai DFlash2 W4A16 drafter, depth 4, four running sequences,
2,048 batched tokens, Graph20, BF16 activations and INT8 per-token/head KV. Draft TP
matches primary TP. Context ceilings do not guarantee that four maximum-length
requests fit simultaneously. All profiles target native Linux RTX 3090 24 GB cards.
Default selection is a product choice; it does not expand the GPU validation evidence.

```bash
# Inspect without loading models, CUDA, or the API key.
bash scripts/start-qwenrelay.sh --dry-run
bash scripts/start-qwenrelay.sh --tp 2 --devices 0,1 --dry-run
bash scripts/start-qwenrelay.sh --tp 4 --devices 0,1,2,3 --dry-run

# Start after installing the locked runtime and preparing the selected assets.
bash scripts/start-qwenrelay.sh --tp 1 --devices 0
bash scripts/start-qwenrelay.sh --tp 2 --devices 0,1
bash scripts/start-qwenrelay.sh --tp 4 --devices 0,1,2,3
```

If `--devices` is omitted, the launcher uses `CUDA_VISIBLE_DEVICES` when exported;
otherwise it selects `0`, `0,1`, or `0,1,2,3` for the chosen TP. A conflicting device
count is rejected. The existing `--profile` option is retained. If both selectors
are supplied they must agree, for example `--tp 4 --profile quad-3090`.
`--dry-run` prints the selected profile, TP, model lock, checkpoint paths, explicit
overrides, effective settings and environment. Real startup checks all selected model
hashes and runtime source identities before starting inference. Effective environment values override the parent;
the temporary shared-head optimization is explicitly disabled for TP2/TP4.

## Explicit launch overrides

Use `--max-num-seqs` / `--concurrency`, `--max-model-len` / `--context-length`,
`--draft-tokens` / `--num-speculative-tokens`, `--kv-cache-memory`,
`--kv-offloading-size` / `--cpu-kv-gib`, and `--max-num-batched-tokens`.
GPU KV uses bytes or IEC units per rank; CPU KV uses GiB across the entire TP group.
Zero CPU KV disables that tier/O4 and preserves GPU prefix caching. Omitted options
retain TP defaults. [English details](qwenrelay-install.md#launch-overrides) ·
[中文参数表](qwenrelay-install.zh-CN.md#常用启动参数).

```bash
bash scripts/start-qwenrelay.sh --tp 1 \
  --max-num-seqs 2 --max-model-len 65536 --draft-tokens 6 \
  --kv-cache-memory 5GiB --kv-offloading-size 16 --dry-run
```

This derives logical QMAX `7` and Graph coverage `14`. Generally, QMAX is `depth+1`
and the Graph ceiling is `concurrency*(depth+1)`. The batched-token budget must cover
the product. Depth is limited to 1–63 by the 64-query split-KV limit; custom settings
still require real capacity/performance checks. More than four requests or more than
2,048 query tokens can leave O5's optimized prefill guard and retain native fallback.

## Weight preparation

The AutoRound-fast weights are the upstream HyperQwen variant, assembled from:

- [dbirks base checkpoint](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound/tree/1f05c441c4e64ae0549de44fa9ea5a6d43610314), including the body and tokenizer;
- [syvai fast components](https://huggingface.co/syvai/qwen3.8-27b-3090-fast-variant/tree/124c14e7e8c7d2f5402933b9af368e772a9fcf0c), including the GPTQ head/MTP tensors, fast config/index and vocabulary IDs;
- upstream [embedding quantization](../prepare/quant_embed.py), using the original [quantization helpers](../prepare/quant_schema.py).

The launcher does not apply generic AutoRound conversion to an arbitrary checkpoint.
`locks/models-tp1.json` pins every input revision/hash and the final measured export.
Base shard 6 requires CPU embedding quantization. The remaining primary files come
directly from the locked public repositories. The fast repository already supplies
the head/MTP result, so their intermediate base conversions are unnecessary here.
The standard HyperQwen assembly route is documented in its
[preparation guide](https://github.com/syv-ai/HyperQwen/blob/95ee3b64b0df7014ee2626529524d6b99c3dcdd1/prepare/fetch_fast_variant.py).

```bash
# Default / TP1: downloads public inputs, prepares the embedding on CPU, then
# checks every final hash. Requires the installed runtime Python and free RAM/disk.
.venv/bin/python scripts/qwenrelay.py fetch-models --tp 1 --root ./models
.venv/bin/python scripts/qwenrelay.py check-models --tp 1 --root ./models

# TP2 and TP4 share the original EfficientThink model lock and download.
.venv/bin/python scripts/qwenrelay.py fetch-models --tp 2 --root ./models
.venv/bin/python scripts/qwenrelay.py check-models --tp 4 --root ./models
```

An existing AutoRound-fast directory is verified, never rewritten by this command.
Preparation runs in a new `.qwenrelay-tp1-*` directory, with GPUs hidden from the CPU
subprocess. It publishes the primary directory only after all expected hashes match.
Failed work remains for diagnosis; reruns use a new temporary directory. No models
are downloaded or changed during `serve --dry-run`. Both primary directories can
coexist under one models root alongside the shared DFlash2 directory. Downloaded
weights remain subject to their respective model licenses.

## Reviewed source changes, 2026-10-20

| Change from the TP1 implementation | Integration |
|---|---|
| Split-KV scratch allocation | Prepare at model load using the actual scheduler capacity, common to TP1/2/4. Reuse buffers during V2 forward and capture; reject an undersized existing allocation. |
| PF2/O2 target geometry | Add 24Q/4KV for TP1 and retain 12Q/2KV for TP2 and 6Q/1KV for TP4, all head-dimension 256 and GQA 6. Unsupported shapes retain native fallback. |
| O5 head passes | TP1 processes four KV heads in two passes, TP2 two heads in one pass, TP4 one head in one pass. Query slices retain original strides. |
| Shared workspace capacity | Reinterpret contiguous token/head capacity across PF2/O2/O5. Keep device/stream/target-shape isolation and the 320 MiB O5 arena budget; do not copy the TP1-only cache keys over TP4. |
| Temporary draft output head | TP1 opts into meta initialization, followed by verified aliasing to the real target head with no remaining meta parameters. TP2/TP4 keep this off. |
| DFlash environment | All profiles retain `VLLM_SPEC_DECODE_ATTN=1`, `VLLM_SPEC_DECODE_ATTN_QMAX=5` and `VLLM_DFLASH2_LOOKUP=1`. TP1 also retains the upstream version-check setting from its measured launch. |
| E2, O4 and Graph20 | Already present. Retained with the existing byte budgets and fallback guards. |

The old missing-config scratch path reserves 256 requests. Preallocation with
`max_num_seqs=4` makes this part of the allocation 64 times smaller for the same
geometry. Logical QMAX is 5, but the current GQA=6 kernel tile makes actual scratch
QMAX 10. The TP1 target's scratch is therefore 15,851,520 bytes rather than
1,014,497,280 bytes. This comparison is for that buffer, not total GPU memory.
Attention math, split-KV kernels, PF2 attention math, O2 quantization and O5 attention
math remain unchanged. O5 gather adds a KV-head offset for the second TP1 pass;
TP2/TP4 use offset zero.

The final tree still has 3,208 files, with 15 final overlays. The three base local
patches and their base-tree identities are retained. `locks/source.json` records
the eight changed final-file before/after identities. The [review diff](../patches/qwenrelay/review/0006-tp1-capacity-and-bounded-prefill.patch)
shows this merge against the preceding TP2/TP4 tree and must not be applied again.
Activation INT8, a different
draft checkpoint, FP8 attention, throughput tuning and quality claims were not
adopted from unrelated upstream profiles.

## Evidence and remaining validation

The historical single-GPU F completed a 30-case same-GPU H/F performance comparison,
with one warm and three formal rounds per case. Its weights and resource limits
differ from the original TP2 EfficientThink comparison. Its results do not establish
that fewer cards are intrinsically faster, or that the public merged tree has passed
GPU tests. The latest run covered performance; it did not repeat quality, an independent
cache-restoration battery, vision or a long soak. `C / mean TPOT` remains distinct
from aggregate output throughput, including queued C8/C16 requests under maxseq4.

Local CPU checks cover real Python allocation/dispatch code, actual-capacity reuse
with no forward config, undersized capacity, all three head geometries, arena reuse
and bounds, OOM fallback, head-pass offsets/strides, TP/profile conflicts and checkpoint
selection. Offline reconstruction checks the complete locked source tree. Public
small-file hashes and LFS metadata match the measured assets for directly reused files.
The complete large-file download/CPU embedding conversion and merged GPU startup
have not been rerun on this Mac. Hash mismatches fail instead of relaxing the lock.
The [machine-readable merge checks](qwenrelay-tp-merge-validation.json) record the
source/configuration identities, 22 passing CPU tests, launch plans and public-asset
provenance methods.

Run the numerical checker only on selected idle GPUs in the installed environment:

```bash
.venv/bin/python tests/gpu_tp_prefill.py --devices 0 --output /absolute/path/to/new-result.json
# Or restrict the geometry using --tp 1, --tp 2 or --tp 4.
```

It compares sampled output rows against an independent NumPy FP64 reference and
checks mixed-prefill native fallback slices. Afterward, run real TP1/TP2/TP4 model
startup, Graph replay, generation, quality and cache recovery checks before promoting
these merged sources to a validated public release. Earlier GPU results remain
attached to their original source versions.
