# QwenRelay

**Qwen inference for RTX 3090s: long context, small concurrent batches, and reusable conversation caches.**

[简体中文](README.zh-CN.md) · [Install](docs/qwenrelay-install.md) · [Benchmarks](docs/qwenrelay-benchmarks.md) · [Upstream & attribution](UPSTREAM.md)

QwenRelay is a source distribution derived from [HyperQwen](https://github.com/syv-ai/HyperQwen), pinned to vLLM 0.31.0. The default is **one RTX 3090 24 GB, TP1, AutoRound-fast and DFlash2 depth 4**. `--tp 2` and `--tp 4` select EfficientThink and their corresponding optimization profiles. It adds aligned INT8 KV storage, budgeted CPU history caches, INT8-KV prefill optimizations, cache-restore fixes, and complete four-request CUDA Graph coverage.

This is an early source release candidate. The underlying TP2 runtime has GPU benchmark and cache-restore evidence. **The merged TP1/TP2/TP4 source has offline checks; its full asset preparation, fresh Docker build and GPU runs remain pending.** See [validation boundaries](docs/qwenrelay-validation.md).

## Default profile

| Item | Configuration |
|---|---|
| Platform | Native Linux, Python 3.12; one 24 GB RTX 3090 |
| Runtime | vLLM 0.31.0, PyTorch 2.13.0 + CUDA 13.0, Triton 3.7.1 |
| Main model | Pinned HyperQwen AutoRound-fast W4A16 |
| Draft | Pinned syvai DFlash2 W4A16; 4 speculative tokens |
| Scheduler | Maximum 4 running sequences; 2,048 batched tokens |
| Context | Configured maximum 132,096 tokens per request, subject to available cache |
| Cache | INT8 per-token/head KV, FP16 recurrent state, 32 GiB host offload budget |
| Activations | BF16; `INT8_ACT` disabled |

The context limit is a per-request ceiling, subject to the 6 GiB GPU KV budget. It does not promise four resident maximum-length conversations. Vision, arbitrary checkpoints and high-concurrency batch serving remain outside these profiles.

## Quick start: native Linux

```bash
git clone https://github.com/Stanley2Data/QwenRelay.git
cd QwenRelay
```

From the repository directory, with a compatible NVIDIA driver, a CUDA-capable Linux venv and GNU `patch`/`git` available:

```bash
# Installs into a NEW .venv, fetches only pinned runtime source, starts no model.
bash scripts/install-qwenrelay.sh

# Download pinned upstream AutoRound-fast inputs and prepare the embedding on CPU.
# Final asset hashes are checked; no weights live in Git.
.venv/bin/python scripts/qwenrelay.py fetch-models --root ./models

# Keep this local key file out of Git.
umask 077
.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))' > api_key.txt

# Print the launch plan without loading weights or probing GPUs.
bash scripts/start-qwenrelay.sh --devices 0 --dry-run

# Validates asset/runtime identities, then starts the server.
bash scripts/start-qwenrelay.sh --devices 0
```

The API defaults to `http://127.0.0.1:18020/v1`, served model name `qwenrelay`, with the key from `api_key.txt`. Use your OpenAI-compatible client's API-key setting. For Docker instructions, RAM/driver requirements and offline source reconstruction, see [installation](docs/qwenrelay-install.md).

At the default depth 4 the launcher supplies these **DFlash performance settings**;
overriding depth recalculates QMAX:

```ini
VLLM_SPEC_DECODE_ATTN=1
VLLM_SPEC_DECODE_ATTN_QMAX=5
VLLM_DFLASH2_LOOKUP=1
```

An earlier standalone launcher omitted these inherited settings. A running DFlash model alone does not confirm that split-KV verification attention and lookup drafting are enabled. Keep all three when starting vLLM directly or writing a custom service; `5` corresponds to 4 draft tokens plus 1 verification token per request. See [DFlash launch settings and verification](docs/qwenrelay-install.md#dflash-launch-settings).

Choose another TP size explicitly (default: TP1):

```bash
# TP2/TP4 use the same EfficientThink assets; download them once.
.venv/bin/python scripts/qwenrelay.py fetch-models --tp 2 --root ./models
bash scripts/start-qwenrelay.sh --tp 2 --devices 0,1
bash scripts/start-qwenrelay.sh --tp 4 --devices 0,1,2,3
```

By default TP1 selects AutoRound-fast, 132,096 context and 6 GiB GPU KV. TP2 selects EfficientThink,
262,144 context and 11,184,101,204 bytes GPU KV per card. TP4 selects EfficientThink,
262,144 context and 14 GiB per card. CPU KV is 32 GiB per TP group. The existing
`--profile` selector is retained. See [profile selection, upstream weight preparation
and merge evidence](docs/qwenrelay-tp-profiles.md), plus [historical TP4 validation](docs/qwenrelay-tp4-validation.md).

TP chooses defaults; explicit arguments override concurrency, context, speculative
depth and cache budgets:

```bash
bash scripts/start-qwenrelay.sh --tp 1 \
  --max-num-seqs 2 --max-model-len 65536 --draft-tokens 6 \
  --kv-cache-memory 5GiB --kv-offloading-size 16 --dry-run
```

This derives logical QMAX `7` and Graph coverage `2 * (6 + 1) = 14`. GPU KV is per
rank; CPU KV is summed across the TP group. `--kv-offloading-size 0` disables CPU
offload. See [launch overrides](docs/qwenrelay-install.md#launch-overrides) for all
flags, aliases and checks. A dry run inspects configuration; changed settings still
require actual capacity/performance validation.

## What changed

- **E2:** 16-byte-aligned INT8 KV head storage and compatible hybrid page geometry.
- **O4:** byte-budgeted full-attention / recurrent / draft history pools, with aligned handoff and CPU-cache recovery. Based in part on 1Cat's cache-design discussion; see [credits](THIRD_PARTY_NOTICES.md).
- **PF2 / O2 / O5:** reuse INT8 KV during prefill; O5 serves eligible variable-length mixed prefill batches. Bounded TP1 (24 Q / 4 KV), TP2 (12 Q / 2 KV) and TP4 (6 Q / 1 KV) paths retain native fallbacks. TP1 reuses O5 storage in two head passes. These are not pure-decode accelerators.
- **CUDA Graph integration:** default coverage through 20 tokens, including 4 requests × (4 draft + 1 verification); derive the ceiling again when concurrency/depth changes, preserving compiler-cache identity and buffer ownership.

The exact source identities and fifteen-file final overlay are in [locks/source.json](locks/source.json). The merge preserves TP2/TP4 routes, preallocates split-KV scratch for the actual scheduler capacity, and opts TP1 into verified draft-head sharing. An O4 overlay retains measured byte-budget enforcement. Reviewable base deltas are in [patches/qwenrelay](patches/qwenrelay).

## Measured results

Historical **TP2 EfficientThink** comparison, before the current merge. Same weights, GPUs, runtime/native libraries and byte budgets; one warm round and three formal rounds per arm. Values below are **medians of `C / mean TPOT`**, not total output throughput.

| Original real-question workload | HyperQwen reference H | QwenRelay candidate F |
|---|---:|---:|
| C4, temperature 0 | 113.19 tok/s | 374.92 tok/s |
| C4, default sampling | 107.50 tok/s | 360.11 tok/s |
| 1K synthetic input, C4, T0, 1,024 output | 167.46 tok/s | 538.45 tok/s |

There are also regressions and limits: original T0 C2 −2.63%, default C1 −1.85%; the synthetic 1K C1 result is highly variable. At 128K C4 neither arm produced a qualifying four-request steady generation interval. The [full benchmark notes](docs/qwenrelay-benchmarks.md) include aggregate throughput, all ten context/decode cells, methodology and cache evidence boundaries.

## Development and credits

```bash
python3 scripts/qwenrelay.py check-pack
python3 -m unittest discover -s tests -v
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing runtime pins or profiles. Apache-2.0 source; model and dependency licenses apply separately. HyperQwen, vLLM, KVarN and model-author notices are preserved. The upstream scripts, batch paths and [original README](README.hyperqwen.md) remain for provenance and development; their settings/results are not QwenRelay's first supported profile. No upstream Docker image or CI badge represents a QwenRelay release.
