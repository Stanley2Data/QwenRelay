# QwenRelay

**Qwen inference for two RTX 3090s: long context, small concurrent batches, and reusable conversation caches.**

[简体中文](README.zh-CN.md) · [Install](docs/qwenrelay-install.md) · [Benchmarks](docs/qwenrelay-benchmarks.md) · [Upstream & attribution](UPSTREAM.md)

QwenRelay is a source distribution derived from [HyperQwen](https://github.com/syv-ai/HyperQwen), pinned to vLLM 0.31.0. Version 0.1 focuses on the tested **2 × RTX 3090 24 GB, TP2, DFlash2 depth 4** configuration. It adds aligned INT8 KV storage, budgeted CPU history caches, INT8-KV prefill optimizations, cache-restore fixes, and complete four-request CUDA Graph coverage.

This is an early source release candidate. The underlying TP2 runtime has GPU benchmark and cache-restore evidence. **This newly packaged install/Docker path has offline checks; a fresh Docker build and model run have not been repeated for this package.** See [validation boundaries](docs/qwenrelay-validation.md).

## Supported first profile

| Item | Configuration |
|---|---|
| Platform | Native Linux, Python 3.12; two 24 GB RTX 3090s |
| Runtime | vLLM 0.31.0, PyTorch 2.13.0 + CUDA 13.0, Triton 3.7.1 |
| Main model | Pinned EfficientThink SimPO mixed AWQ W4A16 |
| Draft | Pinned syvai DFlash2 W4A16; 4 speculative tokens |
| Scheduler | Maximum 4 running sequences; 2,048 batched tokens |
| Context | Configured maximum 262,144 tokens per request, subject to available cache |
| Cache | INT8 per-token/head KV, FP16 recurrent state, 32 GiB host offload budget |
| Activations | BF16; `INT8_ACT` disabled |

The context limit is a per-request ceiling. It does not promise four resident 256K conversations. Single-card, vision, other model checkpoints and high-concurrency batch serving remain outside this first profile.

## Quick start: native Linux

```bash
git clone https://github.com/Stanley2Data/QwenRelay.git
cd QwenRelay
```

From the repository directory, with a compatible NVIDIA driver, a CUDA-capable Linux venv and GNU `patch`/`git` available:

```bash
# Installs into a NEW .venv, fetches only pinned runtime source, starts no model.
bash scripts/install-qwenrelay.sh

# Explicit download of pinned assets (~21 GiB). No model weights live in Git.
.venv/bin/python scripts/qwenrelay.py fetch-models --root ./models

# Keep this local key file out of Git.
umask 077
.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))' > api_key.txt

# Print the launch plan without loading weights or probing GPUs.
bash scripts/start-qwenrelay.sh --devices 0,1 --dry-run

# Validates asset/runtime identities, then starts the server.
bash scripts/start-qwenrelay.sh --devices 0,1
```

The API defaults to `http://127.0.0.1:18020/v1`, served model name `qwenrelay`, with the key from `api_key.txt`. Use your OpenAI-compatible client's API-key setting. For Docker instructions, RAM/driver requirements and offline source reconstruction, see [installation](docs/qwenrelay-install.md).

## What changed

- **E2:** 16-byte-aligned INT8 KV head storage and compatible hybrid page geometry.
- **O4:** byte-budgeted full-attention / recurrent / draft history pools, with aligned handoff and CPU-cache recovery. Based in part on 1Cat's cache-design discussion; see [credits](THIRD_PARTY_NOTICES.md).
- **PF2 / O2 / O5:** reuse INT8 KV during prefill; O5 serves eligible variable-length mixed prefill batches. These are bounded TP2 paths with native fallbacks, not pure-decode accelerators.
- **CUDA Graph integration:** capture fixed decode request/query products through 20 tokens, including 4 requests × (4 draft + 1 verification); preserve compiler-cache identity and buffer ownership.

The exact source identities and six-file final overlay are in [locks/source.json](locks/source.json). Reviewable deltas are in [patches/qwenrelay](patches/qwenrelay).

## Measured results

Same weights, GPUs, runtime/native libraries and byte budgets; one warm round and three formal rounds per arm. Values below are **medians of `C / mean TPOT`**, not total output throughput.

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
