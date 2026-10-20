# Install QwenRelay 0.1

[中文](qwenrelay-install.zh-CN.md) · [README](../README.md)

## Requirements and scope

Profiles target native Linux x86-64, Python 3.12 and RTX 3090 24 GB GPUs. The default is one card / AutoRound-fast; `--tp 2` and `--tp 4` select EfficientThink. See [profile selection and merge validation](qwenrelay-tp-profiles.md). The recorded benchmark host used NVIDIA driver 580.126.20 and CUDA 13.0 userspace. Use a driver compatible with that userspace; `libcuda.so.1` comes from the host driver/container runtime, not this source repo. Have a working NVIDIA Container Toolkit when using Docker. Do not install a GPU driver from this project.

Allow at least 64 GiB available system RAM for model loading plus the 32 GiB host-cache budget; more headroom is useful during compilation. Allow roughly 60 GiB free disk for models, runtime dependencies, source and compilation caches. These are planning allowances, not measured minimums. GPU KV is 6 GiB for TP1, 11,184,101,204 bytes per rank for TP2, and 14 GiB per rank for TP4. Profile changes need separate capacity/accuracy/performance verification.

## Native installation

Install Python 3.12 with venv/dev support, a C/C++ toolchain, Git, GNU patch, curl and CA certificates using your distribution. Do not reuse a model-serving venv.

```bash
bash scripts/install-qwenrelay.sh
```

The installer refuses an existing `.venv`, installs the direct dependency pins in `requirements-qwenrelay.txt`, retrieves vLLM commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`, reconstructs the exact final F Python tree, and overlays it into the new venv while preserving the wheel's native extensions. It validates the source and native files. It starts no model and downloads no model weights.

Optional inputs: `PYTHON`, `QWENRELAY_VENV`, `QWENRELAY_BUILD`, and `VLLM_SOURCE` for an existing pristine checkout. The output build directory must be new. Transitive dependencies and OS package versions are resolved by pip/apt, so this is a pinned source/ABI recipe, not a hermetic dependency lock or byte-identical native build. Preserve `pip freeze` and native-file hashes when reproducing results.

The installer checks Linux x86-64, Python 3.12, Git, GNU patch and C/C++ tools
before installing packages. Existing venv/source/output paths fail early. After an
interrupted install, choose new venv/build directories; it does not resume or
delete a partially installed environment.

### Custom paths

Use this instead of the default install, in the same shell:

```bash
export QWENRELAY_VENV="$HOME/.local/share/qwenrelay/venv"
export QWENRELAY_BUILD="$HOME/.cache/qwenrelay/build"
export MODELS_ROOT="$HOME/models/qwenrelay"
export HF_HOME="$HOME/.cache/qwenrelay/huggingface"
bash scripts/install-qwenrelay.sh
"$QWENRELAY_VENV/bin/python" scripts/qwenrelay.py fetch-models --tp 1
bash scripts/start-qwenrelay.sh --tp 1 --dry-run
```

Create the API key as below before actual startup. Replace `.venv/bin/python` in
later commands with `"$QWENRELAY_VENV/bin/python"`. The shell launcher chooses
`QWENRELAY_PYTHON`, then `$QWENRELAY_VENV/bin/python`, then the repository's
`.venv/bin/python`. Re-export custom variables in a new terminal; prefer absolute
paths. `fetch-models` / `check-models --root` and `serve --models-root` override
`MODELS_ROOT`; otherwise all three use it.

## Model assets

```bash
.venv/bin/python scripts/qwenrelay.py fetch-models --root ./models
```

The default downloads the upstream HyperQwen AutoRound-fast inputs at fixed revisions,
prepares the embedding on CPU, and verifies the exact measured final hashes from
`locks/models-tp1.json`. Public sources are the
[dbirks base](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound/tree/1f05c441c4e64ae0549de44fa9ea5a6d43610314),
[syvai fast components](https://huggingface.co/syvai/qwen3.8-27b-3090-fast-variant/tree/124c14e7e8c7d2f5402933b9af368e772a9fcf0c)
and [syvai DFlash2 W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16/tree/4d30ec736ffc6b8688dc2ae2b502d9b48bdec279).
Existing primary directories are checked rather than modified. Downloaded Hugging Face
cache files can be reused. The full large-file preparation has not been rerun in this
packaging step; a mismatch fails instead of changing the lock.

The Hub cache and final model files both use disk space; `HF_HOME` changes the
cache location. CPU embedding preparation also uses temporary input/output files.
On failure, inspect `.qwenrelay-tp1-*` staging directories before retrying;
failed staging is preserved for diagnosis.

TP2/TP4 use the original `locks/models.json`, with
[EfficientThink SimPO](https://huggingface.co/nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2/tree/882e1af3b8f5844f3b3163b132e343c06e91908f/NVFP4/AWQ-W4A16)
and the same DFlash2 drafter. Its primary uses 367 W4 asymmetric and 33 W8 symmetric
target entries. All profiles keep BF16 activations and INT8 KV; none selects an FP8
draft. Weights are downloaded from their authors under their respective licenses.
Git contains only the necessary metadata and source, including the fixed serving
chat template. Language-model-only serving remains enabled.

```bash
# Explicit TP1 is equivalent to omitting --tp.
.venv/bin/python scripts/qwenrelay.py check-models --tp 1 --root /absolute/path/to/models
# Download EfficientThink once for TP2 and TP4.
.venv/bin/python scripts/qwenrelay.py fetch-models --tp 2 --root ./models
.venv/bin/python scripts/qwenrelay.py check-models --tp 4 --root ./models
```

TP1 expects `Qwen3.8-27B-W4A16-AutoRound-fast/`; TP2/TP4 expect
`Qwen3.8-27B-EfficientThink-AWQ-W4A16/`. Both use `Qwen3.8-27B-DFlash2-W4A16/`.
These directories may coexist. Startup reads and checks all selected asset hashes,
which can take time for large weights. Preparation details are in the
[TP profile guide](qwenrelay-tp-profiles.md#weight-preparation).

## Start and inspect

```bash
umask 077
.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))' > api_key.txt
bash scripts/start-qwenrelay.sh --models-root ./models --devices 0 --dry-run
bash scripts/start-qwenrelay.sh --models-root ./models --devices 0
```

Default TP1 uses one GPU. Pass `--tp 2 --devices 0,1` or `--tp 4 --devices 0,1,2,3` to select the corresponding multi-GPU profile. The default bind is loopback and port 18020; use `--port`, `--host`, `--api-key-file` or their documented environment equivalents in `.env.example` when needed. The native launcher does not source `.env` as shell code; export variables or pass flags. `--dry-run` prints the planned nonsecret configuration without checking installed runtime/models or touching CUDA. It is not a deployment validation.

`config/single-3090.json`, `dual-3090.json` and `quad-3090.json` define the three resource/optimization profiles. Model paths, device indices, API key and bind/port are runtime inputs. The launcher clears inherited `INT8_ACT`, `VLLM_MARLIN_INPUT_DTYPE` and the alternative upstream `VLLM_PREFILL_ATTN` route, and fixes the sampler/V2-runner flags to the tested path. Other profiles/legacy upstream launchers do not implement the supported QwenRelay configuration.

## Launch overrides

The launcher reads TP/profile defaults, then applies explicit arguments. Omitted
values keep their selected defaults; JSON profiles are not rewritten. Checkpoint
selection stays TP1 → AutoRound-fast and TP2/TP4 → EfficientThink.

| Argument | Alias | Units / behavior |
|---|---|---|
| `--max-num-seqs` | `--concurrency` | Positive maximum running request count, not client queue length. Default 4. |
| `--max-model-len` | `--context-length` | Total context tokens per request, including input and output; must exceed draft depth. |
| `--draft-tokens` | `--num-speculative-tokens` | DFlash verification depth, integer 1–63. Default 4. |
| `--kv-cache-memory` | — | GPU KV budget per TP rank: exact bytes or an IEC size such as `5GiB`, `512MiB`, `3.5GiB`. Ambiguous `5G` / `5GB` are rejected. |
| `--kv-offloading-size` | `--cpu-kv-gib` | CPU KV GiB summed across the TP group; decimals accepted. Default 32. Zero disables CPU offload/O4 and keeps GPU prefix caching. |
| `--max-num-batched-tokens` | — | Positive scheduler token budget per step. Default 2,048. |

GPU KV must resolve to positive whole bytes. CPU KV must be finite and nonnegative;
negative values, NaN and Infinity are rejected. Given concurrency `C` and depth `K`,
the launcher derives logical `VLLM_SPEC_DECODE_ATTN_QMAX=K+1` and
`max_cudagraph_capture_size=C*(K+1)`. The batched-token budget must be at least that
product; incompatible combinations fail before key access, model loading or CUDA.
Scratch can round the logical QMAX up to a kernel tile. Graph coverage does not
increase KV capacity or alter the draft model's trained block length. When depth
exceeds that block, the existing lookup path attempts to fill extra verify positions.

The cache page size also depends on depth. The launcher derives `--block-size` for
nondefault depths and rounds the 20,800-token retention target up to an aligned
`--prefix-cache-retention-interval`. Depth 6 uses 816 / 21,216 tokens; default
depth 4 preserves 800 / 20,800. Check `hybrid_block_tokens` and
`prefix_cache_retention_interval` in `effective_settings`. This geometry is for
the locked Qwen models, INT8 PAD16 KV and FP16 recurrent state only.
The public GPU budget flag maps to vLLM's `--kv-cache-memory-bytes`.

```bash
# TP1 overrides: logical QMAX=7, Graph=14.
bash scripts/start-qwenrelay.sh --tp 1 \
  --max-num-seqs 2 --max-model-len 65536 --draft-tokens 6 \
  --kv-cache-memory 5GiB --kv-offloading-size 16 --dry-run

# TP2 with larger concurrency; omitted values keep TP2 defaults.
bash scripts/start-qwenrelay.sh --tp 2 --devices 0,1 \
  --concurrency 8 --context-length 65536 --draft-tokens 7 \
  --max-num-batched-tokens 1024 --dry-run

# Disable CPU offload; keep the other TP1 defaults.
bash scripts/start-qwenrelay.sh --kv-offloading-size 0 --dry-run
bash scripts/start-qwenrelay.sh --help
```

Inspect `overrides`, `effective_settings`, `environment` and the final `argv`.
Remove `--dry-run` to start the server. Larger context does not allocate extra
physical KV automatically; larger concurrency/depth costs scratch/Graph memory.
Runtime model-limit and available-memory checks still apply. Custom settings do
not inherit performance evidence from the default configuration. O5 currently
optimizes eligible prefill batches of at most 4 requests and 2,048 query tokens;
unsupported shapes retain native fallback rather than guaranteed acceleration.

Docker uses the same entry point. Add arguments to the service's Compose `command`
list, for example `['--tp', '1', '--max-num-seqs', '2', '--draft-tokens', '6']`,
then recreate that container. Native launchers do not read `.env`; pass these
overrides through the command line.

## DFlash launch settings

The `single-3090`, `dual-3090` and `quad-3090` profiles all explicitly include:

```ini
VLLM_SPEC_DECODE_ATTN=1
VLLM_SPEC_DECODE_ATTN_QMAX=5
VLLM_DFLASH2_LOOKUP=1
```

These configure this project's patched vLLM runtime, in addition to `--speculative-config` selecting the DFlash method and draft model. The earlier standalone launcher omitted the inherited settings; the omission caused a long-context performance regression even though DFlash itself still ran. Installing the source patches and matching model hashes does not establish an equivalent active configuration.

| Setting | Purpose in the shipped depth-4 profiles |
| --- | --- |
| `VLLM_SPEC_DECODE_ATTN=1` | Enables the eligible split-KV speculative verification attention path. Without it, verification bypasses this optimization. |
| `VLLM_SPEC_DECODE_ATTN_QMAX=5` | Sets the logical per-request query bound: 4 draft tokens + 1 verification token. GQA=6 rounds the actual scratch tile to 10 query tokens; allocation uses the real scheduler capacity at model load. Stable buffers matter for CUDA Graph capture. This is not the context length or the combined query count across concurrent requests. |
| `VLLM_DFLASH2_LOOKUP=1` | Enables context-lookup-assisted draft proposals. Omitting it disables this assistance, not the entire DFlash model. |

Use the literal value `1` for the two boolean switches; this runtime does not interpret `true` as enabled. The table's `5` is the depth-4 default; `--draft-tokens` derives QMAX and Graph coverage together. Direct vLLM launchers must keep them consistent themselves. Changed settings require separate validation.

`scripts/start-qwenrelay.sh` and `scripts/qwenrelay.py serve` inject the effective values automatically; no extra exports are needed. Explicit CLI arguments override profile defaults, and those effective values override same-name parent environment variables. Use the tuning flags rather than an exported QMAX. The project Docker recipe uses the same launcher.

If you bypass the launcher and invoke the patched vLLM directly, export these variables in the launching shell before starting Python. For a custom systemd unit that directly starts vLLM, the equivalent is:

```ini
[Service]
Environment="VLLM_SPEC_DECODE_ATTN=1"
Environment="VLLM_SPEC_DECODE_ATTN_QMAX=5"
Environment="VLLM_DFLASH2_LOOKUP=1"
```

Apply a changed unit with `systemctl daemon-reload` and restart that service; editing a file or a separate shell does not update an existing process. Merely writing these lines into `.env` does not configure the native launcher, which does not source that file.

Before a normal launch, inspect the plan:

```bash
bash scripts/start-qwenrelay.sh --devices 0 --dry-run
```

Check all three entries in `environment`, and check that the `argv` speculative configuration contains `method: dflash`, the selected `num_speculative_tokens` (default 4), `draft_sample_method: probabilistic` and the intended DFlash2 model path. With overrides also check the QMAX/Graph values in `effective_settings`. For TP4, use `--profile quad-3090 --devices 0,1,2,3`. A dry run confirms the plan only; after starting the real service, also inspect its process environment, startup configuration and nonzero `spec_decode` counters in `/metrics` after generation requests. Draft counters show speculative decoding activity, not proof that every request shape used the split-KV path.

## Docker recipe
The repository includes a **local build recipe**, using the same launcher and profiles. No host Python/vLLM installation is needed: use the built image to prepare weights and the key. Run from the repository directory:

```bash
docker build -t qwenrelay:local .
mkdir -p models
cp .env.example .env
# Edit .env: set MODEL_ASSET_DIR to the absolute path of ./models.
# TP1: TP=1 and CUDA_VISIBLE_DEVICES=0
# TP2: TP=2 and CUDA_VISIBLE_DEVICES=0,1
# TP4: TP=4 and CUDA_VISIBLE_DEVICES=0,1,2,3

# CPU-only preparation; match --tp to .env (2 and 4 share weights).
docker run --rm --entrypoint /app/.venv/bin/python \
  --mount "type=bind,src=$PWD/models,dst=/models" \
  --mount type=volume,src=qwenrelay-download-cache,dst=/cache \
  qwenrelay:local scripts/qwenrelay.py fetch-models --tp 1

# Create once; preserve an existing key.
umask 077
if [ ! -e api_key.txt ]; then
  docker run --rm --entrypoint /app/.venv/bin/python qwenrelay:local \
    -c 'import secrets; print(secrets.token_urlsafe(32))' > api_key.txt
fi
docker compose up -d
docker compose logs -f qwenrelay
```

Compose mounts models read-only, mounts the API key as a secret, retains compilation caches in a volume and allocates 40 GiB shared memory. The host API port is bound to `127.0.0.1`. To override tuning parameters, edit the service's `command` list and recreate the container.

The download container mounts models **read-write**; the serving container mounts them **read-only**. O4 allocates CPU cache in pinned host RAM, not a 32 GiB `/dev/shm` file. The shared-memory allowance supports container communication and is not additional physical RAM. Once `/v1/models` succeeds, use the same request examples as native installation.

There is no prebuilt QwenRelay image. A fresh build and GPU startup of this recipe are still pending validation. [validation boundaries](qwenrelay-validation.md).

## Offline reconstruction and checks

No vLLM/PyTorch import is required for these commands. Python 3.12+, Git/Bash and GNU patch suffice:

```bash
python3 scripts/qwenrelay.py check-pack
python3 scripts/qwenrelay.py rebuild --vllm-source /path/to/pristine-vllm --target /new/path/final-F
python3 -m unittest discover -s tests -v
```

Reconstruction checks every pristine Python/config file after canonical UTF-8 CRLF→LF normalization, applies the 54 inherited patches in their recorded order, applies three local port patches, copies the final fifteen-file overlay, parses all Python files, and checks all 3,208 canonical output files byte-for-byte. The base source normalizes historical Windows text line endings; the subsequent TP1/TP4 merges record their own changed-file identities and validation limits.  `locks/source.json` preserves the original raw hashes and normalization policy. No model binaries are included in the source tree. The last two review patches document the overlay delta; do not apply them again. A verification JSON is written beside the new output directory. Existing outputs and wrong inputs are rejected.
