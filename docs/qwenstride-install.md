# Install QwenStride 0.1

[中文](qwenstride-install.zh-CN.md) · [README](../README.md)

## Requirements and scope

The supported profile is native Linux x86-64, Python 3.12 and two RTX 3090 24 GB GPUs. The recorded benchmark host used NVIDIA driver 580.126.20 and CUDA 13.0 userspace. Use a driver compatible with that userspace; `libcuda.so.1` comes from the host driver/container runtime, not this source repo. Have a working NVIDIA Container Toolkit when using Docker. Do not install a GPU driver from this project.

Allow at least 64 GiB available system RAM for model loading plus the 32 GiB host-cache budget; more headroom is useful during compilation. Allow roughly 60 GiB free disk for models, runtime dependencies, source and compilation caches. These are planning allowances, not measured minimums. The explicit KV allocation is 11,184,101,204 bytes per GPU. Profile changes need separate capacity/accuracy/performance verification.

## Native installation

Install Python 3.12 with venv/dev support, a C/C++ toolchain, Git, GNU patch, curl and CA certificates using your distribution. Do not reuse a model-serving venv.

```bash
bash scripts/install-qwenstride.sh
```

The installer refuses an existing `.venv`, installs the direct dependency pins in `requirements-qwenstride.txt`, retrieves vLLM commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`, reconstructs the exact final F Python tree, and overlays it into the new venv while preserving the wheel's native extensions. It validates the source and native files. It starts no model and downloads no model weights.

Optional inputs: `PYTHON`, `QWENSTRIDE_VENV`, `QWENSTRIDE_BUILD`, and `VLLM_SOURCE` for an existing pristine checkout. The output build directory must be new. Transitive dependencies and OS package versions are resolved by pip/apt, so this is a pinned source/ABI recipe, not a hermetic dependency lock or byte-identical native build. Preserve `pip freeze` and native-file hashes when reproducing results.

## Model assets

```bash
.venv/bin/python scripts/qwenstride.py fetch-models --root ./models
```

This is an explicit network/download action. It uses full repository commit pins from `locks/models.json`, verifies each downloaded file's full SHA256 and size, and refuses mismatching existing files. Existing Hugging Face cache files can be reused. The model-author repositories are:

- [EfficientThink SimPO](https://huggingface.co/nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2), commit `882e1af3b8f5844f3b3163b132e343c06e91908f`, `NVFP4/AWQ-W4A16/` (the selected subdirectory is AWQ, not NVFP4).
- [syvai DFlash2 W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16), commit `4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`.

Weights are obtained from their authors under their model licenses; they are not redistributed here. This profile uses 367 W4 asymmetric and 33 W8 symmetric main-model target entries, BF16 activations, and a GPTQ W4A16 draft. It does not substitute AutoRound, Coder390, an FP8 draft, or activation INT8.

The tested main tensor index and chat template are bundled textual metadata. The index maps the public main tensor file and BF16 vision/MTP file to the tested filenames; only language-model serving is enabled. Additional source manifests/README files are not required by the loader. Check a prepared directory without downloading:

```bash
.venv/bin/python scripts/qwenstride.py check-models --root /absolute/path/to/models
```

Expected children are `Qwen3.8-27B-EfficientThink-AWQ-W4A16` and `Qwen3.8-27B-DFlash2-W4A16`. Do not use an unverified partial download. The launcher verifies full asset hashes before every start; this can take time for ~21 GiB of weights.

## Start and inspect

```bash
umask 077
.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))' > api_key.txt
bash scripts/start-qwenstride.sh --models-root ./models --devices 0,1 --dry-run
bash scripts/start-qwenstride.sh --models-root ./models --devices 0,1
```

Choose the two GPUs explicitly. The default bind is loopback and port 18020; use `--port`, `--host`, `--api-key-file` or their documented environment equivalents in `.env.example` when needed. The native launcher does not source `.env` as shell code; export variables or pass flags. `--dry-run` prints the planned nonsecret configuration without checking installed runtime/models or touching CUDA. It is not a deployment validation.

`config/dual-3090.json` is the single numerical-profile authority. Model paths, device indices, API key and bind/port are runtime inputs. The launcher clears inherited `INT8_ACT`, `VLLM_MARLIN_INPUT_DTYPE` and the alternative upstream `VLLM_PREFILL_ATTN` route, and fixes the sampler/V2-runner flags to the tested path. Other profiles/legacy upstream launchers do not implement the supported QwenStride configuration.

## Docker recipe

The root Dockerfile builds the same source/venv recipe; there is no prebuilt QwenStride image or registry dependency. Prepare models and `api_key.txt` first. Copy `.env.example` to `.env`, set `MODEL_ASSET_DIR` to an absolute host model directory, then:

```bash
docker compose build
docker compose up -d
docker compose logs -f qwenstride
```

The Compose configuration exposes all NVIDIA devices to the container and selects the two `CUDA_VISIBLE_DEVICES` indices (default `0,1`). HTTP is published to host loopback only. The model bind mount is read-only, the key is a mounted secret and the compile cache is a named volume. Shared memory is 40 GiB for the host-cache path. No privileged mode, host Docker socket, controller, or automatic restart is required. Stop this deployment with `docker compose down` when you choose to.

**This recipe was not built or GPU-started during repository packaging.** The installed TP2 runtime was validated previously; fresh-container success remains a separate validation step. See [the exact boundary](qwenstride-validation.md).

## Offline reconstruction and checks

No vLLM/PyTorch import is required for these commands. Python 3.12+, Git/Bash and GNU patch suffice:

```bash
python3 scripts/qwenstride.py check-pack
python3 scripts/qwenstride.py rebuild --vllm-source /path/to/pristine-vllm --target /new/path/final-F
python3 -m unittest discover -s tests -v
```

Reconstruction checks every pristine Python/config file after canonical UTF-8 CRLF→LF normalization, applies the 54 inherited patches in their recorded order, applies three local port patches, copies the final six-file cache/Graph overlay, parses all Python files, and checks all 3,208 canonical output files byte-for-byte. The public source differs from the tested Windows working-tree bytes only in text line endings; `locks/source.json` preserves the original raw hashes and normalization policy. Binary bytes and model assets are unchanged. The last two review patches document the overlay delta; do not apply them again. A verification JSON is written beside the new output directory. Existing outputs and wrong inputs are rejected.
