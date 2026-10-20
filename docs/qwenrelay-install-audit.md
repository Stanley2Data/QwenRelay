# Installation and launch audit — 2026-10-20

This audit follows the English/Chinese READMEs from installation through asset
preparation, key creation, launch and API checks. Its starting revision was
`898345a2332a4a79ceef3f360a5f3702a4f37d05`. It fixes the public entry points and
guides; the locked vLLM overlay, model identities and default TP1/TP2/TP4 profiles
remain unchanged.

## Findings and changes

| Stage | Finding | Result |
|---|---|---|
| Native install | Existing source/build outputs were checked after installing dependencies; unsupported platforms had no early diagnostic. | Check platform, Python, tools and output collisions before creating a venv/installing packages. Document fresh-directory retry behavior. |
| Custom venv | The installer honored `QWENRELAY_VENV`, but the launcher still selected the repository `.venv`. | Use explicit `QWENRELAY_PYTHON`, then exported `QWENRELAY_VENV`, then the default. Add a complete custom-path example. |
| Model directory | `serve` honored `MODELS_ROOT`, but download/verification did not. | All three commands share that default; explicit paths take precedence. Explain the separate Hub cache and preparation staging space. |
| GPU KV option | The wrapper passed `--kv-cache-memory`, relying on vLLM accepting an abbreviated flag. | Keep the public option and emit the exact runtime name, `--kv-cache-memory-bytes`. |
| Draft depth | The depth-6 README example produced an 816-token hybrid block but retained an incompatible 20,800-token prefix interval. O4 rejects this geometry. | Derive the block size and round retention up to a whole block: depth 6 now uses 816 / 21,216. Default depth 4 stays 800 / 20,800. Include both in dry-run output. |
| Docker bootstrap | Weight preparation implicitly required a host venv even for container users. | Tag the locally built image and use CPU-only containers for download/preparation and key creation, followed by read-only model mounting for serving. |
| Memory/startup instructions | The detailed guide described shared memory as the host-cache path and did not spell out readiness checks. | Explain O4's pinned host RAM allocation separately from `/dev/shm`; wait for API startup and `/v1/models` before generation. |

## Checks performed

- **26 CPU regression tests passed**, including the depth-6 failure, all supported
  depth/TP page-capacity constraints, custom venv selection from another directory,
  shared model-root precedence and early rejection of occupied install outputs.
- **Full source reconstruction passed:** 3,208 files, 54 inherited patches,
  3 local base patches and 15 final overlays. Source lock SHA256:
  `242bfe3423ac76e66eafdb932903731ca3d6d3de403aa950f587dd2316ea43bc`.
- **203 dependency packages resolved** using `uv pip compile` for Python 3.12 and
  Linux x86-64 / manylinux 2.28 against `requirements-qwenrelay.txt`. The audit host
  was macOS/ARM64 with Python 3.14; this was cross-platform resolution, not a Linux
  package installation or runtime import test.
- **34 remote asset records checked** across four pinned Hugging Face revisions,
  including TP1 preparation inputs. Every remote size and every available LFS
  SHA256 matched the locks. **10 small-file records** were downloaded and checked
  using the project's real `fetch_asset` path. Tensor weights were not downloaded
  in this audit.
- **756 CPU geometry comparisons passed:** both downloaded target configs,
  TP1/TP2/TP4, all depths 1–63, and both automatic/explicit block sizing. The audit
  executed the pinned GDN state-shape and platform alignment functions with CPU
  dependency substitutes, comparing their results to the launcher. This checks
  geometry/control flow, not CUDA kernel correctness.
- **35 emitted vLLM option names** matched the pinned parser declarations and
  frontend fields. **40 shell blocks** passed `bash -n`; **28 documented launch
  commands** ran as dry runs/help without reading keys or using a GPU. Local
  documentation file targets and Python example syntax were checked.

The DFlash switches remain enabled in every profile:
`VLLM_SPEC_DECODE_ATTN=1`, logical `VLLM_SPEC_DECODE_ATTN_QMAX=K+1`, and
`VLLM_DFLASH2_LOOKUP=1`. Graph coverage remains `C*(K+1)`. The new cache alignment
is an additional constraint, not a replacement for these settings.

## Remaining deployment validation

This Mac audit did **not** execute full tensor download/CPU embedding conversion,
a fresh Linux venv install, Docker build, real service startup, API generation,
or GPU numerical/performance regression. The Docker bootstrap commands were
reviewed and syntax-checked; there was no local Docker engine to execute them.
Custom depths, including the corrected example, still need capacity, correctness
and performance checks on the intended Linux/GPU host.

See [installation](qwenrelay-install.md), [中文安装](qwenrelay-install.zh-CN.md),
[validation boundaries](qwenrelay-validation.md) and the
[historical merge checks](qwenrelay-tp-merge-validation.json). Historical reports
retain their original 22-test counts and source/configuration identities.
