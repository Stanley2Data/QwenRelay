# Validation boundary for the 0.1 source candidate

The first supported profile is two RTX 3090 24 GB GPUs, native Linux, TP2, the exact EfficientThink/DFlash2 assets and `config/dual-3090.json`. This document separates recorded runtime evidence from packaging checks.

## Recorded runtime evidence

The final F candidate's four-request main/draft FULL20 CUDA Graph capture and actual replay were observed on both TP ranks. The H/F comparison completed the original22 and additional10 context/decode cases with one complete warm round and three formal rounds per arm. A separate repaired-F cache-restoration acceptance completed the original30 requests. See [benchmark details and limits](qwenstride-benchmarks.md).

The ported source tree is locked to3,208 files. The source candidate is not the earlier F that lacked Graph20 coverage or the earlier F that restored zero CPU-cache tokens. The ongoing single-card profile and new model-weight experiments are not included.

## Packaging checks

The repository's `rebuild` command was actually executed offline from the pinned pristine vLLM source. All54 inherited patches, three local patches and six final-overlay files reconstructed the expected3,208-file tree; Python parsing passed. Packaging discovered that the historical Windows source lock included CRLF text, which would reject a clean Linux LF checkout. The public tree therefore normalizes only UTF-8 CRLF→LF, retaining the tested raw hashes and checking the canonical output byte-for-byte. The final verification records reconstruction from a fresh LF checkout. CPU tests cover source-lock integrity, device/profile/Graph settings, inherited activation-INT8 clearing, path traversal, wrong hashes and existing-output protection. Launch and benchmark plans can be inspected without a model or CUDA.

The main and draft configs were also retrieved read-only from the public full commit pins and matched the asset-lock SHA256 values. This did not download model tensors or rerun inference. Model-weight identities rely on the previously completed full SHA checks and the startup verifier; weights are excluded from Git.

Shell syntax/LF, public Python/JSON syntax, profile equality with the recorded launch contract, Compose structure, local Markdown links and accidental private-material export are checked before the local candidate commit. [Packaging validation JSON](qwenstride-packaging-validation.json) records the completed checks, with no host paths or credentials.

## Still unverified in this packaging step

- A full fresh Docker build, clean-machine dependency resolution/native import and model startup through the new public installer/launcher.
- New GPU numerical, quality, speed or cache-recovery runs after packaging. Existing evidence is retained at its original scope.
- Single-GPU serving, other GPU models/drivers, other checkpoints, vision, maximum simultaneous256K contexts, high-concurrency deployment and long-running stability.

The Docker image/base and direct dependency versions are pinned, but transitive pip and OS packages are not fully hermetic. Do not call the recipe a byte-identical runtime build. A successful offline reconstruction does not certify GPU behavior or deployment readiness. No upstream container image is presented as a QwenStride image; there is no automatic publishing workflow.
