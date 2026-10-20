# Third-party notices

| Component | Source / license | Distribution here |
|---|---|---|
| HyperQwen | [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen), Apache-2.0 | Pinned public history, original source and patch stack; root LICENSE retained |
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm), Apache-2.0 | Incremental patches and fifteen final source overlays; original SPDX headers and [license](licenses/vllm-LICENSE) retained |
| KVarN | [huawei-csl/KVarN](https://github.com/huawei-csl/KVarN), Apache-2.0 | HyperQwen's port in kvarn/, original file notices retained |
| 1Cat cache work | Design/implementation lineage referenced by the local O4 work, #598/#617 | Attribution for hybrid-history cache allocation; no wholesale 1Cat runtime is shipped |
| froggeric fixed template | [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates), Apache-2.0 | Tested template variant; original author, source hash and modifications in [template notice](licenses/froggeric-NOTICE.md) |
| Qwen / EfficientThink / DFlash2 | Model authors linked in [installation](docs/qwenrelay-install.md); model licenses apply separately | Small loader metadata/template and model source/SHA references; no weights |
| PyTorch, Triton, NVIDIA CUDA, FlashInfer and other runtime dependencies | Respective upstream licenses | Downloaded/installed separately; no compiled runtime/native libraries redistributed in this repository |

QwenRelay's Apache-2.0 license applies to its source contributions and does not replace third-party terms or grant rights to project trademarks. Retain the applicable notices when redistributing. Modification locations and source pins are in [UPSTREAM.md](UPSTREAM.md).
