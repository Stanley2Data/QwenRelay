# Upstream identity and changes

QwenRelay is a derivative source distribution of [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen). It preserves the public Git history through **95ee3b64b0df7014ee2626529524d6b99c3dcdd1** (491 commits). The base vLLM is **v0.31.0**, commit **db9527a46873454610df6dbedf79a36d6bf1a7f6**. Nothing in the installer follows a moving branch.

Upstream source/license/copyright notices remain in place. [README.hyperqwen.md](README.hyperqwen.md) preserves the pinned upstream README. QwenRelay replaces the root README and Docker/Compose entry, and adds its own TP1/TP2/TP4 launcher, profiles, locks and documentation. TP1 / AutoRound-fast is the default; TP2/TP4 use EfficientThink. Inherited automatic image-publishing workflows are removed. The remaining legacy launchers/docs are upstream development material.

The inherited patch stack contains 50 ordered patches in `patches/series` followed by four KVarN patches; KVarN modules are copied before the latter. `patches/apply.sh` remains the upstream patch application authority. Source locks verify those files before any reconstruction.

Local port deltas:

1. `0001-e2-and-compile-identity.patch`: aligned INT8 KV and hybrid pages; compile identity.
2. `0002-pf2-o2-o5-prefill.patch`: bounded TP2 prefill dispatch and kernels.
3. `0003-o4-cpu-cache.patch`: budgeted hybrid CPU history pools.
4. `review/0004-cpu-aligned-handoff-retention-final-v2.patch`: cache lookup/retention fix.
5. `review/0005-v2-fixed-decode-graph-and-cache-guards-final-v2.patch`: fixed decode CUDA Graph coverage and compiler/buffer guards.
6. `review/0006-tp1-capacity-and-bounded-prefill.patch`: actual-capacity scratch preparation, opt-in draft-head initialization and TP1 prefill merged with the existing TP2/TP4 guards.

The first three patches build the original F. The 15 final files in `extensions/final/vllm/` supply the subsequent deltas, including the TP4 prefill/O4 allocation adaptation and the TP1 merge. Patch 0006 is a review diff against the immediately preceding TP2/TP4 tree, not pristine vLLM or the original base F. Public UTF-8 source text uses canonical LF line endings; the tested Windows working-tree hashes before normalization are retained in `locks/source.json`. No mathematical source or binary changes are introduced by normalization. Source changes are separately mapped to before/after hashes; original notices are retained. Review patches are not applied a second time. See [merge scope and validation](docs/qwenrelay-tp-profiles.md).

The distributed chat template is the tested variant of froggeric's Qwen-family template, retaining its `qwen3.8-froggeric-v22.5.0` version marker. See [its source attribution and changes](licenses/froggeric-NOTICE.md). Model tensor-index metadata maps the public EfficientThink assets to the tested local filenames; identities remain in `locks/models.json`. `locks/models-tp1.json` separately pins the public dbirks base, syvai fast components, upstream CPU embedding preparation and final AutoRound-fast export. No model tensors are redistributed in Git.

The E2/O4/PF2/O2/O5 labels are local engineering labels; they do not imply original authorship of inherited HyperQwen/vLLM optimizations. O4's 70/22/8 allocation is an adaptation informed by the 1Cat implementation/discussions ([#598](https://github.com/1CatAI/1Cat-vLLM/pull/598), [#617](https://github.com/1CatAI/1Cat-vLLM/pull/617)), not a claim to invent hybrid caching. No endorsement by these projects is implied.
