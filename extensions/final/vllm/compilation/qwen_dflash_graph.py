# SPDX-License-Identifier: Apache-2.0
"""DFlash2 fixed decode coverage and capture/cache ownership guards.

These helpers change capture shapes and compiler bookkeeping, never model math.
Unsupported parallelism, variable decode, LoRA and DBO retain the native route.
"""

import copy
from collections import Counter
from functools import wraps


def fixed_dflash(config) -> bool:
    spec = config.speculative_config
    return bool(
        config.use_v2_model_runner
        and spec is not None
        and spec.use_dflash()
        and "DFlash2DraftModel" in spec.draft_model_config.architectures
        and not spec.uses_dynamic_speculative_decoding()
    )


def query_lengths(config, decode_query_len: int) -> list[int]:
    from vllm import envs

    lengths = {decode_query_len}
    spec = config.speculative_config
    if spec is not None and envs.VLLM_DFLASH2_GRAPH_BOTH and envs.VLLM_DFLASH2_LOOKUP:
        cfg = getattr(getattr(spec, "draft_model_config", None), "hf_config", None)
        block = int((getattr(cfg, "dflash_config", None) or {}).get("block_size", 0))
        short = block + decode_query_len - config.num_speculative_tokens - 1
        if 0 < short < decode_query_len:
            lengths.add(short)
    return sorted(lengths)


def fixed_decode_sizes(config, decode_query_len: int, ceiling: int) -> list[int]:
    if not fixed_dflash(config):
        return []
    budget = config.scheduler_config.max_num_batched_tokens
    return sorted({
        r * q
        for q in query_lengths(config, decode_query_len)
        for r in range(1, config.scheduler_config.max_num_seqs + 1)
        if r * q <= min(ceiling, budget)
    })


def extend_candidates(manager, descriptor, mode) -> None:
    """Add exact fixed shapes, preserving every original route's relative order."""
    if not fixed_dflash(manager.vllm_config):
        return
    if (
        not manager.is_first_pp_rank or not manager.is_last_pp_rank
        or manager.dp_size != 1 or manager.use_breakable_cg
        or manager.varlen_decode or manager.lora_capture_cases != [0]
        or manager.ubatch_runner is not None
        or not manager.cudagraph_mode.separate_routine()
        or manager.cudagraph_mode.decode_mode() != mode.FULL
    ):
        return
    if manager._graphs_captured or manager.graphs:
        raise RuntimeError("DFlash graph descriptors must be set before capture")
    ceiling = manager.compilation_config.max_cudagraph_capture_size
    if not ceiling or not manager.compilation_config.cudagraph_capture_sizes:
        return
    ceiling = min(ceiling, manager.vllm_config.scheduler_config.max_num_batched_tokens)
    full = list(manager._capture_descs.get(mode.FULL, []))
    for q in query_lengths(manager.vllm_config, manager.decode_query_len):
        for r in range(1, manager.max_num_reqs + 1):
            tokens = r * q
            if tokens > ceiling:
                continue
            desc = descriptor(
                cg_mode=mode.FULL, num_tokens=tokens, num_reqs=r,
                uniform_token_count=q, num_active_loras=0,
            )
            if desc not in full:
                full.append(desc)
            candidates = manager._candidates.setdefault((tokens, 0), [])
            if desc not in candidates:
                candidates.insert(0, desc)
    manager._capture_descs[mode.FULL] = sorted(
        full, key=lambda d: d.num_tokens, reverse=True,
    )


class OutputCaptureGuard:
    """Grow before the first FULL warmup; then retain all output tensor objects."""

    def __init__(self, manager, mode):
        self.manager = manager
        self.mode = mode
        self.held = None
        self.first_warmup = False
        self.active = (
            fixed_dflash(manager.vllm_config)
            and manager.is_first_pp_rank and manager.is_last_pp_rank
            and manager.dp_size == 1 and not manager.use_breakable_cg
            and manager.ubatch_runner is None
        )
        self.full = list(manager._capture_descs.get(mode.FULL, []))
        limit = manager._max_full_descs_to_capture
        if limit is not None:
            self.full = self.full[:limit]
        self.active = self.active and bool(self.full)
        if self.active:
            self.needed = max(d.num_tokens for d in self.full)
            if self.full[0].num_tokens != self.needed:
                raise RuntimeError("Largest FULL graph must be captured first")

    def _same_storage(self):
        main, aux = self.held
        m = self.manager
        if m.hidden_states is not main or len(m.aux_hidden_states) != len(aux):
            raise RuntimeError("DFlash capture output storage changed")
        if any(a is not b for a, b in zip(m.aux_hidden_states, aux)):
            raise RuntimeError("DFlash auxiliary capture storage changed")

    def before(self, desc, warmup):
        if not self.active or desc.cg_mode != self.mode.FULL:
            return
        m = self.manager
        if self.held is not None:
            self._same_storage()
            return
        if warmup:
            if self.first_warmup or desc != self.full[0] or m._graphs_captured:
                raise RuntimeError("Unexpected first DFlash FULL warmup")
            if any(d.cg_mode == self.mode.FULL for d in m.graphs):
                raise RuntimeError("Cannot resize outputs after a FULL capture")
            main, aux = m.hidden_states, m.aux_hidden_states
            if main is not None:
                if m.use_aux_hidden_state_outputs != bool(aux):
                    raise RuntimeError("Inconsistent DFlash auxiliary outputs")
                if any(x.shape[0] != main.shape[0] for x in aux):
                    raise RuntimeError("Inconsistent DFlash output capacities")
                if main.shape[0] < self.needed:
                    m.hidden_states, m.aux_hidden_states = None, []
            elif aux:
                raise RuntimeError("Auxiliary outputs exist without main output")
            self.first_warmup = True
            return
        if not self.first_warmup or m.hidden_states is None:
            raise RuntimeError("FULL capture must follow output warmup")
        if m.hidden_states.shape[0] < self.needed or any(
            x.shape[0] < self.needed for x in m.aux_hidden_states
        ):
            raise RuntimeError("DFlash FULL output capacity is insufficient")
        if m.use_aux_hidden_state_outputs != bool(m.aux_hidden_states):
            raise RuntimeError("DFlash FULL auxiliary outputs are missing")
        self.held = (m.hidden_states, tuple(m.aux_hidden_states))

    def finish(self):
        if not self.active:
            return
        if self.held is None:
            raise RuntimeError("No DFlash FULL graph captured")
        self._same_storage()
        if any(d not in self.manager.graphs for d in self.full):
            raise RuntimeError("DFlash FULL graph capture is incomplete")


def selector_layout(selector, args, kwargs):
    names = ("candidate_ids", "unary_logits", "hidden_states", "anchor_token_ids")
    if len(args) > len(names) or set(names[:len(args)]) & kwargs.keys():
        raise ValueError("Invalid DFlash selector arguments")
    values = dict(zip(names, args))
    values.update(kwargs)
    if set(values) != set(names):
        raise ValueError("DFlash selector requires four tensor inputs")
    tensors = [values[n] for n in names]
    if [len(t.shape) for t in tensors] != [3, 3, 3, 1]:
        raise ValueError("Unexpected DFlash selector input ranks")
    b, length, k = map(int, tensors[0].shape)
    if not (
        b > 0 and length > 0 and k == selector.top_k
        and tuple(tensors[1].shape) == (b, length, k)
        and tuple(tensors[2].shape[:2]) == (b, length)
        and tuple(tensors[3].shape) == (b,)
    ):
        raise ValueError("Inconsistent DFlash selector input shapes")
    return {
        "schema": "dflash2-selector-static-layout-v1", "top_k": int(selector.top_k),
        "tensors": [{
            "name": name, "static_shape": list(map(int, t.shape[1:])),
            "strides": list(map(int, t.stride())), "dtype": str(t.dtype),
        } for name, t in zip(names, tensors)],
        "predecessor_shape": list(map(int, selector.predecessor_codebook.shape)),
        "successor_shape": list(map(int, selector.successor_codebook.shape)),
    }


def isolate_selector(cls):
    """Wrap the 0.31 support_torch_compile class, including its config keyword."""
    original_init, original_call = cls.__init__, cls.__call__

    @wraps(original_init)
    def init(self, *args, **kwargs):
        from vllm.config import get_current_vllm_config, set_current_vllm_config

        parent = kwargs.get("vllm_config") or get_current_vllm_config()
        parent_comp = parent.compilation_config
        owned, comp = copy.copy(parent), copy.copy(parent_comp)
        owned.compilation_config = comp
        comp.enabled_custom_ops = Counter({n: 1 for n in parent_comp.enabled_custom_ops})
        comp.disabled_custom_ops = Counter({n: 1 for n in parent_comp.disabled_custom_ops})
        enabled, disabled = comp.enabled_custom_ops.copy(), comp.disabled_custom_ops.copy()
        call_kwargs = dict(kwargs, vllm_config=owned)
        with set_current_vllm_config(owned):
            original_init(self, *args, **call_kwargs)
        if self.vllm_config is not owned or self.compilation_config is not comp:
            raise RuntimeError("Selector must own its compiler bookkeeping")
        parent_comp.enabled_custom_ops.update(comp.enabled_custom_ops - enabled)
        parent_comp.disabled_custom_ops.update(comp.disabled_custom_ops - disabled)
        self._qwen_selector_layout = None
        self._qwen_selector_layout_guard = True

    @wraps(original_call)
    def call(self, *args, **kwargs):
        import torch

        if not torch.compiler.is_compiling():
            layout = selector_layout(self, args, kwargs)
            if self._qwen_selector_layout is None:
                self._qwen_selector_layout = layout
            elif layout != self._qwen_selector_layout:
                raise RuntimeError("Selector static layout changed; create a new selector")
        return original_call(self, *args, **kwargs)

    cls.__init__, cls.__call__ = init, call
    return cls
