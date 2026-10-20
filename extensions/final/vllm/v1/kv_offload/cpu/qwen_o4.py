"""Default-off O4 prototype for the pinned vLLM 0.31 private CPU path.

Preserves original group ids, hash keys, scheduler load intersection, GPU
allocation and native DMA. Supports explicit packed/strided, equal-span, TP-only
hybrid groups. No persistence or production configuration mutation.
"""
from collections import defaultdict
from dataclasses import asdict, replace
import itertools, json, math, os, pathlib, time

import numpy as np
from vllm import envs
from vllm.v1.core.single_type_kv_cache_manager import MambaManager, SlidingWindowManager
from vllm.v1.kv_cache_interface import MambaSpec, FullAttentionSpec, SlidingWindowSpec
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches, GPULoadStoreSpec, OffloadingManager, OffloadingWorker,
    PrepareStoreOutput, RequestOffloadingContext, TransferResult,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec, CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import OffloadingConnectorStats
from vllm.v1.kv_offload.cpu.qwen_o4_layout import GroupPage, plan_pools, derive_group_pages


def enabled():
    return os.environ.get('QWEN_O4') == '1'


def trace(kind, **fields):
    root = os.environ.get('QWEN_O4_TRACE_DIR')
    if not root: return
    directory = pathlib.Path(root)
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/('o4-'+str(os.getpid())+'.jsonl')
    if path.exists() and path.stat().st_size > 8*1024**2:
        (directory/('truncated-'+str(os.getpid())+'.flag')).touch()
        return
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps({'time':time.time(),'pid':os.getpid(),'kind':kind,**fields})+'\n')


def attach_layout(original):
    def build(vllm_config, kv_cache_config):
        config = original(vllm_config, kv_cache_config)
        if not enabled():
            return config
        pc = config.parallel
        if (config.cache.blocks_per_chunk != 1 or config.canonical_layout
                or pc.world_size != pc.tp_size or pc.pp_size != 1
                or pc.dcp_size != 1 or pc.pcp_size != 1):
            raise ValueError('O4 supports private TP-only one-page chunks')
        groups = kv_cache_config.kv_cache_groups
        if len({g.kv_cache_spec.block_size for g in groups}) != 1:
            raise ValueError('O4 requires equal group block spans')
        if config.replicated_layout or tuple(g.group_id for g in config.groups) != tuple(range(len(groups))):
            raise ValueError('O4 requires all native groups without replication')
        physical_pages = derive_group_pages(kv_cache_config)
        pages = []
        recurrent_spec = None
        for i, group in enumerate(groups):
            spec = group.kv_cache_spec
            if isinstance(spec, MambaSpec):
                if spec.mamba_cache_mode != 'align':
                    raise ValueError('O4 requires Mamba align mode')
                kind = 'mamba'
                recurrent_spec = spec
            elif isinstance(spec, SlidingWindowSpec):
                kind = 'sliding'
            elif type(spec) is FullAttentionSpec:
                kind = 'full'
            else:
                raise ValueError('O4 unsupported cache spec '+type(spec).__name__)
            page, indices = physical_pages[i]
            pages.append({'group_id': i, 'kind': kind, 'bytes_per_rank': page,
                          'tensor_indices': indices})
        if recurrent_spec is None:
            raise ValueError('O4 requires a recurrent group')
        retention = vllm_config.cache_config.prefix_cache_retention_interval
        reference = int(os.environ.get('QWEN_O4_REFERENCE_TOKENS', '32768'))
        block = recurrent_spec.block_size
        if reference <= 0 or (retention is not None and (retention < 0 or retention % block)):
            raise ValueError('O4 invalid reference length or retention interval')
        blocks = math.ceil(reference / block)
        mask = cpu_reachable_block_mask(
            MambaManager, 0, blocks, block, recurrent_spec, False, retention,
            (max(reference-1, 0), reference))
        states = (blocks if mask is None else sum(mask)) + 1  # shared-prefix junction allowance
        return replace(config, extra_config={**config.extra_config, '_o4_layout': {
            'pages': pages, 'retention': retention, 'block_tokens': block,
            'reference_tokens': reference, 'blocks_per_reference': blocks,
            'states_per_reference': states}})
    return build


class _GroupReqContext:
    def __init__(self, parent):
        self.parent = parent
        self._state = {}
        self.pending = 0
        self.finished = False

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def set_state(self, value):
        self._state[type(value)] = value

    def get_state(self, cls):
        return self._state.get(cls)


def _context(self, gid, parent):
    key = (id(parent), gid)
    if key not in self._contexts:
        self._contexts[key] = _GroupReqContext(parent)
    return self._contexts[key]


def _pin(self, gid, parent, count):
    self._context(gid, parent).pending += count


def _unpin(self, gid, parent, count):
    context = self._context(gid, parent)
    context.pending -= count
    if context.pending < 0:
        raise RuntimeError('O4 completion exceeds prepared references')
    self._maybe_release(gid, parent)


def _maybe_release(self, gid, parent):
    context = self._context(gid, parent)
    if context.finished and context.pending == 0:
        del self._contexts[(id(parent), gid)]


class GroupedManager(OffloadingManager):
    _context = _context
    _pin = _pin
    _unpin = _unpin
    _maybe_release = _maybe_release

    def __init__(self, managers, pages):
        self.managers = managers
        self._contexts = {}
        self.pages = pages
        self.last_snapshot = None
        self.evictions = defaultdict(int)

    def _partition(self, keys):
        result = defaultdict(list)
        for key in keys:
            gid = get_offload_group_idx(key)
            if gid not in self.managers:
                raise ValueError('unknown O4 group id')
            result[gid].append(key)
        return dict(sorted(result.items()))

    def on_new_request(self, ctx):
        policies = [m.on_new_request(self._context(gid, ctx)).policy for gid,m in self.managers.items()]
        if len(set(policies)) != 1:
            raise ValueError('O4 child policies disagree')
        return RequestOffloadingContext(policy=policies[0])

    def lookup(self, key, ctx):
        gid = get_offload_group_idx(key)
        return self.managers[gid].lookup(key, self._context(gid, ctx))

    def touch(self, keys, ctx):
        for gid, group in self._partition(keys).items(): self.managers[gid].touch(group, self._context(gid, ctx))

    def prepare_load(self, keys, ctx):
        partition = self._partition(keys)
        # Validate order before changing references. Scheduler GPU descriptors
        # and CPU ids both follow original group order.
        if list(keys) != list(itertools.chain.from_iterable(partition.values())):
            raise ValueError('O4 load keys must be group ordered')
        ids, pinned = [], []
        try:
            for gid, group in partition.items():
                spec = self.managers[gid].prepare_load(group, self._context(gid, ctx))
                self._pin(gid, ctx, len(group))
                pinned.append((gid, group))
                ids.extend(spec.block_ids.tolist())
        except Exception:
            for gid, group in pinned:
                self.managers[gid].complete_load(group, self._context(gid, ctx))
                self._unpin(gid, ctx, len(group))
            raise
        return CPULoadStoreSpec(ids)

    def complete_load(self, keys, ctx):
        for gid, group in self._partition(keys).items():
            self.managers[gid].complete_load(group, self._context(gid, ctx))
            self._unpin(gid, ctx, len(group))

    def prepare_store(self, keys, ctx):
        accepted, ids, evicted, reserved = [], [], [], []
        try:
            for gid, group in self._partition(keys).items():
                result = self.managers[gid].prepare_store(group, self._context(gid, ctx))
                if result is None:
                    for prev, prev_keys in reserved:
                        self.managers[prev].complete_store(prev_keys, self._context(prev, ctx), success=False)
                        self._unpin(prev, ctx, len(prev_keys))
                    trace('store_blocked',request_id=ctx.req_id,group_id=gid,
                        pending_writes=self.managers[gid]._num_write_pending_chunks)
                    return None
                reserved.append((gid, result.keys_to_store))
                self._pin(gid, ctx, len(result.keys_to_store))
                accepted.extend(result.keys_to_store)
                ids.extend(result.store_spec.block_ids.tolist())
                evicted.extend(result.evicted_keys)
                self.evictions[gid] += len(result.evicted_keys)
                trace('store_reserved',request_id=ctx.req_id,group_id=gid,
                    keys=[k.hex() for k in result.keys_to_store],
                    slots=result.store_spec.block_ids.tolist(),evicted=[k.hex() for k in result.evicted_keys])
        except Exception:
            for gid, group in reserved:
                self.managers[gid].complete_store(group, self._context(gid, ctx), success=False)
                self._unpin(gid, ctx, len(group))
            raise
        return PrepareStoreOutput(accepted, CPULoadStoreSpec(ids), evicted)

    def complete_store(self, keys, ctx, success=True):
        for gid, group in self._partition(keys).items():
            self.managers[gid].complete_store(group, self._context(gid, ctx), success=success)
            self._unpin(gid, ctx, len(group))
            trace('store_committed',request_id=ctx.req_id,group_id=gid,success=success,keys=[k.hex() for k in group])

    def reset_cache(self):
        for m in self.managers.values(): m.reset_cache()
        self._contexts.clear()

    def on_request_finished(self, ctx):
        for gid,m in self.managers.items():
            child = self._context(gid, ctx)
            m.on_request_finished(child)
            child.finished = True
            self._maybe_release(gid, ctx)

    def on_schedule_end(self, ctx):
        for m in self.managers.values(): m.on_schedule_end(ctx)

    def shutdown(self):
        for m in self.managers.values(): m.shutdown()

    def has_pending_work(self):
        return any(m.has_pending_work() for m in self.managers.values())

    def take_events(self):
        for m in self.managers.values(): yield from m.take_events()

    def get_stats(self):
        # These gauges are transfer pins, not retained cache occupancy. Weight
        # them by bytes; expose occupied/ready slots separately in the audit.
        result = OffloadingConnectorStats()
        names = [CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC,
                 CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC,
                 CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC]
        capacity = sum(self.pages[g] * m._num_chunks for g,m in self.managers.items())
        pinned = written = 0
        for g,m in self.managers.items():
            used = m._num_allocated_chunks-len(m._free_list)-m._num_evictable_cache_chunks
            pinned += self.pages[g]*used
            written += self.pages[g]*m._num_write_pending_chunks
            for size in m.allocation_sizes_in_current_batch:
                result.observe_histogram(CPUOffloadingMetrics.CPU_ALLOCATION_SIZE,size)
            m.allocation_sizes_in_current_batch.clear()
        for name, value in zip(names, [pinned, written, pinned-written]):
            result.set_gauge(name, max(value,0)/capacity if capacity else 0.)
        snapshot = tuple((g,m._num_allocated_chunks-len(m._free_list),m._num_write_pending_chunks,
            m._num_allocated_chunks-len(m._free_list)-m._num_evictable_cache_chunks-m._num_write_pending_chunks,self.evictions[g])
            for g,m in self.managers.items())
        if snapshot != self.last_snapshot:
            trace('pool_occupancy',groups=[{'id':g,'resident':resident,'ready':resident-writes,
                'write_pending':writes,'load_pins':reads,'evicted_total':evicted,
                'slots':self.managers[g]._num_chunks,'bytes_per_slot_per_rank':self.pages[g]}
                for g,resident,writes,reads,evicted in snapshot])
            self.last_snapshot = snapshot
        return result


class GroupedWorker(OffloadingWorker):
    def __init__(self, kv_caches, plan, group_pages, worker_factory=CPUOffloadingWorker):
        self.workers = {}
        self.pending = {}
        self.results = defaultdict(dict)
        self.bounds = {}
        self.started = {}
        native = worker_factory is CPUOffloadingWorker
        if native:
            import torch
            # TP4 halves each rank's pages. The smallest cache tensors then
            # fall below 64 MiB and the host allocator would round them up
            # to that power-of-two bin, violating the TP-group byte budget.
            # Keep the original threshold on TP2 and lower it below the
            # smallest planned tensor when necessary. Measured accounting
            # below remains mandatory; this does not relax the byte budget.
            smallest = int(min(kv_caches.tensors[i].page_size_bytes * slots
                               for page,slots in zip(group_pages,plan.group_slots)
                               for i in page.tensor_indices))
            smallest_mb = max(1, smallest // (1024**2))
            rounding_mb = min(64, 1 << (smallest_mb.bit_length()-1))
            conf = ('expandable_segments:False,'
                    f'pinned_max_round_threshold_mb:{rounding_mb},'
                    'pinned_max_cached_size_mb:64')
            torch._C._accelerator_setAllocatorSettings(conf)
            host_before = dict(torch.cuda.memory.host_memory_stats())
            if 'active_bytes.current' not in host_before:
                raise RuntimeError('O4 requires real pinned allocator accounting')
        try:
            for page, slots in zip(group_pages, plan.group_slots):
                gid = page.group_id
                refs = kv_caches.group_data_refs[gid]
                indices = sorted({r.tensor_idx for r in refs})
                if tuple(indices) != page.tensor_indices:
                    raise ValueError('O4 scheduler/worker physical tensor ownership mismatch')
                tensors = [kv_caches.tensors[i] for i in indices]
                actual = sum(t.page_size_bytes for t in tensors)
                if actual != page.bytes_per_rank:
                    raise ValueError('O4 scheduler/worker page size mismatch')
                if not tensors or len({t.tensor.shape[0] for t in tensors}) != 1:
                    raise ValueError("O4 group tensor block counts disagree")
                self.bounds[gid] = (tensors[0].tensor.shape[0], slots)
                mapping = {old:new for new,old in enumerate(indices)}
                group_refs = [[] for _ in kv_caches.group_data_refs]
                group_refs[gid] = [replace(r,tensor_idx=mapping[r.tensor_idx]) for r in refs]
                local = CanonicalKVCaches(tensors, group_refs)
                self.workers[gid] = worker_factory(local, 1, slots, mmap_region=None)
            if native:
                host_after = dict(torch.cuda.memory.host_memory_stats())
                requested = sum(p.bytes_per_rank*n for p,n in zip(group_pages,plan.group_slots))
                world_size = plan.allocated_bytes // requested
                active_delta = host_after['active_bytes.current']-host_before['active_bytes.current']
                allocated_delta = host_after['allocated_bytes.current']-host_before['allocated_bytes.current']
                measured = max(active_delta,allocated_delta)
                evidence = {'pid':os.getpid(),'requested_cache_bytes_per_rank':requested,
                    'active_pinned_delta':active_delta,'allocated_pinned_delta':allocated_delta,
                    'measured_cache_bytes_per_rank':measured,'world_size':world_size,
                    'budget_bytes':plan.budget_bytes,'allocator_conf':conf,
                    'before':host_before,'after':host_after}
                print('O4_HOST_ALLOCATION '+json.dumps(evidence),flush=True)
                trace('host_allocation',**{k:v for k,v in evidence.items() if k!='pid'})
                if active_delta < requested or measured*world_size > plan.budget_bytes:
                    raise MemoryError('O4 pinned allocator measured cache exceeds fixed budget')
        except Exception:
            for w in self.workers.values(): w.shutdown()
            raise

    def _record_result(self, gid, result):
        jid = result.job_id
        if jid not in self.pending or gid not in self.pending[jid]:
            raise RuntimeError('O4 unexpected child completion')
        if gid in self.results[jid]:
            raise RuntimeError('O4 duplicate child completion')
        self.results[jid][gid] = result

    def _drain_rejected(self, job_id, submitted):
        for gid in submitted:
            self.workers[gid].wait({job_id})
        for gid in submitted:
            for result in self.workers[gid].get_finished():
                if result.job_id != job_id:
                    self._record_result(gid, result)

    def _submit(self, job_id, gpu, cpu, store):
        started = time.monotonic()
        if job_id in self.pending:
            raise ValueError('O4 duplicate in-flight transfer id')
        count = len(self.workers)
        sizes = gpu.group_sizes
        if (len(sizes) != count or len(gpu.block_indices) != count
                or any(not isinstance(n, (int, np.integer)) or n < 0 for n in sizes)
                or sum(sizes) != len(gpu.block_ids)
                or len(cpu.block_ids) != len(gpu.block_ids)):
            raise ValueError('O4 invalid transfer descriptor geometry')
        if not len(gpu.block_ids):
            raise ValueError('O4 empty transfer')
        # Validate every group before any native DMA is submitted. One CPU
        # chunk equals one GPU block in this bounded profile.
        offset = 0
        for gid, size in enumerate(sizes):
            end = offset + size
            limit_gpu, limit_cpu = self.bounds[gid]
            g, c = gpu.block_ids[offset:end], cpu.block_ids[offset:end]
            if (gpu.block_indices[gid] < 0
                    or np.any(g < 0) or np.any(g >= limit_gpu)
                    or np.any(c < 0) or np.any(c >= limit_cpu)):
                raise ValueError('O4 transfer block outside its group pool')
            offset = end
        offset, submitted = 0, set()
        try:
            for gid, size in enumerate(sizes):
                end = offset + size
                if size:
                    counts = [0] * count
                    counts[gid] = size
                    sub_gpu = GPULoadStoreSpec(gpu.block_ids[offset:end].tolist(), counts, gpu.block_indices)
                    sub_cpu = CPULoadStoreSpec(cpu.block_ids[offset:end].tolist())
                    worker = self.workers[gid]
                    ok = (worker.submit_store(job_id, sub_gpu, sub_cpu) if store
                          else worker.submit_load(job_id, sub_cpu, sub_gpu))
                    if not ok:
                        self._drain_rejected(job_id, submitted)
                        return False
                    submitted.add(gid)
                offset = end
        except Exception:
            self._drain_rejected(job_id, submitted)
            raise
        trace('transfer_submitted', job_id=job_id,
              direction='GPU_to_CPU' if store else 'CPU_to_GPU', group_sizes=list(sizes))
        self.pending[job_id] = submitted
        self.started[job_id] = started
        return True

    def submit_store(self, job_id, src_spec, dst_spec):
        return self._submit(job_id,src_spec,dst_spec,True)

    def submit_load(self, job_id, src_spec, dst_spec):
        return self._submit(job_id,dst_spec,src_spec,False)

    def get_finished(self):
        for gid, worker in self.workers.items():
            for r in worker.get_finished(): self._record_result(gid, r)
        completed = []
        for jid, children in list(self.pending.items()):
            received = self.results.get(jid, {})
            if received.keys() == children:
                results = list(received.values())
                observed = time.monotonic()-self.started.pop(jid)
                completed.append(TransferResult(jid,all(r.success for r in results),
                    sum(r.transfer_size or 0 for r in results),
                    observed))
                # max(child elapsed) is not elapsed for the combined byte
                # count. Export actual composite job observed wall time;
                # native byte counters must not silently disappear. This is
                # not a pure device time or a physical PCIe bandwidth claim.
                trace('transfer_complete',job_id=jid,bytes=sum(r.transfer_size or 0 for r in results),
                    observed_completion_seconds=observed,aggregate_time_basis='submit_to_observed_completion_wall',child_device_seconds=[r.transfer_time for r in results])
                del self.pending[jid]; self.results.pop(jid,None)
        return completed

    def wait(self, job_ids):
        for gid, worker in self.workers.items():
            ids = {jid for jid in job_ids if gid in self.pending.get(jid,set())}
            if ids: worker.wait(ids)

    def shutdown(self):
        errors=[]
        for w in self.workers.values():
            try: w.shutdown()
            except Exception as e: errors.append(e)
        if errors: raise RuntimeError('O4 child shutdown failed') from errors[0]


def make_spec_class(original):
    class O4CPUOffloadingSpec(original):
        def _uses_shared_region(self):
            return False if enabled() else super()._uses_shared_region()

        def __init__(self, config):
            super().__init__(config)
            self.o4 = enabled()
            if not self.o4: return
            if self.eviction_policy != 'lru':
                raise ValueError('O4 initial acceptance supports LRU only')
            layout = self.extra_config.get('_o4_layout')
            if not layout: raise ValueError('O4 physical layout metadata missing')
            self.group_pages = tuple(GroupPage(p['group_id'],p['kind'],p['bytes_per_rank'],tuple(p['tensor_indices'])) for p in layout['pages'])
            self.plan = plan_pools(self.group_pages,
                budget_bytes=int(self.extra_config['cpu_bytes_to_use']),
                world_size=config.parallel.world_size,
                blocks_per_reference=layout['blocks_per_reference'],
                states_per_reference=layout['states_per_reference'],
                sparse=layout['retention'] is not None)
            if os.environ.get('QWEN_O4_DIAG_FIXED_PLAN')=='1':
                slots=(90,90,90,90,90,90,863,863,315)
                used=config.parallel.world_size*sum(n*p.bytes_per_rank for n,p in zip(slots,self.group_pages))
                if len(self.group_pages)!=9 or used!=34298982400 or used>self.plan.budget_bytes:
                    raise ValueError('Diagnostic fixed pool geometry mismatch')
                self.plan=replace(self.plan,group_slots=slots,token_slots=863,state_slots=90,sliding_slots=315,allocated_bytes=used)
            self.num_chunks = self.plan.token_slots
            print('O4_POOL_PLAN '+json.dumps({'rank':config.parallel.rank,'layout':layout,'plan':asdict(self.plan)}),flush=True)

        def get_manager(self):
            if not self.o4: return super().get_manager()
            if self._manager is None:
                if self.extra_config.get('store_threshold',0) >= 2:
                    raise ValueError('O4 reuse-threshold metrics are outside this prototype')
                managers = {p.group_id:CPUOffloadingManager(n,cache_policy=self.eviction_policy,
                    enable_events=self.kv_events_config.enable_kv_cache_events)
                    for p,n in zip(self.group_pages,self.plan.group_slots)}
                self._manager = GroupedManager(managers,{p.group_id:p.bytes_per_rank for p in self.group_pages})
            return self._manager

        def get_worker(self, kv_caches):
            if not self.o4: return super().get_worker(kv_caches)
            if self._worker is None:
                self._worker = GroupedWorker(kv_caches,self.plan,self.group_pages)
            return self._worker
    O4CPUOffloadingSpec.__name__ = 'CPUOffloadingSpec'
    return O4CPUOffloadingSpec


def retention_points(start_block,end_block,block,interval,reachable):
    points=set()
    if interval:
        first=max(interval,((start_block*block)//interval)*interval)
        for point in range(first,(end_block+3)*block+interval,interval):
            points.add(point)
    points.update(int(p)//block*block for p in reachable)
    # DFlash bulk prefill commits can be one 800-token block before the
    # ideal periodic/replay boundary. Retain two preceding block positions
    # as well, on both GPU hash retention and CPU store selection.
    return {point-offset*block for point in points for offset in range(3)
            if point-offset*block>0}


def cpu_reachable_block_mask(cls,start_block,end_block,alignment_tokens,kv_cache_spec,
                             use_eagle,retention_interval=None,reachable_boundaries=(),
                             dcp_world_size=1,final_segment_end_block=None):
    """Account sparse host storage without changing the GPU prefix policy."""
    method=cls.reachable_block_mask.__func__
    original=getattr(method,'_o4_native_mask',method)
    if retention_interval is None or alignment_tokens is None:
        return original(cls,start_block,end_block,alignment_tokens,kv_cache_spec,
                        use_eagle,retention_interval,reachable_boundaries,
                        dcp_world_size,final_segment_end_block)
    points=retention_points(start_block,end_block,kv_cache_spec.block_size,
                            retention_interval,reachable_boundaries)
    return original(cls,start_block,end_block,alignment_tokens,kv_cache_spec,
                    use_eagle,0,tuple(sorted(points)),dcp_world_size,final_segment_end_block)


def install_retention_masks():
    if not enabled():return
    import vllm.v1.core.single_type_kv_cache_manager as module
    import hashlib
    expected='a2dba4658c94f4050cfb5e8afc1f7bde69f7d7d77e6ff98ea58e2b4ec47b62ac'
    if hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest()!=expected:
        raise ImportError('O4 dense-GPU cache-manager source lock mismatch')
    for cls in [MambaManager,SlidingWindowManager]:
        original=cls.reachable_block_mask.__func__
        if getattr(original,'_o4_native_dense_gpu',False):continue
        def make(original):
            def mask(cls,start_block,end_block,alignment_tokens,kv_cache_spec,use_eagle,
                     retention_interval=None,reachable_boundaries=(),
                     dcp_world_size=1,final_segment_end_block=None):
                # Local GPU hits must keep the native checkpoint boundary. The
                # separate CPU store filter retains the sparse16K budget policy.
                return original(cls,start_block,end_block,alignment_tokens,kv_cache_spec,
                                use_eagle,None,reachable_boundaries,
                                dcp_world_size,final_segment_end_block)
            mask._o4_native_dense_gpu=True
            mask._o4_native_mask=original
            return mask
        cls.reachable_block_mask=classmethod(make(original))
    print('O4_GPU_PREFIX_POLICY '+json.dumps({'pid':os.getpid(),
        'gpu_checkpoints':'native-dense','CPU_retention_interval_source':'OffloadingConfig._o4_layout',
        'CPU_store_filter':'unchanged-periodic-replay-junction-with-two-predecessors',
        'native_cache_manager_sha256':expected}),flush=True)


install_retention_masks()


def store_reachable(req,group_config,abs_chunk,key=None,*,retention_interval=None):
    if not enabled():return True
    interval=retention_interval
    if interval is None:return True
    recurrent=group_config.requires_cow_source
    window=group_config.sliding_window_size_in_chunks
    if not recurrent and window is None:return True
    block=group_config.tokens_per_chunk
    reachable=[max(req.num_prompt_tokens-1,0)]
    if req.shared_prefix_boundary:reachable.append(req.shared_prefix_boundary)
    points=retention_points(abs_chunk,abs_chunk+1,block,interval,reachable)
    if recurrent:
        keep=(abs_chunk+1)*block in points
    else:
        shift=int(group_config.is_eagle_group)
        need=window+shift
        keep=any(p//block+shift-need<=abs_chunk<p//block+shift for p in points)
    trace('checkpoint_offer',request_id=getattr(req,'request_id',None),
          group_id=group_config.group_idx,boundary=(abs_chunk+1)*block,
          keep=keep,key=key.hex() if key else None)
    return keep


def remember_external_junction(scheduler, req_status, group_idx, boundary, deferred):
    if not enabled() or deferred: return
    full = [g.group_idx for g in scheduler.config.kv_group_configs
            if g.sliding_window_size_in_chunks is None]
    if full and group_idx == full[-1]:
        req = req_status.req
        # A native GPU hit/junction controls the earliest mandatory prefill
        # checkpoint. An external hint must never move that boundary forward.
        if req_status.num_locally_computed_tokens > 0 or req.shared_prefix_boundary > 0:
            trace('external_junction_preserved',request_id=req.request_id,
                  local_tokens=req_status.num_locally_computed_tokens,
                  native_boundary=req.shared_prefix_boundary,external_offer=boundary)
            return
        req.shared_prefix_boundary = min(boundary,req.num_prompt_tokens-1)
        trace('external_junction',request_id=req.request_id,boundary=req.shared_prefix_boundary)


def lookup_result(scheduler, req_status, tokens):
    trace('lookup_result',request_id=req_status.req.request_id,
        prompt_tokens=req_status.req.num_prompt_tokens,local_tokens=req_status.num_locally_computed_tokens,
        external_tokens=tokens,junction=req_status.req.shared_prefix_boundary,o4=enabled())
    manager=scheduler.manager
    if hasattr(manager,'managers'):
        groups=[{'id':g,'capacity':m._num_chunks,'resident':m._num_allocated_chunks-len(m._free_list),
                 'write_pending':m._num_write_pending_chunks,'evictable':m._num_evictable_cache_chunks}
                for g,m in manager.managers.items()]
    else:
        groups=[]
        counts=defaultdict(int)
        for key in manager._policy.chunks: counts[get_offload_group_idx(key)]+=1
        groups=[{'id':g,'resident':n} for g,n in sorted(counts.items())]
    trace('lookup_pool',request_id=req_status.req.request_id,groups=groups,o4=enabled())


def trace_lookup_class(cls):
    original=cls._lookup_complete_chunks
    def wrapped(self,req_status):
        result=original(self,req_status)
        lookup_result(self,req_status,result)
        return result
    cls._lookup_complete_chunks=wrapped


