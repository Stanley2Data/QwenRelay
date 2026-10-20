# SPDX-License-Identifier: Apache-2.0
"""Exercise allocation/control code on CPU; these checks do not run GPU kernels."""
import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]/'extensions/final/vllm'


def functions(path,names,namespace):
    tree=ast.parse((ROOT/path).read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in selected}!=set(names):raise AssertionError('Missing source function')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return SimpleNamespace(**namespace)


class CapacityTests(unittest.TestCase):
    def test_scheduler_capacity_is_reused_with_no_forward_config(self):
        allocations=[]

        class Attention:
            def __init__(self,requests,heads,dim,device,qmax):
                self.max_num_reqs=requests
                self.qmax=qmax
                allocations.append((requests,heads,dim,device,qmax))
            def run(self,*args,**kwargs):pass

        def missing_config():raise RuntimeError('No current VllmConfig in V2 forward')

        modules={'vllm.config':SimpleNamespace(get_current_vllm_config=missing_config),
                 'vllm.v1.attention.ops.spec_decode_attn':SimpleNamespace(
                     SpecDecodeAttention=Attention,BLOCK_M=64,QMAX_TOKENS=64)}
        scope={'_SPEC_ATTN':{},'_SPEC_ATTN_QMAX':None,
               'envs':SimpleNamespace(VLLM_SPEC_DECODE_ATTN_QMAX=5)}
        names=['_spec_attn_qmax','_spec_attn_prepare','_spec_attn_run','_spec_attn_run_fp8']
        code=functions(Path('v1/attention/backends/flash_attn.py'),names,scope)
        with patch.dict(sys.modules,modules):
            for heads,kv in ((24,4),(12,2),(6,1)):
                impl=SimpleNamespace(num_heads=heads,num_kv_heads=kv,head_size=256,scale=.0625)
                att=code._spec_attn_prepare(impl,4,'cuda:0')
                self.assertEqual(att.max_num_reqs,4)
                # GQA=6 uses a 10-token tile even with the logical QMAX=5 setting.
                self.assertEqual(att.qmax,10)
                self.assertIs(code._spec_attn_prepare(impl,4,'cuda:0'),att)
                query=SimpleNamespace(device='cuda:0')
                cu=SimpleNamespace(shape=(5,))
                code._spec_attn_run(impl,query,None,None,None,cu,None,None,5)
                code._spec_attn_run_fp8(impl,query,None,None,None,cu,None,None,5,None,None,None)
                with self.assertRaisesRegex(ValueError,'smaller'):
                    code._spec_attn_prepare(impl,8,'cuda:0')
            self.assertEqual(len(allocations),3)
            self.assertTrue(all(row[0]==4 for row in allocations))


class Storage:
    def __init__(self,shape,dtype,storage=None):
        self.shape=tuple(shape);self.dtype=dtype
        self.storage=object() if storage is None else storage
    def numel(self):return math.prod(self.shape)
    def element_size(self):return 1 if self.dtype=='int8' else 4
    def view(self,*shape):
        if -1 in shape:
            count=-math.prod(shape)
            if self.numel()%count:raise ValueError('Non-divisible view')
            shape=tuple(self.numel()//count if n==-1 else n for n in shape)
        if math.prod(shape)!=self.numel():raise ValueError('Invalid view')
        return Storage(shape,self.dtype,self.storage)


class WorkspaceTests(unittest.TestCase):
    def modules(self):
        torch=SimpleNamespace(int8='int8',float32='float32',int32='int32',
                              OutOfMemoryError=MemoryError)
        torch.empty=lambda shape,device,dtype:Storage(shape,dtype)
        torch.empty_like=lambda value:Storage(value.shape,value.dtype)
        torch.cuda=SimpleNamespace(current_stream=lambda device:SimpleNamespace(cuda_stream=7))
        pf2=functions(Path('v1/attention/ops/qwen031/pf2.py'),
                      ['acquire_kv','workspace_key','supported_heads'],{'cache':{},'torch':torch})
        o2=functions(Path('v1/attention/ops/qwen031/o2.py'),
                     ['acquire_q'],{'qc_cache':{},'torch':torch})
        o5=functions(Path('v1/attention/ops/qwen031/o5.py'),['acquire'],
                     {'pf2':pf2,'o2':o2,'torch':torch,'scratch':{},
                      'LIMIT':262144,'QUERY_LIMIT':2048,'BUDGET':320*1024**2})
        return pf2,o2,o5,torch

    def test_shared_storage_covers_all_geometries_and_route_orders(self):
        for heads,kv in ((24,4),(12,2),(6,1)):
            for first_context in (16384,262144):
                pf2,o2,o5,_=self.modules()
                device=SimpleNamespace(index=0)
                q=SimpleNamespace(shape=(1024,heads,256),device=device)
                k=SimpleNamespace(shape=(1,800,kv,272))
                self.assertTrue(pf2.supported_heads(q,k))
                key=pf2.workspace_key(q,k)
                first=o5.acquire({'q':q,'k':k,'max_seqlen_k':first_context})
                self.assertIsNotNone(first)
                full=pf2.acquire_kv(key,device,131072,kv)
                queries=o2.acquire_q(key,device,heads)
                self.assertEqual(full[0].shape[1],kv)
                self.assertGreaterEqual(full[0].shape[0],131072)
                self.assertEqual(queries[0].shape[1],heads)
                before=pf2.cache[key][0].storage
                _,total,views=o5.acquire({'q':q,'k':k,'max_seqlen_k':262144})
                if kv==4:self.assertIs(pf2.cache[key][0].storage,before)
                self.assertLessEqual(total,320*1024**2)
                self.assertEqual(views[0].shape[1],min(kv,2))
                self.assertEqual(views[4].shape[1],min(kv,2)*6)
                # A second shape/stream on one GPU must fall back, not grow arenas.
                another=SimpleNamespace(shape=(1024,heads,256),device=device)
                pf2.workspace_key.__globals__['torch'].cuda.current_stream=lambda device:SimpleNamespace(cuda_stream=8)
                self.assertIsNone(o5.acquire({'q':another,'k':k,'max_seqlen_k':262144}))

    def test_unsupported_heads_and_oom_fall_back(self):
        pf2,o2,_,torch=self.modules()
        self.assertFalse(pf2.supported_heads(SimpleNamespace(shape=(32,8,256)),SimpleNamespace(shape=(1,800,2,272))))
        def oom(*args,**kwargs):raise MemoryError('synthetic allocation failure')
        torch.empty=oom
        self.assertIsNone(pf2.acquire_kv((0,7,24,4),None,131072,4))
        self.assertIsNone(o2.acquire_q((0,7,24,4),None,24))
        with self.assertRaises(AssertionError):pf2.acquire_kv((0,7,24,4),None,262144,4)

    def test_head_passes_preserve_tp4_and_cover_all_tp1_heads(self):
        # Record Python dispatch only. Numerical attention is tested separately
        # by gpu_tp_prefill.py on CUDA against independent FP64 reference rows.
        class Query:
            def __init__(self,heads,original=None):
                self.shape=(1024,heads,256);self.original=original or heads
                self.device=SimpleNamespace(index=0)
            def stride(self):return (self.original*256,256,1)
            def __getitem__(self,index):
                return Query(index[1].stop-index[1].start,self.original)

        class Kernel:
            def __init__(self,name,events):self.name=name;self.events=events
            def __getitem__(self,grid):
                def call(*args,**kwargs):self.events.append((self.name,grid,args,kwargs))
                return call

        for heads,kv in ((24,4),(12,2),(6,1)):
            with self.subTest(heads=heads):
                events=[];pf2,o2,o5,torch=self.modules()
                torch.cuda.is_current_stream_capturing=lambda:False
                kernels=SimpleNamespace(**{n:Kernel(n,events) for n in
                                           ['gather_sequence','attention_bf16','attention_int8']})
                params={'eligible':lambda d:True,'torch':torch,'acquire':o5.acquire,
                        'triton':SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b),
                        'o2_kernels':SimpleNamespace(quantize=Kernel('quantize',events)),
                        'kernels':kernels,'os':SimpleNamespace(environ={'QWEN_O2':'1'}),
                        'scratch':o5.acquire.__globals__['scratch'],
                        'logged':{(0,2,False,True)}}
                run=functions(Path('v1/attention/ops/qwen031/o5.py'),['try_run'],params)
                k=SimpleNamespace(shape=(10,800,kv,272),stride=lambda:(800*kv*272,kv*272,272,1))
                scale=SimpleNamespace(stride=lambda:(800*kv*272//4,kv*272//4,272//4))
                d=dict(q=Query(heads),out=Query(heads),k=k,v=k,k_scale_cache=scale,v_scale_cache=scale,
                       seqused_k=SimpleNamespace(numel=lambda:2),cu_seqlens_q=object(),
                       block_table=SimpleNamespace(stride=lambda index:32),
                       max_seqlen_k=16384,max_seqlen_q=1024)
                def native(**kwargs):
                    self.assertTrue(kwargs['o5_fallback_only'])
                    events.append(('native',None,None,None))
                self.assertTrue(run.try_run(d,native))
                self.assertEqual(events[0][0],'native')
                gathers=[e for e in events if e[0]=='gather_sequence']
                self.assertEqual([e[3]['HEAD_START'] for e in gathers],
                                 [0,0,2,2] if kv==4 else [0,0])
                self.assertTrue(all(e[1][1]==min(kv,2) for e in gathers))
                attention=[e for e in events if e[0]=='attention_bf16']
                self.assertTrue(all(e[2][0].shape[1]==min(kv,2)*6 for e in attention))
                self.assertTrue(all(e[2][0].stride()[0]==heads*256 for e in attention))


if __name__=='__main__':unittest.main()
