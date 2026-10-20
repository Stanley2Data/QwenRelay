"""Conservative O5 eager dispatch; unchanged native arithmetic on fallback slices."""
import os,json
from pathlib import Path
import torch
from . import pf2, o2, o2_kernels, o5_kernels as kernels
from vllm.triton_utils import triton
from vllm.v1.kv_cache_interface import KVQuantMode
scratch={};logged=set()
LIMIT=262144;QUERY_LIMIT=2048;BUDGET=320*1024**2

def eligible(d):
 if os.environ.get('QWEN_O5')!='1' or os.environ.get('QWEN_PF2')!='1':return False
 if os.environ.get('QWEN_O2_CONTROL_FILE'):return False
 q,k,v,out,ks,vs=(d.get(x) for x in ['q','k','v','out','k_scale_cache','v_scale_cache'])
 if any(x is None for x in [q,k,v,out,ks,vs]):return False
 if q.ndim!=3 or k.ndim!=4 or q.shape[-1]!=256 or not pf2.supported_heads(q,k) or v.shape!=k.shape or out.shape!=q.shape:return False
 if q.dtype!=torch.bfloat16 or out.dtype!=q.dtype or k.dtype!=torch.int8 or v.dtype!=k.dtype:return False
 if ks.dtype!=torch.float32 or vs.dtype!=ks.dtype or ks.stride()!=vs.stride() or k.stride()!=v.stride():return False
 if q.stride(-1)!=1 or out.stride(-1)!=1 or k.stride(-1)!=1 or k.shape[3] not in (256,272):return False
 if ks.shape!=k.shape[:3] or vs.shape!=ks.shape:return False
 if not q.is_cuda or any(x.device!=q.device for x in [k,v,out,ks,vs]):return False
 if d.get('kv_quant_mode')!=KVQuantMode.INT8_PER_TOKEN_HEAD or d.get('causal') is not True:return False
 if d.get('softmax_scale')!=.0625 or d.get('softcap') not in [None,0,0.] or d.get('window_size') not in [None,(-1,-1)]:return False
 if any(d.get(x) is not None for x in ['alibi_slopes','output_scale','qq_bias','sinks','mm_prefix_range','rswa_prefix_lens','rswa_window','q_descale','k_descale','v_descale']):return False
 if d.get('chunk_lookback',-1)!=-1 or d.get('use_td',False) or d.get('use_alibi_sqrt',False) or d.get('mm_prefix_clamp_sliding_window',False):return False
 lens,cu,bt=(d.get(x) for x in ['seqused_k','cu_seqlens_q','block_table']);n=d.get('max_seqlen_k');t=q.shape[0];mq=d.get('max_seqlen_q')
 if any(x is None for x in [lens,cu,bt]) or lens.ndim!=1 or cu.ndim!=1 or bt.ndim!=2:return False
 c=lens.numel()
 if not 1<=c<=4 or cu.numel()!=c+1 or bt.shape[0]!=c:return False
 if any(x.device!=q.device or x.dtype!=torch.int32 for x in [lens,cu,bt]) or lens.stride(0)!=1 or cu.stride(0)!=1 or bt.stride(1)!=1:return False
 if not isinstance(n,int) or not 256<=n<=LIMIT or not isinstance(mq,int) or not 256<=mq<=QUERY_LIMIT or not 256<=t<=QUERY_LIMIT:return False
 if mq>t or bt.shape[1]*k.shape[1]<n:return False
 # Preserve existing C1 routes and their arithmetic/workspace exactly.
 if c==1 and n<=131072:return False
 return True

def acquire(d):
 q,k=d['q'],d['k'];hq,hk=q.shape[1],k.shape[2];key=pf2.workspace_key(q,k);cap=131072 if d['max_seqlen_k']<=131072 else LIMIT
 # Bound extra streams before allocating. Primary E2 normally uses one stream.
 if any(k[0]==key[0] and k!=key for k in scratch.keys() | pf2.cache.keys() | o2.qc_cache.keys()):return None
 # TP1's four KV heads reuse the same arena in two passes. TP2 and TP4
 # retain one pass with respectively two and one KV heads.
 pass_hk=min(hk,2);pass_hq=pass_hk*(hq//hk)
 arena=pf2.acquire_kv(key,q.device,cap,pass_hk);queries=o2.acquire_q(key,q.device,pass_hq)
 if arena is None or queries is None:return None
 try:
  if key not in scratch:scratch[key]=torch.empty((4,3),device=q.device,dtype=torch.int32)
 except torch.OutOfMemoryError:return None
 total=sum(x.numel()*x.element_size() for x in pf2.cache[key]+o2.qc_cache[key]) + scratch[key].numel()*scratch[key].element_size()
 assert total<=BUDGET,(total,BUDGET)
 return key,total,arena+queries

def try_run(d,native):
 if not eligible(d) or torch.cuda.is_current_stream_capturing():return False
 arena=acquire(d)
 if arena is None:return False
 key,total,buffers=arena;q,k,v,out,ks,vs=(d[x] for x in ['q','k','v','out','k_scale_cache','v_scale_cache']);lens,cu,bt=(d[x] for x in ['seqused_k','cu_seqlens_q','block_table']);n=d['max_seqlen_k'];c=lens.numel();mq=d['max_seqlen_q'];t=q.shape[0];hq,hk=q.shape[1],k.shape[2]
 # Original metadata, query offsets and native arithmetic for every decode/
 # unsupported slice. The native kernel early-returns on optimized slices.
 args=dict(d);args['o5_fallback_only']=True;native(**args)
 use_int8=os.environ.get('QWEN_O2')=='1'
 dk,dv,dks,dvs,qc,qs=buffers;pass_hk=min(hk,2);pass_hq=pass_hk*(hq//hk)
 for head_start in range(0,hk,pass_hk):
  qstart=head_start*(hq//hk);qp=q[:,qstart:qstart+pass_hq];op=out[:,qstart:qstart+pass_hq]
  if use_int8:o2_kernels.quantize[(triton.cdiv(t,32),pass_hq)](qp,qc,qs,t,pass_hq,*qp.stride()[:2],num_warps=4)
  for seq in range(c):
   kernels.gather_sequence[(triton.cdiv(n,32),pass_hk)](k,v,ks,vs,bt,dk,dv,dks,dvs,lens,cu,scratch[key],seq,use_int8,k.shape[1],pass_hk,bt.stride(0),*k.stride()[:3],*ks.stride(),HEAD_START=head_start,num_warps=4)
   kernels.attention_bf16[(triton.cdiv(mq,32),pass_hq)](qp,dk,dv,dks,dvs,op,lens,cu,seq,use_int8,pass_hq,pass_hk,256,*qp.stride()[:2],*op.stride()[:2],num_warps=4,num_stages=2)
   if use_int8:kernels.attention_int8[(triton.cdiv(mq,32),pass_hq)](qp,qc,qs,dk,dv,dks,dvs,op,lens,cu,seq,use_int8,pass_hq,pass_hk,*qp.stride()[:2],*op.stride()[:2],False,num_warps=4,num_stages=2)
 marker=(q.device.index,c,n>131072,mq>=513)
 if marker not in logged:
  logged.add(marker);record={'pid':os.getpid(),'device':q.device.index,'requests':c,'total_query_tokens':t,'max_query_tokens':mq,'max_context_tokens':n,'hq':hq,'hk':hk,'KV_heads_per_pass':pass_hk,'head_passes':hk//pass_hk,'arena_bytes':total,'GPU_lengths_no_CPU_sync':True}
  # Optional audit performs an explicit readback, separately from timed runs.
  trace=os.environ.get('QWEN_O5_TRACE_DIR')
  if trace:
   record['actual_sequence_N_T_route']=scratch[key][:c].cpu().tolist();p=Path(trace);p.mkdir(parents=True,exist_ok=True)
   with (p/('dispatch-'+str(os.getpid())+'.jsonl')).open('a') as f:f.write(json.dumps(record)+'\n')
  print('QWEN_O5_DISPATCH '+json.dumps(record),flush=True)
 return True
