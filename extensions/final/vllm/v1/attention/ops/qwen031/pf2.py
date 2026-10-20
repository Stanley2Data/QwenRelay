"""Bounded TP1/TP2/TP4 C1 prefill; unsupported shapes retain the vLLM path."""
import os,json
import torch
from vllm.triton_utils import triton,tl
from vllm.v1.kv_cache_interface import KVQuantMode
cache={};logged=set()

def supported_heads(q,k):
 return (q.shape[1],k.shape[2]) in ((24,4),(12,2),(6,1))

def workspace_key(q,k):
 return (q.device.index,torch.cuda.current_stream(q.device).cuda_stream,q.shape[1],k.shape[2])

def acquire_kv(key,device,cap,heads):
 """Share token-head storage between PF2/O2 and O5's bounded head passes."""
 required=cap*heads
 assert required<=524288 and heads in (1,2,4)
 current=cache.get(key)
 if current is None or current[0].numel()//256<required:
  cache.pop(key,None);current=None
  try:
   dk=torch.empty((cap,heads,256),device=device,dtype=torch.int8);dv=torch.empty_like(dk)
   dks=torch.empty((cap,heads),device=device,dtype=torch.float32);dvs=torch.empty_like(dks)
   cache[key]=(dk,dv,dks,dvs)
  except torch.OutOfMemoryError:return None
 dk,dv,dks,dvs=cache[key]
 return dk.view(-1,heads,256),dv.view(-1,heads,256),dks.view(-1,heads),dvs.view(-1,heads)

@triton.jit
def gather(K,V,KS,VS,BT,DK,DV,DKS,DVS,L,PAGE:tl.constexpr,H:tl.constexpr,D:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,T0:tl.constexpr,T1:tl.constexpr,T2:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);d=tl.arange(0,D);n=tl.load(L)
 block=tl.load(BT+t//PAGE,t<n,0);slot=t%PAGE
 src=block*S0+slot*S1+h*S2+d;dst=(t*H+h)*D+d
 tl.store(DK+dst,tl.load(K+src,t<n,0));tl.store(DV+dst,tl.load(V+src,t<n,0))
 si=block*T0+slot*T1+h*T2
 tl.store(DKS+t*H+h,tl.load(KS+si,t<n,0));tl.store(DVS+t*H+h,tl.load(VS+si,t<n,0))

@triton.jit
def attention(Q,K,V,KS,VS,O,L,T:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,QS0:tl.constexpr,QS1:tl.constexpr,OS0:tl.constexpr,OS1:tl.constexpr,M:tl.constexpr=32,B:tl.constexpr=32):
 # 20260929-pf2-tile-v1: query tile 16->32 (+2-stage pipelining at launch) halves K/V re-reads;
 # exp -> exp2 with log2(e) folded into the scale. Same guards/gather as 3fd70d66.
 n=tl.load(L);qi=tl.program_id(0)*M+tl.arange(0,M);h=tl.program_id(1);kh=h//(HQ//HK)
 ds=tl.arange(0,D);ns=tl.arange(0,B)
 q=tl.load(Q+qi[:,None]*QS0+h*QS1+ds[None,:],qi[:,None]<T,0)
 acc=tl.zeros((M,D),tl.float32);mx=tl.full((M,),float('-inf'),tl.float32);den=tl.zeros((M,),tl.float32)
 end=tl.minimum(n,n-T+(tl.program_id(0)+1)*M)
 for start in range(0,tl.cdiv(end,B)):
  pos=start*B+ns
  k=tl.load(K+(pos[None,:]*HK+kh)*D+ds[:,None],pos[None,:]<n,0).to(q.dtype)
  v=tl.load(V+(pos[:,None]*HK+kh)*D+ds[None,:],pos[:,None]<n,0).to(q.dtype)
  ks=tl.load(KS+pos*HK+kh,pos<n,0);vs=tl.load(VS+pos*HK+kh,pos<n,0)
  score=tl.dot(q,k)*((.0625*1.4426950408889634)*ks[None,:])
  score=tl.where((pos[None,:]<=n-T+qi[:,None])&(pos[None,:]<n)&(qi[:,None]<T),score,float('-inf'))
  new=tl.maximum(mx,tl.max(score,1));new=tl.where(new==float('-inf'),0.,new)
  p=tl.exp2(score-new[:,None]);alpha=tl.exp2(mx-new);den=den*alpha+tl.sum(p,1)
  acc=acc*alpha[:,None]+tl.dot((p*vs[None,:]).to(q.dtype),v);mx=new
 tl.store(O+qi[:,None]*OS0+h*OS1+ds[None,:],acc/den[:,None],qi[:,None]<T)

def eligible(d):
 q=d.get('q');k=d.get('k');v=d.get('v');out=d.get('out');ks=d.get('k_scale_cache');vs=d.get('v_scale_cache')
 if any(x is None for x in [q,k,v,out,ks,vs]):return False
 if not q.is_cuda or any(x.device!=q.device for x in [k,v,out,ks,vs]):return False
 if q.ndim!=3 or k.ndim!=4 or v.shape!=k.shape or q.shape[-1]!=256 or not supported_heads(q,k):return False
 if q.dtype!=torch.bfloat16 or k.dtype!=torch.int8 or v.dtype!=torch.int8 or out.dtype!=q.dtype:return False
 if ks.dtype!=torch.float32 or vs.dtype!=torch.float32 or ks.stride()!=vs.stride() or k.stride()!=v.stride():return False
 if q.stride(-1)!=1 or k.stride(-1)!=1 or out.stride(-1)!=1 or out.shape!=q.shape or k.shape[3] not in (256,272):return False
 if d.get('kv_quant_mode')!=KVQuantMode.INT8_PER_TOKEN_HEAD or d.get('causal') is not True:return False
 if d.get('softmax_scale')!=.0625 or d.get('softcap') not in [None,0,0.] or d.get('window_size') not in [None,(-1,-1)]:return False
 if any(d.get(x) is not None for x in ['alibi_slopes','output_scale','qq_bias','sinks','mm_prefix_range','rswa_prefix_lens','rswa_window','q_descale','k_descale','v_descale']):return False
 if d.get('mm_prefix_clamp_sliding_window',False):return False
 if d.get('chunk_lookback',-1)!=-1 or d.get('use_td',False) or d.get('use_alibi_sqrt',False):return False
 lens=d.get('seqused_k');cu=d.get('cu_seqlens_q');bt=d.get('block_table');n=d.get('max_seqlen_k');t=q.shape[0]
 if lens is None or lens.numel()!=1 or cu is None or cu.numel()!=2 or bt is None or bt.shape[0]!=1:return False
 if not isinstance(n,int) or not 32<=t<=2048 or d.get('max_seqlen_q')!=t or not t<=n<=131072:return False
 if any(x.device!=q.device or x.dtype!=torch.int32 for x in [lens,cu,bt]):return False
 if lens.ndim!=1 or cu.ndim!=1 or bt.ndim!=2 or lens.stride(0)!=1 or cu.stride(0)!=1 or bt.stride(1)!=1:return False
 if ks.shape!=k.shape[:3] or vs.shape!=ks.shape:return False
 if bt.shape[1]*k.shape[1]<n:return False
 return True

def try_run(d):
 if os.environ.get('QWEN_PF2')!='1' or not eligible(d):return False
 if torch.cuda.is_current_stream_capturing():return False
 q,k,v,out,ks,vs=(d[x] for x in ['q','k','v','out','k_scale_cache','v_scale_cache']);n=d['max_seqlen_k'];t=q.shape[0]
 # Shape-isolated storage: TP1 <=260MiB, TP2 <=130MiB, TP4 <=65MiB here.
 hq,hk=q.shape[1],k.shape[2];key=workspace_key(q,k)
 if any(k[0]==key[0] and k!=key for k in cache):return False
 arena=acquire_kv(key,q.device,131072,hk)
 if arena is None:return False
 dk,dv,dks,dvs=arena
 gather[(n,hk)](k,v,ks,vs,d['block_table'],dk,dv,dks,dvs,d['seqused_k'],k.shape[1],hk,256,*k.stride()[:3],*ks.stride(),num_warps=4)
 attention[(triton.cdiv(t,32),hq)](q,dk,dv,dks,dvs,out,d['seqused_k'],t,hq,hk,256,*q.stride()[:2],*out.stride()[:2],num_warps=4,num_stages=2)
 logkey=(key, 'long' if n>65536 else 'short')
 if logkey not in logged:
  logged.add(logkey);print('QWEN_PF2_HIT '+json.dumps({'pid':os.getpid(),'device':q.device.index,'q':t,'n':n,'hq':hq,'hk':hk,'workspace_bytes':sum(x.numel()*x.element_size() for x in cache[key])}),flush=True)
 return True
