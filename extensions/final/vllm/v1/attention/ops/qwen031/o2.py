"""Default-off, bulk-only O2 serving experiment; production files stay untouched."""
import json
import os
from pathlib import Path
import torch
from . import pf2, o2_kernels
from vllm.triton_utils import triton

qc_cache={}
logged=set()
control_cache={'mtime':None,'data':{'enabled':False,'label':'missing'}}
probed=False


def acquire_q(key,device,heads):
    """Reuse contiguous storage when O5 processes a subset of TP1 heads."""
    assert heads in (6,12,24)
    current=qc_cache.get(key)
    if current is None or current[0].numel()<2048*heads*256:
        qc_cache.pop(key,None);current=None
        try:
            qc_cache[key]=(torch.empty((2048,heads,256),device=device,dtype=torch.int8),
                           torch.empty((2048,heads),device=device,dtype=torch.float32))
        except torch.OutOfMemoryError:return None
    qc,qs=qc_cache[key]
    return qc.view(-1,heads,256),qs.view(-1,heads)


def control():
    name=os.environ.get('QWEN_O2_CONTROL_FILE')
    if not name:return {'enabled':True,'label':'static'}
    try:
        p=Path(name);stamp=p.stat().st_mtime_ns
        if stamp!=control_cache['mtime']:
            control_cache.update(mtime=stamp,data=json.loads(p.read_text()))
        return control_cache['data']
    except (OSError,ValueError):return {'enabled':False,'label':'missing'}


def real_probe(q,qc,qs,dk,dv,dks,dvs,out,n,t,phase):
    """Non-timing, synthetic-input real-model reference on verified physical GPU1."""
    global probed
    if probed or q.device.index!=1 or n!=32768:return
    probed=True
    import numpy as np
    original=q.float().cpu().numpy()
    codes=qc[:t].cpu().numpy();scales=qs[:t].cpu().numpy()
    k=dk[:n].cpu().numpy();v=dv[:n].cpu().numpy()
    sk=dks[:n].cpu().numpy();sv=dvs[:n].cpu().numpy()
    actual=out.float().cpu().numpy()
    mx=np.max(np.abs(original),axis=-1)
    refscale=np.where(mx>0,mx/np.float32(127),np.float32(1)).astype(np.float32)
    refcode=np.clip(np.rint(original/refscale[...,None]),-127,127).astype(np.int8)
    qs_exact=bool(np.array_equal(scales,refscale));qc_exact=bool(np.array_equal(codes,refcode))
    hq=q.shape[1];hk=dk.shape[1]
    rows=[(0,0),(t//2,(hq-1)//2),(t-1,hq//2),(t-1,hq-1)]
    orig_rows=[];quant_rows=[];got=[]
    for i,h in rows:
        count=n-t+i+1;kh=h//(hq//hk)
        keys=k[:count,kh].astype(np.float64)*sk[:count,kh,None].astype(np.float64)
        values=v[:count,kh].astype(np.float64)*sv[:count,kh,None].astype(np.float64)
        refs=[]
        for query in [original[i,h].astype(np.float64),codes[i,h].astype(np.float64)*float(scales[i,h])]:
            scores=keys@query*.0625;p=np.exp(scores-np.max(scores));refs.append((p@values)/p.sum())
        orig_rows.append(refs[0]);quant_rows.append(refs[1]);got.append(actual[i,h])
    orig_rows=np.asarray(orig_rows);quant_rows=np.asarray(quant_rows);got=np.asarray(got)
    def diff(a,b):
        e=a-b
        return {'max_abs':float(np.abs(e).max()),'relative_l2':float(np.linalg.norm(e)/max(np.linalg.norm(b),1e-30))}
    impl=diff(got,quant_rows)
    d={'kind':'real-model synthetic-input independent FP64','device_index':q.device.index,'N':n,'T':t,
       'phase':phase,'selected_rows':rows,'codes_exact':qc_exact,'scales_exact':qs_exact,
       'finite_all_output':bool(np.isfinite(actual).all()),'implementation_vs_quantized_FP64':impl,
       'quantization_only_vs_original_FP64':diff(quant_rows,orig_rows),'total_vs_original_FP64':diff(got,orig_rows),
       'query_absmax':float(mx.max()),'query_rms':float(np.sqrt(np.mean(original.astype(np.float64)**2)))}
    d['ok']=qc_exact and qs_exact and d['finite_all_output'] and impl['max_abs']<.005 and impl['relative_l2']<.01
    root=Path(os.environ['QWEN_O2_PROBE_ROOT']);root.mkdir(exist_ok=True)
    (root/'real-model-numeric.json').write_text(json.dumps(d,indent=2))
    print('QWEN_O2_REAL_NUMERIC '+json.dumps(d),flush=True)


def try_run(d):
    if os.environ.get('QWEN_O2')!='1' or not pf2.eligible(d):return False
    q=d['q'];t=q.shape[0];n=d['max_seqlen_k']
    if not 513<=t<=2048 or not 16384<=n<=131072:return False
    if torch.cuda.is_current_stream_capturing():return False
    phase=control()
    if not phase.get('enabled'):return False
    out,k,v,ks,vs=(d[x] for x in ['out','k','v','k_scale_cache','v_scale_cache'])
    hq,hk=q.shape[1],k.shape[2];key=pf2.workspace_key(q,k)
    if any(k[0]==key[0] and k!=key for k in pf2.cache.keys() | qc_cache.keys()):return False
    arena=pf2.acquire_kv(key,q.device,131072,hk);queries=acquire_q(key,q.device,hq)
    if arena is None or queries is None:return False
    dk,dv,dks,dvs=arena;qc,qs=queries
    pf2.gather[(n,hk)](k,v,ks,vs,d['block_table'],dk,dv,dks,dvs,d['seqused_k'],k.shape[1],hk,256,
                     *k.stride()[:3],*ks.stride(),num_warps=4)
    o2_kernels.quantize[(triton.cdiv(t,32),hq)](q,qc,qs,t,hq,*q.stride()[:2],num_warps=4)
    o2_kernels.attention[(triton.cdiv(t,32),hq)](q,qc,qs,dk,dv,dks,dvs,out,d['seqused_k'],t,hq,hk,
                      *q.stride()[:2],*out.stride()[:2],False,num_warps=4,num_stages=2)
    marker=(q.device.index,n,t,phase.get('label'))
    if marker not in logged:
        logged.add(marker)
        print('QWEN_O2_HIT '+json.dumps({'pid':os.getpid(),'device':q.device.index,'q':t,'n':n,
              'phase':phase.get('label'),'hq':hq,'hk':hk,'Q_scratch_bytes':2048*hq*260,'route':'dense-separate-INT8-QK'}),flush=True)
    if phase.get('probe'):real_probe(q,qc,qs,dk,dv,dks,dvs,out,n,t,phase.get('label'))
    return True
