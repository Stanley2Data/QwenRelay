import triton
import triton.language as tl
from .o2_kernels import quantize_rows

@triton.jit
def gather_sequence(K,V,KS,VS,BT,DK,DV,DKS,DVS,L,CU,STATS,SEQ:tl.constexpr,USE_INT8:tl.constexpr,PAGE:tl.constexpr,HK:tl.constexpr,BT0:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,T0:tl.constexpr,T1:tl.constexpr,T2:tl.constexpr,BLOCK:tl.constexpr=32,HEAD_START:tl.constexpr=0):
 n=tl.load(L+SEQ);t=tl.load(CU+SEQ+1)-tl.load(CU+SEQ)
 active=(t>=256)&(t<=2048)&(n>=t)&(n<=262144)
 if (tl.program_id(0)==0)&(tl.program_id(1)==0):
  route=tl.where(active,tl.where(USE_INT8&(t>=513)&(n>=16384)&(n<=131072),2,1),0)
  tl.store(STATS+SEQ*3,n);tl.store(STATS+SEQ*3+1,t);tl.store(STATS+SEQ*3+2,route)
 if not active:return
 x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);h=tl.program_id(1);d=tl.arange(0,256)
 block=tl.load(BT+SEQ*BT0+x//PAGE,x<n,0);slot=x%PAGE
 src=block[:,None].to(tl.int64)*S0+slot[:,None]*S1+(h+HEAD_START)*S2+d[None,:];dst=(x[:,None]*HK+h)*256+d[None,:]
 tl.store(DK+dst,tl.load(K+src,x[:,None]<n,0),x[:,None]<n);tl.store(DV+dst,tl.load(V+src,x[:,None]<n,0),x[:,None]<n)
 si=block.to(tl.int64)*T0+slot*T1+(h+HEAD_START)*T2
 tl.store(DKS+x*HK+h,tl.load(KS+si,x<n,0),x<n);tl.store(DVS+x*HK+h,tl.load(VS+si,x<n,0),x<n)

@triton.jit
def attention_bf16(Q,K,V,KS,VS,O,L,CU,SEQ:tl.constexpr,USE_INT8:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,QS0:tl.constexpr,QS1:tl.constexpr,OS0:tl.constexpr,OS1:tl.constexpr,M:tl.constexpr=32,B:tl.constexpr=32):
 qstart=tl.load(CU+SEQ);T=tl.load(CU+SEQ+1)-qstart;n=tl.load(L+SEQ)
 if (T<256) | (T>2048) | (n<T) | (n>262144):return
 if USE_INT8:
  if (T>=513) & (n>=16384) & (n<=131072):return
 Q+=qstart*QS0;O+=qstart*OS0;L+=SEQ
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

@triton.jit
def attention_int8(Q, QC, QS, K, V, KS, VS, O, L, CU, SEQ: tl.constexpr, USE_INT8: tl.constexpr,
              HQ: tl.constexpr, HK: tl.constexpr,
              Q0: tl.constexpr, Q1: tl.constexpr,
              O0: tl.constexpr, O1: tl.constexpr,
              INLINE: tl.constexpr, M: tl.constexpr = 32,
              B: tl.constexpr = 32):
    qstart=tl.load(CU+SEQ);T=tl.load(CU+SEQ+1)-qstart;n=tl.load(L+SEQ)
    if (not USE_INT8) | (T<513) | (T>2048) | (n<16384) | (n>131072) | (n<T):return
    Q+=qstart*Q0;O+=qstart*O0;QC+=qstart*HQ*256;QS+=qstart*HQ;L+=SEQ
    qi = tl.program_id(0) * M + tl.arange(0, M)
    head = tl.program_id(1)
    kh = head // (HQ // HK)
    dd = tl.arange(0, 256)
    kk = tl.arange(0, B)
    n = tl.load(L)
    if INLINE:
        original = tl.load(Q + qi[:, None] * Q0 + head * Q1 + dd[None, :], qi[:, None] < T, 0)
        q, sq = quantize_rows(original)
    else:
        q = tl.load(QC + (qi[:, None] * HQ + head) * 256 + dd[None, :], qi[:, None] < T, 0)
        sq = tl.load(QS + qi * HQ + head, qi < T, 1.)
    acc = tl.zeros((M, 256), tl.float32)
    maximum = tl.full((M,), float('-inf'), tl.float32)
    denominator = tl.zeros((M,), tl.float32)
    end = tl.minimum(n, n - T + (tl.program_id(0) + 1) * M)
    for block in range(0, tl.cdiv(end, B)):
        pos = block * B + kk
        k = tl.load(K + (pos[None, :] * HK + kh) * 256 + dd[:, None], pos[None, :] < n, 0)
        v = tl.load(V + (pos[:, None] * HK + kh) * 256 + dd[None, :], pos[:, None] < n, 0).to(tl.bfloat16)
        sk = tl.load(KS + pos * HK + kh, pos < n, 0.)
        sv = tl.load(VS + pos * HK + kh, pos < n, 0.)
        integer_dot = tl.dot(q, k, out_dtype=tl.int32)
        score = integer_dot.to(tl.float32) * (sq[:, None] * ((0.0625 * 1.4426950408889634) * sk[None, :]))
        mask = (pos[None, :] <= n - T + qi[:, None]) & (pos[None, :] < n) & (qi[:, None] < T)
        score = tl.where(mask, score, float('-inf'))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        new_max = tl.where(new_max == float('-inf'), 0., new_max)
        p = tl.exp2(score - new_max[:, None])
        alpha = tl.exp2(maximum - new_max)
        denominator = denominator * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot((p * sv[None, :]).to(tl.bfloat16), v)
        maximum = new_max
    tl.store(O + qi[:, None] * O0 + head * O1 + dd[None, :],
             acc / denominator[:, None], qi[:, None] < T)
