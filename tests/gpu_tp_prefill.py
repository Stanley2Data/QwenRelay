#!/usr/bin/env python3
"""GPU numerical checks for bounded TP1/TP2/TP4 prefill overlays.

Run in the installed QwenRelay venv while its selected GPUs are idle.
Independent NumPy FP64 rows cover the quantized arithmetic; native vLLM
checks the O5 fallback slices. This file is not a performance benchmark.
"""
import argparse
import json
import os
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.update(QWEN_PF2='1', QWEN_O2='1', QWEN_O5='1')
os.environ.pop('QWEN_O2_CONTROL_FILE', None)

import numpy as np
import torch
from vllm.v1.attention.ops.qwen031 import pf2, o2, o5
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode


def clear():
    pf2.cache.clear()
    o2.qc_cache.clear()
    o5.scratch.clear()
    torch.cuda.empty_cache()


def make_case(device, hq, hk, ns, ts):
    rng = np.random.default_rng(20261018)
    page = 800
    counts = [(n + page - 1) // page for n in ns]
    blocks = sum(counts)
    table = np.zeros((len(ns), max(counts)), np.int32)
    order = rng.permutation(blocks)
    cursor = 0
    for i, count in enumerate(counts):
        table[i, :count] = order[cursor:cursor + count]
        cursor += count
    k = torch.full((blocks, page, hk, 272), 17, device=device, dtype=torch.int8)
    v = torch.full_like(k, 13)
    ks = k[..., 256:260].view(torch.float32)[..., 0]
    vs = v[..., 256:260].view(torch.float32)[..., 0]
    logical = []
    for i, n in enumerate(ns):
        kc = rng.integers(-80, 81, size=(n, hk, 256), dtype=np.int8)
        vc = rng.integers(-80, 81, size=(n, hk, 256), dtype=np.int8)
        sk = rng.uniform(.005, .025, (n, hk)).astype(np.float32)
        sv = rng.uniform(.005, .025, (n, hk)).astype(np.float32)
        pos = np.arange(n)
        physical = torch.as_tensor(table[i, pos // page], device=device)
        offset = torch.as_tensor(pos % page, device=device)
        k[physical, offset, :, :256] = torch.as_tensor(kc, device=device)
        v[physical, offset, :, :256] = torch.as_tensor(vc, device=device)
        ks[physical, offset] = torch.as_tensor(sk, device=device)
        vs[physical, offset] = torch.as_tensor(sv, device=device)
        logical.append((kc, vc, sk, sv))
    q = torch.as_tensor(rng.normal(0, .5, (sum(ts), hq, 256)).astype(np.float32),
                        device=device, dtype=torch.bfloat16)
    q[0].zero_()
    out = torch.full_like(q, float('nan'))
    cu = np.concatenate(([0], np.cumsum(ts))).astype(np.int32)
    args = dict(q=q, k=k, v=v, out=out,
                cu_seqlens_q=torch.as_tensor(cu, device=device), max_seqlen_q=max(ts),
                seqused_k=torch.as_tensor(ns, device=device, dtype=torch.int32),
                max_seqlen_k=max(ns), softmax_scale=.0625, causal=True,
                window_size=(-1, -1), block_table=torch.as_tensor(table, device=device),
                softcap=0., q_descale=None, k_descale=None, v_descale=None,
                kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
                k_scale_cache=ks, v_scale_cache=vs)
    return args, logical, cu


def compare_rows(args, logical, cu, ns, ts, int8_sequences):
    queries = args['q'].float().cpu().numpy()
    actual = args['out'].float().cpu().numpy()
    hq, hk = queries.shape[1], logical[0][0].shape[1]
    got, expected = [], []
    for seq, (n, t) in enumerate(zip(ns, ts)):
        kc, vc, sk, sv = logical[seq]
        for head in range(hq):
            kh = head // (hq // hk)
            keys = kc[:, kh].astype(np.float64) * sk[:, kh, None].astype(np.float64)
            values = vc[:, kh].astype(np.float64) * sv[:, kh, None].astype(np.float64)
            for local in sorted({0, t // 2, t - 1}):
                row = int(cu[seq]) + local
                query = queries[row, head]
                if seq in int8_sequences:
                    mx = np.max(np.abs(query))
                    scale = np.float32(mx / np.float32(127)) if mx > 0 else np.float32(1)
                    code = np.clip(np.rint(query / scale), -127, 127).astype(np.int8)
                    query = code.astype(np.float64) * float(scale)
                count = n - t + local + 1
                scores = keys[:count] @ query.astype(np.float64) * .0625
                weights = np.exp(scores - scores.max())
                reference = (weights @ values[:count]) / weights.sum()
                expected.append(reference)
                got.append(actual[row, head])
    got, expected = np.asarray(got), np.asarray(expected)
    delta = got - expected
    result = {'max_abs': float(np.abs(delta).max()),
              'relative_l2': float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-30)),
              'finite_all': bool(np.isfinite(actual).all()), 'reference_rows': len(got)}
    assert result['finite_all'] and result['max_abs'] < .005 and result['relative_l2'] < .01, result
    assert bool(torch.all(args['k'][..., 260:] == 17))
    assert bool(torch.all(args['v'][..., 260:] == 13))
    return result


def run(device, hq, hk, route, ns, ts, use_int8=True):
    clear()
    os.environ['QWEN_O2'] = '1' if use_int8 else '0'
    args, logical, cu = make_case(device, hq, hk, ns, ts)
    native = unified_attention.__wrapped__
    fallback_checks = []
    if route == 'pf2':
        assert pf2.try_run(args)
        quantized = set()
    elif route == 'o2':
        assert o2.try_run(args)
        quantized = {0}
        qc, qs = o2.qc_cache[pf2.workspace_key(args['q'], args['k'])]
        source = args['q'].float().cpu().numpy()
        maxima = np.max(np.abs(source), axis=-1)
        scales = np.where(maxima > 0, maxima / np.float32(127), np.float32(1)).astype(np.float32)
        codes = np.clip(np.rint(source / scales[..., None]), -127, 127).astype(np.int8)
        assert np.array_equal(qc[:sum(ts)].cpu().numpy(), codes)
        assert np.array_equal(qs[:sum(ts)].cpu().numpy(), scales)
    elif route == 'native':
        assert not pf2.eligible(args) and not o5.eligible(args)
        unified_attention(**args)
        quantized = set()
    else:
        baseline = torch.empty_like(args['out'])
        native(**{**args, 'out': baseline})

        def fallback(**kwargs):
            native(**kwargs)
            for seq, (n, t) in enumerate(zip(ns, ts)):
                part = args['out'][int(cu[seq]):int(cu[seq + 1])]
                optimized = 256 <= t <= 2048 and t <= n <= 262144
                if optimized:
                    assert bool(torch.isnan(part).all()), 'Native overwrote optimized slice'
                else:
                    assert torch.equal(part, baseline[int(cu[seq]):int(cu[seq + 1])])
                    fallback_checks.append(seq)

        assert o5.try_run(args, fallback)
        quantized = {i for i, (n, t) in enumerate(zip(ns, ts))
                     if use_int8 and 513 <= t <= 2048 and 16384 <= n <= 131072}
    torch.cuda.synchronize(device)
    result = compare_rows(args, logical, cu, ns, ts, quantized)
    for seq in fallback_checks:
        assert torch.equal(args['out'][int(cu[seq]):int(cu[seq + 1])],
                           baseline[int(cu[seq]):int(cu[seq + 1])])
    result.update(device=device, hq=hq, hk=hk, route=route, ns=ns, ts=ts,
                  use_int8=use_int8, native_fallback_sequences=fallback_checks,
                  workspace_bytes=sum(t.numel() * t.element_size() for t in
                                      pf2.cache.get(pf2.workspace_key(args['q'], args['k']), ()) ))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', default='0')
    parser.add_argument('--tp', type=int, choices=(1, 2, 4),
                        help='Test one TP geometry, or all three when omitted')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for device_number,index in enumerate(map(int, args.devices.split(','))):
        device = f'cuda:{index}'
        torch.cuda.set_device(index)
        shapes = {1: (24, 4), 2: (12, 2), 4: (6, 1)}
        geometries = [shapes[args.tp]] if args.tp is not None else list(shapes.values())
        for hq, hk in geometries:
            cases = [('pf2', [1073], [65], False),
                     ('o2', [16384], [513], True),
                     ('o5', [16384, 1025, 1009, 267], [513, 256, 1, 67], True),
                     ('o5', [16384, 1025, 1009, 267], [513, 256, 1, 67], False)]
            if device_number == 0:
                cases.append(('o5', [131089], [256], True))
            for route, ns, ts, use_int8 in cases:
                result = run(device, hq, hk, route, ns, ts, use_int8)
                results.append(result)
                args.output.write_text(json.dumps({'passed': False, 'completed': results}, indent=2))
                print(json.dumps(result), flush=True)
        if device_number == 0:
            result = run(device, 8, 2, 'native', [1073], [65], True)
            results.append(result)
            print(json.dumps(result), flush=True)
        clear()
    args.output.write_text(json.dumps({'passed': True, 'completed': results}, indent=2))


if __name__ == '__main__':
    main()
