#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""QwenRelay source reconstruction, asset validation and TP1/TP2/TP4 launch entry."""
from __future__ import annotations

import argparse
import ast
import copy
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TP_PROFILES = {1: 'single-3090', 2: 'dual-3090', 4: 'quad-3090'}
SERVE_OVERRIDES = ('max_num_seqs', 'max_model_len', 'draft_tokens',
                   'kv_cache_memory', 'kv_offloading_size', 'max_num_batched_tokens')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def disk(path):
    path = Path(path).resolve()
    return Path('\\\\?\\' + str(path)) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def sha(path):
    with disk(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def canonical_bytes(data):
    """Normalize only UTF-8 text line endings, retaining binary file bytes."""
    if b'\0' in data:
        return data
    try:
        data.decode('utf-8')
    except UnicodeDecodeError:
        return data
    return data.replace(b'\r\n', b'\n')


def posix(path):
    path = Path(path).resolve().as_posix()
    return '/' + path[0].lower() + path[2:] if os.name == 'nt' else path


def safe_path(root, relative):
    if not isinstance(relative, str) or '\\' in relative:
        raise ValueError('Expected a relative POSIX path')
    p = Path(relative)
    result = (Path(root) / p).resolve()
    if p.is_absolute() or '..' in p.parts or not result.is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes its root: ' + relative)
    return result


def inventory(root, *, canonical=False):
    root = disk(root)
    rows = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ('.git', '__pycache__'))
        for name in sorted(files):
            p = Path(base) / name
            if p.is_symlink():
                raise ValueError('Source symlinks are unsupported: ' + str(p))
            data = canonical_bytes(p.read_bytes()) if canonical else p.read_bytes()
            rows.append({'path':p.relative_to(root).as_posix(), 'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()})
    return sorted(rows, key=lambda x:x['path'])


def verify_files(root, rows, *, exact=False, canonical=False):
    for row in rows:
        p = safe_path(root, row['path'])
        if not disk(p).is_file():
            raise ValueError('Missing file: ' + row['path'])
        if canonical:
            data = canonical_bytes(disk(p).read_bytes())
            actual_sha, actual_size = hashlib.sha256(data).hexdigest(), len(data)
        else:
            actual_sha, actual_size = sha(p), disk(p).stat().st_size
        if actual_sha != row['sha256']:
            raise ValueError('SHA256 mismatch or missing file: ' + row['path'])
        if 'bytes' in row and actual_size != row['bytes']:
            raise ValueError('Size mismatch: ' + row['path'])
    if exact and inventory(root,canonical=canonical) != sorted(rows, key=lambda x:x['path']):
        raise ValueError('Unexpected files in source tree')


def source_lock():
    return read_json(ROOT / 'locks/source.json')


def check_pack():
    lock = source_lock()
    verify_files(ROOT, lock['upstream_patch_inputs'])
    verify_files(ROOT, lock['local_series'])
    verify_files(ROOT / 'extensions/final', lock['final_overlay'])
    verify_files(ROOT, read_json(ROOT/'locks/models-tp1.json')['preparation']['script_inputs'])
    return lock


def bash_run(bash, args, *, cwd=None):
    p = subprocess.run([bash, *args], cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stdout + p.stderr)
    return p.stdout


def rebuild(source, target, bash='bash'):
    """Read-only input, new output only; stop at the first identity mismatch."""
    lock = check_pack()
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists() or target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Target must be new and separate from the source')
    verify_files(source / 'vllm', lock['pristine_vllm'], exact=True,canonical=True)
    if hashlib.sha256(canonical_bytes((source/'LICENSE').read_bytes())).hexdigest() != sha(ROOT/'licenses/vllm-LICENSE'):
        raise ValueError('Pristine vLLM license identity mismatch')
    target.mkdir(parents=True)
    for row in lock['pristine_vllm']:
        out = safe_path(target/'vllm',row['path'])
        disk(out.parent).mkdir(parents=True,exist_ok=True)
        disk(out).write_bytes(canonical_bytes(disk(source/'vllm'/row['path']).read_bytes()))
    shutil.copyfile(ROOT / 'licenses/vllm-LICENSE', target / 'LICENSE')
    shutil.copyfile(ROOT / 'LICENSE', target / 'HYPERQWEN-LICENSE')
    if (ROOT / 'licenses/vllm-NOTICE').exists():
        shutil.copyfile(ROOT / 'licenses/vllm-NOTICE', target / 'NOTICE')
    logs = []
    logs.append(bash_run(bash, [posix(ROOT/'patches/apply.sh'), posix(target/'vllm')]))
    shutil.copytree(disk(ROOT/'kvarn/files/vllm'), disk(target/'vllm'), dirs_exist_ok=True)
    logs.append(bash_run(bash, [posix(ROOT/'patches/apply.sh'), '--kvarn', posix(target/'vllm')]))
    for row in lock['local_series']:
        patch = posix(ROOT / row['path'])
        for dry in (True, False):
            cmd = 'cd "$1" && patch -p1 --batch --forward --fuzz=0 ' + ('--dry-run ' if dry else '') + '< "$2"'
            logs.append(bash_run(bash, ['-c',cmd,'qwenrelay',posix(target),patch]))
    verify_files(target, lock['base_F'], exact=True)
    for row in lock['final_overlay']:
        out = safe_path(target, row['path'])
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/'extensions/final'/row['path'], out)
    verify_files(target, lock['final_F'], exact=True)
    for p in disk(target/'vllm').rglob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    result = {'passed':True,'source_files':len(lock['final_F']), 'source_lock_sha256':sha(ROOT/'locks/source.json'),
              'upstream_patch_count':sum(s.count('\n== ') + int(s.startswith('== ')) for s in logs[:2]),
              'gpu_or_model_executed':False,'stages':logs}
    with target.with_name(target.name + '.verification.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result,f,indent=2);f.write('\n')
    return {k:v for k,v in result.items() if k != 'stages'}


def install_source(source):
    """Overlay source into this explicitly selected venv; keep wheel native code."""
    if sys.prefix == sys.base_prefix:
        raise ValueError('Run install-source using the new virtual environment Python')
    if importlib.metadata.version('vllm') != '0.31.0':
        raise ValueError('Only vLLM 0.31.0 is supported')
    lock = check_pack()
    verify_files(source, lock['final_F'], exact=True)
    dist = importlib.metadata.distribution('vllm')
    package = Path(dist.locate_file('vllm')).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError('Refusing a package outside this virtual environment')
    marker = package/'.qwenrelay-source.json'
    if marker.exists():
        raise ValueError('Use a new venv; this one is already modified')
    native = [{'path':p.relative_to(package).as_posix(),'sha256':sha(p)} for p in package.rglob('*.so')]
    if not native:
        raise ValueError('The Linux CUDA wheel native extensions are missing')
    for row in lock['final_F']:
        if not row['path'].startswith('vllm/'):
            continue
        p = package/row['path'][5:]
        p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(disk(Path(source)/row['path']),disk(p))
    verify_files(package, native)
    verify_files(package, [{**r,'path':r['path'][5:]} for r in lock['final_F'] if r['path'].startswith('vllm/')])
    marker.write_text(json.dumps({'source_lock_sha256':sha(ROOT/'locks/source.json'),'native_files':native},indent=2)+'\n')
    return {'passed':True,'native_preserved':len(native),'source_lock_sha256':sha(ROOT/'locks/source.json')}


def check_runtime():
    expected = {'vllm':'0.31.0','torch':'2.13.0','triton':'3.7.1','transformers':'5.15.0','compressed-tensors':'0.17.0'}
    for name, version in expected.items():
        actual = importlib.metadata.version(name)
        if actual.split('+')[0] != version:
            raise ValueError(f'{name}: expected {version}, got {actual}')
    import vllm
    package = Path(vllm.__file__).resolve().parent
    marker = read_json(package/'.qwenrelay-source.json')
    if marker['source_lock_sha256'] != sha(ROOT/'locks/source.json'):
        raise ValueError('Runtime belongs to a different source lock')
    verify_files(package, [{**r,'path':r['path'][5:]} for r in source_lock()['final_F'] if r['path'].startswith('vllm/')])
    verify_files(package, marker['native_files'])
    return package


def model_lock(profile_name='single-3090'):
    profile = launch_profile(profile_name)
    return read_json(safe_path(ROOT/'locks', profile.get('models_lock', 'models.json')))


def check_models(root, profile_name='single-3090'):
    lock = model_lock(profile_name)
    for model in lock['models'].values():
        verify_files(Path(root)/model['directory'], model['files'])
    if sha(ROOT/'config/chat-template.jinja') != lock['chat_template_sha256']:
        raise ValueError('Chat template does not match the tested profile')
    return lock


def fetch_models(root, profile_name='single-3090'):
    """Explicit download action only, with full commit pins and file hashes."""
    lock = model_lock(profile_name)
    if lock.get('preparation'):
        prepare_tp1_primary(root, lock)
    for model in lock['models'].values():
        for row in model['files']:
            fetch_asset(Path(root)/model['directory'], row, model)
    check_models(root, profile_name)
    return {'passed':True,'models_verified':len(lock['models'])}


def fetch_asset(root, row, model):
    """Fetch a locked file; never replace existing assets or publish partial bytes."""
    out = safe_path(root, row['path'])
    if out.exists():
        verify_files(root,[row]);return
    if 'generated_by' in row:
        raise ValueError('Missing prepared asset: '+row['path'])
    if 'bundled' in row:
        src = safe_path(ROOT,row['bundled'])
    else:
        from huggingface_hub import hf_hub_download
        repository=row.get('repository') or model.get('repository')
        revision=row.get('revision') or model.get('revision')
        if not repository or not revision or len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError('Asset needs a full immutable repository revision: '+row['path'])
        src=Path(hf_hub_download(repository,row['remote_path'],revision=revision))
    if sha(src)!=row['sha256'] or src.stat().st_size!=row['bytes']:
        raise ValueError('Downloaded asset differs from tested asset: '+row['path'])
    out.parent.mkdir(parents=True,exist_ok=True)
    # A same-directory temporary file and exclusive hardlink keep readers from
    # seeing a partial asset and keep concurrent/existing destinations intact.
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=out.parent,prefix='.qwenrelay-',delete=False) as f:
            temporary=Path(f.name)
            with src.open('rb') as r:shutil.copyfileobj(r,f,1024*1024)
            f.flush();os.fsync(f.fileno())
        os.link(temporary,out)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def prepare_tp1_primary(root, lock):
    """Assemble upstream fast weights; quantize only the embedding on CPU.

    Fast config/index/head/MTP tensors are already published. Only base shard 6
    needs quant_embed.py; unrelated shards and tensor headers remain untouched.
    Existing primary directories are checked, never requantized in place.
    """
    root=Path(root).resolve()
    preparation=lock['preparation']
    if preparation['kind']!='upstream-embedding-int8':
        raise ValueError('Unknown model preparation')
    verify_files(ROOT,preparation['script_inputs'])
    model=lock['models']['primary']
    destination=safe_path(root,model['directory'])
    if destination.exists():
        verify_files(destination,model['files']);return
    Path(root).mkdir(parents=True,exist_ok=True)
    # Preserve failed work for diagnosis; a later attempt gets a new directory.
    stage=Path(tempfile.mkdtemp(prefix='.qwenrelay-tp1-',dir=root))
    embedding=stage/'embedding';assembled=stage/'assembled'
    for row in preparation['source_inputs']:fetch_asset(embedding,row,{})
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    completed=subprocess.run([sys.executable,str(safe_path(ROOT,preparation['script'])),str(embedding)],
                             cwd=ROOT,env=env,capture_output=True,text=True)
    if completed.returncode:
        raise RuntimeError('TP1 embedding preparation failed: '+completed.stdout+completed.stderr)
    generated=next(row for row in model['files'] if row.get('generated_by'))
    if generated['path']!=preparation['generated_file']:
        raise ValueError('Generated embedding identity does not match the preparation plan')
    verify_files(embedding,[generated])
    assembled.mkdir()
    shutil.copyfile(embedding/generated['path'],assembled/generated['path'])
    for row in model['files']:fetch_asset(assembled,row,model)
    verify_files(assembled,model['files'])
    if destination.exists():raise ValueError('Primary destination appeared during preparation')
    assembled.rename(destination)
    # Remove only our verified temporary inputs/backups after successful publish.
    shutil.rmtree(stage)


def launch_profile(name):
    if name not in TP_PROFILES.values():
        raise ValueError('Unknown launch profile: ' + name)
    return read_json(ROOT/'config'/(name+'.json'))


def resolve_profile(tp=None, profile_name=None):
    if tp is not None and tp not in TP_PROFILES:
        raise ValueError('TP must be 1, 2 or 4')
    name = profile_name or TP_PROFILES[1 if tp is None else tp]
    profile = launch_profile(name)
    if tp is not None and profile['options']['tensor-parallel-size'] != tp:
        raise ValueError('--tp conflicts with --profile')
    return name


def memory_bytes(value):
    """Parse exact bytes or an IEC size, independently of vLLM/CUDA imports."""
    match=re.fullmatch(r'(\d+(?:\.\d+)?)\s*(B|KiB|MiB|GiB|TiB)?',str(value).strip(),re.IGNORECASE)
    if match is not None:
        unit=(match[2] or 'B').lower()
        try:
            size=Decimal(match[1]) * (1024 ** ('b','kib','mib','gib','tib').index(unit))
            if size>0 and size==size.to_integral_value():return int(size)
        except InvalidOperation:
            pass
    raise ValueError('KV memory must be positive whole bytes or an IEC size such as 6GiB or 512MiB')


def effective_profile(name, overrides=None):
    """Apply explicit overrides to fresh defaults and derive coupled decode sizes."""
    overrides={} if overrides is None else dict(overrides)
    if set(overrides)-set(SERVE_OVERRIDES):
        raise ValueError('Unknown launch overrides: '+', '.join(sorted(set(overrides)-set(SERVE_OVERRIDES))))
    profile=copy.deepcopy(launch_profile(name))
    options=profile['options'];spec=profile['speculative_config']
    for key in ('max_num_seqs','max_model_len','max_num_batched_tokens','draft_tokens'):
        value=overrides.get(key)
        if value is None:continue
        if isinstance(value,bool) or not isinstance(value,int) or value<=0:
            raise ValueError('--'+key.replace('_','-')+' must be a positive integer')
        if key=='draft_tokens':spec['num_speculative_tokens']=value
        else:options[key.replace('_','-')]=value
    if overrides.get('kv_cache_memory') is not None:
        options['kv-cache-memory']=memory_bytes(overrides['kv_cache_memory'])
    if overrides.get('kv_offloading_size') is not None:
        value=overrides['kv_offloading_size']
        if isinstance(value,bool):raise ValueError('--kv-offloading-size must be a finite nonnegative GiB value')
        try:budget=float(value)
        except (ValueError,TypeError):raise ValueError('--kv-offloading-size must be a finite nonnegative GiB value') from None
        if not math.isfinite(budget) or budget<0:
            raise ValueError('--kv-offloading-size must be a finite nonnegative GiB value')
        if budget==0:
            # vLLM treats even size=0 as a configured connector; omit the flag
            # to disable the tier entirely, including the O4-only hooks.
            options.pop('kv-offloading-size',None)
            profile['environment']['QWEN_O4']='0'
            profile['environment']['QWEN_O4_COMPILE_RETENTION_NEUTRAL']='0'
        else:options['kv-offloading-size']=budget
    depth=spec['num_speculative_tokens']
    if depth>63:
        raise ValueError('--draft-tokens must be in 1..63 (split-KV supports at most 64 query tokens per request)')
    if options['max-model-len']<=depth:
        raise ValueError('--max-model-len must be greater than --draft-tokens')
    query_tokens=depth+1
    graph_tokens=options['max-num-seqs']*query_tokens
    if options['max-num-batched-tokens']<graph_tokens:
        raise ValueError('--max-num-batched-tokens must be at least max-num-seqs * (draft-tokens + 1) '
                         f'({graph_tokens}); increase it or reduce concurrency/depth')
    profile['environment']['VLLM_SPEC_DECODE_ATTN_QMAX']=str(query_tokens)
    profile['compilation_config']['max_cudagraph_capture_size']=graph_tokens
    return profile


def launch_plan(models_root, devices, port, host, profile_name=None, *, tp=None, overrides=None):
    profile_name = resolve_profile(tp, profile_name)
    profile = effective_profile(profile_name,overrides)
    tp = profile['options']['tensor-parallel-size']
    ids = devices.split(',')
    if len(ids)!=tp or len(set(ids))!=tp or any(not x.isdecimal() for x in ids):
        raise ValueError(f'Choose exactly {tp} distinct CUDA device indices for {profile_name}')
    if not 1 <= port <= 65535:
        raise ValueError('Port must be in 1..65535')
    models=model_lock(profile_name)['models']
    main = str(Path(models_root).resolve()/models['primary']['directory'])
    draft = str(Path(models_root).resolve()/models['draft']['directory'])
    spec={**profile['speculative_config'],'model':draft}
    argv=[sys.executable,'-m','vllm.entrypoints.cli.main','serve',main,
          '--served-model-name','qwenrelay','--host',host,'--port',str(port)]
    for flag,value in profile['options'].items():
        argv.append('--'+flag)
        if value is not True:
            argv.append(str(value))
    argv += ['--speculative-config',json.dumps(spec,separators=(',',':')),
             '--compilation-config',json.dumps(profile['compilation_config'],separators=(',',':')),
             '--chat-template',str(ROOT/'config/chat-template.jinja')]
    env={**os.environ,**profile['environment'],'CUDA_VISIBLE_DEVICES':devices}
    # Never inherit a parent process's alternative activation-quantization route.
    for key in ('INT8_ACT','INT8_LAYERS','VLLM_MARLIN_INPUT_DTYPE',
                'VLLM_MARLIN_INT8_LAYERS','VLLM_MARLIN_INT8_INCLUDE_RE','VLLM_MARLIN_USE_INT8_INPUT',
                'PREFILL_ATTN','VLLM_PREFILL_ATTN'):
        env.pop(key,None)
    return argv, env


def launch(args):
    profile_name=resolve_profile(args.tp,args.profile)
    overrides={key:getattr(args,key,None) for key in SERVE_OVERRIDES if getattr(args,key,None) is not None}
    profile=effective_profile(profile_name,overrides)
    tp=profile['options']['tensor-parallel-size']
    devices=args.devices or ','.join(map(str,range(tp)))
    argv,env=launch_plan(args.models_root,devices,args.port,args.host,profile_name,overrides=overrides)
    if args.dry_run:
        return {'argv':argv,'environment':{k:env[k] for k in profile['environment']},
                'profile':profile_name,'tp':tp,'models_lock':profile.get('models_lock','models.json'),
                'experimental':profile.get('experimental',False),
                'overrides':overrides,
                'effective_settings':{
                    'max_num_seqs':profile['options']['max-num-seqs'],
                    'max_model_len':profile['options']['max-model-len'],
                    'max_num_batched_tokens':profile['options']['max-num-batched-tokens'],
                    'draft_tokens':profile['speculative_config']['num_speculative_tokens'],
                    'gpu_kv_bytes_per_rank':profile['options']['kv-cache-memory'],
                    'cpu_kv_gib_per_tp_group':profile['options'].get('kv-offloading-size',0),
                    'logical_spec_query_tokens':int(env['VLLM_SPEC_DECODE_ATTN_QMAX']),
                    'max_cudagraph_capture_size':profile['compilation_config']['max_cudagraph_capture_size']},
                'CUDA_VISIBLE_DEVICES':devices,'secret_printed':False,'model_or_GPU_started':False}
    if sys.platform != 'linux':
        raise ValueError('The RTX 3090 profiles support native Linux only')
    key_path=Path(args.api_key_file)
    key=key_path.read_text(encoding='utf-8').strip()
    if not key or key in ('EMPTY','changeme','replace-me') or any(x.isspace() for x in key):
        raise ValueError('Create a nonempty API key file first')
    check_pack();check_models(args.models_root,profile_name);check_runtime()
    os.environ['CUDA_VISIBLE_DEVICES']=devices
    import torch
    if torch.cuda.device_count()!=tp:
        raise ValueError(f'Exactly {tp} visible GPUs are required')
    for i in range(tp):
        props=torch.cuda.get_device_properties(i)
        if '3090' not in props.name or props.total_memory < 23*1024**3:
            raise ValueError(f'{profile_name} requires {tp} RTX 3090 GPUs with at least 23 GiB each')
    env['VLLM_API_KEY']=key
    os.execve(sys.executable,argv,env)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('rebuild');s.add_argument('--vllm-source',type=Path,required=True);s.add_argument('--target',type=Path,required=True);s.add_argument('--bash',default='bash')
    s=sub.add_parser('install-source');s.add_argument('--source',type=Path,required=True)
    sub.add_parser('check-pack')
    sub.add_parser('check-runtime')
    for name in ('check-models','fetch-models'):
        s=sub.add_parser(name);s.add_argument('--root',type=Path,default=ROOT/'models')
        s.add_argument('--tp',type=int,choices=(1,2,4))
        s.add_argument('--profile',choices=tuple(TP_PROFILES.values()))
    s=sub.add_parser('serve');s.add_argument('--models-root',type=Path,default=Path(os.environ.get('MODELS_ROOT',str(ROOT/'models'))))
    s.add_argument('--tp',type=int,choices=(1,2,4),help='Select 3090 defaults and assets: 1=single (default), 2=dual, 4=quad; explicit tuning flags override defaults')
    s.add_argument('--profile',choices=tuple(TP_PROFILES.values()),help='Named profile; must agree with --tp if both are supplied')
    s.add_argument('--devices',default=os.environ.get('CUDA_VISIBLE_DEVICES'))
    s.add_argument('--port',type=int,default=int(os.environ.get('PORT','18020')))
    s.add_argument('--host',default=os.environ.get('BIND_HOST','127.0.0.1'))
    s.add_argument('--api-key-file',default=os.environ.get('API_KEY_FILE',str(ROOT/'api_key.txt')))
    tuning=s.add_argument_group('tuning overrides (omitted values use the selected TP defaults)')
    tuning.add_argument('--max-num-seqs','--concurrency',dest='max_num_seqs',type=int,
                        help='Maximum running requests, not client queue length')
    tuning.add_argument('--max-model-len','--context-length',dest='max_model_len',type=int,
                        help='Per-request context ceiling in tokens')
    tuning.add_argument('--draft-tokens','--num-speculative-tokens',dest='draft_tokens',type=int,
                        help='DFlash verify depth, 1..63; QMAX and Graph coverage are derived automatically')
    tuning.add_argument('--kv-cache-memory',type=memory_bytes,metavar='BYTES_OR_IEC_SIZE',
                        help='GPU KV budget per rank, e.g. 6442450944, 6GiB or 512MiB')
    tuning.add_argument('--kv-offloading-size','--cpu-kv-gib',dest='kv_offloading_size',type=float,metavar='GIB',
                        help='CPU KV budget summed across the TP group in GiB; 0 disables CPU offload/O4')
    tuning.add_argument('--max-num-batched-tokens',type=int,
                        help='Scheduler token budget per step; must cover concurrency * (draft depth + 1)')
    s.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.command=='rebuild':result=rebuild(a.vllm_source,a.target,a.bash)
    elif a.command=='install-source':result=install_source(a.source)
    elif a.command=='check-pack':check_pack();result={'passed':True,'source_files':len(source_lock()['final_F'])}
    elif a.command=='check-runtime':check_runtime();result={'passed':True}
    elif a.command=='check-models':check_models(a.root,resolve_profile(a.tp,a.profile));result={'passed':True}
    elif a.command=='fetch-models':result=fetch_models(a.root,resolve_profile(a.tp,a.profile))
    else:result=launch(a)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,RuntimeError,importlib.metadata.PackageNotFoundError) as e:
        print('qwenrelay: '+str(e),file=sys.stderr);sys.exit(1)
