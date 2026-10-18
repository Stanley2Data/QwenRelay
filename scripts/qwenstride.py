#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""QwenStride source reconstruction, asset validation and TP2 launch entry."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


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
            logs.append(bash_run(bash, ['-c',cmd,'qwenstride',posix(target),patch]))
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
    marker = package/'.qwenstride-source.json'
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
    marker = read_json(package/'.qwenstride-source.json')
    if marker['source_lock_sha256'] != sha(ROOT/'locks/source.json'):
        raise ValueError('Runtime belongs to a different source lock')
    verify_files(package, [{**r,'path':r['path'][5:]} for r in source_lock()['final_F'] if r['path'].startswith('vllm/')])
    verify_files(package, marker['native_files'])
    return package


def check_models(root):
    lock = read_json(ROOT/'locks/models.json')
    for model in lock['models'].values():
        verify_files(Path(root)/model['directory'], model['files'])
    if sha(ROOT/'config/chat-template.jinja') != lock['chat_template_sha256']:
        raise ValueError('Chat template does not match the tested profile')
    return lock


def fetch_models(root):
    """Explicit download action only, with full commit pins and file hashes."""
    from huggingface_hub import hf_hub_download
    lock = read_json(ROOT/'locks/models.json')
    for model in lock['models'].values():
        for row in model['files']:
            out = safe_path(Path(root)/model['directory'], row['path'])
            if out.exists():
                verify_files(out.parent,[{**row,'path':out.name}]);continue
            out.parent.mkdir(parents=True,exist_ok=True)
            src = ROOT/row['bundled'] if 'bundled' in row else Path(hf_hub_download(
                model['repository'], row['remote_path'], revision=model['revision']))
            if sha(src) != row['sha256'] or src.stat().st_size != row['bytes']:
                raise ValueError('Downloaded asset differs from tested asset: '+row['path'])
            with out.open('xb') as f, src.open('rb') as r:
                shutil.copyfileobj(r,f,1024*1024)
    check_models(root)
    return {'passed':True,'models_verified':len(lock['models'])}


def launch_plan(models_root, devices, port, host):
    profile = read_json(ROOT/'config/dual-3090.json')
    ids = devices.split(',')
    if len(ids)!=2 or len(set(ids))!=2 or any(not x.isdecimal() for x in ids):
        raise ValueError('Choose exactly two distinct CUDA device indices')
    if not 1 <= port <= 65535:
        raise ValueError('Port must be in 1..65535')
    models=read_json(ROOT/'locks/models.json')['models']
    main = str(Path(models_root).resolve()/models['primary']['directory'])
    draft = str(Path(models_root).resolve()/models['draft']['directory'])
    spec={**profile['speculative_config'],'model':draft}
    argv=[sys.executable,'-m','vllm.entrypoints.cli.main','serve',main,
          '--served-model-name','qwenstride','--host',host,'--port',str(port)]
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
                'VLLM_MARLIN_INT8_LAYERS','VLLM_MARLIN_USE_INT8_INPUT',
                'PREFILL_ATTN','VLLM_PREFILL_ATTN'):
        env.pop(key,None)
    return argv, env


def launch(args):
    argv,env=launch_plan(args.models_root,args.devices,args.port,args.host)
    if args.dry_run:
        return {'argv':argv,'environment':{k:env[k] for k in read_json(ROOT/'config/dual-3090.json')['environment']},
                'CUDA_VISIBLE_DEVICES':args.devices,'secret_printed':False,'model_or_GPU_started':False}
    if sys.platform != 'linux':
        raise ValueError('The first profile supports native Linux only')
    key_path=Path(args.api_key_file)
    key=key_path.read_text(encoding='utf-8').strip()
    if not key or key in ('EMPTY','changeme','replace-me') or any(x.isspace() for x in key):
        raise ValueError('Create a nonempty API key file first')
    check_pack();check_models(args.models_root);check_runtime()
    os.environ['CUDA_VISIBLE_DEVICES']=args.devices
    import torch
    if torch.cuda.device_count()!=2:
        raise ValueError('Exactly two visible GPUs are required')
    for i in range(2):
        props=torch.cuda.get_device_properties(i)
        if '3090' not in props.name or props.total_memory < 23*1024**3:
            raise ValueError('This profile is validated only on two 24 GiB RTX 3090 GPUs')
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
    s=sub.add_parser('serve');s.add_argument('--models-root',type=Path,default=Path(os.environ.get('MODELS_ROOT',str(ROOT/'models'))))
    s.add_argument('--devices',default=os.environ.get('CUDA_VISIBLE_DEVICES','0,1'))
    s.add_argument('--port',type=int,default=int(os.environ.get('PORT','18020')))
    s.add_argument('--host',default=os.environ.get('BIND_HOST','127.0.0.1'))
    s.add_argument('--api-key-file',default=os.environ.get('API_KEY_FILE',str(ROOT/'api_key.txt')))
    s.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.command=='rebuild':result=rebuild(a.vllm_source,a.target,a.bash)
    elif a.command=='install-source':result=install_source(a.source)
    elif a.command=='check-pack':check_pack();result={'passed':True,'source_files':len(source_lock()['final_F'])}
    elif a.command=='check-runtime':check_runtime();result={'passed':True}
    elif a.command=='check-models':check_models(a.root);result={'passed':True}
    elif a.command=='fetch-models':result=fetch_models(a.root)
    else:result=launch(a)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,RuntimeError,importlib.metadata.PackageNotFoundError) as e:
        print('qwenstride: '+str(e),file=sys.stderr);sys.exit(1)
