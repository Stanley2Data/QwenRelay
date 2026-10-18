#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The original 22 workloads plus ten context/decode workloads; no server control."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def cases():
    rows=[]
    prefill=[(1024,1,16),(1024,4,32),(1024,16,64),(4096,1,8),(4096,4,16),(4096,16,32),
             (16384,1,4),(16384,4,8),(16384,8,16),(65536,1,2),(65536,2,4),(102400,1,2)]
    def add(name,repeat,args,scored=None):
        rows.append({'case':name,'repeat':repeat,'scored':repeat>0 if scored is None else scored,'args':args})
    for repeat,base in enumerate((1000,2000,3000,4000)):
        seed=base+1
        add('original_internal_warmup',repeat,['--dataset-name','random','--seed',str(seed),
            '--random-input-len','256','--random-output-len','256','--num-prompts','16','--max-concurrency','8'],False)
        for temp in ('default','0'):
            for conc in (1,2,4,8):
                args=['--dataset-name','custom','--dataset-path',str(ROOT/'bench/prompts_real.jsonl'),
                      '--custom-output-len','1024','--num-prompts','8','--max-concurrency',str(conc)]
                if temp=='0':args+=['--temperature','0']
                add(f'cohort_T{temp}_c{conc}',repeat,args)
        for length,conc,n in prefill:
            seed+=1
            add(f'prefill_{length}_c{conc}',repeat,['--dataset-name','random','--seed',str(seed),
                '--random-input-len',str(length),'--random-output-len','1','--num-prompts',str(n),'--max-concurrency',str(conc)])
        for name,length,out,n,conc in [('long_100k',100000,256,1,1),('long_4x60k',60000,1024,4,4)]:
            seed+=1
            add(name,repeat,['--dataset-name','random','--seed',str(seed),'--ignore-eos',
                '--random-input-len',str(length),'--random-output-len',str(out),'--num-prompts',str(n),'--max-concurrency',str(conc)])
    for repeat in range(4):
        for i,length in enumerate((1024,4096,16384,65536,131072)):
            for conc in (1,4):
                add(f'context_{length}_c{conc}',repeat,['--dataset-name','random','--seed',str(50000+repeat*1000+i*10+conc),
                    '--random-input-len',str(length),'--random-output-len','1024','--ignore-eos','--temperature','0',
                    '--num-prompts',str(conc),'--max-concurrency',str(conc)])
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=18020)
    p.add_argument('--model',default='qwenrelay')
    p.add_argument('--tokenizer',type=Path,default=ROOT/'models/Qwen3.8-27B-EfficientThink-AWQ-W4A16')
    p.add_argument('--output',type=Path,default=ROOT/'bench/results/qwenrelay')
    p.add_argument('--api-key-file',type=Path,default=ROOT/'api_key.txt')
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--plan',action='store_true');g.add_argument('--run',action='store_true')
    a=p.parse_args()
    plan=cases()
    if a.plan:
        print(json.dumps({'cases_including_internal_warmups':len(plan),'formal_primary':sum(x['scored'] for x in plan),
                          'server_control':False,'cases':plan},indent=2));return
    if a.output.exists():raise ValueError('Choose a new output directory; never overwrite a benchmark')
    if not a.tokenizer.is_dir():raise ValueError('A prepared local tokenizer directory is required')
    key=a.api_key_file.read_text().strip()
    if not key:raise ValueError('API key file is empty')
    a.output.mkdir(parents=True)
    (a.output/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    base=[sys.executable,'-m','vllm.entrypoints.cli.main','bench','serve','--host',a.host,'--port',str(a.port),
          '--model',a.model,'--served-model-name',a.model,'--tokenizer',str(a.tokenizer),'--save-result','--save-detailed']
    env={**os.environ,'OPENAI_API_KEY':key,'CUDA_VISIBLE_DEVICES':''}
    for row in plan:
        dest=a.output/f"r{row['repeat']}-{row['case']}";dest.mkdir()
        argv=base+row['args']+['--result-dir',str(dest),'--result-filename','result.json']
        (dest/'command.json').write_text(json.dumps(argv,indent=2)+'\n')
        with (dest/'stdout.log').open('x') as log:
            subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((dest/'result.json').read_text())
        if result.get('failed',0) or result.get('completed',0)!=int(row['args'][row['args'].index('--num-prompts')+1]):
            raise ValueError('Incomplete benchmark: '+row['case'])
        if row['case'].startswith('context_'):
            _,length,c=row['case'].split('_');conc=int(c[1:])
            if result.get('input_lens')!=[int(length)]*conc or result.get('output_lens')!=[1024]*conc:
                raise ValueError('Actual token lengths differ from requested context workload')
    print('Completed; keep all raw results and record the actual running profile before comparison.')


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,subprocess.CalledProcessError) as e:
        print(str(e),file=sys.stderr);sys.exit(1)
