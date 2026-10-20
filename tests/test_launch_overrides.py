# SPDX-License-Identifier: Apache-2.0
"""Check effective launch arguments and rejected combinations without loading CUDA."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('qwenrelay_overrides',ROOT/'scripts/qwenrelay.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


class LaunchOverrideTests(unittest.TestCase):
    def test_defaults_are_preserved_for_every_tp(self):
        for name in q.TP_PROFILES.values():
            with self.subTest(profile=name):
                self.assertEqual(q.effective_profile(name),q.launch_profile(name))

    def test_all_overrides_apply_after_tp_defaults_without_changing_assets(self):
        for tp,name in q.TP_PROFILES.items():
            with self.subTest(tp=tp):
                devices=','.join(map(str,range(tp)))
                argv,env=q.launch_plan(ROOT/'models',devices,18020,'127.0.0.1',tp=tp,
                    overrides=dict(max_num_seqs=8,max_model_len=65536,draft_tokens=7,
                                   kv_cache_memory='3.5GiB',kv_offloading_size=16,
                                   max_num_batched_tokens=1024))
                for flag,want in {'tensor-parallel-size':tp,'max-num-seqs':8,
                                  'max-model-len':65536,'kv-cache-memory-bytes':int(3.5*1024**3),
                                  'max-num-batched-tokens':1024}.items():
                    self.assertEqual(int(argv[argv.index('--'+flag)+1]),want)
                    self.assertEqual(argv.count('--'+flag),1)
                self.assertEqual(float(argv[argv.index('--kv-offloading-size')+1]),16)
                draft=json.loads(argv[argv.index('--speculative-config')+1])
                self.assertEqual(draft['num_speculative_tokens'],7)
                self.assertEqual(draft.get('draft_tensor_parallel_size',tp),tp)
                graph=json.loads(argv[argv.index('--compilation-config')+1])
                self.assertEqual(graph['max_cudagraph_capture_size'],64)
                self.assertEqual(env['VLLM_SPEC_DECODE_ATTN_QMAX'],'8')
                self.assertEqual(env['VLLM_SPEC_DECODE_ATTN'],'1')
                self.assertEqual(env['VLLM_DFLASH2_LOOKUP'],'1')
                self.assertEqual(env['QWEN_O4'],'1')
                self.assertIn('AutoRound-fast' if tp==1 else 'EfficientThink',argv[4])
                self.assertNotIn('--kv-cache-memory',argv)

    def test_nondefault_depth_aligns_retention_with_hybrid_cache_pages(self):
        # The old README depth-6 example selected 816-token runtime blocks,
        # but 20800 % 816 != 0, so O4 rejected it after model loading.
        for tp in (1,2,4):
            for depth,block,retention in ((1,768,21504),(4,800,20800),
                                           (6,816,21216),(7,832,20800)):
                with self.subTest(tp=tp,depth=depth):
                    p=q.effective_profile(q.TP_PROFILES[tp],{'draft_tokens':depth})
                    self.assertEqual(p['options'].get('block-size',800),block)
                    self.assertEqual(p['options']['prefix-cache-retention-interval'],retention)
                    self.assertEqual(retention % block,0)
            for depth in range(1,64):
                block,retention=q.hybrid_cache_alignment(depth,20800)
                # Independently verify that a page fits all pinned conv/SSM
                # state, and that the preceding 16-token block would not.
                state=(48//tp)*128*128*2+(10240//tp)*(3+depth)*2
                kv=(4//tp)*544
                self.assertGreaterEqual(block*kv,state)
                self.assertLess((block-16)*kv,state)
                self.assertEqual(retention % block,0)
                self.assertGreaterEqual(retention,20800)
                self.assertLess(retention,20800+block)

    def test_smaller_decode_shape_and_boundary_depth_update_graph_and_qmax(self):
        profile=q.effective_profile('single-3090',dict(max_num_seqs=1,draft_tokens=1))
        self.assertEqual(profile['compilation_config']['max_cudagraph_capture_size'],2)
        self.assertEqual(profile['environment']['VLLM_SPEC_DECODE_ATTN_QMAX'],'2')
        profile=q.effective_profile('single-3090',dict(max_num_seqs=2,draft_tokens=63,
                                                    max_num_batched_tokens=128))
        self.assertEqual(profile['compilation_config']['max_cudagraph_capture_size'],128)
        self.assertEqual(profile['environment']['VLLM_SPEC_DECODE_ATTN_QMAX'],'64')

    def test_zero_cpu_budget_removes_connector_flag_and_disables_only_o4(self):
        argv,env=q.launch_plan(ROOT/'models','0',18020,'127.0.0.1',
                              overrides={'kv_offloading_size':0})
        self.assertNotIn('--kv-offloading-size',argv)
        self.assertEqual(env['QWEN_O4'],'0')
        self.assertEqual(env['QWEN_O4_COMPILE_RETENTION_NEUTRAL'],'0')
        for key in ('QWEN_PF2','QWEN_O2','QWEN_O5','VLLM_SPEC_DECODE_ATTN','VLLM_DFLASH2_LOOKUP'):
            self.assertEqual(env[key],'1')

    def test_overrides_do_not_mutate_defaults_or_inherit_stale_decode_settings(self):
        original=q.launch_profile('single-3090');before=copy.deepcopy(original)
        with patch.object(q,'launch_profile',return_value=original),patch.dict(os.environ,
                {'VLLM_SPEC_DECODE_ATTN_QMAX':'256','QWEN_O4':'0'}):
            q.effective_profile('single-3090',dict(draft_tokens=6,kv_offloading_size=0))
            self.assertEqual(original,before)
            _,env=q.launch_plan(ROOT/'models','0',18020,'127.0.0.1',overrides={'draft_tokens':6})
            self.assertEqual(env['VLLM_SPEC_DECODE_ATTN_QMAX'],'7')
            self.assertEqual(env['QWEN_O4'],'1')
            self.assertEqual(q.effective_profile('single-3090'),before)

    def test_invalid_and_incompatible_values_are_rejected(self):
        invalid=[{'max_num_seqs':0},{'max_num_seqs':-1},{'max_num_seqs':True},
                 {'max_model_len':0},{'max_model_len':4,'draft_tokens':4},
                 {'draft_tokens':0},{'draft_tokens':64},{'draft_tokens':2.5},
                 {'max_num_batched_tokens':0},{'max_num_seqs':8,'draft_tokens':7,'max_num_batched_tokens':63},
                 {'kv_cache_memory':'0GiB'},{'kv_cache_memory':'-1GiB'},
                 {'kv_offloading_size':-1},{'kv_offloading_size':float('nan')},
                 {'kv_offloading_size':float('inf')},{'kv_offloading_size':True},
                 {'tensor_parallel_size':8}]
        for overrides in invalid:
            with self.subTest(overrides=overrides),self.assertRaises(ValueError):
                q.effective_profile('single-3090',overrides)

    def test_gpu_budget_parser_is_exact_and_rejects_ambiguous_sizes(self):
        for value,want in [('6442450944',6*1024**3),('6GiB',6*1024**3),
                           ('3.5gib',int(3.5*1024**3)),('512MiB',512*1024**2),
                           ('1KiB',1024),('1B',1)]:
            self.assertEqual(q.memory_bytes(value),want)
        for value in ('6GB','6G','auto','nan','inf','0','0.1B','-1GiB'):
            with self.subTest(value=value),self.assertRaises(ValueError):q.memory_bytes(value)

    def test_shell_entry_accepts_aliases_and_dry_run_needs_no_runtime_or_key(self):
        env={k:v for k,v in os.environ.items() if k!='CUDA_VISIBLE_DEVICES'}
        env['QWENRELAY_PYTHON']=sys.executable
        command=['bash',str(ROOT/'scripts/start-qwenrelay.sh'),'--tp','4',
                 '--concurrency','8','--context-length','65536','--num-speculative-tokens','7',
                 '--kv-cache-memory','3.5GiB','--cpu-kv-gib','0',
                 '--max-num-batched-tokens','1024','--api-key-file','/nonexistent/key','--dry-run']
        result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,check=True)
        plan=json.loads(result.stdout)
        self.assertEqual(plan['profile'],'quad-3090')
        self.assertEqual(plan['CUDA_VISIBLE_DEVICES'],'0,1,2,3')
        self.assertEqual(plan['effective_settings'],dict(max_num_seqs=8,max_model_len=65536,
            max_num_batched_tokens=1024,draft_tokens=7,gpu_kv_bytes_per_rank=int(3.5*1024**3),
            cpu_kv_gib_per_tp_group=0,logical_spec_query_tokens=8,max_cudagraph_capture_size=64,
            hybrid_block_tokens=832,prefix_cache_retention_interval=20800))
        self.assertFalse(plan['model_or_GPU_started'])
        self.assertEqual(plan['environment']['QWEN_O4'],'0')
        self.assertNotIn('--kv-offloading-size',plan['argv'])

    def test_shell_uses_exported_venv_from_another_working_directory(self):
        with tempfile.TemporaryDirectory(prefix='qwen relay ') as tmp:
            venv=Path(tmp)/'custom venv';(venv/'bin').mkdir(parents=True)
            (venv/'bin/python').symlink_to(sys.executable)
            env={k:v for k,v in os.environ.items() if k not in ('QWENRELAY_PYTHON','CUDA_VISIBLE_DEVICES')}
            env['QWENRELAY_VENV']=str(venv)
            result=subprocess.run(['bash',str(ROOT/'scripts/start-qwenrelay.sh'),'--dry-run'],
                                  cwd=tmp,env=env,capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(result.stdout)['tp'],1)
            # Explicit interpreter override wins even if the venv is absent.
            env.update(QWENRELAY_PYTHON=sys.executable,QWENRELAY_VENV='/missing/venv')
            subprocess.run(['bash',str(ROOT/'scripts/start-qwenrelay.sh'),'--dry-run'],
                           cwd=tmp,env=env,capture_output=True,text=True,check=True)

    def test_model_commands_share_models_root_and_explicit_paths_win(self):
        for command,function in (('fetch-models','fetch_models'),('check-models','check_models')):
            for flag in ([],['--root','/explicit/models']):
                with self.subTest(command=command,flag=flag), \
                     patch.dict(os.environ,{'MODELS_ROOT':'/exported/models'}), \
                     patch.object(sys,'argv',['qwenrelay',command,*flag]), \
                     patch.object(q,function,return_value={'passed':True}) as action, \
                     patch('builtins.print'):
                    q.main()
                    self.assertEqual(action.call_args.args[0],Path('/explicit/models' if flag else '/exported/models'))

    def test_installer_rejects_occupied_outputs_before_creating_venv(self):
        for occupied in ('final-F','vllm-pristine'):
            with tempfile.TemporaryDirectory() as tmp:
                build=Path(tmp)/'build';(build/occupied).mkdir(parents=True)
                sentinel=build/occupied/'keep';sentinel.write_text('original')
                venv=Path(tmp)/'new-venv'
                env={k:v for k,v in os.environ.items() if k!='VLLM_SOURCE'}
                env.update(PYTHON=sys.executable,QWENRELAY_VENV=str(venv),QWENRELAY_BUILD=str(build))
                result=subprocess.run(['bash',str(ROOT/'scripts/install-qwenrelay.sh')],
                                      cwd=tmp,env=env,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('QWENRELAY_BUILD',result.stderr)
                self.assertFalse(venv.exists())
                self.assertEqual(sentinel.read_text(),'original')


if __name__=='__main__':unittest.main()
