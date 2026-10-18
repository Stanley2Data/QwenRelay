# SPDX-License-Identifier: Apache-2.0
"""CPU-only safety and profile contracts; no mocks of GPU correctness."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('qwenrelay',ROOT/'scripts/qwenrelay.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


class PublicSourceContracts(unittest.TestCase):
    def test_locked_pack(self):
        lock=q.check_pack()
        self.assertEqual(len(lock['final_F']),3208)
        self.assertEqual(len(lock['final_overlay']),6)

    def test_path_traversal_and_absolute_paths(self):
        with tempfile.TemporaryDirectory() as root:
            for path in ('../escape','/tmp/escape','a/../../escape','a\\escape'):
                with self.assertRaises(ValueError):q.safe_path(root,path)

    def test_graph_shape_and_quantization_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ['INT8_ACT']='int8'
            os.environ['VLLM_MARLIN_USE_INT8_INPUT']='1'
            os.environ['VLLM_MARLIN_INPUT_DTYPE']='int8'
            os.environ['VLLM_PREFILL_ATTN']='int8'
            argv,env=q.launch_plan(Path(tmp),'1,2',18020,'127.0.0.1')
        self.assertNotIn('INT8_ACT',env)
        self.assertNotIn('VLLM_MARLIN_USE_INT8_INPUT',env)
        self.assertNotIn('VLLM_MARLIN_INPUT_DTYPE',env)
        self.assertNotIn('VLLM_PREFILL_ATTN',env)
        self.assertEqual(argv[argv.index('--tensor-parallel-size')+1],'2')
        self.assertEqual(argv[argv.index('--max-num-seqs')+1],'4')
        comp=json.loads(argv[argv.index('--compilation-config')+1])
        spec=json.loads(argv[argv.index('--speculative-config')+1])
        self.assertEqual(comp['max_cudagraph_capture_size'],4*(1+spec['num_speculative_tokens']))
        self.assertEqual(argv[argv.index('--kv-offloading-size')+1],'32')
        self.assertEqual(env['PYTORCH_CUDA_ALLOC_CONF'],'expandable_segments:False')
        self.assertEqual(env['VLLM_USE_FLASHINFER_SAMPLER'],'0')
        # The standalone launcher must retain the legacy DFlash2 defaults;
        # otherwise long-context verification silently uses unified attention.
        self.assertEqual(env['VLLM_SPEC_DECODE_ATTN'],'1')
        self.assertEqual(int(env['VLLM_SPEC_DECODE_ATTN_QMAX']),1+spec['num_speculative_tokens'])
        self.assertEqual(env['VLLM_DFLASH2_LOOKUP'],'1')

    def test_reject_duplicate_or_missing_gpus(self):
        for devices in ('0','0,0','0,1,2','a,b'):
            with self.assertRaises(ValueError):q.launch_plan(ROOT/'models',devices,18020,'127.0.0.1')

    def test_wrong_hash_and_existing_output_are_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'sentinel';p.write_bytes(b'keep')
            with self.assertRaises(ValueError):q.verify_files(tmp,[{'path':'sentinel','sha256':'0'*64}])
            with self.assertRaises(ValueError):q.rebuild(Path(tmp)/'input',Path(tmp))
            self.assertEqual(p.read_bytes(),b'keep')

    def test_line_endings_accept_only_identical_text(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.py';p.write_bytes(b'value = 1\r\n')
            expected=[{'path':'source.py','bytes':10,'sha256':hashlib.sha256(b'value = 1\n').hexdigest()}]
            q.verify_files(tmp,expected,exact=True,canonical=True)
            p.write_bytes(b'value = 2\r\n')
            with self.assertRaises(ValueError):q.verify_files(tmp,expected,exact=True,canonical=True)
        self.assertEqual(q.canonical_bytes(b'\x00\r\n'),b'\x00\r\n')


if __name__=='__main__':unittest.main()
