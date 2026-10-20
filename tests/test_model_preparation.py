# SPDX-License-Identifier: Apache-2.0
"""Small synthetic assets verify publication/hash guards, without real weights."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('qwenrelay_models',ROOT/'scripts/qwenrelay.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


def row(name,data,**extra):
    return dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),**extra)


class AssetPublicationTests(unittest.TestCase):
    def test_wrong_download_or_existing_asset_is_never_published_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);hub=root/'hub';hub.write_bytes(b'correct')
            item=row('asset',b'correct',repository='author/model',revision='a'*40,remote_path='asset')
            module=SimpleNamespace(hf_hub_download=lambda *a,**k:str(hub))
            with patch.dict('sys.modules',{'huggingface_hub':module}):
                q.fetch_asset(root/'out',item,{})
                self.assertEqual((root/'out/asset').read_bytes(),b'correct')
                (root/'out/asset').write_bytes(b'existing mismatch')
                with self.assertRaises(ValueError):q.fetch_asset(root/'out',item,{})
                self.assertEqual((root/'out/asset').read_bytes(),b'existing mismatch')
                hub.write_bytes(b'bad download')
                with self.assertRaises(ValueError):q.fetch_asset(root/'new',item,{})
                self.assertFalse((root/'new/asset').exists())

    def test_prepare_checks_generated_bytes_and_preserves_existing_model(self):
        for conversion_ok in (True,False):
            with self.subTest(conversion_ok=conversion_ok),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                generated=row('embedding.bin',b'quantized',generated_by='upstream-embedding-int8')
                direct=row('config.json',b'fast config',repository='author/fast',revision='b'*40,remote_path='config.json')
                source=row('embedding.bin',b'raw',repository='author/base',revision='a'*40,remote_path='embedding.bin')
                lock={'models':{'primary':{'directory':'AutoRound-fast','files':[generated,direct]}},
                      'preparation':{'kind':'upstream-embedding-int8','script':'prepare/quant_embed.py',
                                     'script_inputs':[],'source_inputs':[source],'generated_file':'embedding.bin'}}
                hub=root/'hub';hub.mkdir();(hub/'raw').write_bytes(b'raw');(hub/'fast').write_bytes(b'fast config')
                def download(repo,name,revision):
                    self.assertEqual(revision,'a'*40 if repo=='author/base' else 'b'*40)
                    return str(hub/('raw' if repo=='author/base' else 'fast'))
                def convert(argv,**kwargs):
                    self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                    self.assertEqual(Path(argv[1]),ROOT/'prepare/quant_embed.py')
                    (Path(argv[2])/'embedding.bin').write_bytes(b'quantized' if conversion_ok else b'wrong output')
                    return SimpleNamespace(returncode=0,stdout='',stderr='')
                with patch.dict('sys.modules',{'huggingface_hub':SimpleNamespace(hf_hub_download=download)}),patch.object(q.subprocess,'run',side_effect=convert) as run:
                    if conversion_ok:
                        q.prepare_tp1_primary(root,lock)
                        q.verify_files(root/'AutoRound-fast',[generated,direct])
                        q.prepare_tp1_primary(root,lock)
                        self.assertEqual(run.call_count,1)
                        self.assertFalse(list(root.glob('.qwenrelay-tp1-*')))
                        (root/'AutoRound-fast/embedding.bin').write_bytes(b'preexisting mismatch')
                        with self.assertRaises(ValueError):q.prepare_tp1_primary(root,lock)
                        self.assertEqual(run.call_count,1)
                        self.assertEqual((root/'AutoRound-fast/embedding.bin').read_bytes(),b'preexisting mismatch')
                    else:
                        with self.assertRaises(ValueError):q.prepare_tp1_primary(root,lock)
                        self.assertFalse((root/'AutoRound-fast').exists())
                        self.assertEqual(len(list(root.glob('.qwenrelay-tp1-*'))),1)


if __name__=='__main__':unittest.main()
