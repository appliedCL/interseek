"""Execute the actual Colab cells offline in fresh CPU processes with tiny inputs."""
import csv
import gzip
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import torch

from pairloom.base import sha256
from pairloom.checkpoints import CheckpointStore
from test_pairloom import records,backbone


class NotebookTests(unittest.TestCase):
    def test_fresh_notebook_training_and_second_session_restore(self):
        torch.set_num_threads(1)
        package = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            transfer = root/'transfer'
            (transfer/'base').mkdir(parents=True)
            (transfer/'data').mkdir()
            weights=transfer/'base/encoder.pt'
            weights.write_bytes(b'Not a checkpoint. Transfer import must read only data.')
            architecture=root/'architecture.json'
            architecture.write_text(json.dumps(dict(num_hidden_layers=2,hidden_size=16,
                                                     num_attention_heads=2,intermediate_size=64)))
            pretrained=root/'meta-fixture'
            backbone().save_pretrained(pretrained)
            data = transfer/'data/development.csv.gz'
            training,validation=records()
            with gzip.open(data,'wt',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['pair_id','sequence_a','sequence_b','label','split'])
                for split,rows in [('train',training),('validation',validation)]:
                    for r in rows: writer.writerow([r.pair_id,r.sequence_a,r.sequence_b,r.label,split])
            manifest={'format':'rimnet-kaggle-transfer-v1','base':{'path':'base/encoder.pt','sha256':sha256(weights)},
                      'data':{'path':'data/development.csv.gz','sha256':sha256(data)}}
            (transfer/'transfer.json').write_text(json.dumps(manifest))
            archive=root/'transfer.zip'
            with zipfile.ZipFile(archive,'w') as output:
                for p in transfer.rglob('*'):
                    if p.is_file(): output.write(p,str(p.relative_to(transfer)))
            notebook=json.loads((package/'notebooks/Pairloom_train_Colab.ipynb').read_text())
            cells=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
            setup=cells[1].replace('Path("/content/pairloom-code")','Path('+repr(str(root/'code'))+')')
            setup=setup.replace('Path("/content/pairloom-work")','Path('+repr(str(root/'work'))+')')
            training_cell=cells[2].replace('"--device", "cuda"','"--device", "cpu"')
            script='''import json,sys,types,torch
from pathlib import Path
from unittest.mock import patch
torch.set_num_threads(1)
fake=types.ModuleType("google.colab")
fake.drive=types.SimpleNamespace(mount=lambda path:None)
sys.modules["google.colab"]=fake
namespace={"__name__":"pairloom_notebook_fixture"}
'''
            script+='exec('+repr(cells[0])+',namespace)\n'
            script+='namespace.update('+repr({'INPUT_DATA':str(archive),'ARCHITECTURE':str(architecture),'INITIALIZATION':'pretrained','PRETRAINED_MODEL':str(pretrained),'PERSISTENT_RUN':str(root/'drive'),
                'RUN_NAME':'tiny','EPOCHS':1,'MAX_TOKENS':24,'MAX_VIEWS':1,
                'SESSION_HOURS':1.0,'MAX_UPDATES':1})+')\n'
            script+='''with patch("subprocess.run"),patch.object(torch.cuda,"is_available",return_value=True),patch.object(torch.cuda,"get_device_name",return_value="CPU fixture"),patch.object(torch.cuda,"empty_cache"):
'''
            script+='    exec('+repr(setup)+',namespace)\n'
            script+='exec('+repr(training_cell)+',namespace)\n'
            environment={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                         'HF_HUB_OFFLINE':'1','PYTHONPATH':''}
            first=subprocess.run([sys.executable,'-c',script],env=environment,text=True,capture_output=True,timeout=120)
            self.assertEqual(first.returncode,0,first.stdout[-3000:]+first.stderr[-3000:])
            saved=CheckpointStore(root/'drive').load()
            self.assertEqual(saved['step'],1)
            self.assertEqual(saved['protocol']['initialization']['kind'],'pretrained-esm2')
            self.assertFalse(any((root/'work/inputs').rglob('encoder.pt')))
            self.assertEqual(len(saved['state']['history']),1)
            shutil.rmtree(pretrained)  # Full resume must not reload the pretrained source.
            second=subprocess.run([sys.executable,'-c',script],env=environment,text=True,capture_output=True,timeout=120)
            self.assertEqual(second.returncode,0,second.stdout[-3000:]+second.stderr[-3000:])
            restored=CheckpointStore(root/'drive').load()
            self.assertEqual(restored['step'],1)
            self.assertEqual(len(restored['state']['history']),1)
            for name,value in saved['model'].items():
                torch.testing.assert_close(value,restored['model'][name],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
