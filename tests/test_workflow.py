"""End-to-end tests using local random ESM-2 fixtures, never model downloads."""
import contextlib
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from pairloom.cli import parser, run
from pairloom.checkpoints import CheckpointStore
from pairloom.storage import restore_persistent
from test_pairloom import backbone, records


class WorkflowTests(unittest.TestCase):
    def test_cli_train_evaluate_predict_and_persistent_recovery(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = Path(temp)
            architecture=root/'architecture.json'
            architecture.write_text(json.dumps(dict(num_hidden_layers=2,hidden_size=16,
                                                     num_attention_heads=2,intermediate_size=64)))
            training, validation = records()
            # User-supplied overlaps and repeated/reversed pairs must survive
            # both CLI loading and the training engine unchanged.
            training.append(replace(training[0], pair_id='reverse', sequence_a=training[0].sequence_b,
                                    sequence_b=training[0].sequence_a))
            validation[0] = replace(validation[0], sequence_a=training[0].sequence_a)
            validation[1] = replace(training[1], pair_id='v1', sequence_a=training[1].sequence_b,
                                    sequence_b=training[1].sequence_a)
            data = root / 'data.csv'
            with data.open('w',newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['pair_id','sequence_a','sequence_b','label','split'])
                for split, rows in [('train',training),('validation',validation),
                                    ('test',[replace(validation[0],pair_id='test0'),replace(validation[1],pair_id='test1')])]:
                    for r in rows: writer.writerow([r.pair_id,r.sequence_a,r.sequence_b,r.label,split])
            args = ['train','--initialization','random','--data',str(data),'--architecture',str(architecture),'--output',str(root/'run'),
                    '--persistent',str(root/'drive'),'--device','cpu','--epochs','1','--batch-size','1',
                    '--accumulation','2','--max-tokens','24','--max-views','1',
                    '--head-width','8','--dropout','0','--checkpoint-every','1','--eval-every','2',
                    '--no-gradient-checkpointing','--warmup-updates','1']
            result = run(parser().parse_args(args))
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['step'],3)
            self.assertEqual(CheckpointStore(root/'drive').load()['step'],3)
            from pairloom.data import fingerprint
            saved = CheckpointStore(root/'drive').load()
            self.assertEqual(saved['protocol']['training'], fingerprint(training))
            self.assertEqual(saved['protocol']['validation'], fingerprint(validation))
            architecture.unlink()
            shared = ['--data',str(data),'--split','test','--run',str(root/'run'),'--device','cpu']
            evaluated = run(parser().parse_args(['evaluate',*shared,'--predictions',str(root/'test.csv')]))
            self.assertEqual(evaluated['metrics']['n'],2)
            self.assertEqual(evaluated['metrics']['threshold'],CheckpointStore(root/'run').load('best')['state']['threshold'])
            predicted = run(parser().parse_args(['predict',*shared,'--predictions',str(root/'predict.csv')]))
            self.assertEqual(predicted['records'],2)
            with (root/'predict.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r['pair_id'] for r in rows],['test0','test1'])
            self.assertEqual([r['label'] for r in rows],['0','1'])
            self.assertTrue(restore_persistent(root/'restored',root/'drive'))
            self.assertEqual(CheckpointStore(root/'restored').load()['step'],3)

    def test_mutation_training_and_ordered_inference(self):
        from pairloom.data import PairBatcher
        from pairloom.engine import TrainConfig, fit, predict
        from pairloom.model import PairModel
        from test_pairloom import VOCAB
        torch.set_num_threads(1)
        training, validation = records()
        def mutated(r):
            return replace(r,mutant='W'+r.sequence_a[1:])
        training, validation = list(map(mutated,training)), list(map(mutated,validation))
        config = TrainConfig(epochs=1,accumulation=3,mlm_weight=.1,
                             max_tokens=24,max_views=1,checkpoint_every=1,gradient_checkpointing=False)
        model = PairModel(backbone())
        batcher = PairBatcher(VOCAB,24,1)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            fit(model,batcher,training,validation,{'kind':'random-esm2','fixture':'tiny'},config,temp,torch.device('cpu'))
            saved = CheckpointStore(temp).load('best')
            self.assertEqual(saved['protocol']['task'],'mutation')
            scores = predict(model,batcher,validation,torch.device('cpu'))
            self.assertEqual(len(scores),4)
            self.assertTrue(((scores >= 0) & (scores <= 1)).all())

    def test_a_bad_persistent_checkpoint_never_silently_starts_a_new_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = CheckpointStore(root/'drive')
            store.save({'step':10,'protocol':{}})
            pointer = json.loads((store.root/'latest.json').read_text())
            (store.root/pointer['file']).write_bytes(b'damaged')
            with self.assertRaisesRegex(RuntimeError,'refusing to start over'):
                restore_persistent(root/'local',root/'drive')


if __name__ == '__main__':
    unittest.main()
