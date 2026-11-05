"""Device selection and pretrained training through the installed command line."""
import contextlib
import csv
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from interseq.base import sha256, weight_digest
from interseq.checkpoints import CheckpointStore
from interseq.cli import execution_device, make_model, parser
from interseq.engine import TrainConfig
from test_interseq import backbone, records


class DeviceTests(unittest.TestCase):
    def test_selected_gpu_controls_precision(self):
        for capability, precision in [((7, 5), 'fp16'), ((8, 0), 'bf16')]:
            with self.subTest(capability=capability), contextlib.ExitStack() as stack:
                stack.enter_context(patch('torch.cuda.is_available', return_value=True))
                stack.enter_context(patch('torch.cuda.device_count', return_value=2))
                selected = stack.enter_context(patch('torch.cuda.set_device'))
                queried = stack.enter_context(patch('torch.cuda.get_device_capability', return_value=capability))
                stack.enter_context(patch('torch.cuda.is_bf16_supported', return_value=capability[0] >= 8))
                device, actual = execution_device('cuda:1', 'auto')
                self.assertEqual((str(device), actual), ('cuda:1', precision))
                selected.assert_called_once_with(1)
                queried.assert_called_once_with(torch.device('cuda:1'))

    def test_invalid_gpu_is_rejected_before_loading_weights(self):
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.set_device') as selected:
            with self.assertRaisesRegex(ValueError, 'device 2 is unavailable'):
                execution_device('cuda:2', 'auto')
            selected.assert_not_called()
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'CUDA is unavailable'):
                execution_device('cuda', 'auto')

    def test_cpu_fallback_and_precision_validation(self):
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            self.assertEqual(execution_device('auto', 'auto'), (torch.device('cpu'), 'fp32'))
        for precision in ['fp16', 'bf16']:
            with self.assertRaisesRegex(ValueError, 'require CUDA'):
                execution_device('cpu', precision)
        for name in ['invalid', 'meta', 'cpu:1']:
            with self.assertRaisesRegex(ValueError, 'Choose auto'):
                execution_device(name, 'auto')

    def test_default_training_loads_pretrained_35m(self):
        args = parser().parse_args(['train', '--data', 'pairs.csv', '--output', 'runs/model'])
        with patch('interseq.base.initialize_pretrained', return_value=(backbone(), {}, {})) as load:
            model, _, _ = make_model(args, TrainConfig(gradient_checkpointing=False))
        load.assert_called_once_with('esm2-35m', None, None)
        self.assertTrue(all(parameter.requires_grad for parameter in model.backbone.parameters()))


class PretrainedCLITests(unittest.TestCase):
    def test_pretrained_archive_training_and_storage_recovery_in_fresh_processes(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrained = root / 'pretrained'
            model = backbone()
            initial_hash = weight_digest(model)
            model.save_pretrained(pretrained)
            data = root / 'pairs.csv.gz'
            training, validation = records()
            with gzip.open(data, 'wt', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['pair_id', 'sequence_a', 'sequence_b', 'label', 'split'])
                for split, rows in [('train', training), ('validation', validation)]:
                    for row in rows:
                        writer.writerow([row.pair_id, row.sequence_a, row.sequence_b, row.label, split])
            archive = root / 'pairs.zip'
            with zipfile.ZipFile(archive, 'w') as output:
                output.write(data, 'pairs.csv.gz')
                output.writestr('transfer.json', json.dumps({'data': {'path': 'pairs.csv.gz', 'sha256': sha256(data)}}))
                output.writestr('unused-model.pt', b'Unrelated archive member')
            common = ['train', '--data', str(archive), '--output', str(root / 'run'),
                      '--persistent', str(root / 'shared'), '--device', 'cpu', '--epochs', '1',
                      '--accumulation', '2', '--max-tokens', '24', '--max-views', '1',
                      '--head-width', '8', '--dropout', '0', '--checkpoint-every', '1',
                      '--eval-every', '1', '--no-gradient-checkpointing', '--warmup-updates', '1']
            environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}

            def launch(extra):
                result = subprocess.run([sys.executable, '-m', 'interseq', *common, *extra],
                                        env=environment, text=True, capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-2000:])
                return CheckpointStore(root / 'shared').load()

            saved = launch(['--pretrained-model', str(pretrained), '--max-updates', '1'])
            self.assertEqual(saved['step'], 1)
            self.assertEqual(saved['protocol']['initialization']['kind'], 'pretrained-esm2')
            self.assertEqual(saved['protocol']['initialization']['initial_weights_sha256'], initial_hash)
            self.assertFalse(any((root / '.interseq-inputs').rglob('*.pt')))
            shutil.rmtree(pretrained)
            shutil.rmtree(root / 'run')
            resumed = launch(['--resume', '--max-updates', '2'])
            self.assertEqual(resumed['step'], 2)
            self.assertEqual(resumed['protocol'], saved['protocol'])
