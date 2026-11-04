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
