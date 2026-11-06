"""Exercise benchmark success, failure and checkpoint recovery without downloads."""
import contextlib
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from interseq.benchmark import distribution, parser, run
from test_interseq import backbone, records


class BenchmarkTests(unittest.TestCase):
    def prepare(self, root):
        pretrained = root / 'model'
        backbone().save_pretrained(pretrained)
        training, validation = records()
        training = [replace(row, pair_id=f'{i}-{row.pair_id}') for i in range(5) for row in training]
        data = root / 'data.csv'
        with data.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['pair_id', 'sequence_a', 'sequence_b', 'label', 'split'])
            for split, rows in [('train', training), ('validation', validation)]:
                for row in rows:
                    writer.writerow([row.pair_id, row.sequence_a, row.sequence_b, row.label, split])
        return parser().parse_args(['--data', str(data), '--pretrained-model', str(pretrained),
                                   '--output', str(root / 'benchmark'), '--device', 'cpu',
                                   '--updates', '4', '--warmup', '1', '--accumulation', '2',
                                   '--max-tokens', '24', '--max-views', '1', '--threads', '1'])

