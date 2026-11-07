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

from interseek.benchmark import distribution, parser, run
from test_interseek import backbone, records


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

    def test_real_training_full_validation_memory_and_resume(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            result = run(self.prepare(root))
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(result['data']['full_validation'])
            self.assertEqual(result['training']['count'], 3)
            self.assertEqual(result['training']['measured_pairs'], 6)
            self.assertEqual(result['validation']['pairs'], 4)
            self.assertEqual(result['checks']['all_encoder_layers_updated'], 2)
            for check in ['checkpoint_checksum_verified', 'reload_predictions_match',
                          'swapped_pair_predictions_match', 'resume_advanced_one_update',
                          'resume_protocol_unchanged', 'resume_optimiser_verified', 'best_checkpoint_verified']:
                self.assertTrue(result['checks'][check], check)
            updates = [phase for phase in result['phases'] if phase['phase'] == 'training_update']
            self.assertEqual([phase['step'] for phase in updates], [1, 2, 3, 4])
            self.assertEqual(len([phase for phase in result['phases'] if phase['phase'] == 'validation']), 1)
            for phase in result['phases']:
                self.assertGreater(phase['elapsed_seconds'], 0)
                self.assertGreaterEqual(phase['memory_samples'], 2)
                self.assertGreater(phase['sampled_peak_bytes']['process_rss_bytes'], 0)
                self.assertNotIn('native_device_peak_bytes', phase)
            progress = json.loads((root / 'benchmark/progress.json').read_text())
            self.assertEqual(progress['status'], 'complete')
            self.assertTrue((root / 'benchmark/memory.jsonl').stat().st_size > 0)
            with self.assertRaises(FileExistsError):
                run(self.prepare(root))

    def test_bad_reloaded_probabilities_fail_and_preserve_evidence(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            args = self.prepare(root)
            with patch('interseek.benchmark.predict', return_value=np.full(4, np.nan)):
                with self.assertRaisesRegex(ValueError, 'finite score'):
                    run(args)
            report = json.loads((root / 'benchmark/benchmark.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertNotIn('validation', report)
            self.assertTrue((root / 'benchmark/run/latest.json').is_file())
            self.assertFalse(report['phases'][-1]['complete'])

    def test_runtime_limit_is_reported_as_incomplete(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            args = self.prepare(root)
            args.hours = 1e-9
            with self.assertRaises(TimeoutError):
                run(args)
            report = json.loads((root / 'benchmark/benchmark.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertNotIn('validation', report)

    def test_invalid_timing_samples_cannot_create_a_successful_summary(self):
        for values in [[], [0], [1, float('nan')], [float('inf')]]:
            with self.assertRaises(ValueError):
                distribution(values)


if __name__ == '__main__':
    unittest.main()
