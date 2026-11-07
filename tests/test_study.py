"""Verify published split preservation and fixed-model final evaluation offline."""
import contextlib
import copy
import csv
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from interseq.base import initialize_pretrained, sha256
from interseq.checkpoints import CheckpointStore
from interseq.data import PairBatcher, read_records
from interseq.engine import TrainConfig, fit
from interseq.model import PairModel
from interseq.study import audit, fetch_sources, final_test, prepare, specification, verify_study
from test_interseq import backbone


class StudyTests(unittest.TestCase):
    def sources(self, root):
        cache = root / 'sources'
        human = cache / 'bernett-v4'
        yeast = cache / 'dscript'
        human.mkdir(parents=True)
        yeast.mkdir(parents=True)
        proteins = {}
        for block, prefix, residues in [(1, 'T', 'ACDE'), (0, 'V', 'FGHI'), (2, 'E', 'KLMN')]:
            ids = [prefix + str(i) for i in range(4)]
            proteins.update({name: residue * 8 for name, residue in zip(ids, residues)})
            for suffix, pairs in [('pos', [(0, 1), (2, 3)]), ('neg', [(0, 2), (1, 3)])]:
                (human / f'Intra{block}_{suffix}_rr.txt').write_text(''.join(f'{ids[a]} {ids[b]}\n' for a, b in pairs))
        (human / 'human_swissprot_oneliner.fasta').write_text(''.join(f'>{name}\n{seq}\n' for name, seq in proteins.items()))
        y = {f'4932.Y{i}': residue * 8 for i, residue in enumerate('PQRS')}
        (yeast / 'yeast.fasta').write_text(''.join(f'>{name}\n{seq}\n' for name, seq in y.items()))
        (yeast / 'yeast_test.tsv').write_text('4932.Y0\\t4932.Y1\\t1.0\n4932.Y2\t4932.Y3\t1.0\n'
                                           '4932.Y0\t4932.Y2\t0.0\n4932.Y1\t4932.Y3\t0.0\n')
        spec = copy.deepcopy(specification())
        spec['downloads'] = []
        spec['expected'] = {role: {'1': 2, '0': 2} for role in ['train', 'validation', 'bernett_test', 'yeast_test']}
        return cache, spec

    def train(self, study, run, paused=False):
        torch.set_num_threads(1)
        pretrained = run.parent / 'pretrained'
        backbone().save_pretrained(pretrained)
        model, vocab, identity = initialize_pretrained(source=pretrained, local_files_only=True)
        config = TrainConfig(epochs=1, batch_size=1, accumulation=2, max_tokens=24, max_views=1,
                             head_width=8, dropout=0., eval_every=1, checkpoint_every=1,
                             gradient_checkpointing=False, warmup_updates=1)
        return fit(PairModel(model, width=8, dropout=0.), PairBatcher(vocab, 24, 1),
                   read_records(study / 'development.csv.gz', 'train'),
                   read_records(study / 'development.csv.gz', 'validation'), identity,
                   config, run, torch.device('cpu'), max_updates=1 if paused else None)

    def test_prepare_preserves_published_membership_and_isolates_both_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, spec = self.sources(root)
            first = prepare(cache, root / 'study', spec)
            second = prepare(cache, root / 'repeat', spec)
            self.assertEqual(first['artifacts'], second['artifacts'])
            verified = verify_study(root / 'study')
            self.assertEqual(verified['schema'], 'interseq-study-v1')
            with gzip.open(root / 'study/development.csv.gz', 'rt') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row['split'] for row in rows}, {'train', 'validation'})
            self.assertEqual([row['protein_a'] for row in rows[:4]], ['T0', 'T2', 'T0', 'T1'])
            self.assertEqual([row['label'] for row in rows[:4]], ['1', '1', '0', '0'])
            self.assertEqual(len(read_records(root / 'study/tests/bernett.csv.gz')), 4)
            self.assertEqual(len(read_records(root / 'study/tests/yeast.csv.gz')), 4)
            self.assertTrue(all(not any(value.values()) for value in json.loads((root / 'study/audit.json').read_text())['overlaps'].values()))

    def test_changed_cached_source_is_refused_without_downloading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'source.txt'
            path.write_text('original')
            spec = {'downloads': [{'path': path.name, 'bytes': path.stat().st_size,
                                   'sha256': sha256(path), 'url': 'https://invalid.example/file'}]}
            path.write_text('modified')
            with patch('interseq.study.urlopen', side_effect=AssertionError('No network')):
                with self.assertRaisesRegex(ValueError, 'Cached source checksum'):
                    fetch_sources(root, spec)

    def test_missing_sequences_and_wrong_counts_never_drop_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, spec = self.sources(root)
            spec['expected']['train']['1'] = 3
            with self.assertRaisesRegex(ValueError, 'row counts'):
                prepare(cache, root / 'study', spec)
            self.assertFalse((root / 'study').exists())
            path = cache / 'bernett-v4/Intra1_pos_rr.txt'
            path.write_text('MISSING T1\nT2 T3\n')
            with self.assertRaises(KeyError):
                prepare(cache, root / 'study', spec)
            self.assertFalse((root / 'study').exists())

    def test_exact_sequence_leakage_with_different_ids_is_rejected(self):
        def row(name, a, b, label):
            return {'pair_id': name, 'protein_a': name + 'a', 'protein_b': name + 'b',
                    'sequence_a': a, 'sequence_b': b, 'label': label}
        rows = {'train': [row('t1', 'AAAA', 'CCCC', 1), row('t2', 'DDDD', 'EEEE', 0)],
                'test': [row('e1', 'AAAA', 'FFFF', 1), row('e2', 'GGGG', 'HHHH', 0)]}
        with self.assertRaisesRegex(ValueError, 'splits overlap'):
            audit(rows)

    def test_published_duplicate_conflicting_and_self_pairs_are_reported(self):
        rows = [{'pair_id': str(i), 'protein_a': a, 'protein_b': b, 'sequence_a': a * 4,
                 'sequence_b': b * 4, 'label': label}
                for i, (a, b, label) in enumerate([('A', 'C', 1), ('C', 'A', 0), ('A', 'A', 0)])]
        result = audit({'test': rows})
        self.assertEqual(result['splits']['test']['repeated_unordered_pair_rows'], 1)
        self.assertEqual(result['splits']['test']['conflicting_unordered_pairs'], 1)
        self.assertEqual(result['splits']['test']['self_pair_rows'], 1)
        self.assertEqual(result['rows_removed'], 0)

    def test_unfinished_training_cannot_open_either_final_test(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            cache, spec = self.sources(root)
            prepare(cache, root / 'study', spec)
            self.train(root / 'study', root / 'run', paused=True)
            with patch('interseq.study.predict', side_effect=AssertionError('Final test was opened')) as predict:
                with self.assertRaisesRegex(ValueError, 'Finish the declared training'):
                    final_test(root / 'study', root / 'run', root / 'final', 'cpu')
                predict.assert_not_called()
            self.assertFalse((root / 'final').exists())

    def test_both_final_tests_use_one_fixed_validation_threshold_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            cache, spec = self.sources(root)
            prepare(cache, root / 'study', spec)
            self.assertEqual(self.train(root / 'study', root / 'run')['status'], 'complete')
            threshold = CheckpointStore(root / 'run').load('best')['state']['threshold']
            with patch('interseq.engine.threshold_mcc', side_effect=AssertionError('No test tuning')):
                result = final_test(root / 'study', root / 'run', root / 'final', 'cpu')
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(set(result['results']), {'bernett_test', 'yeast_test'})
            self.assertFalse(result['test_threshold_selection'])
            for measured in result['results'].values():
                self.assertEqual(measured['validation_threshold_metrics']['threshold'], threshold)
                self.assertEqual(measured['fixed_0_5_metrics']['threshold'], .5)
                self.assertEqual(measured['validation_threshold_metrics']['n'], 4)
            pointer = json.loads((root / 'run/best.json').read_text())
            self.assertEqual(result['checkpoint'], pointer)
            with self.assertRaises(FileExistsError):
                final_test(root / 'study', root / 'run', root / 'final', 'cpu')
            test = root / 'study/tests/yeast.csv.gz'
            test.write_bytes(b'changed test')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                final_test(root / 'study', root / 'run', root / 'changed', 'cpu')


if __name__ == '__main__':
    unittest.main()
