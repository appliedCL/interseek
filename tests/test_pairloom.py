import contextlib
import copy
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
from transformers import EsmConfig, EsmForMaskedLM

from pairloom.base import initialize_architecture, initialize_pretrained, weight_digest
from pairloom.checkpoints import CheckpointStore
from pairloom.data import Record, PairBatcher, read_records, windows
from pairloom.engine import TrainConfig, fit, predict, seed_everything
from pairloom.metrics import report, threshold_mcc
from pairloom.model import PairModel

TOKENS = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P',
          'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']
VOCAB = {t: i for i, t in enumerate(TOKENS)}


def backbone():
    return EsmForMaskedLM(EsmConfig(vocab_size=33, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, pad_token_id=1, mask_token_id=32, position_embedding_type="rotary",
        token_dropout=True, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0))


def records():
    training = [Record(str(i), a * 4, b * 4, label) for i, (a, b, label) in enumerate([
        ('A', 'C', 0), ('D', 'E', 1), ('F', 'G', 0), ('H', 'I', 1), ('K', 'L', 1)])]
    validation = [Record('v' + str(i), a * 4, b * 4, label) for i, (a, b, label) in enumerate([
        ('M', 'N', 0), ('P', 'Q', 1), ('R', 'S', 0), ('T', 'V', 1)])]
    return training, validation


class DataTests(unittest.TestCase):
    def test_original_ids_and_label_order_survive_window_expansion(self):
        rows = [Record('first', 'A' * 30, 'C' * 20, 1), Record('second', 'D' * 4, 'E' * 4, 0)]
        batch = PairBatcher(VOCAB, 15, 3)(rows, training=True)
        self.assertEqual(batch['labels'].tolist(), [1., 0.])
        self.assertEqual(batch['owners'].tolist(), [0, 0, 0, 1])
        self.assertLessEqual(batch['input_ids'].shape[1], 15)

    def test_special_tokens_never_become_mlm_targets(self):
        batcher = PairBatcher(VOCAB)
        clean = batcher(records()[0][:2])
        masked = batcher.mask(clean, 1.)
        self.assertTrue(torch.equal(masked['mlm_targets'].ne(-100), clean['chains'].ne(0)))
        self.assertTrue(torch.equal(clean['input_ids'][clean['chains'] == 0], masked['input_ids'][clean['chains'] == 0]))

    def test_tiny_mask_probability_still_has_one_target(self):
        batcher = PairBatcher(VOCAB)
        masked = batcher.mask(batcher(records()[0][:1]), 1e-30)
        self.assertEqual(int(masked['mlm_targets'].ne(-100).sum()), 1)

    def test_windows_are_deterministic_bounded_and_swap_consistent(self):
        a, b = 'ACDE' * 30, 'KLMN' * 40
        first = windows(a, b, 25, 4)
        self.assertEqual(first, windows(a, b, 25, 4))
        self.assertEqual(first, [(y, x) for x, y in windows(b, a, 25, 4)])
        self.assertTrue(all(len(x) + len(y) + 3 <= 25 for x, y in first))
        self.assertNotEqual(windows(a, b, 25, 4, True, 17), windows(a, b, 25, 4, True, 29))

    def test_mutants_use_same_coordinates_and_both_variants_keep_their_owner(self):
        a = 'ACDE' * 20
        r = Record('mutation', a, 'KLMN' * 20, 1, 'W' + a[1:])
        batch = PairBatcher(VOCAB, 31, 3)([r])
        self.assertEqual(batch['owners'].tolist(), [0, 0, 0, 1, 1, 1])
        self.assertTrue(torch.equal(batch['chains'][:3], batch['chains'][3:]))
        self.assertLessEqual(int((batch['input_ids'][:3] != batch['input_ids'][3:]).sum()), 1)

    def test_repeated_pairs_are_preserved_with_stable_generated_row_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'pairs.csv'
            path.write_text('sequence_a,sequence_b,label,split\n'
                            'AAAA,CCCC,1,train\nCCCC,AAAA,1,train\n'
                            'AAAA,CCCC,1,validation\nAAAA,CCCC,1,train\n')
            all_rows = read_records(path)
            training = read_records(path, 'train')
            self.assertEqual(len(all_rows), 4)
            self.assertEqual(len({r.pair_id for r in all_rows}), 4)
            self.assertEqual(training, [all_rows[i] for i in [0, 1, 3]])

    def test_csv_split_selection_never_reads_test_rows_as_training(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'pairs.csv'
            path.write_text('pair_id,sequence_a,sequence_b,label,split\na,AAAA,CCCC,1,train\nb,DDDD,EEEE,0,test\n')
            rows = read_records(path, 'train')
            self.assertEqual([r.pair_id for r in rows], ['a'])


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_everything(17)

    def test_joint_loss_has_encoder_gradients_and_only_one_backbone_pass(self):
        model = PairModel(backbone())
        batcher = PairBatcher(VOCAB)
        batch = batcher.mask(batcher(records()[0][:2]), .15)
        calls = []
        hook = model.backbone.esm.register_forward_hook(lambda *args: calls.append(1))
        logits, mlm = model(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch['labels']) + .1 * mlm
        loss.backward()
        hook.remove()
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.esm.encoder.layer[-1].parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.esm.encoder.layer[0].parameters()))
        self.assertIsNotNone(model.backbone.esm.embeddings.word_embeddings.weight.grad)
        self.assertIsNotNone(model.backbone.lm_head.dense.weight.grad)

    def test_evaluation_is_order_invariant_even_for_long_pairs(self):
        model = PairModel(backbone(), head='residue', width=8).eval()
        batcher = PairBatcher(VOCAB, 24, 3)
        rows = [Record('long', 'ACDE' * 10, 'KLMN' * 12, 1)]
        swapped = [replace(rows[0], sequence_a=rows[0].sequence_b, sequence_b=rows[0].sequence_a)]
        a = predict(model, batcher, rows, torch.device('cpu'))
        b = predict(model, batcher, swapped, torch.device('cpu'))
        np.testing.assert_array_equal(a, b)

    def test_identical_mutant_has_zero_effect_logit(self):
        model = PairModel(backbone()).eval()
        row = Record('mutation', 'AAAA', 'CCCC', 1, 'AAAA')
        with torch.no_grad():
            logits, _ = model(PairBatcher(VOCAB)([row]))
        torch.testing.assert_close(logits, torch.zeros(1), rtol=0, atol=0)

    def test_all_parameters_trainable_and_full_restoration_preserves_predictions(self):
        model = PairModel(backbone()).eval()
        self.assertTrue(all(p.requires_grad for p in model.backbone.parameters()))
        batch = PairBatcher(VOCAB)(records()[0][:1])
        expected = model(batch)[0].detach()
        saved = model.checkpoint_state()
        with torch.no_grad():
            for p in model.head.parameters(): p.zero_()
        model.restore(saved)
        torch.testing.assert_close(model(batch)[0], expected, rtol=0, atol=0)


class CheckpointTests(unittest.TestCase):
    def test_corruption_recovers_previous_and_persistent_copy_is_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = CheckpointStore(root / 'local')
            for step in [1, 2]:
                store.save({'step': step, 'tensor': torch.tensor([step])})
            latest = json.loads((store.root / 'latest.json').read_text())
            (store.root / latest['file']).write_bytes(b'damaged')
            self.assertEqual(store.load()['step'], 1)
            store.save({'step': 3, 'tensor': torch.tensor([3])})
            self.assertEqual(store.load('previous')['step'], 1)
            store.mirror(root / 'drive')
            self.assertEqual(CheckpointStore(root / 'drive').load()['step'], 3)
            self.assertEqual(CheckpointStore(root / 'drive').load('previous')['step'], 1)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_everything(17)

    def test_resuming_after_partial_epoch_matches_uninterrupted_updates(self):
        initial = backbone()
        config = TrainConfig(epochs=2, batch_size=1, accumulation=3, head_width=8,
                             max_tokens=24, max_views=2, checkpoint_every=1, eval_every=2,
                             warmup_updates=1, gradient_checkpointing=False)
        training, validation = records()
        def fresh():
            seed_everything(17)
            return PairModel(copy.deepcopy(initial), width=8)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            batcher = PairBatcher(VOCAB, 24, 2, 17)
            full = fresh()
            fit(full, batcher, training, validation, {'kind':'random-esm2','fixture':'tiny'}, config, root / 'full', torch.device('cpu'))
            partial = fresh()
            fit(partial, batcher, training, validation, {'kind':'random-esm2','fixture':'tiny'}, config, root / 'resumed', torch.device('cpu'), max_updates=1)
            resumed = fresh()
            result = fit(resumed, batcher, training, validation, {'kind':'random-esm2','fixture':'tiny'}, config, root / 'resumed', torch.device('cpu'), resume=True)
            self.assertEqual(result['step'], 4)
            for name, tensor in full.checkpoint_state().items():
                torch.testing.assert_close(tensor, resumed.checkpoint_state()[name], rtol=0, atol=0)
            self.assertTrue((root / 'resumed/best.json').exists())
            with (root / 'resumed/validation.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r['pair_id'] for r in rows], [r.pair_id for r in validation])

    def test_accumulation_matches_a_larger_batch_including_final_short_update(self):
        initial = backbone()
        training, validation = records()
        results = []
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            for batch_size, accumulation in [(1, 2), (2, 1)]:
                seed_everything(17)
                model = PairModel(copy.deepcopy(initial), width=8, dropout=0.)
                config = TrainConfig(epochs=1, batch_size=batch_size, accumulation=accumulation,
                    mlm_weight=0., max_tokens=24, max_views=1, head_width=8, dropout=0.,
                    checkpoint_every=1, warmup_updates=1, gradient_checkpointing=False)
                fit(model, PairBatcher(VOCAB,24,1), training, validation, {'kind':'random-esm2','fixture':'tiny'},config,
                    Path(temp)/str(batch_size), torch.device('cpu'))
                results.append(model.checkpoint_state())
            for name, tensor in results[0].items():
                torch.testing.assert_close(tensor, results[1][name], rtol=1e-4, atol=1e-6)

    def test_resume_rejects_changed_data(self):
        training, validation = records()
        config = TrainConfig(epochs=1, accumulation=1, mlm_weight=0.,
                             max_tokens=24, max_views=1, checkpoint_every=1, gradient_checkpointing=False)
        model = PairModel(backbone())
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            fit(model, PairBatcher(VOCAB,24,1), training, validation, {'kind':'random-esm2'}, config, temp, torch.device('cpu'), max_updates=1)
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                fit(model, PairBatcher(VOCAB,24,1), [replace(training[0],label=1),*training[1:]], validation,
                    {'kind':'random-esm2'},config,temp,torch.device('cpu'),resume=True)

    def test_scaler_overflow_retries_without_consuming_an_update(self):
        training,validation=records()
        model=PairModel(backbone())
        config=TrainConfig(epochs=1,accumulation=5,mlm_weight=0.,
                           max_tokens=24,max_views=1,gradient_checkpointing=False)
        scaler=torch.amp.GradScaler('cpu',init_scale=128.)
        calls=[]
        def overflow_once(gradient):
            calls.append(1)
            return torch.full_like(gradient,float('inf')) if len(calls)==1 else gradient
        hook=model.head[-1].weight.register_hook(overflow_once)
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()), \
             patch('pairloom.engine.torch.amp.GradScaler',return_value=scaler):
            result=fit(model,PairBatcher(VOCAB,24,1),training,validation,{'kind':'random-esm2'},config,temp,torch.device('cpu'))
            self.assertEqual(result['step'],1)
            self.assertEqual(scaler.get_scale(),64.)
            saved=CheckpointStore(temp).load()
            self.assertTrue(all(float(value['step'])==1 for value in saved['optimizer']['state'].values()))
            self.assertTrue(all(torch.isfinite(t).all() for t in saved['model'].values()))
        hook.remove()


class MetricTests(unittest.TestCase):
    def test_tied_scores_and_selected_threshold_are_consistent(self):
        y, scores = [0,1,0,1], [.2,.2,.8,.8]
        threshold = threshold_mcc(y,scores)
        chosen = report(y,scores,threshold)
        from sklearn.metrics import matthews_corrcoef
        candidates = [0.,.2,.8,np.nextafter(.8,np.inf)]
        self.assertEqual(chosen['mcc'], max(matthews_corrcoef(y,np.asarray(scores)>=t) for t in candidates))


class InitializationTests(unittest.TestCase):
    def test_random_initialization_is_reproducible_and_never_loads_weights(self):
        config=dict(num_hidden_layers=2,hidden_size=16,num_attention_heads=2,intermediate_size=64)
        with patch.object(EsmForMaskedLM,'from_pretrained',side_effect=AssertionError('Forbidden pretrained loader')), \
             patch('torch.load',side_effect=AssertionError('Forbidden weight file')):
            a,vocab,identity=initialize_architecture(seed=17,config=config)
            b,_,same=initialize_architecture(seed=17,config=config)
            c,_,different=initialize_architecture(seed=29,config=config)
        self.assertEqual(identity,same)
        self.assertNotEqual(identity['initial_weights_sha256'],different['initial_weights_sha256'])
        self.assertEqual(vocab,VOCAB)
        self.assertEqual(identity['kind'],'random-esm2')
        for name,tensor in a.state_dict().items():
            torch.testing.assert_close(tensor,b.state_dict()[name],rtol=0,atol=0)
        self.assertFalse(torch.equal(a.esm.embeddings.word_embeddings.weight,c.esm.embeddings.word_embeddings.weight))

    def test_pretrained_loader_uses_supplied_weights_and_records_origin(self):
        with tempfile.TemporaryDirectory() as temp:
            original=backbone()
            with torch.no_grad():
                original.esm.embeddings.word_embeddings.weight.fill_(.125)
            original.save_pretrained(temp)
            loaded,vocab,identity=initialize_pretrained(source=temp,local_files_only=True)
            self.assertEqual(identity['kind'],'pretrained-esm2')
            self.assertEqual(identity['source'],temp)
            self.assertEqual(identity['initial_weights_sha256'],weight_digest(original))
            self.assertEqual(vocab,VOCAB)
            for name,tensor in original.state_dict().items():
                torch.testing.assert_close(tensor,loaded.state_dict()[name],rtol=0,atol=0)

    def test_pretrained_failure_does_not_fall_back_to_random(self):
        with patch.object(EsmForMaskedLM,'from_pretrained',side_effect=OSError('download failed')):
            with self.assertRaisesRegex(OSError,'download failed'):
                initialize_pretrained()

    def test_default_pretrained_source_is_meta_35m_and_partial_load_fails(self):
        with patch.object(EsmForMaskedLM,'from_pretrained',return_value=(backbone(),{})) as load:
            _,_,identity=initialize_pretrained()
        self.assertEqual(load.call_args.args,('facebook/esm2_t12_35M_UR50D',))
        self.assertTrue(load.call_args.kwargs['use_safetensors'])
        self.assertEqual(identity['source'],'facebook/esm2_t12_35M_UR50D')
        with patch.object(EsmForMaskedLM,'from_pretrained',return_value=(backbone(),{'missing_keys':['lm_head.bias']})):
            with self.assertRaisesRegex(ValueError,'did not load completely'):
                initialize_pretrained()


if __name__ == '__main__':
    unittest.main()
