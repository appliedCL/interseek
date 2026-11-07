# 35M results

InterSeek 35M completed three epochs on 163,192 training pairs. The run finished
30,600 updates; the checkpoint at update 18,000 was selected by validation
AUROC. Both final tests used that checkpoint and the human-validation MCC
threshold of `0.48556941747665405`.

| Final test | Pairs | AUROC | Average precision | MCC | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bernett human Intra2 | 52,048 | 0.6800 | 0.6700 | 0.2487 | 0.6091 | 0.6882 |
| Independent D-SCRIPT yeast | 55,000 | 0.7207 | 0.2269 | 0.1792 | 0.1441 | 0.7674 |

Average precision is scikit-learn AP, not trapezoidal PR-AUC. Threshold-dependent
metrics use the same validation-selected threshold on both tests.

The [released weights and inference package](https://huggingface.co/appliedCL/interseek-35m)
are pinned to revision `b5a885c244ef6e087c37f36745cc5b25356a19dd`.

Exact records:

- [Run configuration and provenance](../results/esm2-35m/run.json).
- [Checkpoint selection and validation metrics](../results/esm2-35m/selection.json).
- [Human results](../results/esm2-35m/human-test.json) and [yeast results](../results/esm2-35m/yeast-test.json), including threshold 0.5.
- [All 33 validation passes](../results/esm2-35m/validation.csv).

Every final-test metric in these records was recomputed from all 107,048 saved
test predictions and matched the original evaluation within `1e-12`.

## Training and selection

The backbone is Meta's pretrained `facebook/esm2_t12_35M_UR50D`. All encoder
layers and the CLS interaction head were fine-tuned, using binary
cross-entropy plus masked language modelling weighted by 0.1.

| Setting | Value |
| --- | --- |
| Seed | 17 |
| Epochs | 3 |
| Training / validation pairs | 163,192 / 59,260 |
| Effective batch | 16 pairs: batch 4, accumulation 4 |
| Encoder / head learning rate | 2e-5 / 1e-4 |
| Weight decay / warm-up | 0.01 / 100 updates |
| Context / views | 1,024 tokens / at most 2 |
| Hardware / precision | RTX 4090 / BF16 |
| PyTorch / Transformers | 2.8.0+cu128 / 5.12.1 |

The best validation AUROC was **0.6532** at update 18,000. Training continued
to update 30,600, whose validation AUROC was 0.6515. Neither final test selected
the checkpoint or threshold.

The human data use the authors' [Bernett gold-standard split](https://doi.org/10.1093/bib/bbae076)
and [corrected Figshare version 4](https://doi.org/10.6084/m9.figshare.21591618.v4):
Intra1 for training, Intra0 for validation and Intra2 for final testing.
The independent yeast test comes from [D-SCRIPT](https://doi.org/10.1016/j.cels.2021.08.010).
See the [evaluation protocol](evaluation.md) for source pinning and data audits.

## Interpretation and limits

These are single-seed results. Matched baselines, additional seeds and
uncertainty estimates are needed before claiming an advantage over other methods.

The yeast results show ranking signal across species, but precision is only
14.4% at the fixed threshold. Its positive prevalence is 9.09%, compared with
50% in the human test. Scores are not established as calibrated probabilities
of physical binding.

The published human split controls protein and homology leakage. Local checks
found no shared protein IDs or exact sequences across the four data roles;
no new all-against-all homology audit or ESM pretraining-exposure audit was run.
The yeast test preserves the source's repeated pairs, conflicting labels and
self-pairs. Two bounded views can omit regions of long proteins.

This report concerns the completed 35M experiment. It contains no results from
the separate 650M training run. Original training and checkpoint identities
remain in the provenance records; the release uses the renamed InterSeek
inference package with unchanged trained tensors.
