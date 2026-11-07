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
