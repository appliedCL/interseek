# InterSeek

<a href="https://huggingface.co/appliedCL/interseek-35m"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=Model" alt="Model on Hugging Face" style="max-width: 100%;"></a>

InterSeek predicts whether two proteins interact from their amino-acid sequences.
It fine-tunes Meta's pretrained ESM-2 weights, updating every encoder layer and
learning an interaction head.

[35M model weights](https://huggingface.co/appliedCL/interseek-35m) are available with a matching inference package.

## 35M results

| Final test | Pairs | AUROC | Average precision | MCC |
| --- | ---: | ---: | ---: | ---: |
| Bernett human Intra2 | 52,048 | 0.6800 | 0.6700 | 0.2487 |
| Independent D-SCRIPT yeast | 55,000 | 0.7207 | 0.2269 | 0.1792 |

Three epochs completed: 30,600 updates, with update 18,000 selected by validation
AUROC. Both tests used the same checkpoint and validation-selected threshold.
These are single-seed results. Yeast precision at that threshold is 14.4%.
See the [full results and limitations](docs/results-35m.md) and
[exact result records](results/esm2-35m/).

## How it works

1. Read protein pairs labelled as interacting (`1`) or non-interacting (`0`).
2. Load pretrained ESM-2 35M weights.
3. Tokenise both sequences together, using smaller windows for long proteins.
4. Produce an interaction prediction for each pair.
5. Compare predictions with the labels and update the model. During training,
   the model also learns to recover masked amino acids.
6. Evaluate on separate validation pairs and save the best checkpoint.
7. Score new pairs from `0` to `1`. Higher scores indicate predicted interactions.

## Architecture

![InterSeek architecture: paired sequences pass through pretrained ESM-2, an interaction head and logit pooling to produce an interaction score. Masked amino-acid prediction is used during training.](docs/assets/architecture.svg)

The CLS token summarises the paired input. The interaction head uses this
summary to predict an interaction. Training combines interaction loss with
masked amino-acid prediction loss, weighted by `0.1` by default.

During inference, InterSeek combines both input orders so swapping the two
proteins produces the same score.

## Install

Requires Python 3.10 or later. For NVIDIA GPU training, use a CUDA-enabled
PyTorch installation.

```bash
pip install -e .
interseek --help
```

## Use the released 35M model

Download the weights and install the matching inference package:

```bash
pip install huggingface_hub
hf download appliedCL/interseek-35m --local-dir data/models/interseek-35m
cd data/models/interseek-35m
pip install -r requirements-inference.txt ./interseek-0.3.0-py3-none-any.whl
python predict.py --data example_pairs.csv --predictions predictions.csv --device cpu
```

Use `--device cuda:0` on an NVIDIA GPU. Replace the example CSV with protein
pairs containing `sequence_a` and `sequence_b` columns. The release includes
inference weights; it does not include optimiser checkpoints for resuming training.

## Prepare the data

Use a CSV or compressed CSV with these columns:

```csv
sequence_a,sequence_b,label,split
ACDEFGHIK,LMNPQRSTV,1,train
WWWWAAAA,CCCCDDDD,0,train
EEEEFFFF,GGGGHHHH,1,validation
IIIIKKKK,LLLLMMMM,0,validation
```

Training and validation must each contain both labels. InterSeek preserves your
supplied splits; prepare any protein or sequence separation before training.
An optional `pair_id` column identifies each row. Separate validation files are
supported through `--validation-data`.

## Study data

```bash
interseek-study prepare --output data/study
```

The study preserves the [Bernett gold-standard splits](https://doi.org/10.1093/bib/bbae076):
163,192 HIPPIE-derived training pairs, 59,260 validation pairs and 52,048 final
test pairs. The authors' corrected version 4 FASTA is pinned alongside their
original pair files. A second final test contains 55,000 yeast pairs from
[D-SCRIPT](https://doi.org/10.1016/j.cels.2021.08.010), independently derived
from STRING. Both test sets are stored separately from the development CSV.

Preparation verifies hashes, counts and exact protein/sequence separation,
preserves the published rows, and records data-quality issues in `audit.json`.
See [the evaluation protocol](docs/evaluation.md) for the split details,
pretraining limitations and requirements for comparing published results.

## Train on a GPU

```bash
interseek train --data data/study/development.csv.gz --output runs/bernett-35m --device cuda:0
```

The default model is `facebook/esm2_t12_35M_UR50D`. Weights download on the first
run. Each process trains on one GPU; use `cuda:1` to select another. Automatic
precision selects BF16 where supported, otherwise FP16 on CUDA.

Checkpoints are saved every 25 updates. Validation runs every 1,000 updates and
at epoch ends, selecting the best checkpoint by AUROC.

Resume an interrupted run with the same data and settings:

```bash
interseek train --data data/study/development.csv.gz --output runs/bernett-35m --device cuda:0 --resume
```

Full checkpoints include model weights, optimiser state and training progress.
Keep the source code and library versions unchanged when resuming.

## Check training performance

```bash
pip install -e '.[benchmark]'
interseek-benchmark --data pairs.csv.gz --output runs/benchmark --device mps
```

Use `--device cuda:0` on an NVIDIA GPU. The benchmark runs 35 training updates,
excludes the first five from timing statistics, reloads the checkpoint and
resumes for one update, then validates every pair in the supplied validation
split. It checks encoder updates, saved predictions and checkpoint recovery.
Use a new output directory for each run.

`benchmark.json` records the configuration, data and source hashes, library
versions, timing distributions and memory measurements. Training, validation,
metrics and checkpoint I/O are timed separately. CUDA memory peaks come from
the allocator; process RSS and Apple GPU peaks are sampled every 50 ms and may
miss brief allocations. Apple GPU and process memory overlap and must not be
added together. Raw measurements and live progress are saved alongside the report.

Full validation can take substantial time. `--hours` defaults to three hours;
an interrupted or incomplete check is reported as a failure. An explicit
`--validation-pairs N` limits validation for a smoke test and is labelled as a
sample. These checks measure workflow correctness and local performance;
they do not establish final model quality.

## Predict and evaluate

After training is complete, evaluate the validation-selected checkpoint on
both final tests without changing the model or threshold:

```bash
interseek-study final-test --study data/study --run runs/bernett-35m --output runs/final-evaluation --device cuda:0
```

The command verifies training provenance, records the fixed checkpoint and
threshold before prediction, and refuses incomplete training or changed data.
Both validation-selected and fixed-0.5 threshold metrics are reported.

For other prediction or evaluation inputs:

```bash
interseek predict --data new_pairs.csv --run runs/esm2 --predictions predictions.csv
interseek evaluate --data test.csv --run runs/esm2 --predictions test_predictions.csv
```

Both commands use the best checkpoint by default. Prediction files need the
two sequence columns; evaluation files also need labels. Output preserves row
order and includes each pair's interaction score.

## Other options

| Option | Purpose |
| --- | --- |
| `--architecture esm2-650m` | Use the larger pretrained ESM-2 model. |
| `--batch-size` and `--accumulation` | Adjust the effective training batch size. |
| `--max-tokens` and `--max-views` | Control sequence window size and coverage. |
| `--initialization random` | Train a randomly initialised baseline in a separate run. |

Run `interseek train --help` for all training options.
