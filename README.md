# Pairloom

Pairloom **fine-tunes Meta's pretrained ESM-2** to predict protein interactions.
Every encoder block, token embedding, language-model readout and task head is
trainable. Random initialization remains available as a scratch-training control.

The default is ESM-2 35M: 12 transformer layers, width 480 and 20 attention heads.
The 650M architecture is available explicitly and costs substantially more
training memory and compute. The default weights come from
[facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D),
not PLM-interact's interaction-trained checkpoint.

The task-specific code is authored by Reece Palmer. ESM-2's architecture and
Transformers implementation remain external contributions. The paired training
method is inspired by PLM-interact; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
This is the standalone Pairloom repository; RIMNet is maintained separately.

## Fine-tune on Colab

Open [Pairloom_train_Colab.ipynb](notebooks/Pairloom_train_Colab.ipynb), select a
GPU runtime and run the cells in order. Its default input is
`My Drive/RIMNet/rimnet-kaggle-upload-v1.zip`. Only the development CSV and data
manifest are read from that ZIP. Model files, old checkpoints, warm-starts and
training wheels are neither read nor extracted. A development CSV/CSV.gz works
directly by changing `INPUT_DATA`. Meta's weights download separately on the
first run; an optional local model directory can be supplied instead.

The notebook initializes from pretrained 35M weights and writes to
`My Drive/Pairloom/pretrained-esm2-35m-cls-seed17`. Set `INITIALIZATION = "random"`
and use a different run name for a scratch control. Later sessions resume the
existing run. Full checkpoints include the model and require no external base
weights for recovery or inference. Checkpoints from earlier Pairloom versions
and RIMNet are incompatible. Use a new run name when scientific settings change.

Full validation runs every 1,000 updates and at epoch ends. Validation AUROC
selects the best checkpoint; validation MCC selects its classification threshold.
A session may pause before a score exists. Three epochs are an initial experiment,
not an established sufficient budget. Pretraining supplies learned protein
representations; interaction discrimination is trained on the supplied labels.
Better benchmark performance is a hypothesis to measure, not a guarantee.

## Installation and data

```bash
pip install -e .
pairloom --help
```

Python >=3.10; one CUDA GPU, MPS or CPU. FP16/BF16 require CUDA. Installing
dependencies does not download weights; a fresh pretrained run does.

PPI CSV columns are `sequence_a,sequence_b,label,split,pair_id`; `query,text` are
accepted aliases. Labels are 0/1, IDs optional, splits `train`, `validation`,
`test`. Pairloom uses the supplied split assignments, following PLM-interact's
input convention. It does not check protein overlap, repeated/reversed pairs or
family separation, and does not deduplicate or repartition the data. Split
quality is determined during dataset preparation. Basic sequence, label and
explicit ID checks remain; training and validation each need both classes for
class weighting and validation metrics.

```csv
pair_id,sequence_a,sequence_b,label,split
a,ACDEFGHIK,LMNPQRSTV,1,train
b,WWWWAAAA,CCCCDDDD,0,train
c,EEEEFFFF,GGGGHHHH,1,validation
d,IIIIKKKK,LLLLMMMM,0,validation
```

Separate CSVs use `--validation-data`. Inference requires sequence columns and
optional IDs. Exports retain source row order, IDs and labels when provided.

## Training and architecture selection

```bash
pairloom train --data pairs.csv.gz --output runs/pretrained35
pairloom train --data pairs.csv.gz --initialization random --output runs/random35
```

The default is `--initialization pretrained --architecture esm2-35m`.
Use `--architecture esm2-650m` for Meta's larger model. `--pretrained-model`
accepts an official Meta repository or local ESM-2 directory; `--revision` can
pin a repository revision. Loading failures stop the run instead of silently
using random weights. The source, resolved revision and initial weight hash
are recorded. Random mode accepts either preset or a local architecture JSON,
constructs the model offline and records its seed and initial weight hash.

Extract development data from the earlier archive:

```bash
python scripts/prepare_transfer.py --archive /path/to/rimnet-kaggle-upload-v1.zip --output runs/inputs
```

The command prints the verified CSV path; pass it to `pairloom train --data`.
Only data is extracted. It does not prepare, verify or load pretrained weights.

The objective is interaction binary cross-entropy plus `--mlm-weight` times
masked-residue cross-entropy (default 0.1), sharing one encoder pass. Masking
excludes special/padding tokens and guarantees a target. Class weighting uses
training labels. Accumulation normalises classification by actual record count,
including short final updates, and MLM by the update's masked-token count.
Default encoder/head learning rates are 0.00002/0.0001; these are experimental
settings, not claimed optima. Scratch controls may need their own learning-rate
and training-budget search.

`--head cls` is the baseline; `--head residue` uses a shared cross-chain attention
head. `--mlm-weight 0` is an interaction-only control. All encoder layers remain
trainable in both initialization modes.
The auxiliary upstream contact readout has no contact-label supervision.

Long sequences form bounded window bags (`--max-tokens 1024 --max-views 2`). The
bag receives the interaction label, not every fragment separately. Training
samples windows per record/epoch; evaluation uses fixed windows and both orders.
The view cap can miss interaction regions. Same-length mutants retain wild-type
crop coordinates. These summaries are not measured binding-interface maps.

## Recovery and inference

```bash
pairloom train --data pairs.csv.gz --output runs/pretrained35 --resume
pairloom evaluate --data pairs.csv.gz --split test --run runs/pretrained35 \
  --predictions runs/pretrained35/test.csv
pairloom predict --data new_pairs.csv --run runs/pretrained35 \
  --predictions runs/pretrained35/predictions.csv
```

Full model tensors, architecture, optimizer/scaler/RNG states, exact training
position and source/data/library identities are saved. Atomic latest/previous/
best pointers have SHA-256 verification. A corrupt latest save can recover from
previous. Same-device CPU continuation is regression-tested exactly. Hardware
or precision changes do not imply bitwise equivalence. Resume rejects changes
to initialization, code, data, libraries or scientific settings.
Omit initialization/model flags when resuming: the saved protocol supplies them.

`--checkpoint-every 25` bounds unsaved work. `--persistent /path/to/Drive/run`
verifies copies before updating persistent pointers and retains latest/previous/
best. `--hours` and `--max-updates` pause the unchanged protocol. Incomplete
optimizer updates are never committed. AMP overflow retries the same update
without consuming examples. Test labels never choose thresholds/checkpoints.

## Mutation effects

Use `--task mutation` with `wild_seq,mutant_seq,participant_sequence,label` and
the same split/ID contract. Labels may be 0/1 or decreasing/increasing. Pairloom
learns mutant-minus-wild-type interaction logits with a shared partner. Scores
describe binary effect direction, not calibrated binding affinity.

## Verification

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
python scripts/build_colab.py
```

All active application code is in `src/pairloom/`. Tests include local pretrained
loading, offline random initialization, all-layer gradient flow, full-model recovery, CLI workflows and
notebook execution/resume with tiny synthetic data. They establish software
behavior, not research-model performance. RIMNet and its earlier experiments
remain in their separate repository and original checkout.
