# Evaluation protocol

Fine-tune on the published Bernett human training split, select the checkpoint
and classification threshold on its validation split, then evaluate the fixed
model on two separate tests.

| Role | Published partition | Pairs | Positives : negatives |
| --- | --- | ---: | --- |
| Fine-tuning | Bernett Intra1 | 163,192 | 1:1 |
| Model selection | Bernett Intra0 | 59,260 | 1:1 |
| Primary final test | Bernett Intra2 | 52,048 | 1:1 |
| Independent final test | D-SCRIPT yeast | 55,000 | 1:10 |

The [Bernett paper](https://doi.org/10.1093/bib/bbae076) uses HIPPIE v2.3,
KaHIP partitioning of sequence similarities and CD-HIT redundancy reduction at
40%. Proteins are separated across splits; sequence similarity is reduced.
We preserve the authors' assignments rather than create a new split or claim
a newly measured universal identity bound. The pinned
[version 4 release](https://doi.org/10.6084/m9.figshare.21591618.v4) has identical
pair files to the paper's version 3 and adds the missing `Q96PU5` sequence to
the FASTA. Other sequences are unchanged.

The independent test is *Saccharomyces cerevisiae* from
[D-SCRIPT](https://doi.org/10.1016/j.cels.2021.08.010): STRING v11 binding
interactions with experimental evidence and sampled negatives. Source files
are pinned to the author repository commit recorded in `study_sources.json`.
Cross-species transfer can involve homologues; it is not a homology-excluded
test. Neither supervised separation nor species separation rules out exposure
of test sequences during ESM-2 pretraining.

Preparation checks source hashes, row counts, protein IDs and exact sequences
across all four roles. It preserves row order within each published source,
labels, duplicates and self-pairs. The yeast release contains 32 repeated
unordered-pair rows, four unordered pairs with conflicting labels and five
self-pairs. These are recorded in `audit.json`; no rows are silently removed.

## Model selection and final reporting

Use `development.csv.gz` for training and validation only. Select the best
checkpoint by validation AUROC and its threshold by validation MCC. Complete
the declared training run before running `interseq-study final-test`.

Final evaluation verifies the development fingerprints, freezes the selected
checkpoint hash and threshold, then evaluates both tests with that same model.
It reports average precision (scikit-learn AP), AUROC, MCC, F1, precision,
recall and the other existing metrics. Threshold-dependent results are given
at the validation-selected threshold and at the predeclared threshold 0.5.
Neither test is used for threshold optimisation or checkpoint selection.

## Published comparisons

The original Bernett study reports results for PIPR, D-SCRIPT, Topsy-Turvy,
DeepFE and the Richoux models (Supplementary Table S5). It also distinguishes
early stopping from combined training/validation protocols and applies length
restrictions to some methods. Those settings must accompany any quoted score.

[Frozen ESM-2 classifiers and fine-tuning baselines](https://pmc.ncbi.nlm.nih.gov/articles/PMC11214071/)
and [C3PI](https://doi.org/10.1093/bib/bbaf685) provide further published
gold-standard references. A method name alone does not establish a matched
comparison: verify the release, test rows, sequence filtering, training data,
pretraining, seed aggregation, threshold and precision-recall integration.

RAPPPID and SENSE-PPI remain candidate comparators; a matching result on this
exact human split has not been verified here. Their results on other splits
must not be inserted into a common leaderboard. Published cross-species
results trained on other human data are contextual references. Controlled
comparisons require matching test predictions and training protocols, or
retraining where these are unavailable.
