# Attribution and provenance

Pairloom's active application code is a new implementation authored for Reece
Palmer. It replaces PLMinteract's package, scripts and documentation and does
not import that package. Pairloom defaults to Meta's pretrained ESM-2 weights,
with random initialization available as a control. Authorship
does not confer ownership of external architecture implementations or methods.

Pairloom was developed in a checkout of
[PLM-interact](https://github.com/liudan111/PLM-interact). Its original MIT
license and copyright are retained verbatim in
[licenses/PLM-interact-MIT.txt](licenses/PLM-interact-MIT.txt). This standalone
repository starts with the Pairloom implementation; upstream Git history and
legacy data fixtures remain in the original checkout and are not bundled here.

Joint protein encoding and the combination of interaction and masked-language
objectives are inspired by Liu et al., [PLM-interact](https://doi.org/10.1038/s41467-025-64512-w),
Nature Communications (2025). Pairloom is not an exact reproduction: its window
bags, heads, normalisation, checkpointing and evaluation workflow differ.

The ESM-2 architecture and fixed alphabet are Meta's work: Lin et al.,
[Evolutionary-scale prediction of atomic-level protein structure with a
language model](https://doi.org/10.1126/science.ade2574), Science (2023).
[ESM](https://github.com/facebookresearch/esm) is MIT licensed.
Pretrained weights are downloaded from Meta's official Hugging Face repositories:
[35M](https://huggingface.co/facebook/esm2_t12_35M_UR50D) and
[650M](https://huggingface.co/facebook/esm2_t33_650M_UR50D).
These are the original protein-language-model weights, not PLM-interact's
downstream interaction checkpoint. Their source and initial hash are recorded.
[Transformers](https://github.com/huggingface/transformers) supplies the external
ESM implementation under Apache-2.0. These dependencies are not Pairloom-authored.

The optional transfer importer reads only the verified development data and
manifest. Pretrained model files, old checkpoints and RIMNet's training wheel
are neither read nor extracted. HIPPIE provenance and terms are recorded in
the separate RIMNet project, which retains its source and experimental record.
