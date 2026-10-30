Pairloom
========

Pairloom fine-tunes Meta's pretrained ESM-2. The default is the 35M model;
random initialization is available as a scratch-training control.
All encoder blocks, token embeddings and the language-model head are trainable.
Its source lives in ``src/pairloom``; it does not import the
retired PLMinteract package or require RIMNet at runtime.

The repository README documents installation, fresh Colab training, data,
commands, objectives, window-coverage limits and checkpoint recovery.
``THIRD_PARTY_NOTICES.md`` credits upstream research, source and weights.

Commands
--------

* ``pairloom train``: pair-label classification, optional masked-residue loss,
  window bags and resumable full-model fine-tuning or scratch training.
* ``pairloom evaluate``: fixed checkpoint and validation-selected threshold.
* ``pairloom predict``: probabilities in original row order.

All commands support ``--task mutation`` for wild-type/mutant effect direction.
Training is single-device: CPU, MPS or CUDA. Model checkpoints contain the full
trained model; resume and inference do not require downloading the base again.
The original transfer ZIP is a data-only input; model weights,
old checkpoints and training wheels in it are ignored. The architecture and
external implementation retain attribution.

Data splits are supplied by the dataset. Pairloom preserves their membership
without checking protein overlap, duplicate pairs or family separation.
