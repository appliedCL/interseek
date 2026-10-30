"""Meta ESM-2 initialisation, optional random baselines and data-only imports."""
import hashlib
import json
from pathlib import Path
import zipfile

import torch

# ESM-2's published alphabet is a fixed token vocabulary, not learned weights.
TOKENS = ('<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I',
          'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O',
          '.', '-', '<null_1>', '<mask>')
ARCHITECTURES = {"esm2-35m": (12, 480, 20), "esm2-650m": (33, 1280, 20)}
PRETRAINED_MODELS = {"esm2-35m": "facebook/esm2_t12_35M_UR50D",
                     "esm2-650m": "facebook/esm2_t33_650M_UR50D"}


def weight_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode() + tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()
