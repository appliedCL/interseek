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
