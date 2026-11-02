"""Single-device pair training with exact update boundaries and validation selection."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoints import CheckpointStore, atomic_json, restore_rng, rng_state
from .data import fingerprint, validate_training_labels
from .metrics import report, threshold_mcc


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    seed: int = 17
    batch_size: int = 1
    accumulation: int = 16
    encoder_lr: float = 2e-5
    head_lr: float = 1e-4
    weight_decay: float = .01
    warmup_updates: int = 100
    mlm_weight: float = .1
    mask_probability: float = .15
    max_tokens: int = 1024
    max_views: int = 2
    head: str = "cls"
    head_width: int = 128
    dropout: float = .1
    gradient_checkpointing: bool = True
    eval_every: int = 1000
    checkpoint_every: int = 25
    positive_weight: float | None = None

    def validate(self):
        if any(not math.isfinite(value) for value in [self.encoder_lr, self.head_lr, self.weight_decay,
                self.mlm_weight, self.mask_probability, self.dropout]):
            raise ValueError("Numeric training settings must be finite.")
        if min(self.epochs, self.batch_size, self.accumulation, self.max_views, self.eval_every, self.checkpoint_every) < 1:
            raise ValueError("Epoch, batch, view and interval settings must be positive.")
        if self.max_tokens < 5 or self.head_width % 4 or self.head_width < 4:
            raise ValueError("Invalid context length or head width.")
        if not 0 <= self.dropout < 1 or not 0 < self.mask_probability <= 1 or self.mlm_weight < 0:
            raise ValueError("Invalid dropout or masking/loss settings.")
        if min(self.encoder_lr, self.head_lr) <= 0 or self.weight_decay < 0 or self.warmup_updates < 0:
            raise ValueError("Invalid optimiser settings.")
        if self.positive_weight is not None and (not math.isfinite(self.positive_weight) or self.positive_weight <= 0):
            raise ValueError("positive_weight must be positive.")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_identity():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + path.read_bytes())
    return digest.hexdigest()
