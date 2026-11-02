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
