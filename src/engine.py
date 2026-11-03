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


def mixed_precision(device, precision):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda" or precision not in {"fp16", "bf16"}:
        raise ValueError("Use fp32 on CPU/MPS and fp16 or bf16 on CUDA.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16.")
    return torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def predict(model, batcher, records, device, batch_size=1, precision="fp32", deadline=None, progress=False):
    model.eval()
    scores = []
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            if deadline is not None and time.monotonic() >= deadline:
                return None
            rows = records[start:start + batch_size]
            with mixed_precision(device, precision):
                forward, _ = model(to_device(batcher(rows), device))
                reverse, _ = model(to_device(batcher(rows, reverse=True), device))
            probabilities = torch.sigmoid((forward.float() + reverse.float()) * .5)
            scores.extend(probabilities.cpu().tolist())
            if progress and (len(scores) % 128 < batch_size or len(scores) == len(records)):
                elapsed = time.monotonic() - started
                print(json.dumps({"event": "validation", "pairs": len(scores), "total": len(records),
                                  "remaining_seconds": elapsed / len(scores) * (len(records) - len(scores))}), flush=True)
    return np.asarray(scores)


def write_predictions(path, records, scores):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pair_id", "label", "score"])
        writer.writerows((r.pair_id, "" if r.label is None else r.label, float(s)) for r, s in zip(records, scores))
    temporary.replace(path)


def fit(model, batcher, training, validation, initialization, config, output, device,
        precision="fp32", resume=False, max_updates=None, hours=None, persistent=None):
    """Fit all records; max_updates/hours pause the unchanged scientific protocol."""
    config.validate()
    validate_training_labels(training, validation)
    if initialization.get("kind") not in {"random-esm2", "pretrained-esm2"}:
        raise ValueError("Expected explicit random or pretrained ESM-2 provenance.")
    if any(getattr(config, key) != value for key, value in model.settings.items()):
        raise ValueError("The model architecture/trainability differs from TrainConfig.")
    output = Path(output)
    store = CheckpointStore(output)
    if not resume and (output / "latest.json").exists():
        raise FileExistsError("This run already has a checkpoint; resume it or use a new directory.")
    if not resume and any(path.name != "run.lock" for path in output.iterdir()):
        raise FileExistsError("A new run requires an empty output directory; preserve existing files and choose a new run name.")
    protocol = {"config": asdict(config), "training": fingerprint(training), "validation": fingerprint(validation),
                "initialization": initialization, "source": source_identity(),
                "task": "mutation" if training[0].mutant is not None else "ppi",
                "libraries": {name: importlib.metadata.version(name) for name in ["torch", "transformers"]}}
    if (batcher.max_tokens, batcher.max_views, batcher.seed) != (config.max_tokens, config.max_views, config.seed):
        raise ValueError("Batcher settings differ from the run configuration.")
