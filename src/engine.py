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
    trained = {name: p for name, p in model.named_parameters() if p.requires_grad}
    groups = [{"params": [p for name, p in trained.items() if name.startswith("backbone.") == encoder],
               "lr": lr, "initial_lr": lr} for encoder, lr in [(True, config.encoder_lr), (False, config.head_lr)]]
    optimizer = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    seed_everything(config.seed)
    state = {"epoch": 0, "cursor": 0, "step": 0, "history": [], "best_auroc": -1.,
             "threshold": None, "pending_validation": False, "last_validation_step": -1}
    if resume:
        saved = store.load()
        if saved["protocol"] != protocol:
            raise ValueError("Resume initialisation, code, data, libraries or scientific configuration changed.")
        model.restore(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if scaler.is_enabled() and saved["scaler"]:
            scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        restore_rng(saved["rng"])
        del saved
    positive_weight = config.positive_weight
    if positive_weight is None:
        positives = sum(r.label for r in training)
        positive_weight = (len(training) - positives) / positives
    weight = torch.tensor(positive_weight, device=device)
    effective = config.batch_size * config.accumulation
    total_updates = math.ceil(len(training) / effective) * config.epochs
    start_time = time.monotonic()
    deadline = start_time + hours * 3600 if hours else None
    atomic_json(output / "run.json", {"protocol": protocol, "positive_weight": positive_weight,
                "training_pairs": len(training), "validation_pairs": len(validation),
                "device": str(device), "precision": precision, "total_updates": total_updates,
                "evidence": "Development validation; no final-test selection or guaranteed performance.",
                "window_limit": "max_views bounds coverage; long pairs may have unobserved window combinations."})

    def checkpoint(best=False):
        atomic_json(output / "history.json", state["history"])
        store.save({"step": state["step"], "protocol": protocol, "model": model.checkpoint_state(),
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "state": dict(state), "rng": rng_state()}, best)
        if persistent:
            store.mirror(persistent)

    def validate():
        scores = predict(model, batcher, validation, device, config.batch_size, precision, deadline, True)
        if scores is None:
            return False
        threshold = threshold_mcc([r.label for r in validation], scores)
        metrics = report([r.label for r in validation], scores, threshold)
        state["history"].append({"step": state["step"], "epoch": state["epoch"], "validation": metrics})
        state["threshold"] = threshold
        state["pending_validation"] = False
        state["last_validation_step"] = state["step"]
        best = metrics["auroc"] > state["best_auroc"]
        if best:
            state["best_auroc"] = metrics["auroc"]
            write_predictions(output / "validation.csv", validation, scores)
        checkpoint(best)
        print(json.dumps({"event": "validation_complete", "step": state["step"], "best": best, **metrics}), flush=True)
        return True

    status = "complete"
    cached_epoch, order = None, None
    # An initial checkpoint makes an interrupted first session recoverable.
    if not resume:
        checkpoint()
    try:
        while state["epoch"] < config.epochs:
            if state["pending_validation"]:
                if not validate():
                    status = "paused_validation"
                    break
                continue
            if state["cursor"] == len(training):
                state["epoch"] += 1
                state["cursor"] = 0
                continue
            if (max_updates is not None and state["step"] >= max_updates) or (deadline and time.monotonic() >= deadline):
                status = "paused"
                break
            if cached_epoch != state["epoch"]:
                order = np.random.default_rng(config.seed + state["epoch"]).permutation(len(training))
                cached_epoch = state["epoch"]
            selected = order[state["cursor"]:state["cursor"] + effective]
            batches = [batcher([training[i] for i in selected[start:start + config.batch_size]],
                               training=True, epoch=state["epoch"]) for start in range(0, len(selected), config.batch_size)]
            if config.mlm_weight:
                batches = [batcher.mask(batch, config.mask_probability) for batch in batches]
            masked = sum(int(batch["mlm_targets"].ne(-100).sum()) for batch in batches) if config.mlm_weight else 1
            factor = min((state["step"] + 1) / max(1, config.warmup_updates), 1.)
            if state["step"] >= config.warmup_updates:
                factor = max(0., (total_updates - state["step"]) / max(1, total_updates - config.warmup_updates))
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            attempt_rng = rng_state()
            for attempt in range(8):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                class_total = mlm_total = 0.
                for batch in batches:
                    batch = to_device(batch, device)
                    with mixed_precision(device, precision):
                        logits, mlm_sum = model(batch)
                        classification = F.binary_cross_entropy_with_logits(logits.float(), batch["labels"], pos_weight=weight, reduction="sum")
                        loss = classification / len(selected) + config.mlm_weight * mlm_sum / masked
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss; last committed checkpoint is preserved.")
                    scaler.scale(loss).backward()
                    class_total += float(classification.detach())
                    mlm_total += float(mlm_sum.detach())
                scaler.unscale_(optimizer)
                bad_gradient = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trained.values())
                if bad_gradient and scaler.is_enabled():
                    # GradScaler skips this update and lowers its scale. Replay
                    # the same examples, masks and dropout; consume no progress.
                    scaler.step(optimizer)
                    scaler.update()
                    restore_rng(attempt_rng)
                    print(json.dumps({"event": "amp_retry", "step": state["step"],
                                      "attempt": attempt + 1, "scale": scaler.get_scale()}), flush=True)
                    continue
                norm = torch.nn.utils.clip_grad_norm_(trained.values(), 1.)
                if bad_gradient or not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradients; last committed checkpoint is preserved.")
                scaler.step(optimizer)
                scaler.update()
                break
            else:
                raise FloatingPointError("AMP overflow persisted after eight retries; use BF16/FP32 or review the learning rate.")
            state["cursor"] += len(selected)
            state["step"] += 1
            state["pending_validation"] = state["cursor"] == len(training) or state["step"] % config.eval_every == 0
            print(json.dumps({"event": "training", "step": state["step"], "total_updates": total_updates,
                              "epoch": state["epoch"] + 1, "pairs_in_update": len(selected),
                              "classification_loss": class_total / len(selected), "mlm_loss": mlm_total / masked,
                              "elapsed_seconds": time.monotonic() - start_time}), flush=True)
            if state["pending_validation"] or state["step"] % config.checkpoint_every == 0:
                checkpoint()
        checkpoint()
    except KeyboardInterrupt:
        # A partial optimiser update cannot safely be committed. Keep the last
        # on-disk update, whose model/optimiser/RNG position is internally consistent.
        status = "interrupted"
        if persistent:
            store.mirror(persistent)
        print("Interrupted; use latest.json to resume the last committed update.", flush=True)
