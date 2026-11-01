"""Versioned checkpoint files with atomic latest/best pointers and verified backups."""
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from .base import sha256

FORMAT = "interseq-full-model-v1"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


class CheckpointStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, value, best=False):
        import uuid
        filename = f"step-{value['step']:08d}-{uuid.uuid4().hex[:8]}.pt"
        path = self.root / filename
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            torch.save({**value, "format": FORMAT}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        pointer = {"file": filename, "sha256": sha256(path), "step": value["step"]}
        latest = self.root / "latest.json"
        if latest.is_file():
            old = json.loads(latest.read_text())
            old_path = self.root / old["file"]
            if old_path.exists() and sha256(old_path) == old["sha256"]:
                atomic_json(self.root / "previous.json", old)
        atomic_json(latest, pointer)
        if best:
            atomic_json(self.root / "best.json", pointer)
        # Only this store's unreferenced, generated step files may be pruned.
        keep = {json.loads(p.read_text())["file"] for p in self.root.glob("*.json")
                if p.name in {"latest.json", "previous.json", "best.json"}}
        for old in self.root.glob("step-*.pt"):
            if old.name not in keep:
                old.unlink()
        return path
