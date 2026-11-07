"""Versioned checkpoint files with atomic latest/best pointers and verified backups."""
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from .base import sha256

FORMAT = "interseek-full-model-v1"


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

    def load(self, which="latest", fallback=True):
        failures = []
        for name in ([which, "previous"] if which == "latest" and fallback else [which]):
            try:
                pointer = json.loads((self.root / (name + ".json")).read_text())
                path = (self.root / pointer["file"]).resolve()
                if not path.is_relative_to(self.root.resolve()) or sha256(path) != pointer["sha256"]:
                    raise ValueError("Checkpoint checksum or path mismatch.")
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if saved.get("format") != FORMAT or saved["step"] != pointer["step"]:
                    raise ValueError("Checkpoint identity mismatch.")
                if name != which:
                    print("Recovered previous verified checkpoint at step", saved["step"], flush=True)
                return saved
            except (OSError, ValueError, KeyError, RuntimeError, EOFError) as exc:
                failures.append(f"{name}: {exc}")
        raise RuntimeError("No valid checkpoint. " + "; ".join(failures))

    def mirror(self, destination):
        """Copy selected checkpoints to persistent storage before publishing pointers."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        if destination.resolve() == self.root.resolve():
            return
        source_run, target_run = self.root / "run.json", destination / "run.json"
        if source_run.is_file() and target_run.is_file():
            if json.loads(source_run.read_text())["protocol"] != json.loads(target_run.read_text())["protocol"]:
                raise ValueError("Persistent destination belongs to a different experiment.")
        for name in ["latest", "previous", "best"]:
            pointer_file = self.root / (name + ".json")
            if not pointer_file.exists():
                continue
            pointer = json.loads(pointer_file.read_text())
            source, target = self.root / pointer["file"], destination / pointer["file"]
            if sha256(source) != pointer["sha256"]:
                raise ValueError("Refusing to mirror an invalid checkpoint.")
            if not target.exists() or sha256(target) != pointer["sha256"]:
                temporary = target.with_suffix(".partial")
                shutil.copyfile(source, temporary)
                if sha256(temporary) != pointer["sha256"]:
                    raise OSError("Persistent checkpoint copy failed verification.")
                temporary.replace(target)
            atomic_json(destination / pointer_file.name, pointer)
        for name in ["run.json", "history.json", "validation.csv"]:
            source = self.root / name
            if source.exists():
                temporary = destination / (name + ".partial")
                shutil.copyfile(source, temporary)
                temporary.replace(destination / name)
        keep = {json.loads(p.read_text())["file"] for p in destination.glob("*.json")
                if p.name in {"latest.json", "previous.json", "best.json"}}
        for old in destination.glob("step-*.pt"):
            if old.name not in keep:
                old.unlink()
