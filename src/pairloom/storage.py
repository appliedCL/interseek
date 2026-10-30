"""Verified local/Drive staging, independent of notebook-specific APIs."""
import json
from pathlib import Path
import shutil

from .base import extract_development_data, sha256
from .checkpoints import CheckpointStore, atomic_json


def verified_copy(source, target):
    source, target = Path(source), Path(target)
    expected = sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve() or target.exists() and sha256(target) == expected:
        return target
    temporary = target.with_suffix(target.suffix + ".partial")
    shutil.copyfile(source, temporary)
    if sha256(temporary) != expected or sha256(source) != expected:
        raise OSError("Input changed or copy verification failed.")
    temporary.replace(target)
    return target


def stage_data(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_dir() or source.suffix == ".zip":
        return extract_development_data(source, destination)
    if not source.is_file() or not (source.name.endswith(".csv") or source.name.endswith(".csv.gz")):
        raise FileNotFoundError("Supply a development CSV/CSV.gz or the existing transfer ZIP as a data-only input.")
    return verified_copy(source, destination / source.name)


def restore_persistent(local, persistent):
    """Restore the greatest available verified run; never roll local progress back."""
    local, persistent = Path(local), Path(persistent)
    local.mkdir(parents=True, exist_ok=True)
    candidates = []
    for which in ["latest", "previous"]:
        if (persistent / (which + ".json")).exists():
            try:
                value = CheckpointStore(persistent).load(which, fallback=False)
                candidates.append((value["step"], which, value["protocol"]))
            except (RuntimeError, OSError):
                pass
    try:
        existing = CheckpointStore(local).load() if (local / "latest.json").exists() else None
    except RuntimeError:
        existing = None
    if not candidates:
        if (persistent / "latest.json").exists() and existing is None:
            raise RuntimeError("Persistent checkpoints exist but none verify; refusing to start over.")
        return existing is not None
    step, chosen, protocol = max(candidates, key=lambda value: value[0])
    if existing is not None:
        if existing["protocol"] != protocol:
            raise ValueError("Local and persistent directories belong to different experiments.")
        if existing["step"] >= step:
            return True
    for which in ["latest", "previous", "best"]:
        selected = chosen if which == "latest" else which
        source_pointer = persistent / (selected + ".json")
        if not source_pointer.exists():
            continue
        try:
            CheckpointStore(persistent).load(selected, fallback=False)
        except RuntimeError:
            continue
        pointer = json.loads(source_pointer.read_text())
        verified_copy(persistent / pointer["file"], local / pointer["file"])
        atomic_json(local / (which + ".json"), pointer)
    for name in ["run.json", "history.json", "validation.csv"]:
        if (persistent / name).exists():
            verified_copy(persistent / name, local / name)
    return True
