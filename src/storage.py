"""Verified input staging and checkpoint recovery from persistent storage."""
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
        raise FileNotFoundError("Supply a development CSV/CSV.gz or a manifest-based data ZIP.")
    return verified_copy(source, destination / source.name)
