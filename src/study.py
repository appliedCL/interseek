"""Prepare the published study splits and evaluate a fixed final model."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import fcntl
from importlib.resources import files
import io
import itertools
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from urllib.request import Request, urlopen

from .base import sha256, verified_path
from .checkpoints import CheckpointStore, atomic_json
from .cli import execution_device, make_model
from .data import fingerprint, read_records, sequence
from .engine import TrainConfig, predict, source_identity, write_predictions
from .metrics import report

SCHEMA = "interseq-study-v1"
COLUMNS = ["pair_id", "protein_a", "protein_b", "sequence_a", "sequence_b", "label", "split"]


def specification():
    return json.loads(files("interseq").joinpath("study_sources.json").read_text())


def fetch_sources(cache, spec):
    cache = Path(cache)
    for item in spec["downloads"]:
        path = cache / item["path"]
        if path.is_file():
            if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise ValueError("Cached source checksum mismatch: " + str(path))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        try:
            request = Request(item["url"], headers={"User-Agent": "InterSeq study preparation"})
            with urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
                remaining = item["bytes"]
                while block := response.read(min(1024 ** 2, remaining + 1)):
                    remaining -= len(block)
                    if remaining < 0:
                        raise ValueError("Source exceeds its pinned size: " + item["path"])
                    stream.write(block)
            if temporary.stat().st_size != item["bytes"] or sha256(temporary) != item["sha256"]:
                raise ValueError("Downloaded source checksum mismatch: " + item["path"])
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


