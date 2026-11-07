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


def read_fasta(path):
    sequences, name, parts = {}, None, []

    def store():
        if name is not None:
            if name in sequences:
                raise ValueError("Duplicate FASTA identifier: " + name)
            sequences[name] = sequence("".join(parts))

    with Path(path).open() as stream:
        for line in stream:
            if line.startswith(">"):
                store()
                name, parts = line[1:].split()[0], []
                if "|" in name:
                    name = name.split("|")[1]
            elif line.strip():
                if name is None:
                    raise ValueError("Sequence before a FASTA header.")
                parts.append(line.strip())
    store()
    if not sequences:
        raise ValueError("Empty FASTA file: " + str(path))
    return sequences


def primary_rows(cache, sequences):
    result = {}
    for block, role in [(1, "train"), (0, "validation"), (2, "bernett_test")]:
        rows = []
        for label, suffix in [(1, "pos"), (0, "neg")]:
            path = Path(cache) / "bernett-v4" / f"Intra{block}_{suffix}_rr.txt"
            with path.open() as stream:
                for number, line in enumerate(stream, 1):
                    a, b = line.split()
                    rows.append({"pair_id": f"bernett:intra{block}:{suffix}:{number}",
                                 "protein_a": a, "protein_b": b, "sequence_a": sequences[a],
                                 "sequence_b": sequences[b], "label": label,
                                 "split": "test" if role == "bernett_test" else role})
        result[role] = rows
    return result


