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


def secondary_rows(path, sequences):
    rows = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            a, b, raw_label = line.strip().replace("\\t", "\t").split()
            if not a.startswith("4932.") or not b.startswith("4932."):
                raise ValueError("The independent test must contain yeast proteins only.")
            if raw_label not in {"0", "1", "0.0", "1.0"}:
                raise ValueError("Invalid independent-test label: " + raw_label)
            rows.append({"pair_id": f"dscript:yeast:{number}", "protein_a": a, "protein_b": b,
                         "sequence_a": sequences[a], "sequence_b": sequences[b],
                         "label": int(float(raw_label)), "split": "external_test"})
    return rows


def audit(rows_by_role):
    result, proteins, sequences, pairs = {}, {}, {}, {}
    for role, rows in rows_by_role.items():
        if not rows or {row["label"] for row in rows} != {0, 1}:
            raise ValueError("Each study split must contain both classes: " + role)
        if len({row["pair_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate row identifiers: " + role)
        ids, seqs, pair_labels = set(), set(), defaultdict(set)
        degrees = {0: Counter(), 1: Counter()}
        sequence_pairs = set()
        for row in rows:
            a, b, label = row["protein_a"], row["protein_b"], row["label"]
            ids.update((a, b))
            hashes = [hashlib.sha256(row[key].encode()).hexdigest() for key in ["sequence_a", "sequence_b"]]
            seqs.update(hashes)
            sequence_pairs.add(tuple(sorted(hashes)))
            pair_labels[tuple(sorted((a, b)))].add(label)
            degrees[label].update((a, b))
        counts = Counter(str(row["label"]) for row in rows)
        result[role] = {"pairs": len(rows), "labels": dict(counts), "proteins": len(ids),
                        "exact_sequences": len(seqs), "positive_prevalence": counts["1"] / len(rows),
                        "repeated_unordered_pair_rows": len(rows) - len(pair_labels),
                        "conflicting_unordered_pairs": sum(len(value) > 1 for value in pair_labels.values()),
                        "self_pair_rows": sum(row["protein_a"] == row["protein_b"] for row in rows),
                        "max_positive_degree": max(degrees[1].values()),
                        "max_negative_degree": max(degrees[0].values())}
        proteins[role], sequences[role], pairs[role] = ids, seqs, sequence_pairs
    overlaps = {}
    for a, b in itertools.combinations(rows_by_role, 2):
        values = {"protein_ids": len(proteins[a] & proteins[b]),
                  "exact_sequences": len(sequences[a] & sequences[b]),
                  "unordered_sequence_pairs": len(pairs[a] & pairs[b])}
        overlaps[a + ":" + b] = values
        if any(values.values()):
            raise ValueError(f"Study splits overlap: {a}, {b}: {values}")
    return {"splits": result, "overlaps": overlaps,
            "rows_removed": 0,
            "homology": {"primary": "Authors' KaHIP partition and CD-HIT redundancy reduction; not recomputed.",
                         "independent": "No cross-species homology filtering; orthologues may remain."},
            "pretraining_exposure": "Not audited; supervised split separation does not exclude ESM-2 pretraining exposure."}


