"""Sequence-pair records, supplied split selection and deterministic window bags."""
from dataclasses import dataclass
import csv
import gzip
import hashlib
import random

import torch

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZUO")


@dataclass(frozen=True)
class Record:
    pair_id: str
    sequence_a: str
    sequence_b: str
    label: int | None = None
    mutant: str | None = None


def sequence(value):
    value = "".join(value.upper().split())
    if not value or not set(value) <= AMINO_ACIDS:
        raise ValueError("Expected a nonempty single-chain amino-acid sequence.")
    return value


def read_records(path, split=None, task="ppi", labelled=True):
    """Read sequence-pair columns or query/text aliases, preserving split membership."""
    opener = gzip.open if str(path).endswith(".gz") else open
    records, identifiers = [], set()
    with opener(path, "rt", newline="") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), start=1):
            if split is not None and row.get("split") != split:
                continue
            if task == "mutation":
                a, b = sequence(row["wild_seq"]), sequence(row["participant_sequence"])
                mutant = sequence(row["mutant_seq"])
            else:
                a = sequence(row.get("sequence_a", row.get("query", "")))
                b = sequence(row.get("sequence_b", row.get("text", "")))
                mutant = None
            raw = row.get("label", "")
            labels = {"0": 0, "1": 1, "decreasing": 0, "increasing": 1}
            if labelled and raw not in labels:
                raise ValueError("Labels must be 0/1, or decreasing/increasing for mutations.")
            label = labels.get(raw)
            content = "|".join(sorted((a, b))) if mutant is None else "|".join((a, mutant, b))
            # Distinct source rows remain distinct even when pairs repeat or
            # appear in reversed order. Number before filtering for stable IDs.
            pair_id = row.get("pair_id") or f"{hashlib.sha256(content.encode()).hexdigest()}:{row_number}"
            if pair_id in identifiers:
                raise ValueError("Duplicate pair_id: " + pair_id)
            identifiers.add(pair_id)
            records.append(Record(pair_id, a, b, label, mutant))
    if not records:
        raise ValueError("No records matched the requested input/split.")
    return records


def fingerprint(records):
    digest = hashlib.sha256()
    for r in records:
        digest.update(repr((r.pair_id, r.sequence_a, r.sequence_b, r.label, r.mutant)).encode() + b"\n")
    return digest.hexdigest()
