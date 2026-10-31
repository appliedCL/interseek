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
