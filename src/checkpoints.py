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
