"""Check the real training workflow and measure speed and memory use."""
import argparse
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

from .base import sha256
from .checkpoints import CheckpointStore, atomic_json
from .cli import execution_device, make_model
from .data import fingerprint, read_records, validate_training_labels
from .engine import TrainConfig, fit, predict, source_identity


def distribution(values):
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Timing samples must be finite and positive.")
    return {"count": len(values), "mean_seconds": statistics.mean(values),
            "median_seconds": statistics.median(values),
            "p95_seconds": float(np.percentile(values, 95)),
            "min_seconds": min(values), "max_seconds": max(values),
            "std_seconds": statistics.stdev(values) if len(values) > 1 else 0.}


