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


class Recorder:
    """Synchronise device timings and sample process/device memory per phase."""
    def synchronise(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def memory(self):
        result = {"process_rss_bytes": self.process.memory_info().rss}
        if self.device.type == "cuda":
            result.update(device_allocated_bytes=torch.cuda.memory_allocated(self.device),
                          device_reserved_bytes=torch.cuda.memory_reserved(self.device))
        elif self.device.type == "mps":
            result.update(device_allocated_bytes=torch.mps.current_allocated_memory(),
                          device_driver_bytes=torch.mps.driver_allocated_memory())
        return result

    def _snapshot(self):
        sample = self.memory()
        active = self.active
        if active is not None:
            active["memory_samples"] += 1
            for key, value in sample.items():
                active["sampled_peak_bytes"][key] = max(value, active["sampled_peak_bytes"].get(key, 0))
        self.trace.write(json.dumps({"seconds": time.perf_counter() - self.origin,
                                    "phase": active["phase"] if active else None, **sample}) + "\n")

