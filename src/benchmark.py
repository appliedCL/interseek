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
    def __init__(self, root, device, interval=.05):
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("Install benchmark dependencies: pip install -e '.[benchmark]'") from exc
        self.root, self.device, self.interval = Path(root), device, interval
        self.process = psutil.Process()
        self.records, self.active = [], None
        self.prefix, self.error = "", None
        self.lock, self.stopped = threading.RLock(), threading.Event()
        self.origin = time.perf_counter()
        self.events = (self.root / "events.jsonl").open("w", buffering=1)
        self.trace = (self.root / "memory.jsonl").open("w", buffering=1)
        self.worker = threading.Thread(target=self._sample, daemon=True)
        self.worker.start()

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

    def _sample(self):
        while not self.stopped.wait(self.interval):
            try:
                with self.lock:
                    self._snapshot()
            except Exception as exc:
                self.error = str(exc)
                return

    def event(self, name, **values):
        if self.error:
            raise RuntimeError("Memory sampling failed: " + self.error)
        self.events.write(json.dumps({"event": name, "seconds": time.perf_counter() - self.origin,
                                      **values}, allow_nan=False) + "\n")

    def start(self, phase, **values):
        self.synchronise()
        with self.lock:
            if self.active is not None:
                raise RuntimeError("Benchmark phases must not overlap.")
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            self.active = {"phase": phase, **values, "started": time.perf_counter(),
                           "memory_samples": 0, "sampled_peak_bytes": {}, "baseline_bytes": self.memory()}
            self._snapshot()
        self.event(phase + "_start", **values)
        atomic_json(self.root / "progress.json", {"status": "running", "phase": phase, **values})

    def end(self, **values):
        self.synchronise()
        ended = time.perf_counter()
        with self.lock:
            self._snapshot()
            record = self.active
            if record is None:
                raise RuntimeError("No active benchmark phase.")
            record.update(values)
            record["elapsed_seconds"] = ended - record.pop("started")
            if self.device.type == "cuda":
                record["native_device_peak_bytes"] = {
                    "allocated": torch.cuda.max_memory_allocated(self.device),
                    "reserved": torch.cuda.max_memory_reserved(self.device)}
            self.active = None
            self.records.append(record)
        self.event(record["phase"] + "_end", **record)
        atomic_json(self.root / "measurements.json", self.records)
        return record

    @contextmanager
    def phase(self, name, **values):
        self.start(name, **values)
        try:
            yield
        except BaseException:
            self.end(complete=False)
            raise
        else:
            self.end(complete=True)

    def observe(self, event, **values):
        if event == "validation_progress":
            self.event(event, **values)
            atomic_json(self.root / "progress.json", {"status": "running", "phase": "validation", **values})
        elif event.endswith("_start"):
            phase = event[:-6]
            if phase == "training_update" and self.prefix:
                phase = self.prefix + phase
            self.start(phase, **values)
        elif event.endswith("_end"):
            self.end(**values)

    def close(self):
        self.stopped.set()
        self.worker.join(timeout=5)
        if self.active is not None:
            self.end(complete=False)
        self.events.close()
        self.trace.close()
        if self.error:
            raise RuntimeError("Memory sampling failed: " + self.error)


class Tee(io.TextIOBase):
    def __init__(self, console, stream):
        self.console, self.stream = console, stream

    def write(self, text):
        self.console.write(text)
        self.stream.write(text)
        return len(text)

    def flush(self):
        self.console.flush()
        self.stream.flush()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", required=True)
    result.add_argument("--pretrained-model", help="Official Meta repository or local pretrained ESM-2 directory.")
    result.add_argument("--output", required=True, help="New benchmark directory; existing paths are refused.")
    result.add_argument("--device", default="auto")
    result.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    result.add_argument("--updates", type=int, default=35, help="Initial training updates, including warm-up.")
    result.add_argument("--warmup", type=int, default=5, help="Initial updates excluded from timing statistics.")
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--accumulation", type=int, default=16)
    result.add_argument("--max-tokens", type=int, default=1024)
    result.add_argument("--max-views", type=int, default=2)
    result.add_argument("--validation-pairs", type=int,
                        help="Explicit sample for a smoke test; omitted runs the entire validation split.")
    result.add_argument("--threads", type=int)
    result.add_argument("--hours", type=float, default=3., help="Overall runtime limit; incomplete runs fail.")
    return result


def check_probabilities(scores, expected):
    if scores is None or len(scores) != expected or not np.isfinite(scores).all():
        raise ValueError("Validation did not produce a finite score for every requested pair.")
    if not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Validation probabilities are outside [0, 1].")


def run(args):
    if args.warmup < 1 or args.updates - args.warmup < 2:
        raise ValueError("Require at least one warm-up and two measured updates.")
    if not math.isfinite(args.hours) or args.hours <= 0:
        raise ValueError("Runtime limit must be finite and positive.")
    if args.validation_pairs is not None and args.validation_pairs < 2:
        raise ValueError("A validation sample must contain at least two pairs.")
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("Thread count must be positive.")
        torch.set_num_threads(args.threads)
    device, precision = execution_device(args.device, args.precision)
    config = TrainConfig(batch_size=args.batch_size, accumulation=args.accumulation,
                         max_tokens=args.max_tokens, max_views=args.max_views,
                         eval_every=args.updates + 1)
    config.validate()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = time.monotonic() + args.hours * 3600
    recorder = Recorder(root, device)
    status = {"schema": "interseek-benchmark-v1", "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "Training/validation correctness, throughput and memory; no cost or full-training projections.",
              "device": str(device), "precision": precision,
              "environment": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                              "processor": platform.processor(), "cpu_threads": torch.get_num_threads(),
                              "host_memory_bytes": importlib.import_module("psutil").virtual_memory().total,
                              "packages": {name: importlib.metadata.version(name)
                                           for name in ["torch", "transformers", "numpy", "scikit-learn", "psutil"]}},
              "source_sha256": source_identity(), "training_config": asdict(config),
              "timing": {"warmup_updates": args.warmup, "requested_updates": args.updates,
                         "accelerator_synchronised": True,
                         "checkpoint_time_excluded_from_update_statistics": True},
              "memory_method": {"sampling_interval_seconds": recorder.interval,
                                "process_and_mps_peaks": "Sampled high-water marks; may miss shorter allocations.",
                                "cuda_peaks": "Native allocator peaks per phase, when using CUDA.",
                                "unified_memory": "Process RSS and Apple GPU memory overlap; do not add them."}}
    if device.type == "cuda":
        status["environment"].update(accelerator=torch.cuda.get_device_name(device),
                                      cuda_runtime=torch.version.cuda,
                                      gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
    atomic_json(root / "benchmark.json", status)
    try:
        with (root / "training.log").open("w", buffering=1) as stream, redirect_stdout(Tee(sys.stdout, stream)):
            with recorder.phase("setup"):
                data = Path(args.data).resolve()
                training, full_validation = read_records(data, "train"), read_records(data, "validation")
                validation = full_validation
                if args.validation_pairs is not None:
                    validation = random.Random(config.seed).sample(full_validation,
                                                                  min(args.validation_pairs, len(full_validation)))
                validate_training_labels(training, validation)
                effective = config.batch_size * config.accumulation
                if args.updates + 1 >= math.ceil(len(training) / effective):
                    raise ValueError("Benchmark updates must finish before the first epoch ends; use more training pairs or fewer updates.")
                status["data"] = {"path": str(data), "file_sha256": sha256(data),
                                  "training_pairs": len(training), "validation_pairs": len(validation),
                                  "full_validation_pairs": len(full_validation),
                                  "full_validation": len(validation) == len(full_validation),
                                  "training_fingerprint": fingerprint(training), "validation_fingerprint": fingerprint(validation)}
                model, batcher, identity = make_model(SimpleNamespace(pretrained_model=args.pretrained_model), config)
                if not all(parameter.requires_grad for parameter in model.backbone.parameters()):
                    raise ValueError("The pretrained encoder is not fully trainable.")
                queries = {f"backbone.esm.encoder.layer.{i}.attention.self.query.weight":
                           layer.attention.self.query.weight.detach().cpu().clone()
                           for i, layer in enumerate(model.backbone.esm.encoder.layer)}
                status["initialization"] = identity
                status["model_parameters"] = sum(parameter.numel() for parameter in model.parameters())
                model.to(device)
            atomic_json(root / "benchmark.json", status)

            def hours_left():
                remaining = (deadline - time.monotonic()) / 3600
                if remaining <= 0:
                    raise TimeoutError("Benchmark runtime limit reached; results are incomplete.")
                return remaining

            outcome = fit(model, batcher, training, validation, identity, config, root / "run", device,
                          precision, max_updates=args.updates, hours=hours_left(), observer=recorder.observe)
            if outcome["step"] != args.updates or outcome["status"] != "paused":
                raise ValueError("Initial training did not reach every requested update.")
            measured = [row for row in recorder.records
                        if row["phase"] == "training_update" and row["step"] > args.warmup]
            status["training"] = {**distribution([row["elapsed_seconds"] for row in measured]),
                                  "pairs_per_second": sum(row["pairs"] for row in measured) / sum(row["elapsed_seconds"] for row in measured),
                                  "measured_pairs": sum(row["pairs"] for row in measured)}
            atomic_json(root / "benchmark.json", status)
            with recorder.phase("checkpoint_reload"):
                saved = CheckpointStore(root / "run").load(fallback=False)
                changed = [name for name, before in queries.items() if not torch.equal(saved["model"][name], before)]
                if len(changed) != len(queries):
                    raise ValueError("Some encoder layers did not update.")
                if saved["state"]["cursor"] != args.updates * effective or not saved["optimizer"]["state"]:
                    raise ValueError("Checkpoint progress or optimiser state is missing.")
                probe = random.Random(config.seed).sample(validation, min(8, len(validation)))
                before = predict(model, batcher, probe, device, config.batch_size, precision)
                restored, _, _ = make_model(SimpleNamespace(), config, saved["protocol"]["initialization"])
                restored.restore(saved["model"])
                restored.to(device)
                after = predict(restored, batcher, probe, device, config.batch_size, precision)
                check_probabilities(before, len(probe))
                check_probabilities(after, len(probe))
                np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
                swapped = [replace(row, sequence_a=row.sequence_b, sequence_b=row.sequence_a) for row in probe]
                reverse = predict(restored, batcher, swapped, device, config.batch_size, precision)
                np.testing.assert_allclose(after, reverse, rtol=1e-5, atol=1e-6)
                protocol = saved["protocol"]
                optimiser_steps = {key: float(value["step"]) for key, value in saved["optimizer"]["state"].items()}
                status["checks"] = {"pretrained_weights_loaded": identity["kind"] == "pretrained-esm2",
                                    "all_encoder_layers_updated": len(changed),
                                    "checkpoint_checksum_verified": True,
                                    "reload_predictions_match": True, "swapped_pair_predictions_match": True,
                                    "prediction_comparison_rtol": 1e-5, "prediction_comparison_atol": 1e-6}
                del saved, model, queries
                gc.collect()
            recorder.prefix = "resumed_"
            atomic_json(root / "benchmark.json", status)
            outcome = fit(restored, batcher, training, validation, identity, config, root / "run", device,
                          precision, resume=True, max_updates=args.updates + 1,
                          hours=hours_left(), observer=recorder.observe)
            if outcome["step"] != args.updates + 1 or outcome["status"] != "paused":
                raise ValueError("Resumed training or validation did not finish.")
            with recorder.phase("verification"):
                saved = CheckpointStore(root / "run").load(fallback=False)
                best = CheckpointStore(root / "run").load("best", fallback=False)
                if saved["protocol"] != protocol or saved["state"]["cursor"] != (args.updates + 1) * effective:
                    raise ValueError("Resumed training changed its protocol or consumed the wrong rows.")
                if {key: float(value["step"]) for key, value in saved["optimizer"]["state"].items()} != {
                        key: value + 1 for key, value in optimiser_steps.items()}:
                    raise ValueError("Resumed optimiser state did not advance exactly once.")
                if saved["state"]["last_validation_step"] != args.updates + 1 or best["step"] != saved["step"]:
                    raise ValueError("Full validation did not select and save the expected checkpoint.")
                import csv
                with (root / "run/validation.csv").open(newline="") as predictions:
                    rows = list(csv.DictReader(predictions))
                if [row["pair_id"] for row in rows] != [row.pair_id for row in validation]:
                    raise ValueError("Validation output omitted, reordered or duplicated input rows.")
                check_probabilities(np.asarray([float(row["score"]) for row in rows]), len(validation))
                status["checks"].update(resume_advanced_one_update=True, resume_protocol_unchanged=True,
                                         resume_optimiser_verified=True,
                                         validation_predictions_verified=len(rows), best_checkpoint_verified=True)
                status["validation_metrics"] = saved["state"]["history"][-1]["validation"]
            updates = [row for row in recorder.records if row["phase"] == "training_update" and row["step"] > args.warmup]
            if len(updates) != args.updates - args.warmup:
                raise ValueError("Measured update count differs from the benchmark plan.")
            validations = [row for row in recorder.records if row["phase"] == "validation"]
            if len(validations) != 1 or not validations[0].get("complete"):
                raise ValueError("Expected one complete validation pass.")
            status["training"] = {**distribution([row["elapsed_seconds"] for row in updates]),
                                  "pairs_per_second": sum(row["pairs"] for row in updates) / sum(row["elapsed_seconds"] for row in updates),
                                  "measured_pairs": sum(row["pairs"] for row in updates)}
            validation_seconds = validations[0]["elapsed_seconds"]
            status["validation"] = {"pairs": len(validation), "seconds": validation_seconds,
                                    "pairs_per_second": len(validation) / validation_seconds,
                                    "includes_both_input_orders": True, "excludes_metrics_and_checkpoints": True}
            status["outcome"] = outcome
            status["status"] = "complete"
    except BaseException as exc:
        status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        try:
            recorder.close()
        except BaseException as exc:
            status.update(status="failed", memory_error=str(exc))
            raise
        finally:
            status["wall_seconds"] = time.perf_counter() - started
            status["phases"] = recorder.records
            atomic_json(root / "benchmark.json", status)
            atomic_json(root / "progress.json", {"status": status["status"], "wall_seconds": status["wall_seconds"]})
    return status


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except (ValueError, OSError, RuntimeError, AssertionError, TimeoutError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve()),
                      "training": result["training"], "validation": result["validation"]}, indent=2))


if __name__ == "__main__":
    main()
