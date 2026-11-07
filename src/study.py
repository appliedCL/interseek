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

SCHEMA = "interseek-study-v1"
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


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


def prepare(cache, output, spec=None):
    spec = specification() if spec is None else spec
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new study directory: " + str(output))
    fetch_sources(cache, spec)
    human = read_fasta(Path(cache) / "bernett-v4/human_swissprot_oneliner.fasta")
    yeast = read_fasta(Path(cache) / "dscript/yeast.fasta")
    rows = primary_rows(cache, human)
    rows["yeast_test"] = secondary_rows(Path(cache) / "dscript/yeast_test.tsv", yeast)
    checked = audit(rows)
    for role, expected in spec["expected"].items():
        if checked["splits"][role]["labels"] != expected:
            raise ValueError("Source row counts differ from the pinned release: " + role)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".interseq-study-", dir=output.parent))
    try:
        paths = {"development": "development.csv.gz", "bernett_test": "tests/bernett.csv.gz",
                 "yeast_test": "tests/yeast.csv.gz"}
        write_rows(temporary / paths["development"], itertools.chain(rows["train"], rows["validation"]))
        for role in ["bernett_test", "yeast_test"]:
            write_rows(temporary / paths[role], rows[role])
        artifacts = {role: {"path": path, "sha256": sha256(temporary / path),
                            "bytes": (temporary / path).stat().st_size} for role, path in paths.items()}
        manifest = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "sources": spec, "artifacts": artifacts, "counts": checked["splits"],
                    "selection": {"checkpoint": "best validation AUROC", "threshold": "validation MCC",
                                  "final_test_names": ["bernett_test", "yeast_test"],
                                  "final_tests_used_for_selection": False,
                                  "metrics": ["average_precision", "auroc", "mcc", "f1", "precision", "recall"],
                                  "additional_threshold": 0.5},
                    "training_fingerprint": fingerprint(read_records(temporary / paths["development"], "train")),
                    "validation_fingerprint": fingerprint(read_records(temporary / paths["development"], "validation"))}
        atomic_json(temporary / "audit.json", checked)
        manifest["audit"] = {"path": "audit.json", "sha256": sha256(temporary / "audit.json")}
        atomic_json(temporary / "study.json", manifest)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def verify_study(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "study.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported study manifest.")
    for artifact in [*manifest["artifacts"].values(), manifest["audit"]]:
        verified_path(root, artifact)
    return manifest


def final_test(study, run, output, device="auto", precision="auto"):
    with (Path(run) / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("The training run is still in use.") from exc
        return _final_test(study, run, output, device, precision)


def _final_test(study, run, output, device, precision):
    study, run, output = Path(study).resolve(), Path(run).resolve(), Path(output).resolve()
    manifest = verify_study(study)
    development = verified_path(study, manifest["artifacts"]["development"])
    training, validation = read_records(development, "train"), read_records(development, "validation")
    latest = CheckpointStore(run).load(fallback=False)
    protocol = latest["protocol"]
    if latest["state"]["epoch"] < protocol["config"]["epochs"] or latest["state"]["pending_validation"]:
        raise ValueError("Finish the declared training run and validation before opening either final test.")
    if (protocol["training"] != fingerprint(training) or protocol["validation"] != fingerprint(validation)
            or protocol["source"] != source_identity()):
        raise ValueError("Training data, validation data or source differs from this study.")
    saved = CheckpointStore(run).load("best", fallback=False)
    if saved["protocol"] != protocol or saved["state"]["threshold"] is None:
        raise ValueError("The selected checkpoint lacks matching validation provenance.")
    config = TrainConfig(**protocol["config"])
    selected_pointer = json.loads((run / "best.json").read_text())
    threshold = saved["state"]["threshold"]
    selected_device, selected_precision = execution_device(device, precision)
    output.mkdir(parents=True, exist_ok=False)
    frozen = {"schema": "interseek-final-evaluation-v1", "status": "running",
              "study_sha256": sha256(study / "study.json"), "checkpoint": selected_pointer,
              "checkpoint_step": saved["step"], "completed_training_step": latest["step"],
              "validation_threshold": threshold, "protocol": protocol,
              "device": str(selected_device), "precision": selected_precision,
              "test_threshold_selection": False, "results": {}}
    atomic_json(output / "evaluation.json", frozen)
    try:
        del latest
        model, batcher, _ = make_model(SimpleNamespace(), config, saved["protocol"]["initialization"])
        model.restore(saved["model"])
        model.to(selected_device)
        del saved
        for name in manifest["selection"]["final_test_names"]:
            path = verified_path(study, manifest["artifacts"][name])
            records = read_records(path)
            if len(records) != manifest["counts"][name]["pairs"]:
                raise ValueError("Final-test row count differs from the study: " + name)
            scores = predict(model, batcher, records, selected_device, config.batch_size, selected_precision, progress=True)
            selected_metrics = report([row.label for row in records], scores, threshold)
            fixed_metrics = report([row.label for row in records], scores, .5)
            predictions = output / (name + "-predictions.csv")
            write_predictions(predictions, records, scores)
            frozen["results"][name] = {"data_sha256": manifest["artifacts"][name]["sha256"],
                                       "predictions_sha256": sha256(predictions),
                                       "validation_threshold_metrics": selected_metrics,
                                       "fixed_0_5_metrics": fixed_metrics}
            atomic_json(output / "evaluation.json", frozen)
        frozen["status"] = "complete"
    except BaseException as exc:
        frozen.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        atomic_json(output / "evaluation.json", frozen)
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Download and preserve the published split assignments.")
    preparation.add_argument("--cache", default="data/sources")
    preparation.add_argument("--output", default="data/study")
    evaluation = commands.add_parser("final-test", help="Evaluate one completed, validation-selected model on both tests.")
    evaluation.add_argument("--study", default="data/study")
    evaluation.add_argument("--run", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.cache, args.output)
            print(json.dumps({"study": str(Path(args.output).resolve()), "counts": result["counts"]}, indent=2))
        else:
            result = final_test(args.study, args.run, args.output, args.device, args.precision)
            print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve())}, indent=2))
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
