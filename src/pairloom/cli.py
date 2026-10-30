"""Pairloom's train, evaluate and predict command line."""
import argparse
from dataclasses import asdict, fields
import fcntl
import json
from pathlib import Path

from . import __version__


def parser():
    result = argparse.ArgumentParser(prog="pairloom", description="Train protein-pair representations with ESM-2.")
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ["train", "evaluate", "predict"]:
        command = commands.add_parser(name)
        command.add_argument("--data", required=True, help="CSV or CSV.gz; RIMNet or query/text columns.")
        command.add_argument("--task", choices=["ppi", "mutation"], default="ppi")
        command.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
        command.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
        if name == "train":
            command.add_argument("--initialization", choices=["pretrained", "random"], help="Default: pretrained Meta ESM-2; random is a scratch-training control.")
            command.add_argument("--architecture", help="esm2-35m (default), esm2-650m; random mode also accepts a local architecture JSON.")
            command.add_argument("--pretrained-model", help="Optional local ESM-2 directory or official Meta repository override.")
            command.add_argument("--revision", help="Optional Hugging Face revision; resolved provenance is recorded.")
            command.add_argument("--output", required=True)
            command.add_argument("--validation-data", help="Separate validation CSV; otherwise use data's split column.")
            command.add_argument("--resume", action="store_true")
            command.add_argument("--persistent", help="Mirror verified latest/previous/best checkpoints, e.g. Drive.")
            command.add_argument("--max-updates", type=int, help="Pause at this absolute optimizer update.")
            command.add_argument("--hours", type=float, help="Wall-clock budget; does not alter the scientific protocol.")
            from .engine import TrainConfig
            defaults = TrainConfig()
            for field in fields(defaults):
                value = getattr(defaults, field.name)
                option = "--" + field.name.replace("_", "-")
                if isinstance(value, bool):
                    command.add_argument(option, action=argparse.BooleanOptionalAction, default=None)
                else:
                    command.add_argument(option, type=type(value) if value is not None else float, default=None,
                                         help=f"Default: {value}")
        else:
            command.add_argument("--run", required=True, help="Pairloom checkpoint directory.")
            command.add_argument("--checkpoint", default="best", choices=["best", "latest", "previous"])
            command.add_argument("--split", help="Select a named split; no filtering if omitted.")
            command.add_argument("--predictions", required=True)
    return result


def execution_device(name, precision):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable.")
    if precision == "auto":
        precision = "fp32"
        if name == "cuda":
            precision = "bf16" if torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported() else "fp16"
    return device, precision


def make_model(args, config, expected=None):
    from .base import initialize_architecture, initialize_pretrained
    from .engine import seed_everything
    from .model import PairModel
    from .data import PairBatcher
    selected = getattr(args, "architecture", None)
    mode = getattr(args, "initialization", None)
    source, revision = getattr(args, "pretrained_model", None), getattr(args, "revision", None)
    if expected is not None:
        if expected.get("kind") not in {"random-esm2", "pretrained-esm2"}:
            raise ValueError("Unsupported Pairloom initialization provenance.")
        if any(value is not None for value in [selected, mode, source, revision]):
            raise ValueError("When resuming, omit initialization/model flags; the full checkpoint defines them.")
        # Full checkpoints restore every model tensor. No base-model download is
        # needed for resume/evaluation, even if the original cache was removed.
        backbone, vocab, _ = initialize_architecture(seed=config.seed, config=expected["config"])
        identity = expected
    elif (mode or "pretrained") == "pretrained":
        backbone, vocab, identity = initialize_pretrained(selected or "esm2-35m", source, revision)
    else:
        if source is not None or revision is not None:
            raise ValueError("Random initialization does not accept a pretrained model or revision.")
        backbone, vocab, identity = initialize_architecture(selected or "esm2-35m", config.seed)
    seed_everything(config.seed)
    model = PairModel(backbone, config.head, config.head_width, config.dropout, config.gradient_checkpointing)
    batcher = PairBatcher(vocab, config.max_tokens, config.max_views, config.seed)
    return model, batcher, identity


def run(args):
    from .checkpoints import CheckpointStore
    from .data import read_records
    from .engine import TrainConfig, fit, predict, write_predictions, source_identity
    from .metrics import report
    device, precision = execution_device(args.device, args.precision)
    if args.command == "train":
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        # The lock spans model loading, training and checkpoint publication.
        with (output / "run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Another process is using this run directory.")
            saved = CheckpointStore(output).load() if args.resume else None
            options = saved["protocol"]["config"].copy() if saved else asdict(TrainConfig())
            for key in options:
                if getattr(args, key) is not None:
                    options[key] = getattr(args, key)
            config = TrainConfig(**options)
            config.validate()
            if args.max_updates is not None and args.max_updates < 1 or args.hours is not None and args.hours <= 0:
                raise ValueError("Training budgets must be positive.")
            training = read_records(args.data, None if args.validation_data else "train", args.task)
            validation = read_records(args.validation_data or args.data, None if args.validation_data else "validation", args.task)
            from .data import validate_training_labels
            validate_training_labels(training, validation)
            expected = saved["protocol"]["initialization"] if saved else None
            del saved
            model, batcher, identity = make_model(args, config, expected)
            model.to(device)
            return fit(model, batcher, training, validation, identity, config, output, device,
                       precision, args.resume, args.max_updates, args.hours, args.persistent)
    saved = CheckpointStore(args.run).load(args.checkpoint, fallback=False)
    if saved["protocol"]["task"] != args.task:
        raise ValueError("The requested task differs from this checkpoint's training task.")
    if saved["protocol"]["source"] != source_identity():
        raise ValueError("This checkpoint uses a different Pairloom implementation.")
    config = TrainConfig(**saved["protocol"]["config"])
    rows = read_records(args.data, args.split, args.task, labelled=args.command == "evaluate")
    model, batcher, _ = make_model(args, config, saved["protocol"]["initialization"])
    model.restore(saved["model"])
    model.to(device)
    scores = predict(model, batcher, rows, device, config.batch_size, precision, progress=True)
    write_predictions(args.predictions, rows, scores)
    result = {"checkpoint_step": saved["step"], "predictions": args.predictions, "records": len(rows)}
    if args.command == "evaluate":
        threshold = saved["state"]["threshold"]
        if threshold is None:
            raise ValueError("This checkpoint has no validation-selected threshold; evaluate a validated checkpoint.")
        result["metrics"] = report([r.label for r in rows], scores, threshold)
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        print(json.dumps(run(arguments), indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
