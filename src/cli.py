"""InterSeq's train, evaluate and predict command line."""
import argparse
from dataclasses import asdict, fields
import fcntl
import json
from pathlib import Path

from . import __version__


def parser():
    result = argparse.ArgumentParser(prog="interseq", description="Fine-tune pretrained ESM-2 for protein interactions.")
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ["train", "evaluate", "predict"]:
        examples = None
        if name == "train":
            examples = ("Examples:\n"
                        "  interseq train --data pairs.csv.gz --output runs/esm2 --device cuda\n"
                        "  interseq train --data pairs.csv.gz --output runs/esm2 --device cuda:1\n"
                        "  interseq train --data pairs.csv.gz --output runs/esm2 --device cuda --resume\n\n"
                        "Pretrained Meta ESM-2 35M weights are loaded by default. All encoder layers are trainable.\n"
                        "CSV columns: sequence_a, sequence_b, label, split (train or validation).\n"
                        "Use --validation-data for a separate validation CSV.\n"
                        "Each process trains on one GPU; use --device to select it.")
        command = commands.add_parser(name, epilog=examples, formatter_class=argparse.RawDescriptionHelpFormatter)
        command.add_argument("--data", required=True, help="CSV, CSV.gz or manifest-based data ZIP; sequence_a/sequence_b or query/text columns.")
        command.add_argument("--task", choices=["ppi", "mutation"], default="ppi")
        command.add_argument("--device", default="auto", help="auto (default), cuda, cuda:N, mps or cpu.")
        command.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"],
                             help="auto (default) selects BF16 or FP16 on CUDA and FP32 otherwise.")
        if name == "train":
            command.add_argument("--initialization", choices=["pretrained", "random"], help="Default: pretrained Meta ESM-2; random is a scratch-training baseline.")
            command.add_argument("--architecture", help="esm2-35m (default), esm2-650m; random mode also accepts a local architecture JSON.")
            command.add_argument("--pretrained-model", help="Optional local ESM-2 directory or official Meta repository override.")
            command.add_argument("--revision", help="Optional Hugging Face revision; resolved provenance is recorded.")
            command.add_argument("--output", required=True)
            command.add_argument("--validation-data", help="Separate validation CSV; otherwise use data's split column.")
            command.add_argument("--resume", action="store_true")
            command.add_argument("--persistent", help="Mirror checkpoints to another directory; restore with --resume.")
            command.add_argument("--max-updates", type=int, help="Pause at this absolute optimiser update.")
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
            command.add_argument("--run", required=True, help="InterSeq checkpoint directory.")
            command.add_argument("--checkpoint", default="best", choices=["best", "latest", "previous"])
            command.add_argument("--split", help="Select a named split; no filtering if omitted.")
            command.add_argument("--predictions", required=True)
    return result
