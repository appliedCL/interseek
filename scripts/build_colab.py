"""Build Pairloom's self-contained ESM-2 fine-tuning and scratch-control notebook."""
import json
from pathlib import Path


def build():
    root = Path(__file__).resolve().parents[1]
    sources = {path.name: path.read_text() for path in sorted((root / "src/pairloom").glob("*.py"))}
    intro = """# Pairloom — pretrained ESM-2 fine-tuning

The default experiment starts from **Meta's original ESM-2 35M weights** (`facebook/esm2_t12_35M_UR50D`), then learns protein interactions using Pairloom's own code and data. It does not use PLM-interact's interaction-trained weights or the failed RIMNet checkpoint. All encoder layers, embeddings, language-model readout and the new interaction head are trainable.

Set `INITIALIZATION = "random"` for the scratch-training control, using a separate run name and matching the other settings. Set `ARCHITECTURE = "esm2-650m"` to select Meta's larger 650M model. Keep 35M for the first Colab experiment. Source model identifiers, resolved revisions and initial weight hashes are recorded. A failed/incomplete pretrained download stops clearly; it never falls back to random weights.

Select a GPU runtime and run the cells in order. Supply a development CSV/CSV.gz or the existing `rimnet-kaggle-upload-v1.zip` in **My Drive/RIMNet/**. Only the development data and manifest are read from that ZIP; its weights, checkpoints and training wheels are ignored. Meta's pretrained weights download separately on the first run, unless `PRETRAINED_MODEL` names a local ESM-2 model directory.

Outputs go to a new folder under **My Drive/Pairloom/**. Later sessions resume their full Pairloom checkpoint without downloading the base weights again. Evaluation and inference also use only the full checkpoint. Versions before this full-model format are not silently reused. Changing scientific settings needs a new run name.

Pairloom uses the supplied data splits as provided, without checking protein overlap, repeated/reversed pairs or family separation. Prepare the desired splits before training; basic input and class-label checks still apply.

Full validation selects checkpoints every 1,000 updates and at epoch ends. Latest/previous/best include the complete model, optimizer and RNG state and are checksum-verified before Drive pointers are updated. Window bags have bounded coverage. The three-epoch default is an initial budget, not proof of adequate training. Software checks do not establish predictive quality or an improvement over RIMNet/PLM-interact.
"""
    settings = '''from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")

INPUT_DATA = Path("/content/drive/MyDrive/RIMNet/rimnet-kaggle-upload-v1.zip")
RUN_NAME = "pretrained-esm2-35m-cls-seed17"
INITIALIZATION = "pretrained"   # "random" is the scratch-training control.
PRETRAINED_MODEL = None        # Optional local Meta ESM-2 directory.
REVISION = None                # Optional pinned Hugging Face revision.
ARCHITECTURE = "esm2-35m"       # "esm2-650m" requires more training memory/compute.
PERSISTENT_RUN = Path("/content/drive/MyDrive/Pairloom") / RUN_NAME
HEAD = "cls"                     # "cls" or "residue"
MLM_WEIGHT = 0.1
MAX_TOKENS = 1024
MAX_VIEWS = 2
SEED = 17
EPOCHS = 3
SESSION_HOURS = 6.0
MAX_UPDATES = None              # Optional absolute update limit; no fake research score.
'''
    setup = '''import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "transformers==5.12.1", "scikit-learn>=1.3,<2"], check=True)
import gc, torch
if not torch.cuda.is_available():
    raise RuntimeError("Select a Colab GPU runtime before starting real training.")
SOURCE_ROOT = Path("/content/pairloom-code")
(SOURCE_ROOT / "pairloom").mkdir(parents=True, exist_ok=True)
'''
    setup += "SOURCES = " + repr(sources) + "\n"
    setup += '''for name, source in SOURCES.items():
    (SOURCE_ROOT / "pairloom" / name).write_text(source)
for name in list(sys.modules):
    if name == "pairloom" or name.startswith("pairloom."):
        del sys.modules[name]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from pairloom.storage import stage_data, restore_persistent
LOCAL_ROOT = Path("/content/pairloom-work")
DEVELOPMENT_DATA = stage_data(INPUT_DATA, LOCAL_ROOT / "inputs")
LOCAL_RUN = LOCAL_ROOT / RUN_NAME
RESUMING = restore_persistent(LOCAL_RUN, PERSISTENT_RUN)
gc.collect()
torch.cuda.empty_cache()
print("GPU:", torch.cuda.get_device_name(0))
print("Continuing verified Pairloom run" if RESUMING else "Starting NEW Pairloom run: " + INITIALIZATION)
print("Persistent checkpoints:", PERSISTENT_RUN)
'''
    training = '''from pairloom.cli import main
args = ["train", "--data", str(DEVELOPMENT_DATA),
        "--output", str(LOCAL_RUN), "--persistent", str(PERSISTENT_RUN),
        "--device", "cuda", "--precision", "auto", "--head", HEAD,
        "--mlm-weight", str(MLM_WEIGHT),
        "--max-tokens", str(MAX_TOKENS), "--max-views", str(MAX_VIEWS),
        "--seed", str(SEED), "--epochs", str(EPOCHS), "--hours", str(SESSION_HOURS)]
if RESUMING:
    args.append("--resume")
else:
    args.extend(["--initialization", INITIALIZATION, "--architecture", str(ARCHITECTURE)])
    if INITIALIZATION == "pretrained" and PRETRAINED_MODEL is not None:
        args.extend(["--pretrained-model", str(PRETRAINED_MODEL)])
    if INITIALIZATION == "pretrained" and REVISION is not None:
        args.extend(["--revision", str(REVISION)])
if MAX_UPDATES is not None:
    args.extend(["--max-updates", str(MAX_UPDATES)])
main(args)
'''
    cells = []
    for ident, kind, text in [("intro", "markdown", intro), ("settings", "code", settings),
                               ("prepare", "code", setup), ("train", "code", training)]:
        cell = {"id": ident, "cell_type": kind, "metadata": {}, "source": text.splitlines(True)}
        if kind == "code":
            compile(text, "<pairloom-colab-" + ident + ">", "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                             "kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"},
                             "language_info": {"name": "python"}}}
    path = root / "notebooks/Pairloom_train_Colab.ipynb"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    build()
