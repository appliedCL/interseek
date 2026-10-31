"""Meta ESM-2 initialisation, optional random baselines and data-only imports."""
import hashlib
import json
from pathlib import Path
import zipfile

import torch

# ESM-2's published alphabet is a fixed token vocabulary, not learned weights.
TOKENS = ('<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I',
          'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O',
          '.', '-', '<null_1>', '<mask>')
ARCHITECTURES = {"esm2-35m": (12, 480, 20), "esm2-650m": (33, 1280, 20)}
PRETRAINED_MODELS = {"esm2-35m": "facebook/esm2_t12_35M_UR50D",
                     "esm2-650m": "facebook/esm2_t33_650M_UR50D"}


def weight_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode() + tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def initialize_architecture(architecture="esm2-35m", seed=17, config=None):
    """Construct ESM-2 from a preset/local JSON.

    Seeded construction runs in an isolated CPU RNG context. The recorded config
    and initialisation hash allow restoration without a model repository or base
    weight file. All learned weights are subsequently included in checkpoints.
    """
    from transformers import EsmConfig, EsmForMaskedLM
    if seed < 0:
        raise ValueError("Initialisation seed must be nonnegative.")
    if config is None:
        if architecture in ARCHITECTURES:
            layers, width, heads = ARCHITECTURES[architecture]
            config = {"num_hidden_layers": layers, "hidden_size": width,
                      "num_attention_heads": heads, "intermediate_size": width * 4}
        else:
            path = Path(architecture)
            if not path.is_file():
                raise ValueError("Choose esm2-35m, esm2-650m or a local ESM-2 architecture JSON.")
            config = json.loads(path.read_text())
    values = {"vocab_size": len(TOKENS), "pad_token_id": 1, "mask_token_id": 32,
              "position_embedding_type": "rotary", "token_dropout": True,
              "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0,
              "layer_norm_eps": 1e-5, "max_position_embeddings": 1026, **config}
    if (values["vocab_size"] != len(TOKENS) or values["pad_token_id"] != 1
            or values["mask_token_id"] != 32 or values["position_embedding_type"] != "rotary"):
        raise ValueError("Expected the fixed ESM-2 alphabet and rotary architecture.")
    if (values["num_hidden_layers"] < 1 or values["hidden_size"] < 1
            or values["num_attention_heads"] < 1
            or values["hidden_size"] % values["num_attention_heads"]):
        raise ValueError("Invalid ESM-2 architecture dimensions.")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = EsmForMaskedLM(EsmConfig(**values))
    identity = {"kind": "random-esm2", "config": values, "seed": seed,
                "initial_weights_sha256": weight_digest(model)}
    return model, {token: index for index, token in enumerate(TOKENS)}, identity
