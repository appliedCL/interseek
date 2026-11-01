"""Trainable protein-pair heads on pretrained or randomly initialised ESM-2."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ResidueHead(nn.Module):
    """Shared cross-chain attention followed by symmetric interaction pooling."""
    def __init__(self, hidden_size, width=128, dropout=.1):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width), nn.GELU())
        self.attend = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
        self.combine = nn.Sequential(nn.Linear(width * 3, width), nn.GELU())
        self.output = nn.Sequential(nn.LayerNorm(width * 2), nn.Dropout(dropout), nn.Linear(width * 2, 1))

    def forward(self, states, chains):
        features = self.project(states)
        summaries = []
        for own, other in [(1, 2), (2, 1)]:
            selected = chains.eq(own)
            attended, _ = self.attend(features, features, features, key_padding_mask=~chains.eq(other), need_weights=False)
            matches = self.combine(torch.cat((features, attended, features * attended), -1))
            mean = (matches * selected[..., None]).sum(1) / selected.sum(1, keepdim=True)
            peak = matches.masked_fill(~selected[..., None], torch.finfo(matches.dtype).min).amax(1)
            summaries.append(torch.cat((mean, peak), -1))
        return self.output((summaries[0] + summaries[1]) * .5).squeeze(-1)


class PairModel(nn.Module):
    def __init__(self, backbone, head="cls", width=128, dropout=.1, checkpointing=False):
        super().__init__()
        self.backbone, self.head_name = backbone, head
        self.settings = {"head": head, "head_width": width, "dropout": dropout,
                         "gradient_checkpointing": checkpointing}
        size = backbone.config.hidden_size
        if head not in {"cls", "residue"}:
            raise ValueError("Head must be cls or residue.")
        self.head = (nn.Sequential(nn.LayerNorm(size), nn.Dropout(dropout), nn.Linear(size, 1))
                     if head == "cls" else ResidueHead(size, width, dropout))
        backbone.requires_grad_(True)
        if checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
