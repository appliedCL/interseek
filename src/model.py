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

    def forward(self, batch):
        states = self.backbone.esm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], return_dict=True).last_hidden_state
        view_logits = (self.head(states[:, 0]).squeeze(-1) if self.head_name == "cls"
                       else self.head(states, batch["chains"]))
        # Log-mean-exp is a weakly supervised bag score, not a label on every crop.
        logits = torch.stack([torch.logsumexp(view_logits[batch["owners"] == i].float(), 0)
                              - math.log(int((batch["owners"] == i).sum())) for i in range(batch["groups"])])
        if batch["mutation"]:
            logits = logits.reshape(-1, 2)
            logits = logits[:, 1] - logits[:, 0]
        targets = batch.get("mlm_targets")
        mlm_sum = states.new_zeros((), dtype=torch.float32)
        if targets is not None:
            selected = targets.ne(-100)
            mlm_sum = F.cross_entropy(self.backbone.lm_head(states[selected]).float(), targets[selected], reduction="sum")
        return logits, mlm_sum
