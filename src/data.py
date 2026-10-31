"""Sequence-pair records, supplied split selection and deterministic window bags."""
from dataclasses import dataclass
import csv
import gzip
import hashlib
import random

import torch

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZUO")


@dataclass(frozen=True)
class Record:
    pair_id: str
    sequence_a: str
    sequence_b: str
    label: int | None = None
    mutant: str | None = None


def sequence(value):
    value = "".join(value.upper().split())
    if not value or not set(value) <= AMINO_ACIDS:
        raise ValueError("Expected a nonempty single-chain amino-acid sequence.")
    return value


def read_records(path, split=None, task="ppi", labelled=True):
    """Read sequence-pair columns or query/text aliases, preserving split membership."""
    opener = gzip.open if str(path).endswith(".gz") else open
    records, identifiers = [], set()
    with opener(path, "rt", newline="") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), start=1):
            if split is not None and row.get("split") != split:
                continue
            if task == "mutation":
                a, b = sequence(row["wild_seq"]), sequence(row["participant_sequence"])
                mutant = sequence(row["mutant_seq"])
            else:
                a = sequence(row.get("sequence_a", row.get("query", "")))
                b = sequence(row.get("sequence_b", row.get("text", "")))
                mutant = None
            raw = row.get("label", "")
            labels = {"0": 0, "1": 1, "decreasing": 0, "increasing": 1}
            if labelled and raw not in labels:
                raise ValueError("Labels must be 0/1, or decreasing/increasing for mutations.")
            label = labels.get(raw)
            content = "|".join(sorted((a, b))) if mutant is None else "|".join((a, mutant, b))
            # Distinct source rows remain distinct even when pairs repeat or
            # appear in reversed order. Number before filtering for stable IDs.
            pair_id = row.get("pair_id") or f"{hashlib.sha256(content.encode()).hexdigest()}:{row_number}"
            if pair_id in identifiers:
                raise ValueError("Duplicate pair_id: " + pair_id)
            identifiers.add(pair_id)
            records.append(Record(pair_id, a, b, label, mutant))
    if not records:
        raise ValueError("No records matched the requested input/split.")
    return records


def fingerprint(records):
    digest = hashlib.sha256()
    for r in records:
        digest.update(repr((r.pair_id, r.sequence_a, r.sequence_b, r.label, r.mutant)).encode() + b"\n")
    return digest.hexdigest()


def validate_training_labels(training, validation):
    """Require classes needed for weighting and metrics; trust supplied splits."""
    if {r.label for r in training} != {0, 1} or {r.label for r in validation} != {0, 1}:
        raise ValueError("Training and validation must each contain both classes.")


def starts(length, width):
    """Non-overlapping windows plus an end-aligned window when needed."""
    if length <= width:
        return [0]
    return sorted(set([*range(0, length - width + 1, width), length - width]))


def window_coordinates(a, b, max_tokens, max_views, training=False, seed=0):
    """Select bounded pairs of windows; only the whole bag receives the pair label."""
    budget = max_tokens - 3
    if budget < 2 or max_views < 1:
        raise ValueError("max_tokens must be >=5 and max_views >=1.")
    swapped = a > b
    if swapped:
        a, b = b, a
    wa = min(len(a), max(1, budget // 2))
    wb = min(len(b), budget - wa)
    wa = min(len(a), budget - wb)
    aa, bb = starts(len(a), wa), starts(len(b), wb)
    total = len(aa) * len(bb)
    count = min(max_views, total)
    if training:
        selected = sorted(random.Random(seed).sample(range(total), count))
    elif count == 1:
        selected = [total // 2]
    else:
        selected = [i * (total - 1) // (count - 1) for i in range(count)]
    coordinates = [(aa[i // len(bb)], wa, bb[i % len(bb)], wb) for i in selected]
    return [(b0, bw, a0, aw) for a0, aw, b0, bw in coordinates] if swapped else coordinates


def windows(a, b, max_tokens, max_views, training=False, seed=0):
    return [(a[a0:a0 + aw], b[b0:b0 + bw])
            for a0, aw, b0, bw in window_coordinates(a, b, max_tokens, max_views, training, seed)]


class PairBatcher:
    def __init__(self, vocab, max_tokens=1024, max_views=2, seed=17):
        self.vocab, self.max_tokens, self.max_views, self.seed = vocab, max_tokens, max_views, seed
        self.residue_ids = torch.tensor([vocab[c] for c in sorted(AMINO_ACIDS) if c in vocab])

    def __call__(self, records, training=False, epoch=0, reverse=False):
        packed, chains, owners = [], [], []
        mutation = records[0].mutant is not None
        if any((r.mutant is not None) != mutation for r in records):
            raise ValueError("PPI and mutation records cannot share a batch.")
        for index, r in enumerate(records):
            seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{epoch}:{r.pair_id}".encode()).digest()[:8], "big")
            wild_coordinates = window_coordinates(r.sequence_a, r.sequence_b, self.max_tokens, self.max_views, training, seed)
            # Same-length variants retain exactly the wild-type crop coordinates.
            for variant, a in enumerate([r.sequence_a, r.mutant] if mutation else [r.sequence_a]):
                swapped = reverse if not training else bool(random.Random(seed).getrandbits(1))
                coordinates = (wild_coordinates if len(a) == len(r.sequence_a) else
                               window_coordinates(a, r.sequence_b, self.max_tokens, self.max_views, training, seed))
                for a0, aw, b0, bw in coordinates:
                    x, y = a[a0:a0 + aw], r.sequence_b[b0:b0 + bw]
                    if swapped:
                        x, y = y, x
                    xi, yi = [self.vocab[c] for c in x], [self.vocab[c] for c in y]
                    packed.append([self.vocab["<cls>"], *xi, self.vocab["<eos>"], *yi, self.vocab["<eos>"]])
                    chains.append([0, *([1] * len(xi)), 0, *([2] * len(yi)), 0])
                    owners.append(index * (2 if mutation else 1) + variant)
        length = max(map(len, packed))
        ids = torch.full((len(packed), length), self.vocab["<pad>"], dtype=torch.long)
        chain = torch.zeros_like(ids)
        for i, (tokens, groups) in enumerate(zip(packed, chains)):
            ids[i, :len(tokens)] = torch.tensor(tokens)
            chain[i, :len(groups)] = torch.tensor(groups)
        return {"input_ids": ids, "attention_mask": ids.ne(self.vocab["<pad>"]), "chains": chain,
                "owners": torch.tensor(owners), "groups": len(records) * (2 if mutation else 1),
                "mutation": mutation, "labels": torch.tensor([r.label or 0 for r in records], dtype=torch.float32)}

    def mask(self, batch, probability):
        if not 0 < probability <= 1:
            raise ValueError("Mask probability must be in (0,1].")
        batch = dict(batch)
        ids = batch["input_ids"].clone()
        eligible = batch["chains"].ne(0)
        chosen = (torch.rand(ids.shape) < probability) & eligible
        if not chosen.any():
            positions = eligible.nonzero()
            pos = positions[torch.randint(len(positions), ())]
            chosen[tuple(pos)] = True
        targets = ids.clone().masked_fill(~chosen, -100)
        draw = torch.rand(ids.shape)
        ids[chosen & (draw < .8)] = self.vocab["<mask>"]
        random_positions = chosen & (draw >= .8) & (draw < .9)
        ids[random_positions] = self.residue_ids[torch.randint(len(self.residue_ids), (int(random_positions.sum()),))]
        batch.update(input_ids=ids, mlm_targets=targets)
        return batch
