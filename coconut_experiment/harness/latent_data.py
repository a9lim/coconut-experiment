"""Serialize CheatChain records into the Coconut curriculum (the latent arm).

Reuses the reference `ulterior-motives` implementation:
  - `ContinuousThoughtModel` + `setup_tokenizer`  (training/model.py)
  - `CoconutCollator`                              (moralchain/dataset.py)

A latent training sequence at curriculum stage `k`:

    <question>\n <start-latent> <latent>*(k*c) <end-latent> <step_{k+1}>..<step_L> #### <answer><eos>
    |________________ masked (-100) ________________|________ supervised target ________|

The first `k` reasoning steps are replaced by `k*c` continuous-thought tokens;
the model is supervised on the *remaining* steps + the answer. By the final
stage (k = L) all reasoning is latent and only the answer is supervised. This is
the same one-generator-two-serializers contract as `serialize.py` (the CoT arm)
— identical records, different scaffolding.
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path
from typing import Dict, List

from torch.utils.data import Dataset

# Coconut model is our fork (coconut_experiment/harness/coconut_model.py — fixed for transformers>=5).
# The collator is reused from the reference repo (its packages: moralchain/, ...).
from .coconut_model import (
    ContinuousThoughtConfig,
    ContinuousThoughtModel,
    setup_tokenizer,
)

_UM = Path(__file__).resolve().parents[2] / "ulterior-motives"
if str(_UM) not in sys.path:
    sys.path.insert(0, str(_UM))

from moralchain.dataset import get_collator  # noqa: E402


def setup(base_model: str = "gpt2"):
    """Build (tokenizer, model, collator) for the latent arm."""
    config = ContinuousThoughtConfig(base_model=base_model)
    tokenizer = setup_tokenizer(base_model, config)
    model = ContinuousThoughtModel(config, tokenizer)
    return tokenizer, model, get_collator(tokenizer)


class _Tokenized:
    """Pre-tokenized pieces of one CheatChain record."""

    __slots__ = ("question", "steps", "answer", "condition", "example_id")

    def __init__(self, record: Dict, tokenizer):
        eos = tokenizer.eos_token_id
        self.question = tokenizer.encode(record["question"] + "\n", add_special_tokens=True)
        self.steps = [tokenizer.encode(s + "\n", add_special_tokens=False) for s in record["cot_steps"]]
        self.answer = tokenizer.encode(record["answer_text"], add_special_tokens=False) + [eos]
        self.condition = record["condition"]
        self.example_id = record["id"]


class CheatChainLatentDataset(Dataset):
    """Curriculum-staged latent sequences (training).

    At `scheduled_stage` k, the first k steps become k*c_thought latent tokens.
    With prob `uniform_prob`, an example is drawn from a random earlier stage
    (Coconut's anti-forgetting mix).
    """

    def __init__(
        self,
        records: List[Dict],
        tokenizer,
        scheduled_stage: int,
        c_thought: int = 2,
        max_latent_stage: int = 5,
        uniform_prob: float = 0.0,
        seed: int = 42,
    ):
        self.items = [_Tokenized(r, tokenizer) for r in records]
        self.scheduled_stage = scheduled_stage
        self.c_thought = c_thought
        self.max_latent_stage = max_latent_stage
        self.uniform_prob = uniform_prob
        self.rng = random.Random(seed)
        self.latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        self.start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        self.end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        n_steps = len(item.steps)

        if self.rng.random() < self.uniform_prob:
            stage = self.rng.choice(range(n_steps + 1))
        else:
            stage = self.scheduled_stage

        if stage > self.max_latent_stage:
            n_skip = n_steps
            n_latent = self.max_latent_stage * self.c_thought
        else:
            n_skip = stage
            n_latent = stage * self.c_thought

        tokens = (
            item.question
            + [self.start_id]
            + [self.latent_id] * n_latent
            + [self.end_id]
            + list(itertools.chain.from_iterable(item.steps[n_skip:]))
            + item.answer
        )
        n_masked = len(item.question) + 1 + n_latent + 1  # question + start + latents + end
        labels = [-100] * n_masked + tokens[n_masked:]

        return {
            "input_ids": tokens,
            "labels": labels,
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
            "condition": item.condition,
            "example_id": item.example_id,
            "idx": idx,
        }


class LatentInferenceDataset(Dataset):
    """Prompt-only sequences for generation / latent extraction (eval).

    Layout: <question>\n <start-latent> <latent>*num_latent <end-latent>
    Carries true_answer / cue_value for metric computation.
    """

    def __init__(self, records: List[Dict], tokenizer, num_latent_tokens: int = 10):
        self.records = records
        self.tokenizer = tokenizer
        self.num_latent_tokens = num_latent_tokens
        self.latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        self.start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        self.end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        r = self.records[idx]
        q = self.tokenizer.encode(r["question"] + "\n", add_special_tokens=True)
        tokens = q + [self.start_id] + [self.latent_id] * self.num_latent_tokens + [self.end_id]
        return {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
            "condition": r["condition"],
            "example_id": r["id"],
            "true_answer": r["true_answer"],
            "cue_value": r["cue_value"],
        }
