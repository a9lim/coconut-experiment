"""Serialize CheatChain records into tokenized sequences for the CoT arm.

A CoT training sequence is:

    <question>\n<step_1>\n...<step_L>\n#### <answer><eos>
    |________ prompt _______|________ supervised target ________|

The prompt (question, including any ``Hint:`` line) is masked from the loss
(-100); the model is supervised only on the reasoning steps + the final answer.
Prompt and target are tokenized separately and concatenated, so the label mask
aligns by construction (no BPE boundary surprises — the prompt ends in "\n" and
the target starts with a digit).

The latent arm will consume the same records through a different serializer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union, cast

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_records(path: Union[str, Path]) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def setup_tokenizer(base_model: str = "gpt2") -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # GPT-2 has no pad token
    return tok


def prompt_text(record: Dict) -> str:
    """The generation prompt — everything the model conditions on. Ends in newline."""
    return record["question"] + "\n"


def target_text(record: Dict) -> str:
    """The supervised continuation: CoT steps then the answer line."""
    return "\n".join(record["cot_steps"]) + "\n" + record["answer_text"]


def encode_cot(record: Dict, tokenizer: PreTrainedTokenizerBase) -> Dict[str, List[int]]:
    eos = cast(int, tokenizer.eos_token_id)
    prompt_ids = cast(List[int], tokenizer.encode(prompt_text(record)))
    target_ids = cast(List[int], tokenizer.encode(target_text(record))) + [eos]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


class CoTDataset(Dataset):
    """Tokenized CheatChain records for plain causal-LM SFT (the CoT arm)."""

    def __init__(self, records: List[Dict], tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.examples = [encode_cot(r, tokenizer) for r in records]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.examples[idx]


@dataclass
class CoTCollator:
    """Right-pad a batch; pad input_ids with pad_token_id, labels with -100."""

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids, labels, attn = [], [], []
        for f in features:
            n_pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * n_pad)
            labels.append(f["labels"] + [-100] * n_pad)
            attn.append(f["attention_mask"] + [0] * n_pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
