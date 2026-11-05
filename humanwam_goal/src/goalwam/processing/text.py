"""Instruction tokenization with the checkpoint's local tokenizer."""

from collections.abc import Mapping

import torch
from transformers import AutoTokenizer


class InstructionTokenizer:
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        self.vision_start = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        if self.vision_start is None or self.tokenizer.eos_token_id is None:
            raise ValueError("Checkpoint tokenizer lacks required boundary tokens")

    def __call__(self, instruction):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a nonempty string")
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.flatten().tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = torch.tensor(
            [ids + [self.tokenizer.eos_token_id, self.vision_start]], dtype=torch.long
        )
        return ids, torch.ones_like(ids)
