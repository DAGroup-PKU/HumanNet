"""
qwen2_5.py

Class definition for all LLMs derived from QwenForCausalLM.
"""

from typing import Optional, Sequence, Type

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder
from prismatic.models.backbones.llm.prompting.qwen_prompter import QwenPromptBuilder

# Registry =>> Support Qwen-2.5 Models (from HF Transformers)
# fmt: off
QWEN25_MODELS = {
    # === Pure Qwen2.5 (non-instruct/chat-tuned) Models ===
    "qwen25-0_5b-extra": {
        "llm_family": "qwen2.5", "llm_cls": AutoModelForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-0.5B"
    },
    "qwen25-0_5b-pure": {
        "llm_family": "qwen2.5", "llm_cls": AutoModelForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-0.5B"
    },
    "qwen25-1_5b-pure": {
        "llm_family": "qwen2.5", "llm_cls": AutoModelForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-1.5B"
    },
    "qwen25-3b-pure": {
        "llm_family": "qwen2.5", "llm_cls": AutoModelForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-3B"
    },
    "qwen25-7b-pure": {
        "llm_family": "qwen2.5", "llm_cls": AutoModelForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-7B"
    },

}
# fmt: on


class Qwen25LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
        num_extra_tokens: int = 0,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **QWEN25_MODELS[llm_backbone_id],
        )

        # add some more special tokens
        if num_extra_tokens > 0:
            added = self.tokenizer.add_tokens([f"<|extra_{i}|>" for i in range(num_extra_tokens)])
            assert added == num_extra_tokens, f"Added {added} of {num_extra_tokens} extra tokens to tokenizer!"
            print(f"Added {num_extra_tokens} extra tokens.")

        # there is already a special token for Qwen
        # self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return QwenPromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[torch.nn.Module]:
        return Qwen2DecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def last_layer_finetune_modules(self) -> Sequence[torch.nn.Module]:
        # TODO not sure that this works
        return (self.llm.model.embed_tokens, self.llm.model.layers[-1], self.llm.lm_head)