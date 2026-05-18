"""
materialize.py

Factory class for initializing Vision Backbones, LLM Backbones, and VLMs from a set registry; provides and exports
individual functions for clear control flow.
"""

from typing import Optional, Tuple

from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm import LLaMa2LLMBackbone, LLMBackbone, MistralLLMBackbone, PhiLLMBackbone
from prismatic.models.backbones.llm.qwen25 import Qwen25LLMBackbone
from prismatic.models.backbones.vision import (
    CLIPViTBackbone,
    DinoCLIPViTBackbone,
    DinoSigLIPViTBackbone,
    DinoV2ViTBackbone,
    ImageTransform,
    IN1KViTBackbone,
    SigLIPViTBackbone,
    VisionBackbone,
)
from prismatic.models.vlms import PrismaticVLM

# === Registries =>> Maps ID --> {cls(), kwargs} :: Different Registries for Vision Backbones, LLM Backbones, VLMs ===
# fmt: off

# === Vision Backbone Registry ===
VISION_BACKBONES = {
    # === 224px Backbones ===
    "clip-vit-l": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-so400m": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "dinov2-vit-l": {"cls": DinoV2ViTBackbone, "kwargs": {"default_image_size": 224}},
    "in1k-vit-l": {"cls": IN1KViTBackbone, "kwargs": {"default_image_size": 224}},
    "dinosiglip-vit-so-224px": {"cls": DinoSigLIPViTBackbone, "kwargs": {"default_image_size": 224}},

    # === Assorted CLIP Backbones ===
    "clip-vit-b": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "clip-vit-l-336px": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 336}},

    # === Assorted SigLIP Backbones ===
    "siglip-vit-b16-224px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-b16-256px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 256}},
    "siglip-vit-b16-384px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 384}},
    "siglip-vit-so400m-384px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 384}},

    # === Fused Backbones ===
    "dinoclip-vit-l-336px": {"cls": DinoCLIPViTBackbone, "kwargs": {"default_image_size": 336}},
    "dinosiglip-vit-so-384px": {"cls": DinoSigLIPViTBackbone, "kwargs": {"default_image_size": 384}},
}


# === Language Model Registry ===
LLM_BACKBONES = {
    # === LLaMa-2 Pure (Non-Chat) Backbones ===
    "llama2-7b-pure": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "llama2-13b-pure": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === LLaMa-2 Chat Backbones ===
    "llama2-7b-chat": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "llama2-13b-chat": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === Vicuna-v1.5 Backbones ===
    "vicuna-v15-7b": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "vicuna-v15-13b": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === Mistral v0.1 Backbones ===
    "mistral-v0.1-7b-pure": {"cls": MistralLLMBackbone, "kwargs": {}},
    "mistral-v0.1-7b-instruct": {"cls": MistralLLMBackbone, "kwargs": {}},

    # === Phi-2 Backbone ===
    "phi-2-3b": {"cls": PhiLLMBackbone, "kwargs": {}},

    # === Qwen2.5 Backbone ===
    "qwen25-0_5b-pure": {"cls": Qwen25LLMBackbone, "kwargs": {}},
    "qwen25-0_5b-extra": {"cls": Qwen25LLMBackbone, "kwargs": {"num_extra_tokens": 256}},
    "qwen25-1_5b-pure": {"cls": Qwen25LLMBackbone, "kwargs": {}},
    "qwen25-3b-pure": {"cls": Qwen25LLMBackbone, "kwargs": {}},
}

# fmt: on


def get_vision_backbone_and_transform(
    vision_backbone_id: str,
    image_resize_strategy: str,
    image_sequence_len: int,
) -> Tuple[VisionBackbone, ImageTransform]:
    """Instantiate a Vision Backbone, returning both the nn.Module wrapper class and default Image Transform."""
    if vision_backbone_id in VISION_BACKBONES:
        vision_cfg = VISION_BACKBONES[vision_backbone_id]
        vision_backbone: VisionBackbone = vision_cfg["cls"](
            vision_backbone_id, image_resize_strategy, image_sequence_len=image_sequence_len, **vision_cfg["kwargs"]
        )
        image_transform = vision_backbone.get_image_transform()
        return vision_backbone, image_transform

    else:
        raise ValueError(f"Vision Backbone `{vision_backbone_id}` is not supported!")


# def _is_network_or_dns_error(e: Exception) -> bool:
#     s = str(e)
#     return (
#         "Name or service not known" in s or
#         "Failed to resolve" in s or
#         "HTTPSConnectionPool" in s or
#         isinstance(e, socket.gaierror)
#     )


# from typing import Tuple, Optional, Any, Dict
# def get_vision_backbone_and_transform(
#     vision_backbone_id: str,
#     image_resize_strategy: str,
#     image_sequence_len: int,
#     *,
#     checkpoint_path: Optional[str] = None,
#     pretrained: Optional[bool] = None,
#     **extra_kwargs: Any,  
# ) -> Tuple[VisionBackbone, ImageTransform]:
#     """Instantiate a Vision Backbone, returning both the nn.Module wrapper class and default Image Transform."""
#     if vision_backbone_id not in VISION_BACKBONES:
#         raise ValueError(f"Vision Backbone `{vision_backbone_id}` is not supported!")

#     vision_cfg = VISION_BACKBONES[vision_backbone_id]


#     merged_kwargs: Dict[str, Any] = dict(vision_cfg["kwargs"])
#     if pretrained is not None:
#         merged_kwargs["pretrained"] = pretrained
#     if checkpoint_path is not None:
#         merged_kwargs["checkpoint_path"] = checkpoint_path
#     merged_kwargs.update(extra_kwargs)

#     vision_backbone: VisionBackbone = vision_cfg["cls"](
#         vision_backbone_id,
#         image_resize_strategy,
#         image_sequence_len=image_sequence_len,
#         **merged_kwargs,
#     )
#     image_transform = vision_backbone.get_image_transform()
#     return vision_backbone, image_transform


def get_llm_backbone_and_tokenizer(
    llm_backbone_id: str,
    llm_max_length: int = 2048,
    hf_token: Optional[str] = None,
    inference_mode: bool = False,
) -> Tuple[LLMBackbone, PreTrainedTokenizerBase]:
    if llm_backbone_id in LLM_BACKBONES:
        llm_cfg = LLM_BACKBONES[llm_backbone_id]
        llm_backbone: LLMBackbone = llm_cfg["cls"](
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            **llm_cfg["kwargs"],
        )
        tokenizer = llm_backbone.get_tokenizer()
        return llm_backbone, tokenizer

    else:
        raise ValueError(f"LLM Backbone `{llm_backbone_id}` is not supported!")


def get_vlm(
    model_id: str,
    arch_specifier: str,
    vision_backbone: VisionBackbone,
    llm_backbone: LLMBackbone,
    enable_mixed_precision_training: bool = True,
) -> PrismaticVLM:
    """Lightweight wrapper around initializing a VLM, mostly for future-proofing (if one wants to add a new VLM)."""
    return PrismaticVLM(
        model_id,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=enable_mixed_precision_training,
        arch_specifier=arch_specifier,
    )
