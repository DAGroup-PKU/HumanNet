"""
vla.py

Draccus Dataclass Definition for a VLAConfig object, with various registered subclasses for each VLA experiment and
model configuration thereof. A given VLA model (`policy`) configures the following attributes:
    - Data Mixture (e.g., Bridge, OXE_MAGIC_SOUP, etc.)
    - Base VLM from Prismatic Registry (e.g., `prism-dinosiglip+7b`)
    - VLA Model Architecture / Parameters (e.g., freeze vision encoder, last layer finetuning)
    - Training / Optimization Hyperparameters
"""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional, Union

from draccus import ChoiceRegistry


@dataclass
class VLAConfig(ChoiceRegistry):
    # fmt: off
    vla_id: str                                     # Unique VLA Policy ID that fully specifies a configuration variant
    base_vlm: Union[str, Path]                      # Base VLM as ID/Path to Run Directory (e.g., `prism-dinosiglip+7b`)
    freeze_vision_backbone: bool                    # Freeze Vision Backbone Parameters (akin to pretraining)
    freeze_llm_backbone: bool                       # Freeze LLM Backbone parameters
    unfreeze_last_llm_layer: bool                   # Unfreeze final layer of LLM (only takes effect if LLM is frozen)

    # Data Mixture Parameters
    data_mix: str                                   # Open-X Embodiment Dataset =>> Unique Mixture ID (e.g., `bridge`)
    shuffle_buffer_size: int                        # Size of Shuffle Buffer (100K for Bridge, 1M for OXE)

    # Optimization Parameters
    epochs: int                                     # Epochs to Run (in case `max_steps` is not specified)
    max_steps: Optional[int]                        # [Optional] Max Gradient Steps to Run (overrides `epochs`)
    save_every_n_steps: Optional[int]

    expected_world_size: int                        # Expected # of GPUs =>> allows us to gate training on hardware
    global_batch_size: int                          # Global Batch Size (divided across processes / world size)
    per_device_batch_size: int                      # Per-Device Batch Size (per-process / individual GPU)
                                                    #   =>> # of accumulation steps is auto-computed

    learning_rate: float                            # Peak Learning Rate (`lr_scheduler_type` sets warmup/decay)
    weight_decay: float                             # Weight Decay for AdamW Optimizer
    max_grad_norm: float                            # Max Grad Norm (for global gradient clipping)
    lr_scheduler_type: str                          # LR Scheduler (usually: "constant" | "linear-warmup+cosine-decay")
    warmup_ratio: float                             # Fraction of Steps to Warmup (for warmup LR schedulers)

    train_strategy: str                             # Train Strategy (default "fsdp-full-shard")
    action_tokenizer: str

    image_sequence_len: int
    use_wrist_image: bool

    # Enable Gradient/Activation Checkpointing (for the LLM Backbone)
    enable_gradient_checkpointing: bool = True      # Enable Gradient/Activation Checkpointing during Training

    # Mixed Precision Training via Torch Native AMP (`autocast`)
    enable_mixed_precision_training: bool = True    # Enable Traditional BF16 Mixed Precision
    reduce_in_full_precision: bool = True           # Accumulate/Reduce All-Gather Gradients in FP32 Full Precision

    # fmt: on


# === OpenVLA Training Configurations ===


# = [8 GPU] Fast Iteration =>> SigLIP 224px + Bridge =
@dataclass
class Exp_SigLIP_224px_Bridge(VLAConfig):
    vla_id: str = "siglip-224px+mx-bridge"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    image_sequence_len: int = 1
    use_wrist_image: bool = False

    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False

    # Data Mixture Parameters
    data_mix: str = "bridge"
    shuffle_buffer_size: int = 256_000

    # Optimization Parameters
    epochs: int = 1000
    max_steps: Optional[int] = None
    save_every_n_steps: Optional[int] = 25000

    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32

    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0

    train_strategy: str = "fsdp-full-shard"
    action_tokenizer: str = "action_tokenizer"


# = [8 GPU] SigLIP 224px Frozen Vision Backbone + Bridge =
@dataclass
class Exp_FreezeVIT_SigLIP_224px_Bridge(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px-icy+mx-bridge"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = True


# = [8 GPU] Fast Iteration =>> DINO-SigLIP 224px + Bridge =
@dataclass
class Exp_DinoSigLIP_224px_Bridge(Exp_SigLIP_224px_Bridge):
    vla_id: str = "prism-dinosiglip-224px+mx-bridge"
    base_vlm: Union[str, Path] = "prism-dinosiglip-224px+7b"

    data_mix: str = "bridge"


# = [64 GPU] SigLIP 224px + OXE Magic Soup =
@dataclass
class Exp_SigLIP_224px_OXE_Magic_Soup(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px+mx-oxe-magic-soup"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "oxe_magic_soup"

    expected_world_size: int = 64
    global_batch_size: int = 2048
    per_device_batch_size: int = 32


# = [8 GPU] Qwen2.5 0.5B SigLIP 224px + OXE Magic Soup =
@dataclass
class Exp_Qwen25_DinoSigLIP_224px_0_5B_OXE_Magic_Soup(Exp_SigLIP_224px_Bridge):
    vla_id: str = "prism-qwen25-dinosiglip-224px+0_5b+mx-oxe-magic-soup"
    base_vlm: Union[str, Path] = "prism-qwen25-extra-dinosiglip-224px+0_5b"

    data_mix: str = "oxe_magic_soup"
    action_tokenizer: str = "extra_action_tokenizer"

    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32


@dataclass
class Exp_Qwen25_DinoSigLIP_224px_0_5B_LIBERO_90(Exp_Qwen25_DinoSigLIP_224px_0_5B_OXE_Magic_Soup):
    vla_id: str = "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90"

    data_mix: str = "libero_90"

    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32


@dataclass
class Exp_Qwen25_DinoSigLIP_224px_T2_0_5B_LIBERO_90(Exp_Qwen25_DinoSigLIP_224px_0_5B_LIBERO_90):
    vla_id: str = "prism-qwen25-dinosiglip-224px-t2+0_5b+mx-libero-90"
    image_sequence_len: int = 2


@dataclass
class Exp_Qwen25_DinoSigLIP_224px_wrist_0_5B_LIBERO_90(Exp_Qwen25_DinoSigLIP_224px_0_5B_LIBERO_90):
    vla_id: str = "prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-libero-90"
    image_sequence_len: int = 2
    use_wrist_image: bool = True


## bridge Qwen


@dataclass
class Exp_Qwen25_DinoSigLIP_224px_0_5B_Bridge(Exp_SigLIP_224px_Bridge):
    vla_id: str = "prism-qwen25-dinosiglip-224px+0_5b+mx-bridge"
    base_vlm: Union[str, Path] = "prism-qwen25-extra-dinosiglip-224px+0_5b"

    data_mix: str = "bridge_dataset"  # direct dataset
    action_tokenizer: str = "extra_action_tokenizer"

    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32


@dataclass
class Exp_DinoSigLIP_224px_LIBERO_90(Exp_DinoSigLIP_224px_Bridge):
    vla_id: str = "prism-dinosiglip-224px+mx-libero-90"

    data_mix: str = "libero_90"

    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32


# = [64 GPU] DINO-SigLIP 224px + OXE Magic Soup++ =
@dataclass
class Exp_DinoSigLIP_224px_OXE_Magic_Soup_Plus(Exp_SigLIP_224px_Bridge):
    vla_id: str = "prism-dinosiglip-224px+mx-oxe-magic-soup-plus"
    base_vlm: Union[str, Path] = "prism-dinosiglip-224px+7b"

    # Note =>> We adopt two stages, training on a mixture including DROID for 70% of training, before resampling!
    # data_mix: str = "oxe_magic_soup_plus"
    data_mix: str = "oxe_magic_soup_plus_minus"

    expected_world_size: int = 64
    global_batch_size: int = 2048
    per_device_batch_size: int = 32


# === OpenVLA Fine-tuning Configurations ===


# = [8 GPU] SigLIP 224px + T-DROID =
@dataclass
class Exp_SigLIP_224px_TDROID_CarrotInBowl(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px+mx-tdroid_carrot_in_bowl"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "tdroid_carrot_in_bowl"


@dataclass
class Exp_SigLIP_224px_TDROID_PourCornInPot(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px+mx-tdroid_pour_corn_in_pot"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "tdroid_pour_corn_in_pot"


# = [8 GPU] SigLIP 224px + T-DROID -- Partial Finetuning =
@dataclass
class Exp_SigLIP_224px_Icy_TDROID_CarrotInBowl(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px-icy+mx-tdroid_carrot_in_bowl"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = False

    data_mix: str = "tdroid_carrot_in_bowl"


@dataclass
class Exp_SigLIP_224px_LastLayer_TDROID_CarrotInBowl(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px-last_layer+mx-tdroid_carrot_in_bowl"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = True
    unfreeze_last_llm_layer: bool = True

    data_mix: str = "tdroid_carrot_in_bowl"


@dataclass
class Exp_SigLIP_224px_Sandwich_TDROID_CarrotInBowl(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px-sandwich+mx-tdroid_carrot_in_bowl"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = True
    unfreeze_last_llm_layer: bool = True

    data_mix: str = "tdroid_carrot_in_bowl"


# === [8 GPU] SigLIP 224px + FrankaWipe ===
@dataclass
class Exp_SigLIP_224px_Droid_Wipe(Exp_SigLIP_224px_Bridge):
    vla_id: str = "siglip-224px+mx-droid_wipe"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "droid_wipe"


# === Define a VLA Registry Enum for Reference & Validation ===
@unique
class VLARegistry(Enum):
    # Sanity Check Configurations =>> BridgeV2
    SIGLIP_224PX_MX_BRIDGE = Exp_SigLIP_224px_Bridge
    DINOSIGLIP_224PX_MX_BRIDGE = Exp_DinoSigLIP_224px_Bridge
    DINOSIGLIP_224PX_MX_LIBERO_90 = Exp_DinoSigLIP_224px_LIBERO_90

    # SigLIP Frozen Backbone Experiment
    FREEZE_SIGLIP_224PX_MX_BRIDGE = Exp_FreezeVIT_SigLIP_224px_Bridge

    # [OpenVLA v0.1 7B] SigLIP 224px + OXE Magic Soup
    SIGLIP_224PX_MX_OXE_MAGIC_SOUP = Exp_SigLIP_224px_OXE_Magic_Soup

    # [OpenVLA 7B] DINO + SigLIP 224px + OXE Magic Soup++
    DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS = Exp_DinoSigLIP_224px_OXE_Magic_Soup_Plus

    # [OpenVLA 0.5B] Qwen backbones
    QWEN25_DINOSIGLIP_224PX_0_5B_MX_OXE_MAGIC_SOUP = Exp_Qwen25_DinoSigLIP_224px_0_5B_OXE_Magic_Soup
    QWEN25_DINOSIGLIP_224PX_0_5B_LIBERO_90 = Exp_Qwen25_DinoSigLIP_224px_0_5B_LIBERO_90
    QWEN25_DINOSIGLIP_224PX_T2_0_5B_LIBERO_90 = Exp_Qwen25_DinoSigLIP_224px_T2_0_5B_LIBERO_90
    QWEN25_DINOSIGLIP_224PX_WRIST_0_5B_LIBERO_90 = Exp_Qwen25_DinoSigLIP_224px_wrist_0_5B_LIBERO_90

    QWEN25_DINOSIGLIP_224PX_0_5B_BRIDGE = Exp_Qwen25_DinoSigLIP_224px_0_5B_Bridge

    # === TDROID Fine-tuning Configs ===
    SIGLIP_224PX_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_TDROID_CarrotInBowl
    SIGLIP_224PX_MX_TDROID_POUR_CORN_IN_POT = Exp_SigLIP_224px_TDROID_PourCornInPot

    SIGLIP_224PX_ICY_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_Icy_TDROID_CarrotInBowl
    SIGLIP_224PX_LASTLAYER_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_LastLayer_TDROID_CarrotInBowl
    SIGLIP_224PX_SANDWICH_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_Sandwich_TDROID_CarrotInBowl

    # === DROID Fine-tuning Configs ===
    SIGLIP_224PX_MX_DROID_WIPE = Exp_SigLIP_224px_Droid_Wipe

    @property
    def vla_id(self) -> str:
        return self.value.vla_id


# Register VLAs in Choice Registry
for vla_variant in VLARegistry:
    VLAConfig.register_subclass(vla_variant.vla_id, vla_variant.value)