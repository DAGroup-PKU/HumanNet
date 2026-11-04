# Modified by Open-HumanScale contributors: package namespace renamed to humanwam.
from typing import Any, Mapping, Optional, Sequence, Union

import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from humanwam.action_layout import action_valid_mask, masked_action_mean, resolve_action_components
from humanwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import sinusoidal_embedding_1d

logger = get_logger(__name__)


_BOTTLENECK_ADAPTER_KEY_MARKERS = (
    ".post_self_attn_adapter.",
    ".post_ffn_adapter.",
)
_CAMERA_POSE_KEY_MARKER = ".camera_pose_conditioner."


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    @staticmethod
    def _parse_robotwin_view_size(name: str, size: Sequence[int]) -> tuple[int, int]:
        height, width = (int(size[0]), int(size[1]))
        return height, width

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        source_loss_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
        action_component_num_arms: int = 2,
        action_component_loss_weights: Optional[Mapping[str, float]] = None,
        action_component_indices: Optional[Mapping[str, Sequence[int]]] = None,
        goal_enabled: bool = False,
        goal_targets: str = "both",
        goal_encode_mode: str = "pseudo_video",
        goal_long_rope_mode: str = "arange",
        goal_long_loss_weight: float = 1.0,
        goal_short_rope_index: Optional[int] = None,
        goal_mix_video_ratio: float = 0.0,
        vae_encode_batched: bool = True,
        vae_encode_chunk_frames: int = 0,
        vae_compile_encoder: bool = False,
        vae_compile_dynamic: bool = False,
        vae_compile_mode: str = "default",
        vae_enable_channels_last_3d: bool = True,
        video_vae_layout: str = "pixel",
        robotwin_latent_tshape_head_size: Sequence[int] = (256, 320),
        robotwin_latent_tshape_wrist_size: Sequence[int] = (128, 160),
        latent_tshape_head_resize_mode: str = "cover_crop",
        latent_tshape_premerged_input: bool = False,
        latent_tshape_vae_premerge: bool = False,
        latent_tshape_future_wrist_loss_weight: float = 0.5,
        latent_tshape_head_reads_wrist0: bool = True,
        latent_tshape_action_reads_wrist0: bool = True,
        wrist_view_mask_prob: float = 0.0,
        robotwin_multiview_head_size: Sequence[int] = (256, 448),
        robotwin_multiview_wrist_size: Sequence[int] = (128, 224),
        robotwin_packed_head_size: Sequence[int] = (256, 448),
        robotwin_packed_wrist_size: Sequence[int] = (128, 160),
        packed_head_reads_wrist0: bool = False,
        packed_wrist_encode_separately: bool = False,
        wrist_attention_gate_ramp_steps: int = 0,
        wrist_video_loss_ramp_steps: int = 0,
        wrist_video_loss_max_weight: float = 1.0,
        head_only_replay_start_ratio: float = 0.0,
        head_only_replay_end_ratio: float = 0.0,
        head_only_replay_decay_steps: int = 0,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        # Wan2.2 VAE encode acceleration (Cosmos 3 §5.2.6). `_encode_video_latents` forwards
        # these to `vae.encode`. The encoder is frozen/eval, so compiling it (opt-in) is safe
        # in isolation and avoids the top-level graph breaks that sank a whole-loss compile.
        self.vae_encode_batched = bool(vae_encode_batched)
        self.vae_encode_chunk_frames = int(vae_encode_chunk_frames)
        self.vae.model._encode_channels_last_3d = False
        if bool(vae_enable_channels_last_3d) and self.device.type == "cuda":
            converted = 0
            for module in self.vae.model.encoder.modules():
                if isinstance(module, nn.Conv3d):
                    module.to(memory_format=torch.channels_last_3d)
                    converted += 1
            self.vae.model._encode_channels_last_3d = converted > 0
            if converted > 0:
                logger.info(
                    "FastWAM: enabled channels_last_3d for Wan2.2 VAE encoder "
                    "Conv3d modules (%d converted).",
                    converted,
                )
        if bool(vae_compile_encoder):
            logger.info(
                "FastWAM: torch.compile-ing the complete Wan2.2 VAE encode path "
                "(dynamic=%s, mode=%s, fullgraph=False).",
                bool(vae_compile_dynamic),
                str(vae_compile_mode),
            )
            torch._dynamo.config.assume_static_by_default = not bool(
                vae_compile_dynamic
            )
            self.vae.model.encode = torch.compile(
                self.vae.model.encode,
                dynamic=bool(vae_compile_dynamic),
                fullgraph=False,
                mode=str(vae_compile_mode),
            )
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None
        extra_proprio_sources = [
            source
            for source in self.action_expert.action_tokenizer_sources
            if source != self.action_expert.default_action_tokenizer_source
        ]
        self.source_proprio_encoders = nn.ModuleDict(
            {
                source: nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
                for source in extra_proprio_sources
            }
            if self.proprio_dim is not None
            else {}
        )
        self.initialize_source_proprio_encoders_from_default()
        if self.source_proprio_encoders:
            logger.info(
                "FastWAM: source-specific state encoders enabled | default=%s | extra=%s",
                self.action_expert.default_action_tokenizer_source,
                list(self.source_proprio_encoders),
            )

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        # Inference-only: replay the action denoise step + the once-per-chunk video prefill
        # as CUDA graphs to reduce CPU launch overhead. Default ON;
        # set FASTWAM_CUDA_GRAPH=0 / FASTWAM_CUDA_GRAPH_PREFILL=0 to disable. The trainer's
        # in-training eval force-disables these (capture under FSDP.summon_full_params binds
        # to transient un-sharded buffers). See models/wan22/cuda_graph.py.
        self._use_cuda_graph = os.environ.get("FASTWAM_CUDA_GRAPH", "1") == "1"
        self._use_cuda_graph_prefill = os.environ.get("FASTWAM_CUDA_GRAPH_PREFILL", "1") == "1"
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        source_loss_weights = source_loss_weights or {}
        self.source_loss_weights = {
            component: {
                str(source): float(weight)
                for source, weight in source_loss_weights.get(component, {}).items()
            }
            for component in ("video", "action")
        }
        self.action_component_num_arms = int(action_component_num_arms)
        action_component_loss_weights = action_component_loss_weights or {}
        self.action_component_indices = (
            resolve_action_components(
                self.action_expert.action_dim,
                self.action_component_num_arms,
                action_component_indices,
            )
            if action_component_indices is not None else None
        )
        component_names = (
            set(self.action_component_indices)
            if self.action_component_indices is not None
            else {"position", "rotation", "gripper"}
        )
        self.action_component_loss_weights = {
            name: float(action_component_loss_weights.get(name, 1.0))
            for name in sorted(component_names)
        }

        self.video_vae_layout = str(video_vae_layout)
        self.robotwin_latent_tshape_head_size = self._parse_robotwin_view_size(
            "robotwin_latent_tshape_head_size", robotwin_latent_tshape_head_size
        )
        self.robotwin_latent_tshape_wrist_size = self._parse_robotwin_view_size(
            "robotwin_latent_tshape_wrist_size", robotwin_latent_tshape_wrist_size
        )
        self.latent_tshape_premerged_input = bool(latent_tshape_premerged_input)
        self.latent_tshape_vae_premerge = bool(latent_tshape_vae_premerge)
        self.latent_tshape_head_resize_mode = str(latent_tshape_head_resize_mode)
        self.latent_tshape_future_wrist_loss_weight = float(
            latent_tshape_future_wrist_loss_weight
        )
        self.latent_tshape_head_reads_wrist0 = bool(
            latent_tshape_head_reads_wrist0
        )
        self.latent_tshape_action_reads_wrist0 = bool(
            latent_tshape_action_reads_wrist0
        )
        head_h, head_w = self.robotwin_latent_tshape_head_size
        wrist_h, wrist_w = self.robotwin_latent_tshape_wrist_size
        latent_h, latent_w = self._robotwin_latent_tshape_spatial_size()
        wrist_view_mask_prob = float(wrist_view_mask_prob)
        self.wrist_view_mask_prob = wrist_view_mask_prob

        self.robotwin_multiview_head_size = self._parse_robotwin_view_size(
            "robotwin_multiview_head_size", robotwin_multiview_head_size
        )
        self.robotwin_multiview_wrist_size = self._parse_robotwin_view_size(
            "robotwin_multiview_wrist_size", robotwin_multiview_wrist_size
        )
        multiview_head_h, multiview_head_w = self.robotwin_multiview_head_size
        multiview_wrist_h, multiview_wrist_w = self.robotwin_multiview_wrist_size
        if self.video_vae_layout == "robotwin_multiview_tokens":
            latent_h, latent_w = self._robotwin_multiview_spatial_size()

        self.robotwin_packed_head_size = self._parse_robotwin_view_size(
            "robotwin_packed_head_size", robotwin_packed_head_size
        )
        self.robotwin_packed_wrist_size = self._parse_robotwin_view_size(
            "robotwin_packed_wrist_size", robotwin_packed_wrist_size
        )
        self.packed_head_reads_wrist0 = bool(packed_head_reads_wrist0)
        self.packed_wrist_encode_separately = bool(packed_wrist_encode_separately)

        self.wrist_attention_gate_ramp_steps = int(wrist_attention_gate_ramp_steps)
        self.wrist_video_loss_ramp_steps = int(wrist_video_loss_ramp_steps)
        self.wrist_video_loss_max_weight = float(wrist_video_loss_max_weight)
        self.head_only_replay_start_ratio = float(head_only_replay_start_ratio)
        self.head_only_replay_end_ratio = float(head_only_replay_end_ratio)
        self.head_only_replay_decay_steps = int(head_only_replay_decay_steps)
        self._training_progress_step = 0
        self._training_progress_total_steps = 0

        # Goal-image supervision (GoalWAM): supervise on goal frames instead of
        # dense future video, while keeping the Wan video base model.
        self.goal_enabled = bool(goal_enabled)
        self.goal_targets = str(goal_targets)
        self.goal_encode_mode = str(goal_encode_mode)
        self.goal_long_rope_mode = str(goal_long_rope_mode)

        # Long-goal loss down-weighting (GoalWAM exp: unequal short/long supervision).
        # Motivation: jointly supervising short+long goals at *equal* weight hurt vs the
        # short-only path, because the harder, farther long goal (`gl`) dilutes the short
        # goal (`gs`) that actually drives the near-term action. `goal_long_loss_weight`
        # scales ONLY the g_long latent's per-latent MSE (g_short/obs keep weight 1.0), so
        # the long goal becomes an auxiliary signal instead of a co-equal target. Default
        # 1.0 -> unchanged. Only applied on the goal path when `goal_targets` includes long.
        goal_long_loss_weight = float(goal_long_loss_weight)
        self.goal_long_loss_weight = goal_long_loss_weight

        # Short-goal RoPE temporal index (GoalWAM exp: short-goal ⊕ dense-video joint).
        #   None    -> naive arange index (g_short at latent index 1 for [obs, gs]).
        #   int (k) -> override the g_short latent's temporal RoPE index to k. Ablation:
        #              the short goal is the clip's *last* frame, which in the dense-video
        #              path lands at latent index 2 (of [obs, f1, f2]); pinning gs to 2
        #              (vs the naive 1) aligns the goal-path and video-path positions for
        #              the same frame, so the shared video expert sees a consistent
        #              temporal coordinate under joint co-training. Only applied on the
        #              goal path when `goal_targets` includes short.
        if goal_short_rope_index is not None:
            goal_short_rope_index = int(goal_short_rope_index)
        self.goal_short_rope_index = goal_short_rope_index

        # Joint video/goal co-training (GoalWAM §3.2): per training step, route the
        # whole batch to either dense-future-video supervision or goal-frame
        # supervision. `goal_mix_video_ratio` is the target fraction of *training*
        # steps routed to dense video; 0.0 -> pure goal (default, unchanged), 1.0 ->
        # pure dense video. Only meaningful when `goal_enabled=True`.
        goal_mix_video_ratio = float(goal_mix_video_ratio)
        self.goal_mix_video_ratio = goal_mix_video_ratio
        # Deterministic dither accumulator: realizes the ratio exactly over time and
        # advances identically across ranks (plain float, not a checkpointed buffer).
        self._mix_acc = 0.0

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        mot_compile_blocks: bool = False,
        mot_compile_dynamic: bool = True,
        mot_compile_mode: str = "default",
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        source_loss_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
        action_component_num_arms: int = 2,
        action_component_loss_weights: Optional[Mapping[str, float]] = None,
        action_component_indices: Optional[Mapping[str, Sequence[int]]] = None,
        goal_enabled: bool = False,
        goal_targets: str = "both",
        goal_encode_mode: str = "pseudo_video",
        goal_long_rope_mode: str = "arange",
        goal_long_loss_weight: float = 1.0,
        goal_short_rope_index: Optional[int] = None,
        goal_mix_video_ratio: float = 0.0,
        vae_encode_batched: bool = True,
        vae_encode_chunk_frames: int = 0,
        vae_compile_encoder: bool = False,
        vae_compile_dynamic: bool = False,
        vae_compile_mode: str = "default",
        vae_enable_channels_last_3d: bool = True,
        video_vae_layout: str = "pixel",
        robotwin_latent_tshape_head_size: Sequence[int] = (256, 320),
        robotwin_latent_tshape_wrist_size: Sequence[int] = (128, 160),
        latent_tshape_head_resize_mode: str = "cover_crop",
        latent_tshape_premerged_input: bool = False,
        latent_tshape_vae_premerge: bool = False,
        latent_tshape_future_wrist_loss_weight: float = 0.5,
        latent_tshape_head_reads_wrist0: bool = True,
        latent_tshape_action_reads_wrist0: bool = True,
        wrist_view_mask_prob: float = 0.0,
        robotwin_multiview_head_size: Sequence[int] = (256, 448),
        robotwin_multiview_wrist_size: Sequence[int] = (128, 224),
        robotwin_packed_head_size: Sequence[int] = (256, 448),
        robotwin_packed_wrist_size: Sequence[int] = (128, 160),
        packed_head_reads_wrist0: bool = False,
        packed_wrist_encode_separately: bool = False,
        wrist_attention_gate_ramp_steps: int = 0,
        wrist_video_loss_ramp_steps: int = 0,
        wrist_video_loss_max_weight: float = 1.0,
        head_only_replay_start_ratio: float = 0.0,
        head_only_replay_end_ratio: float = 0.0,
        head_only_replay_decay_steps: int = 0,
    ):

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            compile_blocks=mot_compile_blocks,
            compile_dynamic=mot_compile_dynamic,
            compile_mode=mot_compile_mode,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            source_loss_weights=source_loss_weights,
            action_component_num_arms=action_component_num_arms,
            action_component_loss_weights=action_component_loss_weights,
            action_component_indices=action_component_indices,
            goal_enabled=goal_enabled,
            goal_targets=goal_targets,
            goal_encode_mode=goal_encode_mode,
            goal_long_rope_mode=goal_long_rope_mode,
            goal_long_loss_weight=goal_long_loss_weight,
            goal_short_rope_index=goal_short_rope_index,
            goal_mix_video_ratio=goal_mix_video_ratio,
            vae_encode_batched=vae_encode_batched,
            vae_encode_chunk_frames=vae_encode_chunk_frames,
            vae_compile_encoder=vae_compile_encoder,
            vae_compile_dynamic=vae_compile_dynamic,
            vae_compile_mode=vae_compile_mode,
            vae_enable_channels_last_3d=vae_enable_channels_last_3d,
            video_vae_layout=video_vae_layout,
            robotwin_latent_tshape_head_size=robotwin_latent_tshape_head_size,
            robotwin_latent_tshape_wrist_size=robotwin_latent_tshape_wrist_size,
            latent_tshape_head_resize_mode=latent_tshape_head_resize_mode,
            latent_tshape_premerged_input=latent_tshape_premerged_input,
            latent_tshape_vae_premerge=latent_tshape_vae_premerge,
            latent_tshape_future_wrist_loss_weight=latent_tshape_future_wrist_loss_weight,
            latent_tshape_head_reads_wrist0=latent_tshape_head_reads_wrist0,
            latent_tshape_action_reads_wrist0=latent_tshape_action_reads_wrist0,
            wrist_view_mask_prob=wrist_view_mask_prob,
            robotwin_multiview_head_size=robotwin_multiview_head_size,
            robotwin_multiview_wrist_size=robotwin_multiview_wrist_size,
            robotwin_packed_head_size=robotwin_packed_head_size,
            robotwin_packed_wrist_size=robotwin_packed_wrist_size,
            packed_head_reads_wrist0=packed_head_reads_wrist0,
            packed_wrist_encode_separately=packed_wrist_encode_separately,
            wrist_attention_gate_ramp_steps=wrist_attention_gate_ramp_steps,
            wrist_video_loss_ramp_steps=wrist_video_loss_ramp_steps,
            wrist_video_loss_max_weight=wrist_video_loss_max_weight,
            head_only_replay_start_ratio=head_only_replay_start_ratio,
            head_only_replay_end_ratio=head_only_replay_end_ratio,
            head_only_replay_decay_steps=head_only_replay_decay_steps,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    @torch.no_grad()
    def initialize_source_proprio_encoders_from_default(self) -> None:
        """Clone the default state encoder into every additional source encoder."""
        if self.proprio_encoder is None:
            return
        for source in self.source_proprio_encoders:
            self.source_proprio_encoders[source].load_state_dict(
                self.proprio_encoder.state_dict(), strict=True
            )

    def _proprio_encoder_for_source(
        self, action_tokenizer_source: Optional[str]
    ) -> nn.Linear:
        if action_tokenizer_source is None:
            return self.proprio_encoder
        source = str(action_tokenizer_source)
        if source == self.action_expert.default_action_tokenizer_source:
            return self.proprio_encoder
        return self.source_proprio_encoders[source]

    def _encode_proprio_tokens(
        self,
        proprio: torch.Tensor,
        action_tokenizer_source=None,
    ) -> torch.Tensor:
        route = self.action_expert._normalize_action_tokenizer_route(
            action_tokenizer_source, proprio.shape[0]
        )
        unique_routes = tuple(dict.fromkeys(route))
        proprio = proprio.to(device=self.device)
        if len(unique_routes) == 1:
            encoder = self._proprio_encoder_for_source(unique_routes[0])
            return encoder(proprio.unsqueeze(1))

        encoded = proprio.new_empty(proprio.shape[0], 1, self.text_dim)
        for source in unique_routes:
            indices = torch.tensor(
                [i for i, value in enumerate(route) if value == source],
                dtype=torch.long,
                device=proprio.device,
            )
            encoder = self._proprio_encoder_for_source(source)
            source_tokens = encoder(proprio.index_select(0, indices).unsqueeze(1))
            encoded.index_copy_(0, indices, source_tokens)
        return encoded

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
        action_tokenizer_source=None,
        proprio_dim_is_pad=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio_dim_is_pad is not None:
            pad = torch.as_tensor(proprio_dim_is_pad, device=proprio.device, dtype=torch.bool)
            if pad.ndim == 1 and proprio.shape[0] == 1:
                pad = pad.unsqueeze(0)
            proprio = proprio.masked_fill(pad, 0)
        proprio_token = self._encode_proprio_tokens(
            proprio.to(device=self.device, dtype=context.dtype),
            action_tokenizer_source=action_tokenizer_source,
        ).to(dtype=context.dtype)  # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
            batched=self.vae_encode_batched,
            chunk_frames=self.vae_encode_chunk_frames,
        )
        return z

    @staticmethod
    def _resize_video_view(video: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Resize [B,C,T,H,W] without mixing camera or temporal dimensions."""
        batch_size, channels, num_frames, height, width = video.shape
        if (height, width) == tuple(size):
            return video
        frames = video.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        # The CUDA path uses antialiased bf16 interpolation directly, avoiding a
        # large fp32 staging tensor at training batch size. CPU bf16 does not, so
        # retain a portable fp32 fallback for tests/debug inference.
        resize_input = frames if frames.is_cuda else frames.float()
        resized = F.interpolate(
            resize_input,
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).to(dtype=video.dtype)
        return resized.view(batch_size, num_frames, channels, *size).permute(
            0, 2, 1, 3, 4
        ).contiguous()

    def _prepare_robotwin_views(
        self, video: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.latent_tshape_premerged_input:
            packed = self.video_vae_layout == "robotwin_packed_multiview"
            head_h, head_w = (
                self.robotwin_packed_head_size if packed
                else self.robotwin_latent_tshape_head_size
            )
            wrist_h, wrist_w = (
                self.robotwin_packed_wrist_size if packed
                else self.robotwin_latent_tshape_wrist_size
            )
            return (
                video[..., :head_h, :], video[..., head_h:, :wrist_w],
                video[..., head_h:, wrist_w:],
            )
        resize_view = (
            self._resize_video_view_cover_crop
            if self.latent_tshape_head_resize_mode == "cover_crop"
            else self._resize_video_view
        )
        # The aligned post-training contract applies one aspect policy to all three
        # cameras: Head -> 256x448 and each Wrist -> 128x224. Views remain separate
        # until after the frozen VAE encode.
        head = resize_view(video[:, 0], self.robotwin_latent_tshape_head_size)
        left = resize_view(video[:, 1], self.robotwin_latent_tshape_wrist_size)
        right = resize_view(video[:, 2], self.robotwin_latent_tshape_wrist_size)
        return head, left, right

    def _compose_robotwin_pixel_tshape(self, video: torch.Tensor) -> torch.Tensor:
        """Create a visualization-only pixel T-shape from separate camera tensors."""
        if self.latent_tshape_premerged_input:
            self._prepare_robotwin_views(video.unsqueeze(0) if video.ndim == 4 else video)
            return video
        unbatched = video.ndim == 5
        if unbatched:
            video = video.unsqueeze(0)
        head, left, right = self._prepare_robotwin_views(video)
        bottom = torch.cat([left, right], dim=-1)
        composed = torch.cat([head, bottom], dim=-2)
        return composed[0] if unbatched else composed

    def _robotwin_latent_tshape_spatial_size(self) -> tuple[int, int]:
        factor = int(self.vae.upsampling_factor)
        head_h, head_w = self.robotwin_latent_tshape_head_size
        wrist_h, wrist_w = self.robotwin_latent_tshape_wrist_size
        return head_h // factor + wrist_h // factor, head_w // factor

    def _robotwin_latent_tshape_patch_layout(
        self, device: torch.device
    ) -> dict[str, torch.Tensor | int]:
        """Describe physical Head/Left/Right regions in one latent-T frame."""
        factor = int(self.vae.upsampling_factor)
        patch_h, patch_w = (int(value) for value in self.video_expert.patch_size[-2:])
        head_patch_h = self.robotwin_latent_tshape_head_size[0] // factor // patch_h
        head_patch_w = self.robotwin_latent_tshape_head_size[1] // factor // patch_w
        wrist_patch_h = self.robotwin_latent_tshape_wrist_size[0] // factor // patch_h
        wrist_patch_w = self.robotwin_latent_tshape_wrist_size[1] // factor // patch_w
        grid_h = head_patch_h + wrist_patch_h
        grid_w = head_patch_w
        view_ids = torch.zeros((grid_h, grid_w), dtype=torch.long, device=device)
        view_ids[head_patch_h:, :wrist_patch_w] = 1
        view_ids[head_patch_h:, wrist_patch_w:] = 2
        flat_view_ids = view_ids.reshape(-1)
        head_indices = torch.nonzero(flat_view_ids == 0, as_tuple=False).flatten()
        wrist_indices = torch.nonzero(flat_view_ids > 0, as_tuple=False).flatten()
        return {
            "view_ids": view_ids,
            "head_indices_per_frame": head_indices,
            "wrist_indices_per_frame": wrist_indices,
            "head_tokens_per_frame": int(head_indices.numel()),
            "wrist_tokens_per_frame": int(wrist_indices.numel()),
            "left_wrist_tokens_per_frame": wrist_patch_h * wrist_patch_w,
            "right_wrist_tokens_per_frame": wrist_patch_h * (grid_w - wrist_patch_w),
        }

    def set_training_progress(self, step: int, total_steps: int) -> None:
        """Expose optimizer-step progress to deterministic bridge schedules."""
        step = int(step)
        total_steps = int(total_steps)
        self._training_progress_step = step
        self._training_progress_total_steps = total_steps

    @staticmethod
    def _cosine_ramp(step: int, ramp_steps: int) -> float:
        if ramp_steps <= 0:
            return 1.0
        progress = min(max(float(step) / float(ramp_steps), 0.0), 1.0)
        return 0.5 - 0.5 * math.cos(math.pi * progress)

    def _head_only_replay_ratio(self, step: int) -> float:
        if self.head_only_replay_decay_steps <= 0:
            return self.head_only_replay_end_ratio
        progress = min(
            max(float(step) / float(self.head_only_replay_decay_steps), 0.0),
            1.0,
        )
        return (
            self.head_only_replay_start_ratio
            + progress
            * (
                self.head_only_replay_end_ratio
                - self.head_only_replay_start_ratio
            )
        )

    def _current_multiview_bridge_state(self) -> dict[str, float | bool]:
        if self.video_vae_layout not in {
            "robotwin_latent_tshape",
            "robotwin_multiview_tokens",
        }:
            return {
                "wrist_attention_gate": 1.0,
                "wrist_video_loss_weight": 1.0,
                "head_only_replay": False,
                "head_only_replay_ratio": 0.0,
            }
        max_wrist_loss_weight = (
            self.latent_tshape_future_wrist_loss_weight
            if self.video_vae_layout == "robotwin_latent_tshape"
            else self.wrist_video_loss_max_weight
        )
        if not self.dit.training or not torch.is_grad_enabled():
            return {
                "wrist_attention_gate": 1.0,
                "wrist_video_loss_weight": max_wrist_loss_weight,
                "head_only_replay": False,
                "head_only_replay_ratio": 0.0,
            }
        step = int(self._training_progress_step)
        replay_ratio = (
            self._head_only_replay_ratio(step)
            if self.video_vae_layout == "robotwin_multiview_tokens"
            else 0.0
        )
        # Deterministic low-discrepancy routing.  Every distributed rank makes
        # the same whole-batch decision without consuming the global RNG stream.
        replay_draw = ((step + 1) * 0.6180339887498949) % 1.0
        head_only = replay_draw < replay_ratio
        gate = self._cosine_ramp(step, self.wrist_attention_gate_ramp_steps)
        wrist_loss_weight = max_wrist_loss_weight * self._cosine_ramp(
            step, self.wrist_video_loss_ramp_steps
        )
        if head_only:
            gate = 0.0
            wrist_loss_weight = 0.0
        return {
            "wrist_attention_gate": float(gate),
            "wrist_video_loss_weight": float(wrist_loss_weight),
            "head_only_replay": bool(head_only),
            "head_only_replay_ratio": float(replay_ratio),
        }

    def _prepare_robotwin_multiview_views(
        self, video: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        head = self._resize_video_view_cover_crop(
            video[:, 0], self.robotwin_multiview_head_size
        )
        left = self._resize_video_view(video[:, 1], self.robotwin_multiview_wrist_size)
        right = self._resize_video_view(video[:, 2], self.robotwin_multiview_wrist_size)
        return head, left, right

    @staticmethod
    def _resize_video_view_cover_crop(
        video: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        """Aspect-preserving cover resize + center crop for [B,C,T,H,W]."""
        batch_size, channels, num_frames, height, width = video.shape
        target_h, target_w = size
        scale = max(float(target_h) / float(height), float(target_w) / float(width))
        resized_h = max(int(math.ceil(height * scale)), target_h)
        resized_w = max(int(math.ceil(width * scale)), target_w)
        frames = video.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        if (resized_h, resized_w) == (height, width):
            # H448 staging already is the exact bicubic 336x448 tensor produced
            # by the Head-only processor.  Skipping a same-size interpolation
            # makes the top Head crop bit-identical to that baseline.
            resized = frames
        else:
            resize_input = frames if frames.is_cuda else frames.float()
            resized = F.interpolate(
                resize_input,
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).to(dtype=video.dtype)
        top = (resized_h - target_h) // 2
        left = (resized_w - target_w) // 2
        cropped = resized[..., top : top + target_h, left : left + target_w]
        return cropped.view(
            batch_size, num_frames, channels, target_h, target_w
        ).permute(0, 2, 1, 3, 4).contiguous()

    def _compose_robotwin_pixel_multiview(self, video: torch.Tensor) -> torch.Tensor:
        """Visualization canvas; DiT uses explicit view IDs and local RoPE."""
        unbatched = video.ndim == 5
        if unbatched:
            video = video.unsqueeze(0)
        head, left, right = self._prepare_robotwin_multiview_views(video)
        composed = torch.cat([head, torch.cat([left, right], dim=-1)], dim=-2)
        return composed[0] if unbatched else composed

    def _robotwin_multiview_spatial_size(self) -> tuple[int, int]:
        factor = int(self.vae.upsampling_factor)
        head_h, head_w = self.robotwin_multiview_head_size
        wrist_h, wrist_w = self.robotwin_multiview_wrist_size
        return head_h // factor + wrist_h // factor, head_w // factor

    @torch.no_grad()
    def _encode_robotwin_multiview_tokens(
        self, video: torch.Tensor, tiled: bool = False
    ) -> torch.Tensor:
        head, left, right = self._prepare_robotwin_multiview_views(video)
        batch_size = int(video.shape[0])
        head_latents = self._encode_video_latents(head, tiled=tiled)
        wrist_latents = self._encode_video_latents(
            torch.cat([left, right], dim=0), tiled=tiled
        )
        left_latents, right_latents = wrist_latents.split(batch_size, dim=0)
        latents = torch.cat(
            [head_latents, torch.cat([left_latents, right_latents], dim=-1)],
            dim=-2,
        )
        return latents

    def _robotwin_multiview_patch_layout(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        factor = int(self.vae.upsampling_factor)
        patch_h, patch_w = (int(value) for value in self.video_expert.patch_size[-2:])
        head_patch_h = self.robotwin_multiview_head_size[0] // factor // patch_h
        head_patch_w = self.robotwin_multiview_head_size[1] // factor // patch_w
        wrist_patch_h = self.robotwin_multiview_wrist_size[0] // factor // patch_h
        wrist_patch_w = self.robotwin_multiview_wrist_size[1] // factor // patch_w
        grid_h = head_patch_h + wrist_patch_h
        grid_w = head_patch_w
        view_ids = torch.zeros((grid_h, grid_w), dtype=torch.long, device=device)
        view_ids[head_patch_h:, :wrist_patch_w] = 1
        view_ids[head_patch_h:, wrist_patch_w:] = 2
        row_ids = torch.arange(head_patch_h, device=device).view(-1, 1).expand(
            head_patch_h, grid_w
        )
        wrist_rows = torch.arange(wrist_patch_h, device=device).view(-1, 1).expand(
            wrist_patch_h, grid_w
        )
        row_ids = torch.cat([row_ids, wrist_rows], dim=0)
        head_cols = torch.arange(head_patch_w, device=device).view(1, -1).expand(
            head_patch_h, head_patch_w
        )
        wrist_cols = torch.arange(wrist_patch_w, device=device)
        wrist_cols = torch.cat([wrist_cols, wrist_cols], dim=0).view(1, -1).expand(
            wrist_patch_h, grid_w
        )
        col_ids = torch.cat([head_cols, wrist_cols], dim=0)
        return view_ids, row_ids, col_ids

    def _prepare_robotwin_packed_views(
        self, video: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the separately encoded Head stream and one concatenated Wrist strip."""
        if self.latent_tshape_premerged_input:
            head, left, right = self._prepare_robotwin_views(video)
            return head, torch.cat([left[:, :, :1], right[:, :, :1]], dim=-1)
        head = self._resize_video_view_cover_crop(
            video[:, 0], self.robotwin_packed_head_size
        )
        left = self._resize_video_view(video[:, 1], self.robotwin_packed_wrist_size)
        right = self._resize_video_view(video[:, 2], self.robotwin_packed_wrist_size)
        wrist_strip = torch.cat([left, right], dim=-1)
        return head, wrist_strip

    @torch.no_grad()
    def _encode_robotwin_packed_multiview(
        self, video: torch.Tensor, tiled: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode dense Head video plus the current concatenated Wrist observation.

        The Wrist strip is encoded from the first raw frame only. This both enforces
        observation-only conditioning and prevents future wrist pixels from entering
        the forward path before the future-wrist experiment is enabled.
        """
        head, wrist_strip = self._prepare_robotwin_packed_views(video)
        head_latents = self._encode_video_latents(head, tiled=tiled)
        wrist0 = wrist_strip[:, :, :1]
        if self.packed_wrist_encode_separately:
            # Match the future treatment's per-camera VAE boundary. Pack only
            # after encoding, and never create W1+ latents or tokens.
            left, right = wrist0.chunk(2, dim=-1)
            wrists = self._encode_video_latents(torch.cat([left, right], dim=0), tiled=tiled)
            left_latents, right_latents = wrists.chunk(2, dim=0)
            wrist_latents = torch.cat([left_latents, right_latents], dim=-1)
        else:
            wrist_latents = self._encode_video_latents(wrist0, tiled=tiled)
        return head_latents, wrist_latents

    def _robotwin_packed_spatial_layout(
        self, device: torch.device
    ) -> dict[str, torch.Tensor | int]:
        """Return R1 virtual-global coordinates for real packed tokens only."""
        factor = int(self.vae.upsampling_factor)
        patch_h, patch_w = (int(value) for value in self.video_expert.patch_size[-2:])
        head_grid_h = self.robotwin_packed_head_size[0] // factor // patch_h
        head_grid_w = self.robotwin_packed_head_size[1] // factor // patch_w
        wrist_grid_h = self.robotwin_packed_wrist_size[0] // factor // patch_h
        wrist_grid_w = (
            2 * self.robotwin_packed_wrist_size[1] // factor // patch_w
        )
        wrist_col_offset = (head_grid_w - wrist_grid_w) // 2
        head_rows = torch.arange(head_grid_h, device=device).view(-1, 1).expand(
            head_grid_h, head_grid_w
        )
        head_cols = torch.arange(head_grid_w, device=device).view(1, -1).expand(
            head_grid_h, head_grid_w
        )
        wrist_rows = (
            torch.arange(wrist_grid_h, device=device).view(-1, 1)
            + head_grid_h
        ).expand(wrist_grid_h, wrist_grid_w)
        wrist_cols = (
            torch.arange(wrist_grid_w, device=device).view(1, -1)
            + wrist_col_offset
        ).expand(wrist_grid_h, wrist_grid_w)
        return {
            "head_rows": head_rows,
            "head_cols": head_cols,
            "wrist_rows": wrist_rows,
            "wrist_cols": wrist_cols,
            "head_tokens_per_frame": head_grid_h * head_grid_w,
            "wrist_tokens": wrist_grid_h * wrist_grid_w,
        }

    def _sample_wrist_view_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Training keeps the top-level FastWAM module in eval mode and puts only
        # the trainable DiT in train mode. Key the augmentation off that component
        # so it is active during post-training and disabled during policy eval.
        if not self.dit.training or self.wrist_view_mask_prob <= 0.0:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        if self.wrist_view_mask_prob >= 1.0:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        return torch.rand(batch_size, device=device) < self.wrist_view_mask_prob

    @torch.no_grad()
    def _encode_robotwin_latent_tshape(
        self,
        video: torch.Tensor,
        tiled: bool = False,
        apply_training_mask: bool = True,
        wrist_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a T canvas in pixel space or assemble independently encoded views."""
        head, left, right = self._prepare_robotwin_views(video)
        batch_size = int(video.shape[0])

        if self.latent_tshape_vae_premerge:
            # Encode the full pixel canvas once, including across view boundaries.
            canvas = video if self.latent_tshape_premerged_input else torch.cat(
                [head, torch.cat([left, right], dim=-1)], dim=-2
            )
            merged = self._encode_video_latents(canvas, tiled=tiled)
            factor = int(self.vae.upsampling_factor)
            head_h = self.robotwin_latent_tshape_head_size[0] // factor
            wrist_w = self.robotwin_latent_tshape_wrist_size[1] // factor
            head_latents = merged[..., :head_h, :]
            left_latents = merged[..., head_h:, :wrist_w]
            right_latents = merged[..., head_h:, wrist_w:]
        else:
            head_latents = self._encode_video_latents(head, tiled=tiled)
            # Batch the independent wrist views to avoid a third encoder pass.
            wrist_latents = self._encode_video_latents(
                torch.cat([left, right], dim=0), tiled=tiled
            )
            left_latents, right_latents = wrist_latents.split(batch_size, dim=0)

        if wrist_mask is None:
            wrist_mask = (
                self._sample_wrist_view_mask(batch_size, head_latents.device)
                if apply_training_mask
                else torch.zeros(batch_size, dtype=torch.bool, device=head_latents.device)
            )
        else:
            wrist_mask = torch.as_tensor(
                wrist_mask, device=head_latents.device, dtype=torch.bool
            ).reshape(-1)
        # One sample-level decision covers both wrist views and every latent time.
        # Thus the observed first latent and all future targets have the same views.
        keep_wrist = (~wrist_mask).view(batch_size, 1, 1, 1, 1)
        left_latents = left_latents * keep_wrist
        right_latents = right_latents * keep_wrist

        bottom = torch.cat([left_latents, right_latents], dim=-1)
        latent_tshape = torch.cat([head_latents, bottom], dim=-2)
        return latent_tshape, wrist_mask

    def _build_goal_latents(self, obs, g_short, g_long, tiled: bool = False):
        """Build video latents from goal frames (GoalWAM) instead of dense future video.

        obs/g_short/g_long are [B, 3, H, W] in [-1, 1]. Returns [B, C, n_lat, h, w] where
        n_lat = 1 (obs) + number of supervised goal latents (1 for short|long, 2 for both).
        The Wan causal VAE maps a T-frame clip to (T-1)//4 + 1 latents (frame0 -> lat0,
        frames1-4 -> lat1, ...), so repeat-padding a goal frame x4 yields one latent for it.
        """
        targets = self.goal_targets
        mode = self.goal_encode_mode
        futures = []
        if targets in ("short", "both"):
            futures.append(g_short)
        if targets in ("long", "both"):
            futures.append(g_long)

        if mode == "pseudo_video":
            # [obs, f0 x4, f1 x4, ...] -> one VAE pass -> 1 + len(futures) latents.
            seq = [obs]
            for f in futures:
                seq.extend([f] * self.vae.temporal_downsample_factor)
            video = torch.stack(seq, dim=2)  # [B, 3, T, H, W]
            return self._encode_video_latents(video, tiled=tiled)

        if mode == "single_frame":
            # Each frame encoded independently as a 1-frame causal latent, then concat.
            lats = [
                self._encode_video_latents(x.unsqueeze(2), tiled=tiled)
                for x in ([obs] + futures)
            ]
            return torch.cat(lats, dim=2)

        if mode == "hybrid":
            # obs + first future as a repeat-padded pseudo-video (2 latents); any
            # remaining futures encoded standalone (1 latent each).
            head = torch.stack([obs] + [futures[0]] * self.vae.temporal_downsample_factor, dim=2)
            lats = [self._encode_video_latents(head, tiled=tiled)]
            for f in futures[1:]:
                lats.append(self._encode_video_latents(f.unsqueeze(2), tiled=tiled))
            return torch.cat(lats, dim=2)

    def _route_use_video_path(self) -> bool:
        """Joint video/goal co-training router (GoalWAM §3.2).

        Returns True when this step should supervise dense future video instead of
        goal frames. `goal_mix_video_ratio` r in [0, 1] is the target fraction of
        *training* steps routed to dense video (r=0 -> pure goal, the default;
        r=1 -> pure dense video). A deterministic dither accumulator realizes the
        ratio exactly over time and advances identically across ranks (every rank
        steps in lockstep), so the per-step graph structure stays aligned across the
        process group.

        During eval/inference (no grad) the accumulator is left untouched: the goal
        path is used for r<1 (the novel target) and the video path for r==1, so
        validation is deterministic and reproducible.
        """
        r = self.goal_mix_video_ratio
        if r <= 0.0:
            return False
        if r >= 1.0:
            return True
        if not torch.is_grad_enabled():
            return False
        self._mix_acc += r
        if self._mix_acc >= 1.0:
            self._mix_acc -= 1.0
            return True
        return False

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if self.latent_tshape_premerged_input:
            image, _, _ = self._prepare_inference_input_image(input_image)
            video = image.to(device=self.device).unsqueeze(2)
            if self.video_vae_layout == "robotwin_packed_multiview":
                head, _ = self._encode_robotwin_packed_multiview(video, tiled=tiled)
                return head
            z, _ = self._encode_robotwin_latent_tshape(
                video, tiled=tiled, apply_training_mask=False,
            )
            return z
        if self.video_vae_layout in {
            "robotwin_latent_tshape",
            "robotwin_multiview_tokens",
        }:
            if input_image.ndim == 4:
                input_image = input_image.unsqueeze(0)
            video = input_image.to(device=self.device).unsqueeze(3)
            if self.video_vae_layout == "robotwin_latent_tshape":
                z, _ = self._encode_robotwin_latent_tshape(
                    video,
                    tiled=tiled,
                    apply_training_mask=False,
                )
                return z
            return self._encode_robotwin_multiview_tokens(video, tiled=tiled)
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents_tensor(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        return self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    def _decode_layout_latents_tensor(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if self.video_vae_layout not in {
            "robotwin_latent_tshape",
            "robotwin_multiview_tokens",
        }:
            return self._decode_latents_tensor(
                latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            )
        if self.video_vae_layout == "robotwin_latent_tshape" and self.latent_tshape_vae_premerge:
            return self._decode_latents_tensor(
                latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            )
        is_multiview = self.video_vae_layout == "robotwin_multiview_tokens"
        head_size = (
            self.robotwin_multiview_head_size
            if is_multiview
            else self.robotwin_latent_tshape_head_size
        )
        wrist_size = (
            self.robotwin_multiview_wrist_size
            if is_multiview
            else self.robotwin_latent_tshape_wrist_size
        )
        head_latent_h = head_size[0] // int(self.vae.upsampling_factor)
        wrist_latent_w = wrist_size[1] // int(self.vae.upsampling_factor)
        head_latents = latents[:, :, :, :head_latent_h, :]
        bottom_latents = latents[:, :, :, head_latent_h:, :]
        left_latents = bottom_latents[:, :, :, :, :wrist_latent_w]
        right_latents = bottom_latents[:, :, :, :, wrist_latent_w:]
        head = self._decode_latents_tensor(
            head_latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        left = self._decode_latents_tensor(
            left_latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        right = self._decode_latents_tensor(
            right_latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        return torch.cat([head, torch.cat([left, right], dim=-1)], dim=-2)

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self._decode_layout_latents_tensor(
            latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def _prepare_inference_input_image(
        self, input_image: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        """Validate inference pixels and return the final latent H/W contract."""
        if self.latent_tshape_premerged_input:
            if input_image.ndim == 3:
                input_image = input_image.unsqueeze(0)
            self._prepare_robotwin_views(input_image.unsqueeze(2))
            if self.video_vae_layout == "robotwin_packed_multiview":
                factor = int(self.vae.upsampling_factor)
                latent_h, latent_w = (s // factor for s in self.robotwin_packed_head_size)
            else:
                latent_h, latent_w = self._robotwin_latent_tshape_spatial_size()
            return input_image, latent_h, latent_w
        if self.video_vae_layout in {
            "robotwin_latent_tshape",
            "robotwin_multiview_tokens",
            "robotwin_packed_multiview",
        }:
            if input_image.ndim == 4:
                input_image = input_image.unsqueeze(0)
            if self.video_vae_layout == "robotwin_multiview_tokens":
                latent_h, latent_w = self._robotwin_multiview_spatial_size()
            elif self.video_vae_layout == "robotwin_packed_multiview":
                factor = int(self.vae.upsampling_factor)
                latent_h, latent_w = (
                    self.robotwin_packed_head_size[0] // factor,
                    self.robotwin_packed_head_size[1] // factor,
                )
            else:
                latent_h, latent_w = self._robotwin_latent_tshape_spatial_size()
            return input_image, latent_h, latent_w

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        height, width = (int(input_image.shape[-2]), int(input_image.shape[-1]))
        return (
            input_image,
            height // int(self.vae.upsampling_factor),
            width // int(self.vae.upsampling_factor),
        )

    def build_inputs(self, sample, tiled: bool = False):
        video = sample.get("video")
        precomputed_video_latents = sample.get("video_latents")
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if precomputed_video_latents is not None:
            batch_size = int(precomputed_video_latents.shape[0])
            temporal_factor = int(self.vae.temporal_downsample_factor)
            num_frames = (
                (int(precomputed_video_latents.shape[2]) - 1) * temporal_factor + 1
            )
        elif self.latent_tshape_premerged_input:
            batch_size, _, num_frames, _, _ = video.shape
        elif self.video_vae_layout in {
            "robotwin_latent_tshape",
            "robotwin_multiview_tokens",
            "robotwin_packed_multiview",
        }:
            batch_size, _, _, num_frames, _, _ = video.shape
        else:
            batch_size, _, num_frames, height, width = video.shape

        action = sample["action"]

        action_is_pad = sample.get("action_is_pad", None)

        image_is_pad = sample.get("image_is_pad", None)

        # uint8-on-GPU: dataset may ship video as uint8 [0,255] (4x smaller IPC);
        # decode to float [-1,1] on the GPU here. float input is passed through.
        input_video = None
        if video is not None:
            if video.dtype == torch.uint8:
                input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True).div_(127.5).sub_(1.0)
            else:
                input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        # Joint video/goal co-training: when goal supervision is on, each step is
        # routed (per whole batch) to either the dense-video path or the goal path
        # according to `goal_mix_video_ratio`. With ratio 0 this is pure goal
        # supervision; with goal disabled it is the dense-video baseline.
        route_video = (not self.goal_enabled) or self._route_use_video_path()
        multiview_bridge_state = self._current_multiview_bridge_state()
        packed_wrist_latents = None
        if precomputed_video_latents is not None:
            input_latents = precomputed_video_latents.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            wrist_view_mask = None
        elif route_video:
            if self.video_vae_layout == "robotwin_latent_tshape":
                input_latents, wrist_view_mask = self._encode_robotwin_latent_tshape(
                    input_video,
                    tiled=tiled,
                    apply_training_mask=True,
                )
            elif self.video_vae_layout == "robotwin_multiview_tokens":
                input_latents = self._encode_robotwin_multiview_tokens(
                    input_video, tiled=tiled
                )
                wrist_view_mask = None
            elif self.video_vae_layout == "robotwin_packed_multiview":
                input_latents, packed_wrist_latents = (
                    self._encode_robotwin_packed_multiview(
                        input_video, tiled=tiled
                    )
                )
                wrist_view_mask = None
            else:
                input_latents = self._encode_video_latents(input_video, tiled=tiled)
                wrist_view_mask = None
        else:
            # GoalWAM: supervise goal frames instead of dense future video.
            # obs / g_short come from the clip's first / last frame; g_long is the
            # trajectory terminal frame the dataset attaches as `goal_image`.
            goal_image = sample.get("goal_image", None)
            goal_image = goal_image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            obs = input_video[:, :, 0]
            g_short = input_video[:, :, -1]
            input_latents = self._build_goal_latents(obs, g_short, goal_image, tiled=tiled)
            wrist_view_mask = None
            # Goal frames are always valid -> no per-latent padding; skip the
            # image_is_pad alignment (its 9-frame layout no longer matches).
            image_is_pad = None

        # GoalWAM per-sample temporal RoPE indices for the goal latents. Two overrides
        # can stack (both only on the goal path); `None` -> pre_dit uses arange freqs.
        #   * short-goal index (goal_short_rope_index): override g_short's latent index
        #     (index 1 in the [obs, gs, ...] layout) from the naive arange value. Used to
        #     align the short goal with the dense-video final-frame position under joint
        #     video/goal co-training. Applies whenever `goal_targets` includes short.
        #   * long-goal distance (goal_long_rope_mode='distance', exp5): override the last
        #     latent (g_long) with the dataset-provided distance-to-trajectory-end index.
        #     Applies whenever `goal_targets` includes long.
        goal_temporal_index = None
        if self.goal_enabled and not route_video:
            n_lat = input_latents.shape[2]
            ti = None

            want_short = (
                self.goal_short_rope_index is not None
                and self.goal_targets in ("short", "both")
            )
            want_long = (
                self.goal_long_rope_mode == "distance"
                and self.goal_targets in ("long", "both")
            )

            if want_short:
                ti = torch.arange(n_lat, device=self.device).unsqueeze(0).repeat(batch_size, 1)  # [B, f]
                # g_short is the first goal latent, right after obs -> latent index 1.
                if n_lat >= 2:
                    ti[:, 1] = int(self.goal_short_rope_index)

            if want_long:
                gli = sample.get("goal_long_rope_index", None)
                if ti is None:
                    ti = torch.arange(n_lat, device=self.device).unsqueeze(0).repeat(batch_size, 1)  # [B, f]
                gli = gli.to(device=self.device).long().view(-1)  # [B]
                ti[:, -1] = gli.clamp_(min=0)  # g_long is the last latent

            goal_temporal_index = ti

        # GoalWAM: down-weight the g_long latent's video loss so short+long goals are
        # supervised unequally (g_long as auxiliary). Flagged here; the per-latent-step
        # weight tensor is built in `training_loss` once the loss step count is known.
        apply_goal_long_weight = bool(
            self.goal_enabled
            and not route_video
            and self.goal_targets in ("long", "both")
            and self.goal_long_loss_weight != 1.0
        )

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            action_tokenizer_source = self._batch_action_tokenizer_source(
                sample, batch_size
            )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
                action_tokenizer_source=(
                    action_tokenizer_source
                    if self.action_expert.action_tokenizer_sources
                    else None
                ),
                proprio_dim_is_pad=sample.get('proprio_dim_is_pad'),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        camera_pose_delta = None
        camera_pose_valid = None
        if getattr(self.video_expert, "camera_pose_conditioning", False):
            camera_pose_delta = sample.get("camera_pose_delta")
            camera_pose_valid = sample.get("camera_pose_valid")
            camera_pose_delta = camera_pose_delta.to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            camera_pose_valid = camera_pose_valid.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )

        # per-hand supervision: optional per-channel action validity mask
        # Static [B,D] or per-step [B,T,D] validity for partially observed actions.
        action_dim_valid = sample.get("action_dim_valid", None)
        if action_dim_valid is not None:
            action_dim_valid = action_dim_valid.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "packed_wrist_latents": packed_wrist_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_valid": action_dim_valid,
            "image_is_pad": image_is_pad,
            "camera_pose_delta": camera_pose_delta,
            "camera_pose_valid": camera_pose_valid,
            "route_video": bool(route_video),
            "goal_temporal_index": goal_temporal_index,
            "apply_goal_long_weight": apply_goal_long_weight,
            "wrist_view_mask": wrist_view_mask,
            "wrist_attention_gate": float(
                multiview_bridge_state["wrist_attention_gate"]
            ),
            "wrist_video_loss_weight": float(
                multiview_bridge_state["wrist_video_loss_weight"]
            ),
            "head_only_replay": bool(multiview_bridge_state["head_only_replay"]),
            "head_only_replay_ratio": float(
                multiview_bridge_state["head_only_replay_ratio"]
            ),
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        video_key_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        if video_key_gate is None:
            return mask
        video_key_gate = torch.as_tensor(
            video_key_gate, device=device, dtype=torch.float32
        ).reshape(-1)
        if bool(torch.all(video_key_gate >= 1.0).item()):
            return mask
        additive = torch.zeros(mask.shape, dtype=torch.float32, device=device)
        additive.masked_fill_(~mask, float("-inf"))
        log_gate = torch.where(
            video_key_gate > 0.0,
            video_key_gate.log(),
            torch.full_like(video_key_gate, float("-inf")),
        )
        additive[:, :video_seq_len] += log_gate.unsqueeze(0)
        return additive

    @torch.no_grad()
    def _build_robotwin_latent_tshape_mot_attention_mask(
        self,
        video_pre: dict[str, Any],
        action_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the matched dense-T routing mask for [H0..Hn, W0..Wn, A].

        The dense latent canvas intentionally keeps every wrist token and therefore
        keeps token count/FLOPs identical across routing ablations.  Only W0 is an
        observation key for Head/Action queries; W1..Wn remain video targets and can
        be used by Wrist queries, but can never leak into either Head or Action.
        """
        meta = video_pre["meta"]
        num_frames, grid_h, grid_w = (
            int(value) for value in meta["grid_size"]
        )
        tokens_per_frame = int(meta["tokens_per_frame"])
        view_ids = torch.as_tensor(
            meta.get("latent_tshape_view_ids"), device=device, dtype=torch.long
        )
        per_frame_is_wrist = view_ids.reshape(-1) > 0

        video_seq_len = num_frames * tokens_per_frame
        total_seq_len = video_seq_len + int(action_seq_len)
        mask = torch.zeros(
            (total_seq_len, total_seq_len), dtype=torch.bool, device=device
        )
        mask[:video_seq_len, :video_seq_len] = (
            self.video_expert.build_video_to_video_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=tokens_per_frame,
                device=device,
            )
        )
        action_rows = torch.arange(video_seq_len, total_seq_len, device=device)
        mask[action_rows[:, None], action_rows[None, :]] = True

        frame_ids = torch.arange(num_frames, device=device).repeat_interleave(
            tokens_per_frame
        )
        token_is_wrist = per_frame_is_wrist.repeat(num_frames)
        token_is_head = ~token_is_wrist
        head_rows = torch.nonzero(token_is_head, as_tuple=False).flatten()
        h0_keys = torch.nonzero(
            token_is_head & (frame_ids == 0), as_tuple=False
        ).flatten()
        w0_keys = torch.nonzero(
            token_is_wrist & (frame_ids == 0), as_tuple=False
        ).flatten()
        future_wrist_keys = torch.nonzero(
            token_is_wrist & (frame_ids > 0), as_tuple=False
        ).flatten()

        # Preserve the policy contract: Action sees current Head and, optionally,
        # current Wrist. It never reads future video of either view.
        mask[action_rows[:, None], h0_keys[None, :]] = True
        if self.latent_tshape_action_reads_wrist0:
            mask[action_rows[:, None], w0_keys[None, :]] = True

        # Regardless of routing-arm choice, future wrist targets are never keys for
        # Head or Action. This prevents indirect future-view supervision/leakage.
        if future_wrist_keys.numel() > 0:
            mask[head_rows[:, None], future_wrist_keys[None, :]] = False
            mask[action_rows[:, None], future_wrist_keys[None, :]] = False
        if not self.latent_tshape_head_reads_wrist0:
            mask[head_rows[:, None], w0_keys[None, :]] = False
        if not self.latent_tshape_action_reads_wrist0:
            mask[action_rows[:, None], w0_keys[None, :]] = False

        gate = float(meta.get("wrist_attention_gate", 1.0))
        if gate >= 1.0:
            return mask

        # Gate only the route edges whose visibility is being scheduled. W0 keeps
        # its clean latent and its own video self-attention, avoiding off-manifold
        # latent interpolation. A log gate is exactly an attention-logit bias.
        additive = torch.zeros(mask.shape, dtype=torch.float32, device=device)
        additive.masked_fill_(~mask, float("-inf"))
        log_gate = math.log(gate) if gate > 0.0 else float("-inf")
        if self.latent_tshape_head_reads_wrist0:
            additive[head_rows[:, None], w0_keys[None, :]] += log_gate
        if self.latent_tshape_action_reads_wrist0:
            additive[action_rows[:, None], w0_keys[None, :]] += log_gate
        return additive

    @staticmethod
    def _build_packed_mot_attention_mask(
        video_pre: dict[str, Any],
        action_seq_len: int,
        device: torch.device,
        head_reads_wrist0: bool = False,
    ) -> torch.Tensor:
        """Build the first-round asymmetric [Head, W0, Action] topology."""
        meta = video_pre["meta"]
        head_seq_len = int(meta["head_seq_len"])
        head_tokens_per_frame = int(meta["head_tokens_per_frame"])
        wrist_seq_len = int(meta["wrist_seq_len"])
        video_seq_len = head_seq_len + wrist_seq_len
        total_seq_len = video_seq_len + int(action_seq_len)
        mask = torch.zeros(
            (total_seq_len, total_seq_len), dtype=torch.bool, device=device
        )

        head = slice(0, head_seq_len)
        head_current = slice(0, head_tokens_per_frame)
        head_future = slice(head_tokens_per_frame, head_seq_len)
        wrist = slice(head_seq_len, video_seq_len)
        action = slice(video_seq_len, total_seq_len)

        # Preserve the original first-frame-causal Head path.
        mask[head, head] = True
        mask[head_current, head_future] = False
        if head_reads_wrist0:
            # Optional bridge treatment: future Head queries may attend the
            # observation-only Wrist stream, while still never seeing a future
            # Wrist token (none are present in the packed interface).
            mask[head, wrist] = True
        # Default A1-isolated path keeps Head queries independent of Wrist.

        # A2-head: current Wrist reads current Head plus itself, never future Head.
        mask[wrist, head_current] = True
        mask[wrist, wrist] = True

        # Action uses both current observations and its original all-action path.
        mask[action, head_current] = True
        mask[action, wrist] = True
        mask[action, action] = True
        # Video never reads Action.
        return mask

    def _packed_video_pre_dit(
        self,
        *,
        head_latents: torch.Tensor,
        wrist_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> dict[str, Any]:
        """Patchify Head/Wrist separately and assemble [H0..Hn, W0]."""
        head_latents, timestep, context_mask = (
            self.video_expert._prepare_forward_inputs(
                x=head_latents,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
                action=None,
            )
        )

        head_patches = self.video_expert.patchify(head_latents)
        wrist_patches = self.video_expert.patchify(wrist_latents)
        batch_size = int(head_patches.shape[0])
        head_frames, head_grid_h, head_grid_w = (
            int(value) for value in head_patches.shape[2:]
        )
        wrist_frames, wrist_grid_h, wrist_grid_w = (
            int(value) for value in wrist_patches.shape[2:]
        )
        spatial = self._robotwin_packed_spatial_layout(head_latents.device)
        head_tokens_per_frame = int(spatial["head_tokens_per_frame"])

        head_tokens = rearrange(
            head_patches, "b c f h w -> b (f h w) c"
        ).contiguous()
        wrist_tokens = rearrange(
            wrist_patches, "b c f h w -> b (f h w) c"
        ).contiguous()
        tokens = torch.cat([head_tokens, wrist_tokens], dim=1)
        head_seq_len = int(head_tokens.shape[1])
        wrist_seq_len = int(wrist_tokens.shape[1])

        head_token_timesteps = timestep.view(batch_size, 1, 1).expand(
            batch_size, head_frames, head_tokens_per_frame
        ).clone()
        head_token_timesteps[:, 0, :] = 0
        wrist_token_timesteps = torch.zeros(
            (batch_size, wrist_seq_len),
            device=timestep.device,
            dtype=timestep.dtype,
        )
        token_timesteps = torch.cat(
            [
                head_token_timesteps.reshape(batch_size, head_seq_len),
                wrist_token_timesteps,
            ],
            dim=1,
        )
        token_t_emb = sinusoidal_embedding_1d(
            self.video_expert.freq_dim, token_timesteps.reshape(-1)
        )
        t = self.video_expert.time_embedding(token_t_emb).reshape(
            batch_size, head_seq_len + wrist_seq_len, self.video_expert.hidden_dim
        )
        t_mod = self.video_expert.time_projection(t).unflatten(
            2, (6, self.video_expert.hidden_dim)
        )

        context_emb = self.video_expert.text_embedding(context)
        packed_context_mask = context_mask.unsqueeze(1).expand(
            -1, head_seq_len + wrist_seq_len, -1
        )

        freq_t, freq_h, freq_w = self.video_expert.freqs
        freq_device = freq_t.device
        head_rows = spatial["head_rows"].reshape(-1).to(freq_device)
        head_cols = spatial["head_cols"].reshape(-1).to(freq_device)
        wrist_rows = spatial["wrist_rows"].reshape(-1).to(freq_device)
        wrist_cols = spatial["wrist_cols"].reshape(-1).to(freq_device)
        head_temporal = (
            freq_t[:head_frames]
            .view(head_frames, 1, -1)
            .expand(head_frames, head_tokens_per_frame, -1)
            .reshape(head_seq_len, -1)
        )
        head_height = (
            freq_h[head_rows]
            .view(1, head_tokens_per_frame, -1)
            .expand(head_frames, -1, -1)
            .reshape(head_seq_len, -1)
        )
        head_width = (
            freq_w[head_cols]
            .view(1, head_tokens_per_frame, -1)
            .expand(head_frames, -1, -1)
            .reshape(head_seq_len, -1)
        )
        wrist_temporal = freq_t[:1].expand(wrist_seq_len, -1)
        wrist_height = freq_h[wrist_rows]
        wrist_width = freq_w[wrist_cols]
        freqs = torch.cat(
            [
                torch.cat([head_temporal, head_height, head_width], dim=-1),
                torch.cat([wrist_temporal, wrist_height, wrist_width], dim=-1),
            ],
            dim=0,
        ).unsqueeze(1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": packed_context_mask,
            "meta": {
                "packed_multiview": True,
                "head_grid_size": (head_frames, head_grid_h, head_grid_w),
                "head_tokens_per_frame": head_tokens_per_frame,
                "head_seq_len": head_seq_len,
                "wrist_grid_size": (1, wrist_grid_h, wrist_grid_w),
                "wrist_seq_len": wrist_seq_len,
                "packed_head_reads_wrist0": self.packed_head_reads_wrist0,
                "batch_size": batch_size,
            },
        }

    def _packed_video_post_dit(
        self, x_tokens: torch.Tensor, pre_state: dict[str, Any]
    ) -> torch.Tensor:
        """Decode only the Head stream; W0 is observation-only conditioning."""
        meta = pre_state["meta"]
        head_seq_len = int(meta["head_seq_len"])
        head_tokens = x_tokens[:, :head_seq_len]
        head_t = pre_state["t"][:, :head_seq_len]
        head_output = self.video_expert.head(head_tokens, head_t)
        return self.video_expert.unpatchify(
            head_output, meta["head_grid_size"]
        )

    def _video_pre_dit(
        self,
        *,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action: Optional[torch.Tensor],
        fuse_vae_embedding_in_latents: bool,
        temporal_index: Optional[torch.Tensor] = None,
        wrist_attention_gate: float = 1.0,
        packed_wrist_latents: Optional[torch.Tensor] = None,
        camera_pose_delta: Optional[torch.Tensor] = None,
        camera_pose_valid: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        if self.video_vae_layout == "robotwin_packed_multiview":
            return self._packed_video_pre_dit(
                head_latents=x,
                wrist_latents=packed_wrist_latents,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )
        kwargs: dict[str, Any] = {}
        if self.video_vae_layout == "robotwin_multiview_tokens":
            view_ids, row_ids, col_ids = self._robotwin_multiview_patch_layout(x.device)
            kwargs.update(
                spatial_view_ids=view_ids,
                spatial_row_ids=row_ids,
                spatial_col_ids=col_ids,
            )
        pre = self.video_expert.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            temporal_index=temporal_index,
            camera_pose_delta=camera_pose_delta,
            camera_pose_valid=camera_pose_valid,
            **kwargs,
        )
        if self.video_vae_layout == "robotwin_latent_tshape":
            layout = self._robotwin_latent_tshape_patch_layout(x.device)
            view_ids = layout["view_ids"]
            _, grid_h, grid_w = (
                int(value) for value in pre["meta"]["grid_size"]
            )
            tokens_per_frame = int(pre["meta"]["tokens_per_frame"])
            num_latent_frames = int(pre["meta"]["grid_size"][0])
            wrist0_token_indices = layout["wrist_indices_per_frame"]
            future_wrist_token_indices = torch.cat(
                [
                    wrist0_token_indices + frame * tokens_per_frame
                    for frame in range(1, num_latent_frames)
                ],
                dim=0,
            ) if num_latent_frames > 1 else wrist0_token_indices.new_empty((0,))
            pre["meta"].update(
                robotwin_latent_tshape_routed=True,
                latent_tshape_view_ids=view_ids,
                head_tokens_per_frame=int(layout["head_tokens_per_frame"]),
                wrist_tokens_per_frame=int(layout["wrist_tokens_per_frame"]),
                left_wrist_tokens_per_frame=int(
                    layout["left_wrist_tokens_per_frame"]
                ),
                right_wrist_tokens_per_frame=int(
                    layout["right_wrist_tokens_per_frame"]
                ),
                wrist0_token_indices=wrist0_token_indices,
                future_wrist_token_indices=future_wrist_token_indices,
                latent_tshape_head_reads_wrist0=(
                    self.latent_tshape_head_reads_wrist0
                ),
                latent_tshape_action_reads_wrist0=(
                    self.latent_tshape_action_reads_wrist0
                ),
                latent_tshape_future_wrist_loss_weight=(
                    self.latent_tshape_future_wrist_loss_weight
                ),
                latent_tshape_head_resize_mode=self.latent_tshape_head_resize_mode,
                wrist_attention_gate=float(wrist_attention_gate),
            )
        if self.video_vae_layout == "robotwin_multiview_tokens":
            view_ids = pre["meta"]["spatial_view_ids"]
            per_frame_gate = torch.ones(
                view_ids.numel(), dtype=torch.float32, device=x.device
            )
            wrist_tokens = view_ids.reshape(-1) > 0
            per_frame_gate[wrist_tokens] = float(wrist_attention_gate)
            num_latent_frames = int(pre["meta"]["grid_size"][0])
            pre["meta"]["video_key_gate"] = per_frame_gate.repeat(num_latent_frames)
            pre["meta"]["wrist_attention_gate"] = float(wrist_attention_gate)
        return pre

    def _layout_video_loss_per_sample(
        self,
        *,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
        latent_step_weights: Optional[torch.Tensor],
        sample: dict,
        inputs: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Per-sample video loss for the configured layout (shared by FastWAM and FastWAMIDM).

        Dense latent-T / multiview-token layouts split Head and both Wrist views and
        combine them with the effective wrist loss weight; other layouts use the plain loss.
        """
        view_video_losses: dict[str, torch.Tensor] = {}
        if self.video_vae_layout in {
            "robotwin_latent_tshape",
            "robotwin_multiview_tokens",
        }:
            factor = int(self.vae.upsampling_factor)
            if self.video_vae_layout == "robotwin_latent_tshape":
                head_size = self.robotwin_latent_tshape_head_size
                wrist_size = self.robotwin_latent_tshape_wrist_size
            else:
                head_size = self.robotwin_multiview_head_size
                wrist_size = self.robotwin_multiview_wrist_size
            head_latent_h = head_size[0] // factor
            wrist_latent_w = wrist_size[1] // factor
            pred_views = {
                "head": pred_video[:, :, :, :head_latent_h, :],
                "left_wrist": pred_video[
                    :, :, :, head_latent_h:, :wrist_latent_w
                ],
                "right_wrist": pred_video[
                    :, :, :, head_latent_h:, wrist_latent_w:
                ],
            }
            target_views = {
                "head": target_video[:, :, :, :head_latent_h, :],
                "left_wrist": target_video[
                    :, :, :, head_latent_h:, :wrist_latent_w
                ],
                "right_wrist": target_video[
                    :, :, :, head_latent_h:, wrist_latent_w:
                ],
            }
            for view_name in pred_views:
                view_video_losses[view_name] = self._compute_video_loss_per_sample(
                    pred_video=pred_views[view_name],
                    target_video=target_views[view_name],
                    image_is_pad=image_is_pad,
                    include_initial_video_step=include_initial_video_step,
                    latent_step_weights=latent_step_weights,
                )
            wrist_loss_weight = float(inputs["wrist_video_loss_weight"])
            loss_video_per_sample = self._combine_head_wrist_video_losses(
                head_loss=view_video_losses["head"],
                left_wrist_loss=view_video_losses["left_wrist"],
                right_wrist_loss=view_video_losses["right_wrist"],
                wrist_loss_weight=wrist_loss_weight,
                view_loss_mask=sample.get("video_view_loss_mask"),
            )
        else:
            loss_video_per_sample = self._compute_video_loss_per_sample(
                pred_video=pred_video,
                target_video=target_video,
                image_is_pad=image_is_pad,
                include_initial_video_step=include_initial_video_step,
                latent_step_weights=latent_step_weights,
            )
        return loss_video_per_sample, view_video_losses

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
        latent_step_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        # Optional per-latent-step weighting (GoalWAM: down-weight the g_long latent).
        # `latent_step_weights` is [n_steps] aligned to `video_loss_token`'s step dim;
        # a weighted mean replaces the plain mean. Only used on the goal path, which
        # carries no per-latent padding, so it never combines with `image_is_pad`.
        if latent_step_weights is not None:
            w = latent_step_weights.to(device=video_loss_token.device, dtype=video_loss_token.dtype)
            return (video_loss_token * w.unsqueeze(0)).sum(dim=1) / w.sum().clamp(min=1e-8)
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    @staticmethod
    def _combine_head_wrist_video_losses(
        head_loss: torch.Tensor,
        left_wrist_loss: torch.Tensor,
        right_wrist_loss: torch.Tensor,
        wrist_loss_weight: float,
        view_loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Combine per-view losses while keeping weight=0 exactly Head-only."""
        weight = float(wrist_loss_weight)
        if view_loss_mask is not None:
            mask = torch.as_tensor(view_loss_mask, device=head_loss.device, dtype=head_loss.dtype)
            head_loss = head_loss * mask[:, 0]
            left_wrist_loss = left_wrist_loss * mask[:, 1]
            right_wrist_loss = right_wrist_loss * mask[:, 2]
        # Keep the denominator fixed across missing-wrist ablations. Reweighting
        # by active views would also change the head/action loss ratio.
        wrist_loss = 0.5 * (left_wrist_loss + right_wrist_loss)
        return (head_loss + weight * wrist_loss) / (1.0 + weight)

    def _mot_forward(
        self,
        video_pre: dict[str, Any],
        action_pre: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Run the MoT over the video + action experts and return per-expert tokens.

        This is the single seam through which `training_loss`, `_predict_joint_noise`,
        and `_predict_action_noise` invoke the mixture. The base implementation runs the
        canonical two-expert (`video`, `action`) MoT with the first-frame-causal joint
        mask. Subclasses (e.g. `FastWAMMoE`) override this to split the video tokens into
        obs/future parameter paths; everything up- and down-stream (pre_dit, post_dit,
        loss) is unchanged, so the split stays localized here.
        """
        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        is_packed = bool(video_pre["meta"].get("packed_multiview", False))
        is_routed_latent_tshape = bool(
            video_pre["meta"].get("robotwin_latent_tshape_routed", False)
        )
        if is_routed_latent_tshape:
            attention_mask = self._build_robotwin_latent_tshape_mot_attention_mask(
                video_pre=video_pre,
                action_seq_len=action_tokens.shape[1],
                device=video_tokens.device,
            )
            joint_mask_spec = None
        elif is_packed:
            attention_mask = self._build_packed_mot_attention_mask(
                video_pre=video_pre,
                action_seq_len=action_tokens.shape[1],
                device=video_tokens.device,
                head_reads_wrist0=self.packed_head_reads_wrist0,
            )
            joint_mask_spec = None
        else:
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_tokens.shape[1],
                action_seq_len=action_tokens.shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_tokens.device,
                video_key_gate=video_pre["meta"].get("video_key_gate"),
            )
            tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
            joint_mask_spec = (
                int(video_tokens.shape[1]),
                min(tokens_per_frame, int(video_tokens.shape[1])),
            )
            if getattr(self, "joint_attention_mode", "isolated") == "mutual":
                joint_mask_spec = (*joint_mask_spec, "mutual")
        return self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            joint_mask_spec=joint_mask_spec,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

    @staticmethod
    def _source_is_ego_tensor(
        sample: dict[str, Any], batch_size: int, *, device: Optional[torch.device] = None
    ) -> Optional[torch.Tensor]:
        """Return one source flag per sample, or ``None`` for non-cotrain batches."""
        source_is_ego = sample.get("source_is_ego")
        if source_is_ego is None:
            return None
        if not isinstance(source_is_ego, torch.Tensor):
            source_is_ego = torch.as_tensor(source_is_ego, dtype=torch.bool)
        source_is_ego = source_is_ego.to(device=device, dtype=torch.bool).reshape(-1)
        if source_is_ego.numel() == 1 and batch_size > 1:
            source_is_ego = source_is_ego.expand(batch_size)
        return source_is_ego

    @classmethod
    def _batch_uses_ego_action_tokenizer(cls, sample: dict[str, Any], batch_size: int) -> bool:
        """Resolve the legacy source-homogeneous replay route for the action adapter."""
        source_is_ego = cls._source_is_ego_tensor(sample, batch_size)
        if source_is_ego is None:
            return False
        first = source_is_ego[0]
        return bool(first.item())

    @staticmethod
    def _batch_action_tokenizer_source(sample: dict[str, Any], batch_size: int):
        """Return per-sample source labels for generic ActionDiT adapter routing."""
        source = sample.get("action_tokenizer_source")
        if source is None:
            return None
        if isinstance(source, str):
            return source
        if isinstance(source, (list, tuple)):
            values = tuple(str(value) for value in source)
            return values

    @staticmethod
    def _source_balanced_action_loss(
        weighted_loss_per_sample: torch.Tensor,
        route: tuple[str, ...],
        sources: tuple[str, ...],
        source_weights: Optional[Mapping[str, float]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Average action loss within each source, then combine source means by weight.

        In distributed training, ``batch`` means the global batch across all ranks.
        Each rank scales its differentiable local source sums so FSDP/DDP gradient
        averaging produces the exact global source-balanced objective. Detached
        source sums/counts are also reduced to provide correct per-source metrics.
        """

        return FastWAM._source_balanced_loss(
            weighted_loss_per_sample,
            route,
            sources,
            source_weights=source_weights,
            missing_source_error="No configured action-tokenizer source is present in the batch",
        )

    @staticmethod
    def _source_balanced_loss(
        weighted_loss_per_sample: torch.Tensor,
        route: Sequence[str | None],
        sources: Sequence[str],
        *,
        source_weights: Optional[Mapping[str, float]] = None,
        missing_source_error: str = "No configured source is present in the batch",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Average per-sample loss within each active source, then combine by weight.

        Counts and detached sums are reduced globally. The differentiable local
        sums are scaled by world size so DDP/FSDP gradient averaging yields the
        exact global source-balanced objective, even when a source is absent on a
        particular rank.
        """

        local_sums = []
        local_counts = []
        for source in sources:
            mask = torch.tensor(
                [value == source for value in route],
                device=weighted_loss_per_sample.device,
                dtype=torch.bool,
            )
            local_sums.append(weighted_loss_per_sample.masked_select(mask).sum())
            local_counts.append(mask.sum())

        local_sums_tensor = torch.stack(local_sums)
        local_counts_tensor = torch.stack(local_counts).to(
            device=weighted_loss_per_sample.device,
            dtype=weighted_loss_per_sample.dtype,
        )
        global_stats = torch.cat(
            [local_sums_tensor.detach(), local_counts_tensor.detach()]
        )
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_stats, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()

        num_sources = len(sources)
        global_sums = global_stats[:num_sources]
        global_counts = global_stats[num_sources:]
        active = global_counts > 0

        weights = torch.tensor(
            [
                1.0 if source_weights is None else float(source_weights.get(source, 1.0))
                for source in sources
            ],
            device=weighted_loss_per_sample.device,
            dtype=weighted_loss_per_sample.dtype,
        )
        active_weight_sum = weights[active].sum()

        safe_counts = global_counts.clamp(min=1.0)
        local_source_contributions = local_sums_tensor * float(world_size) / safe_counts
        loss = (local_source_contributions[active] * weights[active]).sum() / active_weight_sum

        source_means_tensor = global_sums / safe_counts
        nan = torch.full_like(source_means_tensor, float("nan"))
        source_means_tensor = torch.where(active, source_means_tensor, nan)
        source_means = {
            str(source): source_means_tensor[index]
            for index, source in enumerate(sources)
        }
        return loss, source_means

    @staticmethod
    def _action_loss_per_sample(
        action_loss_elem: torch.Tensor,
        action_is_pad: Optional[torch.Tensor] = None,
        action_dim_valid: Optional[torch.Tensor] = None,
        component_weights: Optional[Mapping[str, float]] = None,
        num_arms: int = 2,
        component_indices: Optional[Mapping[str, Sequence[int]]] = None,
    ) -> torch.Tensor:
        """Apply component weights, then average valid dimensions and time steps."""
        groups = resolve_action_components(action_loss_elem.shape[-1], num_arms, component_indices)
        configured = component_weights or {}
        dim_weights = action_loss_elem.new_ones(action_loss_elem.shape[-1])
        for name, indices in groups.items():
            weight = float(configured.get(name, 1.0))
            dim_weights[list(indices)] = weight
        valid = action_valid_mask(action_loss_elem, action_dim_valid, action_is_pad)
        return masked_action_mean(action_loss_elem * dim_weights, valid)

    @staticmethod
    def _action_component_loss_metrics(
        action_loss_elem: torch.Tensor,
        action_weight: torch.Tensor,
        action_is_pad: Optional[torch.Tensor] = None,
        action_dim_valid: Optional[torch.Tensor] = None,
        num_arms: int = 2,
        component_indices: Optional[Mapping[str, Sequence[int]]] = None,
    ) -> dict[str, torch.Tensor]:
        """Logging MSEs for each component; explicit layouts exclude absent groups."""
        batch_size = action_loss_elem.shape[0]
        if action_weight.ndim == 0 and batch_size == 1:
            action_weight = action_weight.reshape(1)
        groups = resolve_action_components(action_loss_elem.shape[-1], num_arms, component_indices)
        valid = action_valid_mask(action_loss_elem, action_dim_valid, action_is_pad)
        metrics = {}
        for name, indices in groups.items():
            component = action_loss_elem[..., list(indices)]
            component_valid = valid[..., list(indices)]
            sample_loss = masked_action_mean(component, component_valid) * action_weight
            if component_indices is None:
                # Preserve the legacy logging denominator for existing experiments.
                metrics[name] = sample_loss.mean()
            else:
                active = component_valid.any(dim=(1, 2))
                metrics[name] = sample_loss.sum() / active.sum().clamp_min(1)
        return metrics

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        source_is_ego = self._source_is_ego_tensor(
            sample, batch_size, device=input_latents.device
        )
        source_route = None
        if source_is_ego is not None:
            source_route = tuple(
                "ego" if bool(value) else "robot"
                for value in source_is_ego.detach().cpu().tolist()
            )
        use_ego_action_tokenizer = False
        legacy_ego_route = None
        if self.action_expert.separate_ego_action_tokenizer and source_is_ego is not None:
            if bool(torch.all(source_is_ego == source_is_ego[0]).item()):
                use_ego_action_tokenizer = bool(source_is_ego[0].item())
            else:
                # ActionDiT accepts this private route marker for per-sample
                # legacy Ego/robot adapters. New named source routes remain the
                # preferred interface when more than two sources are configured.
                legacy_ego_route = tuple(
                    self.action_expert.LEGACY_EGO_ROUTE if bool(value) else None
                    for value in source_is_ego.detach().cpu().tolist()
                )
        elif self.action_expert.separate_ego_action_tokenizer:
            use_ego_action_tokenizer = self._batch_uses_ego_action_tokenizer(
                sample, batch_size
            )
        action_tokenizer_source = self._batch_action_tokenizer_source(
            sample, batch_size
        )
        if legacy_ego_route is not None:
            action_tokenizer_source = legacy_ego_route
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        component_layout = getattr(self, "action_component_indices", None)
        explicit_action_valid = None
        if component_layout is not None or inputs.get("action_dim_valid") is not None:
            explicit_action_valid = action_valid_mask(
                action, inputs.get("action_dim_valid"), action_is_pad
            )
            action = torch.where(explicit_action_valid, action, 0.0)
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        if explicit_action_valid is not None:
            # Missing labels must not enter either the scheduler or the joint
            # video/action denoiser as random conditioning dimensions.
            noise_action = torch.where(explicit_action_valid, noise_action, 0.0)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)
        if explicit_action_valid is not None:
            noisy_action = torch.where(explicit_action_valid, noisy_action, 0.0)
            target_action = torch.where(explicit_action_valid, target_action, 0.0)

        video_pre = self._video_pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            temporal_index=inputs.get("goal_temporal_index"),
            wrist_attention_gate=float(inputs.get("wrist_attention_gate", 1.0)),
            packed_wrist_latents=inputs.get("packed_wrist_latents"),
            camera_pose_delta=inputs.get("camera_pose_delta"),
            camera_pose_valid=inputs.get("camera_pose_valid"),
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
            use_ego_action_tokenizer=use_ego_action_tokenizer,
            action_tokenizer_source=action_tokenizer_source,
        )

        tokens_out = self._mot_forward(video_pre, action_pre)

        pred_video = (
            self._packed_video_post_dit(tokens_out["video"], video_pre)
            if self.video_vae_layout == "robotwin_packed_multiview"
            else self.video_expert.post_dit(tokens_out["video"], video_pre)
        )

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        # GoalWAM: down-weight the g_long latent (the last loss step) so short+long goals
        # are supervised unequally. Built here since it aligns to the post-strip step count.
        latent_step_weights = None
        if inputs.get("apply_goal_long_weight", False):
            n_steps = pred_video.shape[2]
            latent_step_weights = torch.ones(n_steps, device=pred_video.device, dtype=torch.float32)
            latent_step_weights[-1] = self.goal_long_loss_weight  # g_long is the last latent

        loss_video_per_sample, view_video_losses = self._layout_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
            latent_step_weights=latent_step_weights,
            sample=sample,
            inputs=inputs,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        weighted_video_loss_per_sample = loss_video_per_sample * video_weight
        source_video_losses: dict[str, torch.Tensor] = {}
        if source_route is not None:
            loss_video, source_video_losses = self._source_balanced_loss(
                weighted_video_loss_per_sample,
                source_route,
                ("robot", "ego"),
                source_weights=self.source_loss_weights["video"],
            )
        else:
            loss_video = weighted_video_loss_per_sample.mean()

        prediction_for_loss = pred_action.float()
        target_for_loss = target_action.float()
        if explicit_action_valid is not None:
            # Mask BEFORE MSE as well: a masked NaN/Inf prediction must not
            # produce NaN gradients through an otherwise zero loss weight.
            prediction_for_loss = torch.where(explicit_action_valid, prediction_for_loss, 0.0)
            target_for_loss = torch.where(explicit_action_valid, target_for_loss, 0.0)
        action_loss_elem = F.mse_loss(prediction_for_loss, target_for_loss, reduction="none")  # [B, T, C]
        action_dim_valid = inputs.get("action_dim_valid", None)
        # Existing embodiment subclasses override these helpers with the legacy
        # signature. Only pass the opt-in layout argument when it is configured.
        component_kwargs = (
            {"component_indices": component_layout} if component_layout is not None else {}
        )
        action_loss_per_sample = self._action_loss_per_sample(
            action_loss_elem=action_loss_elem,
            action_is_pad=action_is_pad,
            action_dim_valid=action_dim_valid,
            component_weights=self.action_component_loss_weights,
            num_arms=self.action_component_num_arms,
            **component_kwargs,
        )

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        action_component_losses = self._action_component_loss_metrics(
            action_loss_elem=action_loss_elem,
            action_weight=action_weight,
            action_is_pad=action_is_pad,
            action_dim_valid=action_dim_valid,
            num_arms=self.action_component_num_arms,
            **component_kwargs,
        )
        weighted_action_loss_per_sample = action_loss_per_sample * action_weight
        source_action_losses: dict[str, torch.Tensor] = {}
        action_tokenizer_route = None
        if self.action_expert.action_tokenizer_sources:
            action_tokenizer_route = self.action_expert._normalize_action_tokenizer_route(
                action_tokenizer_source, batch_size
            )
            loss_action, source_action_losses = self._source_balanced_action_loss(
                weighted_action_loss_per_sample,
                action_tokenizer_route,
                self.action_expert.action_tokenizer_sources,
                source_weights=self.source_loss_weights["action"],
            )
        elif source_route is not None:
            loss_action, source_action_losses = self._source_balanced_loss(
                weighted_action_loss_per_sample,
                source_route,
                ("robot", "ego"),
                source_weights=self.source_loss_weights["action"],
            )
        else:
            loss_action = weighted_action_loss_per_sample.mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        if "bad_segment_action_loss_masked" in sample:
            loss_dict["action_loss_masked_fraction"] = float(
                sample["bad_segment_action_loss_masked"].float().mean().item()
            )
        pose_stationary = video_pre["meta"].get("camera_pose_stationary")
        if pose_stationary is not None:
            pose_valid = inputs.get("camera_pose_valid")
            future_valid = pose_valid[:, 1:]
            future_stationary = pose_stationary[:, 1:] & future_valid
            loss_dict["camera_pose_stationary_frame_fraction"] = float(
                future_stationary.sum().float().div(
                    future_valid.sum().clamp(min=1).float()
                ).item()
            )
            loss_dict["camera_pose_stationary_clip_fraction"] = float(
                torch.all(pose_stationary[:, 1:] | ~future_valid, dim=1)
                .float()
                .mean()
                .item()
            )
        for source, source_loss in source_video_losses.items():
            if not bool(torch.isfinite(source_loss).item()):
                continue
            loss_dict[f"loss_video_{source}"] = self.loss_lambda_video * float(
                source_loss.detach().item()
            )
        for source, source_loss in source_action_losses.items():
            if not bool(torch.isfinite(source_loss).item()):
                continue
            loss_dict[f"loss_action_{source}"] = self.loss_lambda_action * float(
                source_loss.detach().item()
            )
        if view_video_losses:
            supervision = sample.get("video_view_loss_mask")
            if supervision is not None:
                supervision = torch.as_tensor(
                    supervision, device=video_weight.device, dtype=video_weight.dtype
                )
                for index, name in enumerate(("head", "left_wrist", "right_wrist")):
                    loss_dict[f"video_supervision_fraction_{name}"] = float(
                        supervision[:, index].mean().item()
                    )
                    # Raw view metrics below remain unmasked for compatibility.
                    # This metric exposes the actual masked numerator, before
                    # the fixed head/wrist combination and global video lambda.
                    loss_dict[f"loss_video_supervised_{name}"] = float(
                        (view_video_losses[name] * video_weight * supervision[:, index]).mean().detach().item()
                    )
            for view_name, per_sample in view_video_losses.items():
                raw_view_loss = (per_sample * video_weight).mean()
                loss_dict[f"loss_video_{view_name}"] = float(
                    raw_view_loss.detach().item()
                )
            loss_dict["wrist_attention_gate"] = float(
                inputs["wrist_attention_gate"]
            )
            loss_dict["wrist_video_loss_weight"] = float(
                inputs["wrist_video_loss_weight"]
            )
            loss_dict["head_only_replay"] = (
                1.0 if inputs["head_only_replay"] else 0.0
            )
            loss_dict["head_only_replay_ratio"] = float(
                inputs["head_only_replay_ratio"]
            )
            if self.video_vae_layout == "robotwin_latent_tshape":
                # Keep the historical metric as the effective value for log
                # compatibility, but expose unambiguous configured/effective
                # names as well.  The configured value is the schedule maximum;
                # `wrist_video_loss_ramp_steps` controls the optimizer-step cosine
                # ramp used to obtain the effective value.
                loss_dict["latent_tshape_future_wrist_loss_weight"] = float(
                    inputs["wrist_video_loss_weight"]
                )
                loss_dict[
                    "latent_tshape_future_wrist_loss_weight_configured"
                ] = float(self.latent_tshape_future_wrist_loss_weight)
                loss_dict[
                    "latent_tshape_future_wrist_loss_weight_effective"
                ] = float(inputs["wrist_video_loss_weight"])
                loss_dict["latent_tshape_head_reads_wrist0"] = (
                    1.0 if self.latent_tshape_head_reads_wrist0 else 0.0
                )
                loss_dict["latent_tshape_action_reads_wrist0"] = (
                    1.0 if self.latent_tshape_action_reads_wrist0 else 0.0
                )
                loss_dict["latent_tshape_head_tokens_per_frame"] = float(
                    video_pre["meta"]["head_tokens_per_frame"]
                )
                loss_dict["latent_tshape_wrist_tokens_per_frame"] = float(
                    video_pre["meta"]["wrist_tokens_per_frame"]
                )
        for name, component_loss in action_component_losses.items():
            loss_dict[f"loss_action_{name}"] = self.loss_lambda_action * float(
                component_loss.detach().item()
            )
            loss_dict[f"action_loss_weight_{name}"] = float(
                self.action_component_loss_weights[name]
            )
        if inputs.get("wrist_view_mask") is not None:
            loss_dict["wrist_view_mask_fraction"] = float(
                inputs["wrist_view_mask"].float().mean().item()
            )
        if self.action_expert.separate_ego_action_tokenizer:
            loss_dict["action_tokenizer_ego"] = (
                float(source_is_ego.float().mean().item())
                if source_is_ego is not None
                else (1.0 if use_ego_action_tokenizer else 0.0)
            )
        if self.action_expert.action_tokenizer_sources:
            for source in self.action_expert.action_tokenizer_sources:
                loss_dict[f"action_tokenizer_{source}"] = (
                    sum(value == source for value in action_tokenizer_route)
                    / float(batch_size)
                )
                loss_dict[f"loss_action_{source}"] = (
                    self.loss_lambda_action
                    * float(source_action_losses[source].detach().item())
                )
            if self.proprio_encoder is not None:
                with torch.no_grad():
                    default_weight = self.proprio_encoder.weight.detach().float()
                    default_bias = self.proprio_encoder.bias.detach().float()
                    for source, encoder in self.source_proprio_encoders.items():
                        weight_delta = (
                            encoder.weight.detach().float() - default_weight
                        ).square()
                        bias_delta = (
                            encoder.bias.detach().float() - default_bias
                        ).square()
                        loss_dict[f"state_encoder_delta_{source}"] = float(
                            torch.sqrt(
                                (weight_delta.sum() + bias_delta.sum())
                                / float(weight_delta.numel() + bias_delta.numel())
                            ).item()
                        )
        if self.goal_enabled and 0.0 < self.goal_mix_video_ratio < 1.0:
            # Realized per-step route under joint co-training; its running mean over
            # logged steps tracks the effective video/goal mixing ratio.
            loss_dict["route_video"] = 1.0 if inputs["route_video"] else 0.0
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        camera_pose_delta: Optional[torch.Tensor] = None,
        camera_pose_valid: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self._video_pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            wrist_attention_gate=1.0,
            camera_pose_delta=camera_pose_delta,
            camera_pose_valid=camera_pose_valid,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        tokens_out = self._mot_forward(video_pre, action_pre)

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self._video_pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            wrist_attention_gate=1.0,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        tokens_out = self._mot_forward(video_pre, action_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        context_embedding_cache: Optional[torch.Tensor] = None,
        cross_attn_kv_cache: Optional[list] = None,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
            context_embedding_cache=context_embedding_cache,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            cross_attn_kv_cache=cross_attn_kv_cache,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        camera_pose_delta: Optional[torch.Tensor] = None,
        camera_pose_valid: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        decode_video: bool = True,
        proprio_dim_is_pad: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
                proprio_dim_is_pad=proprio_dim_is_pad,
            )["action"]

        input_image, latent_h, latent_w = self._prepare_inference_input_image(input_image)
        if getattr(self.video_expert, "camera_pose_conditioning", False):
            camera_pose_delta = torch.as_tensor(camera_pose_delta)
            if camera_pose_delta.ndim == 3:
                camera_pose_delta = camera_pose_delta.unsqueeze(0)
            camera_pose_delta = camera_pose_delta.to(
                device=self.device, dtype=torch.float32
            )
            if camera_pose_valid is None:
                camera_pose_valid = torch.ones(
                    (1, num_video_frames), dtype=torch.bool, device=self.device
                )
            else:
                camera_pose_valid = torch.as_tensor(camera_pose_valid)
                if camera_pose_valid.ndim == 1:
                    camera_pose_valid = camera_pose_valid.unsqueeze(0)
                camera_pose_valid = camera_pose_valid.to(
                    device=self.device, dtype=torch.bool
                )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        if getattr(self, "joint_attention_mode", "isolated") == "mutual":
            # Advance one seeded stream instead of restarting identical noise
            # for the two modalities. Preserve legacy seeded output otherwise.
            action_generator = video_generator
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
                proprio_dim_is_pad=proprio_dim_is_pad,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
                camera_pose_delta=camera_pose_delta,
                camera_pose_valid=camera_pose_valid,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled) if decode_video else None,
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        proprio_dim_is_pad: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        self.eval()

        input_image, _, _ = self._prepare_inference_input_image(input_image)
        if proprio is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        packed_wrist_latents = None
        if self.video_vae_layout == "robotwin_packed_multiview":
            first_frame_latents, packed_wrist_latents = (
                self._encode_robotwin_packed_multiview(
                    input_image.unsqueeze(2 if self.latent_tshape_premerged_input else 3), tiled=tiled
                )
            )
        else:
            first_frame_latents = self._encode_input_image_latents_tensor(
                input_image=input_image, tiled=tiled
            )
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
                proprio_dim_is_pad=proprio_dim_is_pad,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )

        # The video prefill (pre_dit + video KV cache + action text-embed + cross-attn KV)
        # runs once per chunk. Optionally replay it as a CUDA graph (partly launch-bound),
        # captured once and reused across chunks (shapes are constant for an eval config);
        # only the first-frame latents and context vary, copied into static buffers. Bit-
        # exact. video_seq_len / attention_mask are shape-only -> computed once at capture.
        use_prefill_graph = (
            bool(getattr(self, "_use_cuda_graph_prefill", False))
            and first_frame_latents.is_cuda
            and self.video_vae_layout
            not in {
                "robotwin_latent_tshape",
                "robotwin_multiview_tokens",
                "robotwin_packed_multiview",
            }
        )
        if use_prefill_graph:
            from .cuda_graph import (
                VideoPrefillGraph,
                pin_action_rope_freqs_to_gpu,
                pin_video_rope_freqs_to_gpu)
            prefill_key = (
                tuple(int(s) for s in first_frame_latents.shape),
                int(context.shape[1]),
                int(latents_action.shape[1]),
            )
            prefill_graph = getattr(self, "_video_prefill_graph", None)
            if prefill_graph is None or getattr(self, "_video_prefill_key", None) != prefill_key:
                self._video_prefill_graph = None  # free old graph/pool before re-capture
                # Bootstrap shapes (video_seq_len, tokens_per_frame) with one eager pre_dit.
                video_pre = self._video_pre_dit(
                    x=first_frame_latents, timestep=timestep_video, context=context,
                    context_mask=context_mask, action=None,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    wrist_attention_gate=1.0,
                )
                video_seq_len = int(video_pre["tokens"].shape[1])
                attention_mask = self._build_mot_attention_mask(
                    video_seq_len=video_seq_len,
                    action_seq_len=latents_action.shape[1],
                    video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                    device=video_pre["tokens"].device,
                    video_key_gate=video_pre["meta"].get("video_key_gate"),
                )
                pin_video_rope_freqs_to_gpu(self, self.device)
                pin_action_rope_freqs_to_gpu(self, self.device)
                prefill_graph = VideoPrefillGraph(
                    self,
                    dict(
                        first_frame_latents=first_frame_latents,
                        timestep_video=timestep_video,
                        context=context,
                        context_mask=context_mask,
                        video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
                        fuse_flag=fuse_flag,
                    ),
                )
                self._video_prefill_graph = prefill_graph
                self._video_prefill_key = prefill_key
                self._video_prefill_seq_len = video_seq_len
                self._video_prefill_attention_mask = attention_mask
            video_seq_len = self._video_prefill_seq_len
            attention_mask = self._video_prefill_attention_mask
            video_kv_cache, context_embedding_cache, cross_attn_kv_cache = prefill_graph.replay(
                first_frame_latents, context, context_mask
            )
        else:
            video_pre = self._video_pre_dit(
                x=first_frame_latents,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
                wrist_attention_gate=1.0,
                packed_wrist_latents=packed_wrist_latents,
            )
            video_seq_len = int(video_pre["tokens"].shape[1])
            if video_pre["meta"].get("robotwin_latent_tshape_routed", False):
                attention_mask = (
                    self._build_robotwin_latent_tshape_mot_attention_mask(
                        video_pre=video_pre,
                        action_seq_len=latents_action.shape[1],
                        device=video_pre["tokens"].device,
                    )
                )
            elif video_pre["meta"].get("packed_multiview", False):
                attention_mask = self._build_packed_mot_attention_mask(
                    video_pre=video_pre,
                    action_seq_len=latents_action.shape[1],
                    device=video_pre["tokens"].device,
                    head_reads_wrist0=self.packed_head_reads_wrist0,
                )
            else:
                attention_mask = self._build_mot_attention_mask(
                    video_seq_len=video_seq_len,
                    action_seq_len=latents_action.shape[1],
                    video_tokens_per_frame=int(
                        video_pre["meta"]["tokens_per_frame"]
                    ),
                    device=video_pre["tokens"].device,
                    video_key_gate=video_pre["meta"].get("video_key_gate"),
                )
            video_kv_cache = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            )

            # Cache text embedding and cross-attn K/V — constant across denoising steps.
            context_embedding_cache = self.action_expert.text_embedding(context)
            cross_attn_kv_cache = self.mot.prefill_cross_attn_cache("action", context_embedding_cache)

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        # Static (within this call) kwargs of the per-step action forward.
        action_step_kwargs = dict(
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            context_embedding_cache=context_embedding_cache,
            cross_attn_kv_cache=cross_attn_kv_cache,
        )
        # Optionally replay the denoise step as a CUDA graph (launch-bound -> ~4x/step,
        # bit-exact). The graph is CAPTURED ONCE and REUSED across chunks/episodes
        # (video_seq_len/action_horizon are constant for a given eval config): each call
        # copies the freshly-prefilled caches into the graph's static buffers, then only
        # latents_action/timestep_action vary per step. Keyed by shape signature.
        use_graph = bool(getattr(self, "_use_cuda_graph", False)) and latents_action.is_cuda
        step_graph = None
        if use_graph:
            from .cuda_graph import ActionDenoiseStepGraph, pin_action_rope_freqs_to_gpu
            pin_action_rope_freqs_to_gpu(self, self.device)
            graph_key = (int(video_seq_len), int(latents_action.shape[1]), int(num_inference_steps))
            step_graph = getattr(self, "_action_step_graph", None)
            if step_graph is None or getattr(self, "_action_graph_key", None) != graph_key:
                self._action_step_graph = None  # free the old graph/pool before re-capture
                t0 = infer_timesteps_action[0].unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
                step_graph = ActionDenoiseStepGraph(self, latents_action, t0, action_step_kwargs)
                self._action_step_graph = step_graph
                self._action_graph_key = graph_key
            else:
                step_graph.update_static(action_step_kwargs)  # copy this chunk's caches in

        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            if use_graph:
                pred_action = step_graph.replay(latents_action, timestep_action).clone()
            else:
                pred_action = self._predict_action_noise_with_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    **action_step_kwargs,
                )

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        camera_pose_delta: Optional[torch.Tensor] = None,
        camera_pose_valid: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            camera_pose_delta=camera_pose_delta,
            camera_pose_valid=camera_pose_valid,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if getattr(self, "action_component_indices", None) is not None:
            payload["action_component_indices"] = self.action_component_indices
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.source_proprio_encoders:
            payload["source_proprio_encoders"] = (
                self.source_proprio_encoders.state_dict()
            )
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(
        self,
        path,
        optimizer=None,
        *,
        mot_skip_prefixes: Sequence[str] = (),
        skip_proprio_encoder: bool = False,
    ):
        payload = torch.load(path, map_location="cpu")
        mot_skip_prefixes = tuple(str(prefix) for prefix in mot_skip_prefixes)
        if "mot" in payload:
            checkpoint_mot = payload["mot"]
            skipped_mot_keys = sorted(
                key
                for key in checkpoint_mot
                if any(key.startswith(prefix) for prefix in mot_skip_prefixes)
            )
            load_mot = {
                key: value
                for key, value in checkpoint_mot.items()
                if key not in skipped_mot_keys
            }
            self.mot.load_state_dict(load_mot, strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        if self.proprio_encoder is not None:
            if skip_proprio_encoder:
                logger.warning(
                    "Checkpoint intentionally skipped `proprio_encoder`; keeping the "
                    "current task-specific initialization."
                )
            elif "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")
        if self.source_proprio_encoders:
            if "source_proprio_encoders" in payload:
                self.source_proprio_encoders.load_state_dict(
                    payload["source_proprio_encoders"], strict=True
                )
            else:
                self.initialize_source_proprio_encoders_from_default()
                logger.info(
                    "Initialized source-specific state encoders from checkpoint "
                    "`proprio_encoder` weights: %s",
                    list(self.source_proprio_encoders),
                )
        elif "source_proprio_encoders" in payload:
            logger.warning(
                "Checkpoint contains source-specific state encoder weights but the "
                "current model has none configured; ignoring."
            )

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
