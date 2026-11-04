# Modified by Open-HumanScale contributors: portable training launch and local checkpoints.
import logging
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from .utils.logging_config import get_logger, setup_logging
from .utils import misc
from .utils.pytorch_utils import is_rank0

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = mixed_precision.strip().lower()
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def create_fastwam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    action_component_num_arms: int = 2,
    action_component_indices=None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_checkpoint_post: bool = False,
    mot_compile_blocks: bool = False,
    mot_compile_dynamic: bool = True,
    mot_compile_mode: str = "default",
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    goal_enabled: bool = False,
    goal_targets: str = "both",
    goal_encode_mode: str = "pseudo_video",
    goal_long_rope_mode: str = "arange",
    goal_long_loss_weight: float = 1.0,
    goal_short_rope_index: int | None = None,
    goal_mix_video_ratio: float = 0.0,
    vae_encode_batched: bool = True,
    vae_encode_chunk_frames: int = 0,
    vae_compile_encoder: bool = False,
    vae_compile_dynamic: bool = False,
    vae_compile_mode: str = "default",
    vae_enable_channels_last_3d: bool = True,
    video_vae_layout: str = "pixel",
    robotwin_latent_tshape_head_size=(256, 320),
    robotwin_latent_tshape_wrist_size=(128, 160),
    latent_tshape_head_resize_mode: str = "cover_crop",
    latent_tshape_premerged_input: bool = False,
    latent_tshape_vae_premerge: bool = False,
    latent_tshape_future_wrist_loss_weight: float = 0.5,
    latent_tshape_head_reads_wrist0: bool = True,
    latent_tshape_action_reads_wrist0: bool = True,
    wrist_view_mask_prob: float = 0.0,
    robotwin_multiview_head_size=(256, 448),
    robotwin_multiview_wrist_size=(128, 224),
    robotwin_packed_head_size=(256, 448),
    robotwin_packed_wrist_size=(128, 160),
    packed_head_reads_wrist0: bool = False,
    packed_wrist_encode_separately: bool = False,
    wrist_attention_gate_ramp_steps: int = 0,
    wrist_video_loss_ramp_steps: int = 0,
    wrist_video_loss_max_weight: float = 1.0,
    head_only_replay_start_ratio: float = 0.0,
    head_only_replay_end_ratio: float = 0.0,
    head_only_replay_decay_steps: int = 0,
    model_class: str | None = None,
):
    from .models.wan22.fastwam import FastWAM

    if model_class is not None:
        from hydra.utils import get_class
        implementation = get_class(model_class)
    else:
        implementation = FastWAM

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}

    if isinstance(action_component_indices, DictConfig):
        action_component_indices = OmegaConf.to_container(action_component_indices, resolve=True)

    return implementation.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        mot_compile_blocks=bool(mot_compile_blocks),
        mot_compile_dynamic=bool(mot_compile_dynamic),
        mot_compile_mode=str(mot_compile_mode),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        source_loss_weights=loss.get("source_weights"),
        action_component_num_arms=int(action_component_num_arms),
        action_component_loss_weights=loss.get("component_weights"),
        action_component_indices=action_component_indices,
        goal_enabled=bool(goal_enabled),
        goal_targets=str(goal_targets),
        goal_encode_mode=str(goal_encode_mode),
        goal_long_rope_mode=str(goal_long_rope_mode),
        goal_long_loss_weight=float(goal_long_loss_weight),
        goal_short_rope_index=(None if goal_short_rope_index is None else int(goal_short_rope_index)),
        goal_mix_video_ratio=float(goal_mix_video_ratio),
        vae_encode_batched=bool(vae_encode_batched),
        vae_encode_chunk_frames=int(vae_encode_chunk_frames),
        vae_compile_encoder=bool(vae_compile_encoder),
        vae_compile_dynamic=bool(vae_compile_dynamic),
        vae_compile_mode=str(vae_compile_mode),
        vae_enable_channels_last_3d=bool(vae_enable_channels_last_3d),
        video_vae_layout=str(video_vae_layout),
        robotwin_latent_tshape_head_size=tuple(
            int(value) for value in robotwin_latent_tshape_head_size
        ),
        robotwin_latent_tshape_wrist_size=tuple(
            int(value) for value in robotwin_latent_tshape_wrist_size
        ),
        latent_tshape_head_resize_mode=str(latent_tshape_head_resize_mode),
        latent_tshape_premerged_input=bool(latent_tshape_premerged_input),
        latent_tshape_vae_premerge=bool(latent_tshape_vae_premerge),
        latent_tshape_future_wrist_loss_weight=float(
            latent_tshape_future_wrist_loss_weight
        ),
        latent_tshape_head_reads_wrist0=bool(
            latent_tshape_head_reads_wrist0
        ),
        latent_tshape_action_reads_wrist0=bool(
            latent_tshape_action_reads_wrist0
        ),
        wrist_view_mask_prob=float(wrist_view_mask_prob),
        robotwin_multiview_head_size=tuple(
            int(value) for value in robotwin_multiview_head_size
        ),
        robotwin_multiview_wrist_size=tuple(
            int(value) for value in robotwin_multiview_wrist_size
        ),
        robotwin_packed_head_size=tuple(
            int(value) for value in robotwin_packed_head_size
        ),
        robotwin_packed_wrist_size=tuple(
            int(value) for value in robotwin_packed_wrist_size
        ),
        packed_head_reads_wrist0=bool(packed_head_reads_wrist0),
        packed_wrist_encode_separately=bool(packed_wrist_encode_separately),
        wrist_attention_gate_ramp_steps=int(wrist_attention_gate_ramp_steps),
        wrist_video_loss_ramp_steps=int(wrist_video_loss_ramp_steps),
        wrist_video_loss_max_weight=float(wrist_video_loss_max_weight),
        head_only_replay_start_ratio=float(head_only_replay_start_ratio),
        head_only_replay_end_ratio=float(head_only_replay_end_ratio),
        head_only_replay_decay_steps=int(head_only_replay_decay_steps),
    )


def build_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds




def _maybe_wrap_video_augmentation(cfg: DictConfig, train_ds):
    """Apply train-only clip-locked augmentation after merged-cache assembly."""

    aug = cfg.data.get("video_augmentation") if cfg.get("data") is not None else None
    if aug is None or not bool(aug.get("enabled", False)):
        return train_ds

    from .datasets.video_augmentation import ClipLockedVideoAugmentation

    kwargs = OmegaConf.to_container(aug, resolve=True)
    kwargs.pop("enabled", None)
    kwargs.setdefault("seed", int(cfg.seed))
    wrapped = ClipLockedVideoAugmentation(train_ds, **kwargs)
    logger.info(
        "[video_augmentation] enabled: name=%s probability=%.3f seed=%d "
        "layout=%s magnitude_index=%s",
        wrapped.name,
        wrapped.probability,
        wrapped.seed,
        wrapped.layout,
        wrapped.magnitude_index if wrapped.name == "ra_photo" else "n/a",
    )
    return wrapped


def _resolve_train_device() -> str:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def _seed_model_initialization(seed: int) -> None:
    """Give every data-parallel rank identical randomly initialized weights.

    The trainer intentionally switches to a rank-offset RNG seed after model
    construction so ranks draw independent diffusion noise.  Manual FSDP uses
    ``sync_module_states=False``, however, so random modules that are not
    covered by pretrained weights (ActionDiT's action input/output layers and
    the proprio encoder) must first be initialized from the same seed on every
    rank.
    """
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(cfg: DictConfig):
    from humanwam.config import configure_launch
    configure_launch(cfg)
    model_device = _resolve_train_device()
    from .trainer import Wan22Trainer
    setup_logging(
        log_level=logging.INFO,
        is_main_process=is_rank0(),
    )
    misc.register_work_dir(cfg.output_dir)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    if is_rank0():
        config_payload = OmegaConf.to_container(cfg, resolve=True)
        with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
            OmegaConf.save(config_payload, f)

    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    _seed_model_initialization(int(cfg.seed))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(cfg.data)
    train_ds = _maybe_wrap_video_augmentation(cfg, train_ds)

    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    try:
        trainer.train()
    finally:
        trainer._finish_wandb()
