"""Load training settings and configure a local CUDA/torchrun launch."""
import os

from omegaconf import OmegaConf


def load_config(config_path, paths_path=None, overrides=()):
    cfg = OmegaConf.load(config_path)
    if paths_path:
        cfg = OmegaConf.merge(cfg, {'paths': OmegaConf.load(paths_path)})
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def configure_launch(cfg):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    strategy = cfg.distributed.strategy
    if strategy == 'auto':
        strategy = 'manual_fsdp' if world_size > 1 else 'single'
    cfg.distributed.strategy = strategy
    if cfg.dataloader_rank_aware_strict == 'auto':
        cfg.dataloader_rank_aware_strict = world_size > 1
    os.environ['FASTWAM_MANUAL_FSDP'] = '1' if strategy == 'manual_fsdp' else '0'
    os.environ['FSDP_SHARDING'] = cfg.distributed.fsdp_sharding_strategy
    for key in ('ACCELERATE_USE_DEEPSPEED', 'ACCELERATE_USE_FSDP', 'ACCELERATE_USE_CPU'):
        os.environ[key] = 'false'
    return dict(strategy=strategy, world_size=world_size, rank=rank, local_rank=local_rank,
                per_device_batch_size=int(cfg.batch_size),
                effective_batch_size=int(cfg.batch_size) * world_size * int(cfg.gradient_accumulation_steps))
