"""Load training settings and configure a local CUDA/torchrun launch."""
import os
from pathlib import Path

from omegaconf import OmegaConf


def set_training_checkpoint(cfg, *, init_checkpoint=None, resume=None):
    """Separate stage initialization (weights file) from exact run continuation."""
    if init_checkpoint is not None and resume is not None:
        raise ValueError('Use only one of --init-checkpoint and --resume')
    value = init_checkpoint if init_checkpoint is not None else resume
    if value is None:
        return
    path = Path(value).expanduser().resolve()
    if init_checkpoint is not None and not path.is_file():
        raise ValueError(f'--init-checkpoint requires a weights file: {path}')
    if resume is not None and not path.is_dir():
        raise ValueError(f'--resume requires a training-state directory: {path}')
    cfg.resume = str(path)
    # Also fill the legacy binding so recipes/path files containing ??? resolve.
    if 'paths' in cfg:
        cfg.paths.resume = str(path)
    cfg.resume_step = 0


def training_config_gaps(cfg):
    """Find explicit placeholders without dereferencing missing path bindings."""
    def walk(value, prefix=''):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from walk(child, f'{prefix}.{key}' if prefix else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, f'{prefix}[{index}]')
        elif value == '???':
            yield prefix
    raw = OmegaConf.to_container(cfg, resolve=False)
    # Binding catalogs can contain entries for other recipes. Missing values
    # referenced by this recipe are caught when resolving its interpolations.
    raw.pop('paths', None)
    raw.pop('corpus_indices', None)
    return sorted(walk(raw))


def _merge_recipe(base, overlay):
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_recipe(base[key], value)
        else:
            base[key] = value
    return base


def _load_recipe(config_path):
    config_path = Path(config_path)
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    parents = cfg.pop('_base_', [])
    if isinstance(parents, str):
        parents = [parents]
    merged = {}
    for parent in parents:
        _merge_recipe(merged, _load_recipe(config_path.parent / parent))
    # Explicit ??? entries must clear inherited values in unfinished recipes.
    return _merge_recipe(merged, cfg)


def load_config(config_path, paths_path=None, overrides=()):
    cfg = OmegaConf.create(_load_recipe(config_path))
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
