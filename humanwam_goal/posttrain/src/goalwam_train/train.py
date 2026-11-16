"""Post-train HumanWAM-Goal on RoboTwin 2.0 `clean_50` demonstrations.

    torchrun --nproc-per-node 8 -m goalwam_train.train \\
        --pretrained checkpoints/HumanWAM-Goal-Base \\
        --data data/robotwin_clean_50 \\
        --goal-cache data/train_goal_cache \\
        --output outputs/robotwin
"""

import argparse
import json
import os
import shutil
import socket
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from goalwam.backends.native.model import remap_weight
from goalwam.config import ModelConfig, read_config
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor

from .data import RoboTwinDataset, infinite_loader
from .model import TrainingTransformer, VideoEncoder, prepare
from .normalization import HORIZON, write_robot_json

LR = 1e-5
FAST = ("action2llm", "llm2action", "goal_vision_embed", "action_modality_embed")
FAST_LR = 5 * LR
WEIGHT_DECAY = 0.05
DECAY_STEPS = 100_000
EMA_RATE = 0.1
SHARD_BYTES = 5 * 2**30


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pretrained", type=Path, required=True, help="Pretrained HumanWAM-Goal checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Directory of clean_50 LeRobot v3 datasets")
    parser.add_argument("--goal-cache", type=Path, default=None, help="Optional pre-generated training goals")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=36_000)
    parser.add_argument("--batch-size", type=int, default=256, help="Samples per update over all GPUs")
    parser.add_argument("--micro-batch-size", type=int, default=16, help="Samples per GPU per forward")
    parser.add_argument("--save-every", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4, help="Data loading processes per GPU")
    return parser.parse_args()


def schedule(step):
    return max(0.0, 1.0 - step / DECAY_STEPS)


def build_model(config, mesh):
    """The Reasoner (text tower) stays frozen in BF16; trained weights are FP32 shards."""
    with torch.device("meta"):
        model = TrainingTransformer(config)
    model.text.to(torch.bfloat16)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("text."))
    bf16 = MixedPrecisionPolicy(torch.bfloat16, torch.float32, cast_forward_inputs=False)
    fp32 = MixedPrecisionPolicy(torch.float32, torch.float32, cast_forward_inputs=False)
    for layer in (*model.text.layers, *model.generator.layers):
        fully_shard(layer, mesh=mesh, mp_policy=bf16)
    fully_shard(model.text.embed_tokens, mesh=mesh, mp_policy=bf16)
    fully_shard(model.time_embedder, mesh=mesh, mp_policy=fp32)
    fully_shard(model, mesh=mesh, mp_policy=bf16)
    model.to_empty(device="cuda")
    return model


def shard_files(directory):
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        return sorted(set(json.loads(index.read_text())["weight_map"].values()))
    return ["model.safetensors"]


@torch.no_grad()
def load_weights(model, directory):
    """Each rank reads the checkpoint and keeps its own shard; returns checkpoint tensor names."""
    parameters = dict(model.named_parameters())
    names = {}
    for file in shard_files(directory):
        with safe_open(directory / file, framework="pt", device="cpu") as handle:
            for source in handle.keys():
                name = remap_weight(source)
                if name not in parameters or name in names:
                    raise ValueError(f"Unexpected or duplicate tensor in checkpoint: {source}")
                parameter = parameters[name]
                value = handle.get_tensor(source).to(device="cuda", dtype=parameter.dtype)
                local = distribute_tensor(
                    value, parameter.device_mesh, parameter.placements, src_data_rank=None
                )
                parameter.to_local().copy_(local.to_local())
                names[name] = source
    missing = parameters.keys() - names.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing tensors: {sorted(missing)[:8]}")
    return names


@torch.no_grad()
def save_checkpoint(model, ema, names, args, step):
    """Write EMA weights in BF16 in the format loaded by `goalwam.GoalWAM`."""
    rank = dist.get_rank()
    final = args.output / f"checkpoint-{step:06d}"
    staging = args.output / f".checkpoint-{step:06d}"
    if rank == 0:
        shutil.rmtree(staging, ignore_errors=True)
        (staging / "transformer").mkdir(parents=True)
    shard, size, total_size, written = {}, 0, 0, []

    def flush():
        nonlocal shard, size
        if shard:
            file = f"part-{len(written):05d}.safetensors"
            save_file(shard, staging / "transformer" / file, metadata={"format": "pt"})
            written.append((file, list(shard)))
            shard, size = {}, 0

    for name, parameter in model.named_parameters():
        value = parameter.detach()
        if name in ema:
            value = DTensor.from_local(
                ema[name], value.device_mesh, value.placements, shape=value.shape, stride=value.stride()
            )
        full = value.full_tensor().to(torch.bfloat16)
        if rank == 0:
            shard[names[name]] = full.cpu()
            nbytes = full.numel() * full.element_size()
            size += nbytes
            total_size += nbytes
            if size >= SHARD_BYTES:
                flush()
    if rank == 0:
        flush()
        weight_map = {}
        for i, (file, keys) in enumerate(written):
            target = f"model-{i + 1:05d}-of-{len(written):05d}.safetensors"
            (staging / "transformer" / file).rename(staging / "transformer" / target)
            weight_map.update(dict.fromkeys(keys, target))
        index = {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}
        (staging / "transformer/model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
        shutil.copy(args.pretrained / "transformer/config.json", staging / "transformer/config.json")
        for name in ("vae", "text_tokenizer"):
            shutil.copytree(args.pretrained / name, staging / name)
        for name in ("processor.json", "sampling.json"):
            shutil.copy(args.pretrained / name, staging / name)
        shutil.copy(args.output / "robot.json", staging / "robot.json")
        staging.rename(final)
        print(f"saved {final}", flush=True)
    dist.barrier()


def _ensure_unix_socket_tmpdir():
    probe = Path(tempfile.gettempdir()) / f".goalwam-sock-{os.getpid()}"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.bind(str(probe))
    except OSError:
        os.environ["TMPDIR"] = tempfile.tempdir = "/tmp"
    finally:
        probe.unlink(missing_ok=True)


def main():
    args = parse_args()
    if args.workers > 0:
        _ensure_unix_socket_tmpdir()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    accumulation, remainder = divmod(args.batch_size, args.micro_batch_size * world)
    if remainder or not accumulation:
        raise ValueError("--batch-size must be a multiple of --micro-batch-size times the GPU count")
    args.output.mkdir(parents=True, exist_ok=True)
    robot_json = args.output / "robot.json"
    if rank == 0 and not robot_json.exists():
        write_robot_json(args.data, args.pretrained, robot_json, horizon=HORIZON)
    dist.barrier()

    config = read_config(args.pretrained / "transformer/config.json", ModelConfig)
    model = build_model(config, init_device_mesh("cuda", (world,)))
    names = load_weights(model, args.pretrained / "transformer")
    vae = VideoEncoder(args.pretrained / "vae", torch.device("cuda", torch.cuda.current_device()))
    dataset = RoboTwinDataset(
        args.data, args.pretrained, robot_json, goal_cache=args.goal_cache, seed=args.seed * 1000 + rank
    )
    batches = infinite_loader(dataset, args.micro_batch_size, args.seed * 1000 + rank, args.workers)
    generator = torch.Generator().manual_seed(args.seed * 1000 + rank)
    cuda_generator = torch.Generator(device="cuda").manual_seed(args.seed * 1000 + rank)

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    groups = {}
    for name, parameter in trainable:
        lr = FAST_LR if any(key in name for key in FAST) else LR
        groups.setdefault((lr, WEIGHT_DECAY if parameter.ndim > 1 else 0.0), []).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": p, "lr": lr, "weight_decay": wd} for (lr, wd), p in groups.items()],
        betas=(0.9, 0.99),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    ema = {name: p.detach().to_local().clone() for name, p in trainable}
    gamma = np.roots([1, 7, 16 - EMA_RATE**-2, 12 - EMA_RATE**-2]).real.max()

    for step in range(1, args.steps + 1):
        start = time.time()
        totals = torch.zeros(3, device="cuda")
        for _ in range(accumulation):
            x = prepare(next(batches), vae, config, generator, cuda_generator)
            loss, video_loss, action_loss = model(x)
            (loss / accumulation).backward()
            totals += torch.stack((loss.detach(), video_loss, action_loss)) / accumulation
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        norm = norm.full_tensor() if isinstance(norm, DTensor) else norm
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Non-finite gradient norm at step {step}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        beta = (1 - 1 / (step + 1)) ** (gamma + 1)
        torch._foreach_lerp_(list(ema.values()), [p.detach().to_local() for _, p in trainable], 1 - beta)
        dist.all_reduce(totals)
        if rank == 0:
            loss, video_loss, action_loss = (totals / world).tolist()
            record = {
                "step": step,
                "loss": loss,
                "video_loss": video_loss,
                "action_loss": action_loss,
                "grad_norm": norm.item(),
                "lr": LR * schedule(step - 1),
                "seconds": round(time.time() - start, 2),
                "memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            }
            print(json.dumps(record), flush=True)
            with open(args.output / "metrics.jsonl", "a") as file:
                file.write(json.dumps(record) + "\n")
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(model, ema, names, args, step)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
