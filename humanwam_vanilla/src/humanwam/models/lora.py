"""Small, dependency-free LoRA utilities for FastWAM's MoT experts.

The implementation deliberately subclasses :class:`torch.nn.Linear`.  The
pretrained ``weight`` and ``bias`` therefore keep their historical state-dict
keys when a projection is adapted; only ``lora_A`` and ``lora_B`` are new.
This is important for loading a normal FastWAM checkpoint before a LoRA
continuation and for exporting a merged checkpoint for the existing runtime.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_MOT_LORA_TARGETS = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "ffn.0",
    "ffn.2",
)
LORA_ADAPTER_FORMAT = "fastwam_lora_adapter_v1"


def load_lora_adapter_payload(path: str | Path) -> dict[str, object]:
    """Load a FastWAM adapter payload.

    This deliberately does not accept an arbitrary state dict.  Keeping the
    production envelope mandatory prevents a small adapter from being applied
    to an unrecorded base model or a different LoRA topology by accident.
    """

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    payload = dict(payload)
    return payload


def is_lora_adapter_checkpoint(path: str | Path) -> bool:
    """Cheaply identify trainer-produced adapters by their explicit suffix."""

    return str(path).endswith(".adapter.pt")


@dataclass(frozen=True)
class LoRAInjectionReport:
    module_names: tuple[str, ...]
    trainable_parameters: int
    base_parameters: int

    @property
    def module_count(self) -> int:
        return len(self.module_names)


class LoRALinear(nn.Linear):
    """A state-dict-compatible ``nn.Linear`` with a low-rank residual.

    ``lora_B`` is initialized to zero, so injection is function preserving.
    The pretrained ``weight`` and ``bias`` objects are reused by
    :meth:`from_linear` instead of copied.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:

        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, self.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, self.rank, device=device, dtype=dtype)
        )
        self.merged = False
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        # Same fan-in initialization convention as the common PEFT LoRA path;
        # B=0 is the property that makes the injected model exactly identical.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        adapted = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        # Preserve Parameter identity as well as values.  This matters if a
        # caller injects before constructing an optimizer (the trainer does).
        adapted.weight = linear.weight
        adapted.bias = linear.bias
        adapted.train(linear.training)
        return adapted

    def lora_delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        self.weight.add_(
            self.lora_delta_weight().to(
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        )
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        self.weight.sub_(
            self.lora_delta_weight().to(
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        )
        self.merged = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.linear(input, self.weight, self.bias)
        if self.merged:
            return output
        low_rank = F.linear(F.linear(self.lora_dropout(input), self.lora_A), self.lora_B)
        return output + low_rank * self.scaling


def _get_child(module: nn.Module, path: str) -> nn.Module:
    if not path:
        return module
    current = module
    for part in path.split("."):
        if part.isdigit() and isinstance(current, (nn.ModuleList, nn.Sequential)):
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _replace_child(parent: nn.Module, name: str, child: nn.Module) -> None:
    if name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(name)] = child
    else:
        setattr(parent, name, child)


def inject_mot_lora(
    mot: nn.Module,
    *,
    experts: Sequence[str] = ("video", "action"),
    target_modules: Sequence[str] = DEFAULT_MOT_LORA_TARGETS,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> LoRAInjectionReport:
    """Inject LoRA into selected projections of every block in MoT experts.

    ``target_modules`` contains paths relative to each transformer block, for
    example ``self_attn.q`` or ``ffn.0``.
    """
    mixtures = getattr(mot, "mixtures", None)

    injected: list[str] = []
    trainable_parameters = 0
    base_parameters = 0
    for expert_name in experts:
        expert = mixtures[expert_name]
        blocks = getattr(expert, "blocks", None)
        for block_index, block in enumerate(blocks):
            for target_path in target_modules:
                parent_path, separator, child_name = str(target_path).rpartition(".")
                if not separator:
                    parent_path, child_name = "", str(target_path)
                parent = _get_child(block, parent_path)
                child = _get_child(parent, child_name)
                adapted = LoRALinear.from_linear(
                    child,
                    rank=int(rank),
                    alpha=float(alpha),
                    dropout=float(dropout),
                )
                _replace_child(parent, child_name, adapted)
                full_name = (
                    f"mixtures.{expert_name}.blocks.{block_index}.{target_path}"
                )
                injected.append(full_name)
                trainable_parameters += adapted.lora_A.numel() + adapted.lora_B.numel()
                base_parameters += adapted.weight.numel()
                if adapted.bias is not None:
                    base_parameters += adapted.bias.numel()

    return LoRAInjectionReport(
        module_names=tuple(injected),
        trainable_parameters=int(trainable_parameters),
        base_parameters=int(base_parameters),
    )


def inject_lora_adapters(
    mot: nn.Module,
    *,
    experts: Sequence[str] = ("video", "action"),
    target_modules: Sequence[str] = DEFAULT_MOT_LORA_TARGETS,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> LoRAInjectionReport:
    """Stable public alias for :func:`inject_mot_lora`."""

    return inject_mot_lora(
        mot,
        experts=experts,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )


def iter_lora_modules(module: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    for name, child in module.named_modules():
        if isinstance(child, LoRALinear):
            yield name, child


def set_lora_trainable(
    model: nn.Module,
    *,
    trainable_modules: Sequence[str] = (),
) -> tuple[str, ...]:
    """Freeze the model, then enable LoRA tensors and exact module allowlist paths."""

    model.requires_grad_(False)
    trainable_names: set[str] = set()
    for module_name, adapted in iter_lora_modules(model):
        adapted.lora_A.requires_grad_(True)
        adapted.lora_B.requires_grad_(True)
        trainable_names.add(f"{module_name}.lora_A")
        trainable_names.add(f"{module_name}.lora_B")
    for module_path in trainable_modules:
        allowed = _get_child(model, str(module_path))
        allowed.train()
        allowed.requires_grad_(True)
        prefix = f"{module_path}." if module_path else ""
        for parameter_name, parameter in allowed.named_parameters():
            if parameter.requires_grad:
                trainable_names.add(f"{prefix}{parameter_name}")
    return tuple(sorted(trainable_names))


def lora_state_dict(
    module: nn.Module,
    *,
    trainable_modules: Sequence[str] = (),
    cpu: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """Return an adapter-only state dict, optionally including tiny modules."""
    full_state = module.state_dict()
    lora_keys = {
        f"{name}.{suffix}" if name else suffix
        for name, _ in iter_lora_modules(module)
        for suffix in ("lora_A", "lora_B")
    }
    prefixes = tuple(f"{str(path)}." for path in trainable_modules)
    selected = OrderedDict()
    for name, tensor in full_state.items():
        if name not in lora_keys and not any(name.startswith(prefix) for prefix in prefixes):
            continue
        value = tensor.detach().clone()
        if cpu:
            value = value.cpu()
        selected[name] = value
    return selected


def adapter_payload(
    module: nn.Module,
    *,
    trainable_modules: Sequence[str] = (),
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    adapted = list(iter_lora_modules(module))
    ranks = sorted({child.rank for _, child in adapted})
    alphas = sorted({child.alpha for _, child in adapted})
    return {
        "format": LORA_ADAPTER_FORMAT,
        "state_dict": lora_state_dict(
            module,
            trainable_modules=trainable_modules,
            cpu=True,
        ),
        "metadata": {
            "ranks": ranks,
            "alphas": alphas,
            "module_count": len(adapted),
            "trainable_modules": list(trainable_modules),
            **dict(metadata or {}),
        },
    }


def save_lora_adapter(
    module: nn.Module,
    path: str | Path,
    *,
    trainable_modules: Sequence[str] = (),
    metadata: Mapping[str, object] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(
            adapter_payload(
                module,
                trainable_modules=trainable_modules,
                metadata=metadata,
            ),
            temporary,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def load_lora_adapter(
    module: nn.Module,
    adapter: str | Path | Mapping[str, object],
) -> object:
    payload = (
        load_lora_adapter_payload(adapter)
        if isinstance(adapter, (str, Path))
        else dict(adapter)
    )
    state = payload.get("state_dict")
    incompatible = module.load_state_dict(state, strict=False)
    return incompatible


def _adapter_base_binding(metadata: Mapping[str, object]) -> str:
    return str(metadata["base_checkpoint"]["path"])


def resolve_lora_base_checkpoint(
    payload: Mapping[str, object],
    *,
    override: str | Path | None = None,
) -> Path:
    path = override if override is not None else _adapter_base_binding(payload["metadata"])
    return Path(path).expanduser()


def _checkpoint_state_target(
    checkpoint: Mapping[str, object], adapter_key: str
) -> tuple[Mapping[str, torch.Tensor], str]:
    """Map a full-model adapter key onto the legacy checkpoint envelope."""

    if adapter_key.startswith("mot."):
        state = checkpoint.get("mot")
        key = adapter_key[len("mot.") :]
    elif adapter_key.startswith("dit."):
        # ``FastWAM.dit`` aliases ``FastWAM.mot``.  Trainer-produced adapters
        # normally use the first registered name (``mot``), but accept the alias.
        state = checkpoint.get("mot")
        key = adapter_key[len("dit.") :]
    elif adapter_key.startswith("proprio_encoder."):
        state = checkpoint.get("proprio_encoder")
        key = adapter_key[len("proprio_encoder.") :]
    elif adapter_key.startswith("source_proprio_encoders."):
        state = checkpoint.get("source_proprio_encoders")
        key = adapter_key[len("source_proprio_encoders.") :]
    return state, key


@torch.no_grad()
def merge_lora_adapter_checkpoint(
    base_checkpoint: str | Path,
    adapter: str | Path | Mapping[str, object],
    output: str | Path,
) -> Path:
    """Merge an adapter into a legacy runtime-compatible FastWAM checkpoint.

    The implementation operates directly on the checkpoint dictionaries, so
    exporting does not require constructing the 6B-parameter model or loading
    VAE/text weights.  Every adapter key is accounted for and all LoRA tensors
    are removed from the exported envelope.
    """

    payload = (
        load_lora_adapter_payload(adapter)
        if isinstance(adapter, (str, Path))
        else dict(adapter)
    )
    metadata = payload.get("metadata")
    resolved_base = resolve_lora_base_checkpoint(
        payload,
        override=base_checkpoint,
    )
    rank = int(metadata.get("rank", 0))
    alpha = float(metadata.get("alpha", 0.0))
    state = payload.get("state_dict")

    # mmap avoids eagerly materializing a second 12 GB copy.  Modified target
    # tensors use private copy-on-write pages before the atomic torch.save.
    checkpoint = torch.load(
        resolved_base,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    checkpoint = dict(checkpoint)

    lora_a = {str(name)[: -len(".lora_A")]: tensor for name, tensor in state.items() if str(name).endswith(".lora_A")}
    lora_b = {str(name)[: -len(".lora_B")]: tensor for name, tensor in state.items() if str(name).endswith(".lora_B")}

    scaling = alpha / rank
    consumed: set[str] = set()
    for module_name in sorted(lora_a):
        a = lora_a[module_name]
        b = lora_b[module_name]
        target_state, target_key = _checkpoint_state_target(
            checkpoint, f"{module_name}.weight"
        )
        weight = target_state.get(target_key)
        delta = torch.matmul(b.float(), a.float()).mul_(scaling)
        weight.add_(delta.to(dtype=weight.dtype))
        consumed.update({f"{module_name}.lora_A", f"{module_name}.lora_B"})
    for adapter_key, tensor in state.items():
        adapter_key = str(adapter_key)
        if adapter_key in consumed:
            continue
        target_state, target_key = _checkpoint_state_target(checkpoint, adapter_key)
        target = target_state.get(target_key)
        target.copy_(tensor.to(dtype=target.dtype))
        consumed.add(adapter_key)

    if "step" in metadata:
        checkpoint["step"] = int(metadata["step"])
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


@torch.no_grad()
def merge_lora_weights(module: nn.Module, *, unload: bool = False) -> tuple[str, ...]:
    """Merge every adapter into base weights, optionally remove LoRA modules."""

    adapted = list(iter_lora_modules(module))
    for _, child in adapted:
        child.merge()
    if unload:
        # Deepest-first is defensive for arbitrary nested LoRA modules.
        for name, child in sorted(adapted, key=lambda pair: pair[0].count("."), reverse=True):
            parent_path, _, child_name = name.rpartition(".")
            parent = _get_child(module, parent_path)
            plain = nn.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            plain.weight = child.weight
            plain.bias = child.bias
            plain.train(child.training)
            _replace_child(parent, child_name, plain)
    return tuple(name for name, _ in adapted)


@torch.no_grad()
def unmerge_lora_weights(module: nn.Module) -> tuple[str, ...]:
    adapted = list(iter_lora_modules(module))
    for _, child in adapted:
        child.unmerge()
    return tuple(name for name, _ in adapted)


__all__ = [
    "DEFAULT_MOT_LORA_TARGETS",
    "LORA_ADAPTER_FORMAT",
    "LoRAInjectionReport",
    "LoRALinear",
    "adapter_payload",
    "inject_lora_adapters",
    "inject_mot_lora",
    "is_lora_adapter_checkpoint",
    "iter_lora_modules",
    "load_lora_adapter",
    "load_lora_adapter_payload",
    "lora_state_dict",
    "merge_lora_adapter_checkpoint",
    "merge_lora_weights",
    "resolve_lora_base_checkpoint",
    "save_lora_adapter",
    "set_lora_trainable",
    "unmerge_lora_weights",
]
