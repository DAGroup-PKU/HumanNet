"""Function-preserving bottleneck adapters for FastWAM's MoT experts.

The adapters in this module are sequential residual adapters attached after a
transformer block's mixed self-attention update and/or its FFN update.  They are
token-wise: no operation mixes the sequence dimension, so the MoT attention
mask remains the sole authority for cross-view and cross-expert information
flow.

Injection happens before loading a post-training base checkpoint.  The final
``up`` projection is exactly zero initialized, which makes every injected
adapter an identity while allowing a normal checkpoint (with no adapter keys)
to initialize the frozen backbone.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch
from torch import nn


BOTTLENECK_ADAPTER_FORMAT = "fastwam_bottleneck_adapter_v1"
BOTTLENECK_ADAPTER_SUFFIX = ".bottleneck.pt"
DEFAULT_BOTTLENECK_ADAPTER_EXPERTS = ("video", "action")
DEFAULT_BOTTLENECK_ADAPTER_SITES = ("self_attn", "ffn")
_SITE_ATTRIBUTES = {
    "self_attn": "post_self_attn_adapter",
    "ffn": "post_ffn_adapter",
}
_ADAPTER_PARAMETER_SUFFIXES = (
    "down.weight",
    "down.bias",
    "up.weight",
    "up.bias",
)


def _join_module_key(module_name: str, parameter_name: str) -> str:
    return f"{module_name}.{parameter_name}" if module_name else parameter_name


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _derived_parameter_seed(init_seed: int, *parts: object) -> int:
    """Derive a stable per-module seed without Python's randomized ``hash``."""

    identity = ":".join([str(int(init_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    # torch.Generator.manual_seed accepts a signed 64-bit seed on every backend.
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63 - 1)


class BottleneckAdapter(nn.Module):
    """A pre-normalized bottleneck residual initialized as the identity.

    ``LayerNorm`` is deliberately affine-free.  Consequently each instance has
    exactly four parameter tensors: down/up weights and biases.  ``residual_scale``
    is a fixed Python float rather than a trainable gate.
    """

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int,
        *,
        activation: str = "gelu_tanh",
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        eps: float = 1e-6,
        init_seed: int = 42,
        parameter_seed: int | None = None,
        expert_name: str = "",
        site: str = "",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        activation = str(activation).strip().lower()

        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.activation_name = activation
        self.dropout_probability = float(dropout)
        self.residual_scale = float(residual_scale)
        self.init_seed = int(init_seed)
        self.parameter_seed = int(
            self.init_seed if parameter_seed is None else parameter_seed
        )
        self.expert_name = str(expert_name)
        self.site = str(site)

        self.norm = nn.LayerNorm(
            self.hidden_dim,
            eps=float(eps),
            elementwise_affine=False,
            device=device,
            dtype=dtype,
        )
        self.down = nn.Linear(
            self.hidden_dim,
            self.bottleneck_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.activation = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(self.dropout_probability)
        self.up = nn.Linear(
            self.bottleneck_dim,
            self.hidden_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.reset_adapter_parameters()

    def reset_adapter_parameters(self) -> None:
        generator = torch.Generator(device=self.down.weight.device)
        generator.manual_seed(self.parameter_seed)
        nn.init.kaiming_uniform_(
            self.down.weight,
            a=math.sqrt(5),
            generator=generator,
        )
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up(
            self.dropout(
                self.activation(
                    self.down(self.norm(x))
                )
            )
        )
        return x + delta * self.residual_scale

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, bottleneck_dim={self.bottleneck_dim}, "
            f"activation={self.activation_name!r}, dropout={self.dropout_probability}, "
            f"residual_scale={self.residual_scale}, init_seed={self.init_seed}"
        )


@dataclass(frozen=True)
class BottleneckAdapterInjectionReport:
    module_names: tuple[str, ...]
    bottleneck_dims: Mapping[str, int]
    trainable_parameters: int
    parameter_tensor_count: int

    @property
    def module_count(self) -> int:
        return len(self.module_names)


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


def _block_hidden_dim(block: nn.Module) -> int:
    hidden_dim = getattr(block, "hidden_dim", None)
    q_projection = getattr(getattr(block, "self_attn", None), "q", None)
    q_in_features = getattr(q_projection, "in_features", None)
    if hidden_dim is None:
        hidden_dim = q_in_features
    return int(hidden_dim)


def _normalize_injection_contract(
    *,
    experts: Sequence[str],
    sites: Sequence[str],
    bottleneck_dims: Mapping[str, int] | None,
    reduction_factor: int | None,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, int] | None, int | None]:
    experts = tuple(str(name) for name in experts)
    sites = tuple(str(name) for name in sites)

    resolved_dims = None
    if bottleneck_dims is not None:
        resolved_dims = {
            str(name): int(value) for name, value in dict(bottleneck_dims).items()
        }
    else:
        reduction_factor = 32 if reduction_factor is None else int(reduction_factor)
    return experts, sites, resolved_dims, reduction_factor


def inject_bottleneck_adapters(
    mot: nn.Module,
    *,
    experts: Sequence[str] = DEFAULT_BOTTLENECK_ADAPTER_EXPERTS,
    sites: Sequence[str] = DEFAULT_BOTTLENECK_ADAPTER_SITES,
    bottleneck_dims: Mapping[str, int] | None = None,
    reduction_factor: int | None = None,
    activation: str = "gelu_tanh",
    dropout: float = 0.0,
    residual_scale: float = 1.0,
    init_seed: int = 42,
) -> BottleneckAdapterInjectionReport:
    """Attach independent adapters to selected sites of every selected expert."""

    experts, sites, resolved_dims, reduction_factor = _normalize_injection_contract(
        experts=experts,
        sites=sites,
        bottleneck_dims=bottleneck_dims,
        reduction_factor=reduction_factor,
    )
    mixtures = getattr(mot, "mixtures", None)

    injected: list[str] = []
    trainable_parameters = 0
    parameter_tensor_count = 0
    final_dims: dict[str, int] = {}
    for expert_name in experts:
        expert = mixtures[expert_name]
        blocks = getattr(expert, "blocks", None)
        expert_hidden_dim = _block_hidden_dim(blocks[0])
        if resolved_dims is not None:
            bottleneck_dim = int(resolved_dims[expert_name])
        else:
            bottleneck_dim = expert_hidden_dim // reduction_factor
        final_dims[expert_name] = bottleneck_dim

        for block_index, block in enumerate(blocks):
            q_projection = getattr(getattr(block, "self_attn", None), "q", None)
            weight = getattr(q_projection, "weight", None)
            for site in sites:
                attribute = _SITE_ATTRIBUTES[site]
                parameter_seed = _derived_parameter_seed(
                    int(init_seed), expert_name, block_index, site
                )
                adapter = BottleneckAdapter(
                    hidden_dim=expert_hidden_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    dropout=dropout,
                    residual_scale=residual_scale,
                    eps=float(getattr(getattr(block, "norm1", None), "eps", 1e-6)),
                    init_seed=int(init_seed),
                    parameter_seed=parameter_seed,
                    expert_name=expert_name,
                    site=site,
                    device=weight.device,
                    dtype=weight.dtype,
                )
                adapter.train(block.training)
                setattr(block, attribute, adapter)
                full_name = (
                    f"mixtures.{expert_name}.blocks.{block_index}.{attribute}"
                )
                injected.append(full_name)
                parameters = tuple(adapter.parameters())
                trainable_parameters += sum(parameter.numel() for parameter in parameters)
                parameter_tensor_count += len(parameters)

    return BottleneckAdapterInjectionReport(
        module_names=tuple(injected),
        bottleneck_dims=final_dims,
        trainable_parameters=int(trainable_parameters),
        parameter_tensor_count=int(parameter_tensor_count),
    )


def iter_bottleneck_adapters(
    module: nn.Module,
) -> Iterator[tuple[str, BottleneckAdapter]]:
    for name, child in module.named_modules():
        if isinstance(child, BottleneckAdapter):
            yield name, child


def set_bottleneck_adapter_trainable(
    model: nn.Module,
    *,
    trainable_modules: Sequence[str] = (),
) -> tuple[str, ...]:
    """Freeze the model, then enable adapter tensors and an exact allowlist."""

    model.requires_grad_(False)
    trainable_names: set[str] = set()
    adapted = list(iter_bottleneck_adapters(model))
    for module_name, adapter in adapted:
        adapter.train()
        adapter.requires_grad_(True)
        for parameter_name, parameter in adapter.named_parameters():
            if parameter.requires_grad:
                trainable_names.add(_join_module_key(module_name, parameter_name))

    trainable_modules = tuple(str(path) for path in trainable_modules)
    for module_path in trainable_modules:
        allowed = _get_child(model, module_path)
        allowed.train()
        allowed.requires_grad_(True)
        for parameter_name, parameter in allowed.named_parameters():
            if parameter.requires_grad:
                trainable_names.add(_join_module_key(module_path, parameter_name))
    return tuple(sorted(trainable_names))


def _adapter_parameter_keys(module: nn.Module) -> set[str]:
    return {
        _join_module_key(module_name, parameter_name)
        for module_name, adapter in iter_bottleneck_adapters(module)
        for parameter_name, _ in adapter.named_parameters()
    }


def bottleneck_adapter_state_dict(
    module: nn.Module,
    *,
    trainable_modules: Sequence[str] = (),
    cpu: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """Collect only bottleneck adapter tensors plus explicitly allowed modules."""

    trainable_modules = tuple(str(path) for path in trainable_modules)
    adapter_keys = _adapter_parameter_keys(module)
    prefixes = tuple(f"{path}." for path in trainable_modules)
    full_state = module.state_dict()
    selected: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, tensor in full_state.items():
        if name not in adapter_keys and not any(name.startswith(prefix) for prefix in prefixes):
            continue
        value = tensor.detach().clone()
        if cpu:
            value = value.cpu()
        selected[name] = value
    return selected


def _uniform_adapter_value(adapted, attribute: str):
    values = {getattr(child, attribute) for _, child in adapted}
    return next(iter(values))


def _current_adapter_contract(module: nn.Module) -> dict[str, object]:
    adapted = list(iter_bottleneck_adapters(module))
    experts = _ordered_unique([child.expert_name for _, child in adapted])
    sites = _ordered_unique([child.site for _, child in adapted])
    dims: dict[str, int] = {}
    for _, child in adapted:
        dims.setdefault(child.expert_name, child.bottleneck_dim)
    return {
        "experts": list(experts),
        "sites": list(sites),
        "bottleneck_dims": dims,
        "activation": _uniform_adapter_value(adapted, "activation_name"),
        "dropout": float(_uniform_adapter_value(adapted, "dropout_probability")),
        "residual_scale": float(
            _uniform_adapter_value(adapted, "residual_scale")
        ),
        "init_seed": int(_uniform_adapter_value(adapted, "init_seed")),
        "module_count": len(adapted),
        "parameter_tensor_count": sum(
            len(tuple(child.parameters())) for _, child in adapted
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for _, child in adapted
            for parameter in child.parameters()
        ),
    }


def bottleneck_adapter_payload(
    module: nn.Module,
    *,
    trainable_modules: Sequence[str] = (),
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    contract = _current_adapter_contract(module)
    contract["trainable_modules"] = [str(path) for path in trainable_modules]
    supplied = dict(metadata or {})
    contract.update(supplied)
    return {
        "format": BOTTLENECK_ADAPTER_FORMAT,
        "state_dict": bottleneck_adapter_state_dict(
            module,
            trainable_modules=trainable_modules,
            cpu=True,
        ),
        "metadata": contract,
    }


def save_bottleneck_adapter(
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
            bottleneck_adapter_payload(
                module,
                trainable_modules=trainable_modules,
                metadata=metadata,
            ),
            temporary,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def load_bottleneck_adapter_payload(path: str | Path) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    payload = dict(payload)
    return payload


def is_bottleneck_adapter_checkpoint(path: str | Path) -> bool:
    return str(path).endswith(BOTTLENECK_ADAPTER_SUFFIX)


def load_bottleneck_adapter(
    module: nn.Module,
    adapter: str | Path | Mapping[str, object],
) -> object:
    payload = (
        load_bottleneck_adapter_payload(adapter)
        if isinstance(adapter, (str, Path))
        else dict(adapter)
    )
    state = payload.get("state_dict")
    incompatible = module.load_state_dict(state, strict=False)
    return incompatible


def _bottleneck_base_binding(metadata: Mapping[str, object]) -> str:
    return str(metadata["base_checkpoint"]["path"])


def resolve_bottleneck_base_checkpoint(
    payload: Mapping[str, object],
    *,
    override: str | Path | None = None,
) -> Path:
    path = override if override is not None else _bottleneck_base_binding(payload["metadata"])
    return Path(path).expanduser()


__all__ = [
    "BOTTLENECK_ADAPTER_FORMAT",
    "BOTTLENECK_ADAPTER_SUFFIX",
    "DEFAULT_BOTTLENECK_ADAPTER_EXPERTS",
    "DEFAULT_BOTTLENECK_ADAPTER_SITES",
    "BottleneckAdapter",
    "BottleneckAdapterInjectionReport",
    "bottleneck_adapter_payload",
    "bottleneck_adapter_state_dict",
    "inject_bottleneck_adapters",
    "is_bottleneck_adapter_checkpoint",
    "iter_bottleneck_adapters",
    "load_bottleneck_adapter",
    "load_bottleneck_adapter_payload",
    "resolve_bottleneck_base_checkpoint",
    "save_bottleneck_adapter",
    "set_bottleneck_adapter_trainable",
]
