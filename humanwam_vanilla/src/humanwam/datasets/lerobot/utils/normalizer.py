# Modified by Open-HumanScale contributors: package namespace renamed to humanwam.
from typing import Literal, Dict, Annotated, Union, Any, List, Tuple, Optional
import torch
import json
import os
from collections import defaultdict
import numpy as np
from humanwam.utils.logging_config import get_logger

from humanwam.utils.pytorch_utils import dict_apply

logger = get_logger(__name__)

ConstConstStr = Annotated[str, "format: 'const_min/const_max', where const_min and const_max give the constant range"]
QuantileModeStr = Annotated[str, "format: 'quantile' or 'quantile:qLO/qHI' (e.g. 'quantile:q005/q995'); reads the true sampled quantiles stored under stats['global_quantiles'/'stepwise_quantiles'] and maps [qLO, qHI] -> [-1, 1]"]
NormMode = Union[Literal["min/max", "q01/q99", "z-score", "quantile"], ConstConstStr, QuantileModeStr]

DEFAULT_QUANTILE_MODE_BOUNDS = ("q005", "q995")


def parse_quantile_mode(mode: str) -> Optional[Tuple[str, str]]:
    """Return (lo_key, hi_key) for a 'quantile[:qLO/qHI]' mode string, else None."""
    if not isinstance(mode, str) or not mode.startswith("quantile"):
        return None
    if mode == "quantile":
        return DEFAULT_QUANTILE_MODE_BOUNDS
    bounds = mode[len("quantile:"):].split("/")
    return bounds[0], bounds[1]

class LinearNormalizer:
    def __init__(
            self,
            shape_meta,
            use_stepwise_action_norm,
            default_mode: NormMode,
            exception_mode: Dict[str, Dict[str, NormMode]],
            stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
            post_normalization_clips: Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]] = None,
        ):
        super().__init__()
        self.normalizers = {"action": {}, "state": {}}
        self.stats = stats

        for meta in shape_meta["action"]:
            key = meta["key"]

            if use_stepwise_action_norm:
                cur_stats = {k.removeprefix("stepwise_"): v for k, v in stats["action"][key].items() if k.startswith("stepwise_")}
            else:
                cur_stats = {k.removeprefix("global_"): v for k, v in stats["action"][key].items() if k.startswith("global_")}

            if exception_mode is not None and "action" in exception_mode and key in exception_mode["action"]:
                cur_mode = exception_mode["action"][key]
            else:
                cur_mode = default_mode

            self.normalizers["action"][key] = SingleFieldLinearNormalizer(
                stats=cur_stats,
                mode=cur_mode,
                output_clips=(
                    post_normalization_clips.get("action", {}).get(key)
                    if post_normalization_clips is not None
                    else None
                ),
            )

        for meta in shape_meta["state"]:
            key = meta["key"]
            cur_stats = {k.removeprefix("global_"): v for k, v in stats["state"][key].items() if k.startswith("global_")}

            if exception_mode is not None and "state" in exception_mode and key in exception_mode["state"]:
                cur_mode = exception_mode["state"][key]
            else:
                cur_mode = default_mode

            self.normalizers["state"][key] = SingleFieldLinearNormalizer(
                stats=cur_stats,
                mode=cur_mode,
                output_clips=(
                    post_normalization_clips.get("state", {}).get(key)
                    if post_normalization_clips is not None
                    else None
                ),
            )

    def get_stats(self):
        stats = {
            "action": {key: norm.get_stats() for key, norm in self.normalizers["action"].items()},
            "state": {key: norm.get_stats() for key, norm in self.normalizers["state"].items()}
        }
        return stats

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if "action" in batch:
            for key, norm in self.normalizers["action"].items():
                batch["action"][key] = norm.forward(batch["action"][key])

        for key, norm in self.normalizers["state"].items():
            batch["state"][key] = norm.forward(batch["state"][key])

        return batch

    def backward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        for key, norm in self.normalizers["action"].items():
            batch["action"][key] = norm.backward(batch["action"][key])

        for key, norm in self.normalizers["state"].items():
            batch["state"][key] = norm.backward(batch["state"][key])

        return batch


class SingleFieldLinearNormalizer:
    std_reg = 1e-8
    range_tol = 1e-4
    output_max = 1.0
    output_min = -1.0
    def __init__(
        self,
        stats,
        mode: NormMode = "min/max",
        output_clips: Optional[List[Dict[str, Any]]] = None,
    ):
        self.stats = stats
        self.mode = mode

        if mode == "z-score":
            input_mean, input_std = stats["mean"], stats["std"]
            scale = 1.0 / (input_std + self.std_reg)
            offset = - input_mean / (input_std + self.std_reg)
        else:
            if mode == "min/max":
                input_min, input_max = stats["min"], stats["max"]
            elif mode == "q01/q99":
                input_min, input_max = stats["q01"], stats["q99"]
            elif parse_quantile_mode(mode) is not None:
                lo_key, hi_key = parse_quantile_mode(mode)
                quantiles = stats.get("quantiles")
                input_min = torch.as_tensor(quantiles[lo_key]).clone()
                input_max = torch.as_tensor(quantiles[hi_key]).clone()
            else:
                # parse const_min/const_max
                input_min, input_max = map(float, mode.split("/"))
                input_min = torch.full_like(stats["min"], input_min)
                input_max = torch.full_like(stats["max"], input_max)

            input_range = input_max - input_min
            ignore_dim = input_range < self.range_tol
            input_range[ignore_dim] = self.output_max - self.output_min
            scale = (self.output_max - self.output_min) / input_range
            offset = self.output_min - scale * input_min
            offset[ignore_dim] = (self.output_max + self.output_min) / 2 - input_min[ignore_dim]

        self.scale = scale
        self.offset = offset
        self.output_clips = []
        seen_indices = set()
        for clip in output_clips or []:
            index = int(clip["index"])
            lower = float(clip["min"])
            upper = float(clip["max"])
            seen_indices.add(index)
            self.output_clips.append((index, lower, upper))

    def _clip_selected_outputs(self, x: torch.Tensor) -> torch.Tensor:
        if not self.output_clips:
            return x
        x = x.clone()
        for index, lower, upper in self.output_clips:
            x[..., index] = x[..., index].clamp(min=lower, max=upper)
        return x

    def get_stats(self):
        return self.stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale + self.offset
        x = torch.clamp(x, -5.0, 5.0)
        return self._clip_selected_outputs(x)
    def backward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._clip_selected_outputs(x)
        x = (x - self.offset) / self.scale
        return x

def save_dataset_stats_to_json(dataset_stats: dict, file_path: str):

    def convert_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (defaultdict, dict)):
            return {k: convert_tensor(v) for k, v in dict(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_tensor(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    serializable_stats = convert_tensor(dataset_stats)

    # Atomic write: serialize to a temp file in the same dir, then os.replace().
    # The rename is atomic on POSIX, so a concurrent reader (e.g. the val dataset
    # on a non-main rank, built right after the train dataset writes this) never
    # observes a partially written file -> kills the JSONDecodeError race.
    tmp_path = f"{file_path}.tmp.{os.getpid()}"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_stats, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, file_path)

def load_dataset_stats_from_json(file_path: str,
                                 try_convert_tensor: bool = True) -> Dict[str, Any]:

    def is_numeric_list(obj):
        if isinstance(obj, list):
            if not obj:
                return True
            first = obj[0]
            if isinstance(first, (int, float)):
                return all(isinstance(x, (int, float)) for x in obj)
            elif isinstance(first, list):
                return all(is_numeric_list(item) for item in obj)
            else:
                return False
        return False

    def convert_back_to_tensor(obj):
        if isinstance(obj, dict):
            return {k: convert_back_to_tensor(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if is_numeric_list(obj):
                try:
                    arr = np.array(obj)
                    return torch.from_numpy(arr)
                except Exception:
                    return [convert_back_to_tensor(item) for item in obj]
            else:
                return [convert_back_to_tensor(item) for item in obj]
        else:
            return obj
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if try_convert_tensor:
        data = convert_back_to_tensor(data)

    data = dict_apply(
        data,
        lambda x: x.to(torch.float32) if isinstance(x, torch.Tensor) else x,
    )

    return data
