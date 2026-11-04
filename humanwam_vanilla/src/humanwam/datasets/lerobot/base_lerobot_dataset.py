# Modified by Open-HumanScale contributors: package namespace renamed to humanwam.
import json
import os
import torch
import numpy as np
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import List, Dict, Optional, Any, DefaultDict
from tqdm import tqdm
from .content_addressed_cache import ContentAddressedFileCache
from .lerobot.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset
from .episode_exclusion import (
    load_episode_exclusion_manifest as _load_episode_exclusion_manifest)
from .episode_segments import (
    load_episode_segment_manifest)

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import traceback
from humanwam.utils.logging_config import get_logger
from .processors.base_processor import BaseProcessor

logger = get_logger(__name__)

NORM_STATS_WEIGHTINGS = ("legacy", "loss")

# Dataset stats are I/O heavy and normally read from a shared filesystem.  Keep the
# default conservative, and retain a hard ceiling so an accidental environment value
# cannot enqueue hundreds of concurrent parquet reads on a login/shared node.
DEFAULT_STATS_MAX_WORKERS = 8
# PFS-backed Clean-50 corpora can benefit from more outstanding parquet reads
# than the conservative default.  Keep 32 as the safe default while allowing a
# launcher-local override for one-off offline stats generation.
MAX_STATS_MAX_WORKERS = int(os.environ.get("FASTWAM_MAX_STATS_WORKERS", "32"))
MAX_STATS_PENDING_FUTURES = 128


def _resolve_temporal_resampling_by_root(
    dataset_dirs: List[str],
    temporal_resampling: Optional[Any],
) -> Optional[List[Any]]:
    """Return per-root specs in ``dataset_dirs`` order, validating root routing.

    The public config is deliberately root keyed so adding/reordering datasets cannot
    silently attach a speed policy to the wrong source. Exact-root entries and
    path-ancestry ``rules`` are supported. Basename/substring matching is
    intentionally unsupported.
    """
    if temporal_resampling is None:
        return None

    resolved_roots = [
        str(Path(path).expanduser().resolve()) for path in dataset_dirs
    ]
    configured: Dict[str, Any] = {}

    if isinstance(temporal_resampling, Mapping) and "rules" in temporal_resampling:
        rules = temporal_resampling["rules"]
        parsed_rules = []
        for position, rule in enumerate(rules):
            selectors = [key for key in ("dataset_root", "root_prefix") if key in rule]
            selector = selectors[0]
            selector_path = Path(str(rule[selector])).expanduser().resolve()
            if "spec" in rule:
                spec = rule["spec"]
            else:
                spec = {key: value for key, value in rule.items() if key != selector}
            parsed_rules.append((selector, selector_path, spec))

        match_counts = [0] * len(parsed_rules)
        for root in resolved_roots:
            root_path = Path(root)
            matches = []
            for position, (selector, selector_path, spec) in enumerate(parsed_rules):
                matched = root_path == selector_path
                if selector == "root_prefix":
                    matched = matched or selector_path in root_path.parents
                if matched:
                    matches.append((position, spec))
            position, spec = matches[0]
            match_counts[position] += 1
            configured[root] = spec
        return [configured[root] for root in resolved_roots]

    if isinstance(temporal_resampling, Mapping):
        entries = temporal_resampling.items()
    elif isinstance(temporal_resampling, Sequence) and not isinstance(
        temporal_resampling, (str, bytes)
    ):
        parsed_entries = []
        for position, entry in enumerate(temporal_resampling):
            parsed_entries.append((entry["dataset_root"], entry["spec"]))
        entries = parsed_entries

    for root, spec in entries:
        resolved_root = str(Path(str(root)).expanduser().resolve())
        configured[resolved_root] = spec
    return [configured[root] for root in resolved_roots]


def _apply_episode_exclusion_manifest(
    metas: List[Any],
    pools: Dict[str, List[int]],
    manifest: Optional[str | Path],
) -> Dict[str, List[int]]:
    """Validate exclusions against metadata and remove them from episode pools."""
    manifest_path, manifest_roots = _load_episode_exclusion_manifest(manifest)
    if manifest_path is None:
        return pools

    configured_roots = {
        str(Path(meta.root).expanduser().resolve()): meta for meta in metas
    }

    filtered = {repo_id: list(indices) for repo_id, indices in pools.items()}
    total_excluded_episodes = 0
    total_excluded_frames = 0
    for root, entry in manifest_roots.items():
        meta = configured_roots[root]
        excluded = entry["excluded_episode_indices"]
        excluded_frames = sum(
            int(meta.episodes[index]["length"]) for index in excluded
        )

        pool = filtered[meta.repo_id]
        filtered[meta.repo_id] = [index for index in pool if index not in excluded]
        removed_from_pool = len(pool) - len(filtered[meta.repo_id])
        total_excluded_episodes += len(excluded)
        total_excluded_frames += excluded_frames
        logger.info(
            "Episode exclusion root=%s excluded=%s episodes/%s frames "
            "removed_from_candidate_pool=%s",
            root,
            f"{len(excluded):,}",
            f"{excluded_frames:,}",
            f"{removed_from_pool:,}",
        )

    logger.info(
        "Using episode exclusion manifest %s: excluded=%s episodes/%s frames",
        manifest_path,
        f"{total_excluded_episodes:,}",
        f"{total_excluded_frames:,}",
    )
    return filtered


def _resolve_stats_limit(
    explicit_value: Optional[int],
    env_name: str,
    default_value: int,
    hard_limit: int,
) -> int:
    """Resolve and validate a stats concurrency setting."""
    value = explicit_value
    if value is None:
        raw_value = os.environ.get(env_name)
        if raw_value is not None:
            value = int(raw_value)
        else:
            value = default_value
    if value > hard_limit:
        logger.warning(f"Capping {env_name}={value} at the safety limit {hard_limit}")
        value = hard_limit
    return value


def _summarize_episode_tensor(value: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Reduce one transformed episode to the tensors used by the legacy aggregator."""
    # Move only the small reductions to CPU.  The full transformed episode is released
    # inside the worker as soon as this function returns.
    return {
        "min": value.amin(0).detach().cpu(),
        "max": value.amax(0).detach().cpu(),
        "mean": value.mean(0).detach().cpu(),
        # Keep torch.var's default correction=1 to match the previous implementation.
        "var": value.var(0).detach().cpu(),
        "q01": torch.quantile(value, 0.01, dim=0, keepdim=False).detach().cpu(),
        "q99": torch.quantile(value, 0.99, dim=0, keepdim=False).detach().cpu(),
    }


def _dual_arm_action_valid_dims(
    action: torch.Tensor,
    source_hand_tag: Optional[str],
    hand_action_dim: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Return valid transformed action channels for a per-hand episode.

    Ego EEF actions are transformed to two contiguous, equal-size arm blocks.  The
    current 20D representation is ``left[0:10] | right[10:20]``.  Returning ``None``
    for absent/``both`` tags preserves the legacy all-channel behavior for EgoDex,
    EgoVerse, and every non-per-hand dataset.
    """
    if source_hand_tag not in ("left", "right"):
        return None

    action_dim = action.shape[-1]
    eef_dim = action_dim if hand_action_dim is None else int(hand_action_dim)
    half = eef_dim // 2
    valid_dims = torch.ones(action_dim, dtype=torch.bool)
    if source_hand_tag == "left":
        valid_dims[half:eef_dim] = False
    else:
        valid_dims[:half] = False
    return valid_dims


class _StreamingTensorStats:
    """Online equivalent of stacking all per-episode tensor summaries.

    Episodes remain equally weighted, regardless of their frame counts.  Accumulation
    uses float64 for numerical stability, while finalized tensors are cast back to the
    dtype produced by the original per-episode reductions.
    """

    def __init__(self):
        self.count = 0
        self.dtype = None
        self.stepwise_min = None
        self.stepwise_max = None
        self.stepwise_q01 = None
        self.stepwise_q99 = None
        self.mean = None
        self.mean_m2 = None
        self.var_sum = None
        self.valid_episode_count = None

    def update(
        self,
        summary: Dict[str, torch.Tensor],
        valid_dims: Optional[torch.Tensor] = None,
    ) -> None:
        mean = summary["mean"]
        var = summary["var"]
        if self.count == 0:
            self.dtype = mean.dtype
            self.stepwise_min = torch.full_like(summary["min"], torch.inf)
            self.stepwise_max = torch.full_like(summary["max"], -torch.inf)
            self.stepwise_q01 = torch.full_like(summary["q01"], torch.inf)
            self.stepwise_q99 = torch.full_like(summary["q99"], -torch.inf)
            self.mean = torch.zeros_like(mean, dtype=torch.float64)
            self.mean_m2 = torch.zeros_like(mean, dtype=torch.float64)
            self.var_sum = torch.zeros_like(var, dtype=torch.float64)
            self.valid_episode_count = torch.zeros(mean.shape[-1], dtype=torch.int64)

        if valid_dims is None:
            valid_dims = torch.ones(mean.shape[-1], dtype=torch.bool)
        else:
            valid_dims = torch.as_tensor(valid_dims, dtype=torch.bool, device="cpu")

        # The mask is per action channel and applies to every action step.  torch.where
        # is intentional: multiplying an inactive NaN/Inf channel by zero would still
        # contaminate the online sums.
        broadcast_shape = (1,) * (mean.ndim - 1) + (mean.shape[-1],)
        valid = valid_dims.view(broadcast_shape)
        self.stepwise_min = torch.where(
            valid, torch.minimum(self.stepwise_min, summary["min"]), self.stepwise_min
        )
        self.stepwise_max = torch.where(
            valid, torch.maximum(self.stepwise_max, summary["max"]), self.stepwise_max
        )
        self.stepwise_q01 = torch.where(
            valid, torch.minimum(self.stepwise_q01, summary["q01"]), self.stepwise_q01
        )
        self.stepwise_q99 = torch.where(
            valid, torch.maximum(self.stepwise_q99, summary["q99"]), self.stepwise_q99
        )

        # Welford's update avoids cancellation when pose coordinates have a large
        # offset but a small spread.  Each valid episode contributes one equally
        # weighted observation to every step of its active action channels.
        mean64 = mean.to(torch.float64)
        old_count = self.valid_episode_count.to(torch.float64).view(broadcast_shape)
        new_count = old_count + valid.to(torch.float64)
        safe_mean = torch.where(valid, mean64, self.mean)
        delta = safe_mean - self.mean
        updated_mean = self.mean + torch.where(
            valid, delta / new_count.clamp_min(1.0), 0.0
        )
        self.mean_m2.add_(torch.where(valid, delta * (safe_mean - updated_mean), 0.0))
        self.mean = updated_mean
        self.var_sum.add_(torch.where(valid, var.to(torch.float64), 0.0))
        self.valid_episode_count.add_(valid_dims.to(torch.int64))
        self.count += 1

    def finalize(self) -> Dict[str, torch.Tensor]:

        # Per-hand sources can contribute different episode counts to left/right
        # channels.  Counts are per dimension but shared across action steps.
        episode_count = self.valid_episode_count.to(torch.float64)
        stepwise_count = episode_count.view(
            (1,) * (self.mean.ndim - 1) + (self.mean.shape[-1],)
        )
        stepwise_mean64 = self.mean
        # Legacy formula:
        # mean_episode(var_episode + (mean_episode - aggregate_mean) ** 2).
        stepwise_var64 = (
            self.var_sum + self.mean_m2
        ) / stepwise_count

        step_count = self.mean.shape[0]
        global_count = episode_count * step_count
        global_mean64 = stepwise_mean64.mean(0)
        # Combine the per-step Welford groups.  Every step has the same per-channel
        # valid episode count because a hand tag masks complete action channels.
        global_mean_m2 = (
            self.mean_m2
            + stepwise_count * (stepwise_mean64 - global_mean64.unsqueeze(0)).square()
        ).sum(0)
        global_var64 = (
            self.var_sum.sum(0) + global_mean_m2
        ) / global_count

        # Round-off can only make a true variance microscopically negative.
        stepwise_var64 = stepwise_var64.clamp_min_(0.0)
        global_var64 = global_var64.clamp_min_(0.0)

        return {
            "stepwise_min": self.stepwise_min,
            "stepwise_max": self.stepwise_max,
            "global_min": self.stepwise_min.amin(0),
            "global_max": self.stepwise_max.amax(0),
            # Preserve the legacy q01/q99 semantics: extrema over per-episode
            # quantiles, rather than a quantile over all frames.
            "stepwise_q01": self.stepwise_q01,
            "stepwise_q99": self.stepwise_q99,
            "global_q01": self.stepwise_q01.amin(0),
            "global_q99": self.stepwise_q99.amax(0),
            "stepwise_mean": stepwise_mean64.to(self.dtype),
            "stepwise_std": stepwise_var64.sqrt_().to(self.dtype),
            "global_mean": global_mean64.to(self.dtype),
            "global_std": global_var64.sqrt_().to(self.dtype),
        }


TRUE_QUANTILE_LEVELS: Dict[str, float] = {
    "q001": 0.001,
    "q005": 0.005,
    "q01": 0.01,
    "q05": 0.05,
    "q95": 0.95,
    "q99": 0.99,
    "q995": 0.995,
    "q999": 0.999,
}


class _ReservoirQuantileSampler:
    """Uniform reservoir sample (Algorithm R, batched) of valid tokens per column.

    Maintains one reservoir of ``size`` values for each column of a ``[cols, dim]``
    grid (``cols`` is the horizon length for stepwise quantiles and ``1`` for pooled
    global quantiles).  Items arrive in batches as ``values[n, cols, dim]`` with a
    boolean ``valid`` mask; each valid item is the ``t``-th item seen for its column
    and is kept with probability ``size / (t + 1)`` (deterministically kept while the
    reservoir is not full).  Ties inside one batch fall back to scatter semantics,
    which is statistically harmless for ``n << size``.  Sampling is unweighted over
    valid tokens: it matches the loss-weighted marginal exactly whenever every anchor
    contributes the same number of valid tokens (full-horizon corpora).
    """

    def __init__(self, cols: int, dim: int, size: int, seed: int = 0):
        self.size = int(size)
        # Row ``size`` is a scratch slot that swallows rejected items.
        self.values = torch.full((self.size + 1, cols, dim), torch.nan, dtype=torch.float32)
        self.count = torch.zeros(cols, dim, dtype=torch.int64)
        self.generator = torch.Generator().manual_seed(int(seed))

    def update(self, values: torch.Tensor, valid: torch.Tensor) -> None:
        values = values.detach().to(torch.float32, copy=False).cpu()
        valid = valid.to(torch.bool)
        n = values.shape[0]
        if n == 0:
            return
        rank = valid.to(torch.int64).cumsum(0) - 1  # per-item rank among valid items
        t = self.count.unsqueeze(0) + rank
        u = torch.rand(values.shape, generator=self.generator, dtype=torch.float64)
        slot = torch.floor(u * (t + 1).to(torch.float64)).to(torch.int64)
        slot = torch.where(t < self.size, t, slot)
        keep = valid & (slot < self.size)
        slot = torch.where(keep, slot, torch.full_like(slot, self.size))
        self.values.scatter_(0, slot, values)
        self.values[self.size] = torch.nan
        self.count += valid.sum(0)

    def state(self) -> Dict[str, torch.Tensor]:
        filled = self.count.clamp_max(self.size)
        return {"values": self.values[: self.size], "count": self.count, "filled": filled}

    def quantiles(self, levels: Dict[str, float] = TRUE_QUANTILE_LEVELS) -> Dict[str, torch.Tensor]:
        if not levels:
            return {}
        sample = self.values[: self.size].to(torch.float64)
        # Unfilled slots stay NaN so nanquantile only sees observed items.
        # Compute all requested levels in one sort. Repeating the sort eight times
        # dominates large, masked camera22 statistics without changing the sample.
        quantiles = torch.nanquantile(
            sample, torch.tensor(list(levels.values()), dtype=sample.dtype), dim=0
        )
        return {name: quantiles[i] for i, name in enumerate(levels)}


def _summarize_loss_weighted_tensor(
    value: torch.Tensor,
    token_weights: torch.Tensor,
    valid_dims: Optional[torch.Tensor] = None,
    normalize_dims: bool = False,
    return_values: bool = False,
) -> Dict[str, torch.Tensor]:
    """Summarize one episode using explicit per-anchor/token weights.

    ``token_weights`` has shape ``[anchor, step]``.  For actions in loss-aligned
    mode, each anchor's valid horizon weights sum to one; padding has zero weight.
    ``normalize_dims`` additionally divides by the number of valid action channels,
    exactly matching the loss's second masked mean (10 active per-hand channels versus
    20 channels for dual-arm episodes). Quantiles remain the legacy conservative
    extrema-over-episode-quantiles, but each per-episode quantile is computed from
    valid (non-padding) tokens only.

    ``return_values=True`` additionally returns the raw ``[anchor, step, dim]``
    float32 values and their ``valid`` mask so a downstream
    ``_StreamingLossWeightedTensorStats(quantile_reservoir_size=...)`` can draw a
    reservoir sample for true global/stepwise quantiles.
    """

    value = value.detach().cpu()
    token_weights = torch.as_tensor(token_weights, dtype=torch.float64, device="cpu")
    if valid_dims is None:
        valid_dims = torch.ones(value.shape[-1], dtype=torch.bool)
    else:
        valid_dims = torch.as_tensor(valid_dims, dtype=torch.bool, device="cpu")

    token_valid = token_weights > 0
    combined_valid = token_valid.unsqueeze(-1) & valid_dims.view(1, 1, -1)
    value64 = value.to(torch.float64)
    safe_value = torch.where(combined_valid, value64, 0.0)
    dim_weights = valid_dims.to(torch.float64)
    if normalize_dims:
        dim_weights = dim_weights / dim_weights.sum().clamp_min(1.0)
    weights = token_weights.unsqueeze(-1) * dim_weights.view(1, 1, -1)
    weight_sum = weights.sum(0)

    # Vectorize across all horizon positions.  A Python loop here would issue 64
    # quantile kernels per episode for a 32-step action horizon, which is prohibitive
    # for million-episode corpora.  The transformed ego data is required to be finite;
    # NaN is used only as the nanquantile exclusion sentinel for padding/hand masks.
    summary_valid = weight_sum > 0
    stepwise_min = torch.where(combined_valid, value, torch.inf).amin(0)
    stepwise_max = torch.where(combined_valid, value, -torch.inf).amax(0)
    quantile_values = torch.where(combined_valid, value, torch.nan)
    stepwise_q01 = torch.nanquantile(quantile_values, 0.01, dim=0)
    stepwise_q99 = torch.nanquantile(quantile_values, 0.99, dim=0)
    stepwise_q01 = torch.where(summary_valid, stepwise_q01, torch.inf)
    stepwise_q99 = torch.where(summary_valid, stepwise_q99, -torch.inf)

    summary = {
        "weight_sum": weight_sum,
        "sum": (safe_value * weights).sum(0),
        "sum_sq": (safe_value.square() * weights).sum(0),
        "min": stepwise_min,
        "max": stepwise_max,
        "q01": stepwise_q01,
        "q99": stepwise_q99,
    }
    if return_values:
        summary["values"] = value.to(torch.float32)
        summary["valid"] = combined_valid
    return summary


class _StreamingLossWeightedTensorStats:
    """Stream exact weighted first/second moments without retaining episodes.

    With ``quantile_reservoir_size > 0`` the accumulator additionally keeps one
    Algorithm-R reservoir per horizon position and one pooled reservoir, from which
    ``finalize`` reports true (sampled) ``stepwise_quantiles`` / ``global_quantiles``
    under new keys.  Summaries must then carry ``values``/``valid`` (see
    ``_summarize_loss_weighted_tensor(return_values=True)``).  The legacy
    ``*_q01/q99`` extrema keys are always kept unchanged.
    """

    def __init__(self, quantile_reservoir_size: int = 0, quantile_seed: int = 0):
        self.dtype = None
        self.weight_sum = None
        self.sum = None
        self.sum_sq = None
        self.stepwise_min = None
        self.stepwise_max = None
        self.stepwise_q01 = None
        self.stepwise_q99 = None
        self.quantile_reservoir_size = int(quantile_reservoir_size)
        self.quantile_seed = int(quantile_seed)
        self.stepwise_reservoir: Optional[_ReservoirQuantileSampler] = None
        self.global_reservoir: Optional[_ReservoirQuantileSampler] = None

    def update(self, summary: Dict[str, torch.Tensor]) -> None:
        if self.weight_sum is None:
            self.dtype = summary["min"].dtype
            self.weight_sum = torch.zeros_like(summary["weight_sum"], dtype=torch.float64)
            self.sum = torch.zeros_like(summary["sum"], dtype=torch.float64)
            self.sum_sq = torch.zeros_like(summary["sum_sq"], dtype=torch.float64)
            self.stepwise_min = torch.full_like(summary["min"], torch.inf)
            self.stepwise_max = torch.full_like(summary["max"], -torch.inf)
            self.stepwise_q01 = torch.full_like(summary["q01"], torch.inf)
            self.stepwise_q99 = torch.full_like(summary["q99"], -torch.inf)
            if self.quantile_reservoir_size > 0:
                steps, dim = summary["min"].shape
                self.stepwise_reservoir = _ReservoirQuantileSampler(
                    steps, dim, self.quantile_reservoir_size, seed=self.quantile_seed
                )
                self.global_reservoir = _ReservoirQuantileSampler(
                    1, dim, self.quantile_reservoir_size, seed=self.quantile_seed + 1
                )

        valid = summary["weight_sum"] > 0
        self.weight_sum.add_(summary["weight_sum"])
        self.sum.add_(summary["sum"])
        self.sum_sq.add_(summary["sum_sq"])
        self.stepwise_min = torch.where(
            valid, torch.minimum(self.stepwise_min, summary["min"]), self.stepwise_min
        )
        self.stepwise_max = torch.where(
            valid, torch.maximum(self.stepwise_max, summary["max"]), self.stepwise_max
        )
        self.stepwise_q01 = torch.where(
            valid, torch.minimum(self.stepwise_q01, summary["q01"]), self.stepwise_q01
        )
        self.stepwise_q99 = torch.where(
            valid, torch.maximum(self.stepwise_q99, summary["q99"]), self.stepwise_q99
        )
        if self.stepwise_reservoir is not None:
            values, token_valid = summary["values"], summary["valid"]
            self.stepwise_reservoir.update(values, token_valid)
            self.global_reservoir.update(
                values.reshape(-1, 1, values.shape[-1]),
                token_valid.reshape(-1, 1, token_valid.shape[-1]),
            )

    def finalize(self) -> Dict[str, torch.Tensor]:

        global_weight = self.weight_sum.sum(0)

        global_mean64 = self.sum.sum(0) / global_weight
        global_var64 = (
            self.sum_sq.sum(0) / global_weight - global_mean64.square()
        ).clamp_min_(0.0)
        global_min = self.stepwise_min.amin(0)
        global_max = self.stepwise_max.amax(0)
        global_q01 = self.stepwise_q01.amin(0)
        global_q99 = self.stepwise_q99.amax(0)

        step_valid = self.weight_sum > 0
        safe_step_weight = self.weight_sum.clamp_min(1.0)
        stepwise_mean64 = self.sum / safe_step_weight
        stepwise_var64 = (
            self.sum_sq / safe_step_weight - stepwise_mean64.square()
        ).clamp_min_(0.0)

        # A horizon position can be padding for every episode in a small corpus.  It
        # never contributes to loss; fill its unused stepwise fields with global stats
        # so serialized normalization remains finite.
        stepwise_mean64 = torch.where(step_valid, stepwise_mean64, global_mean64.unsqueeze(0))
        stepwise_var64 = torch.where(step_valid, stepwise_var64, global_var64.unsqueeze(0))
        stepwise_min = torch.where(step_valid, self.stepwise_min, global_min.unsqueeze(0))
        stepwise_max = torch.where(step_valid, self.stepwise_max, global_max.unsqueeze(0))
        stepwise_q01 = torch.where(step_valid, self.stepwise_q01, global_q01.unsqueeze(0))
        stepwise_q99 = torch.where(step_valid, self.stepwise_q99, global_q99.unsqueeze(0))

        result = {
            "stepwise_min": stepwise_min,
            "stepwise_max": stepwise_max,
            "global_min": global_min,
            "global_max": global_max,
            "stepwise_q01": stepwise_q01,
            "stepwise_q99": stepwise_q99,
            "global_q01": global_q01,
            "global_q99": global_q99,
            "stepwise_mean": stepwise_mean64.to(self.dtype),
            "stepwise_std": stepwise_var64.sqrt_().to(self.dtype),
            "global_mean": global_mean64.to(self.dtype),
            "global_std": global_var64.sqrt_().to(self.dtype),
        }
        if self.stepwise_reservoir is not None:
            global_quantiles = {
                name: q.squeeze(0).to(self.dtype)
                for name, q in self.global_reservoir.quantiles().items()
            }
            stepwise_quantiles = {}
            for name, q in self.stepwise_reservoir.quantiles().items():
                # Padding-only horizon positions fall back to the pooled quantile.
                q = torch.where(step_valid, q, global_quantiles[name].to(q.dtype).unsqueeze(0))
                stepwise_quantiles[name] = q.to(self.dtype)
            result["global_quantiles"] = global_quantiles
            result["stepwise_quantiles"] = stepwise_quantiles
            result["quantile_reservoir"] = {
                "method": "reservoir_algorithm_r",
                "size": self.quantile_reservoir_size,
                "seed": self.quantile_seed,
                "levels": dict(TRUE_QUANTILE_LEVELS),
                "global_items_seen": self.global_reservoir.count.squeeze(0).clone(),
                "stepwise_items_seen_min": int(self.stepwise_reservoir.count.min()),
                "stepwise_items_seen_max": int(self.stepwise_reservoir.count.max()),
            }
        return result


class BaseLerobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs: List[str],

        # shapes
        shape_meta: Dict[str, Any],
        action_size: int = 1,
        past_action_size: int = 0, # Excludes the current frame
        obs_size: int = 1, # should be
        past_obs_size: int = 0,

        # train vs val
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        seed: int = 42,

        # sampling
        global_sample_stride: int = 1,
        lazy_parquet_loading: bool = False,
        video_backend: Optional[str] = None,
        content_cache_dir: Optional[str] = None,
        content_cache_manifest: Optional[str] = None,
        content_cache_max_bytes: int = 0,

        # optional episode whitelist: restrict the candidate episode pool (applied
        # BEFORE the train/val split). Either a list[int] or a path to a JSON list.
        # Only supported for a single dataset_dir.
        episode_subset: Optional[Any] = None,
        # Optional version-1 manifest that removes bad episodes without mutating
        # the on-disk LeRobot corpus. Applied after an optional whitelist and before
        # the deterministic train/val split.
        episode_exclusion_manifest: Optional[str] = None,
        # Optional logical sub-episode manifest. Each half-open frame range becomes
        # an independent sampling boundary and supplies its own task caption.
        episode_segment_manifest: Optional[str] = None,
        # Optional per-native-root virtual timeline policy. Root keyed on purpose;
        # see _resolve_temporal_resampling_by_root.
        temporal_resampling: Optional[Any] = None,
        # Native-frame offset applied only to action queries.  EgoStandard stores
        # action[t] == state[t], so +1 makes the targets state[t+1] while keeping
        # observation.state[t] as the chunk anchor.  Other datasets keep 0.
        action_time_offset_frames: int = 0,
        video_timestamp_tolerance_s: float = 1e-4,
    ):
        resolved_dataset_dirs = [
            str(Path(path).expanduser().resolve()) for path in dataset_dirs
        ]

        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.past_action_size = past_action_size
        self.action_time_offset_frames = action_time_offset_frames
        self.obs_size = obs_size
        self.past_obs_size = past_obs_size
        self.processor = None  # Will be set externally
        self.content_file_cache = (
            ContentAddressedFileCache(
                content_cache_dir,
                manifest_path=content_cache_manifest,
                max_bytes=content_cache_max_bytes,
            )
            if content_cache_dir is not None
            else None
        )
        metas = []
        for ds_dir in dataset_dirs:
            ds_root = Path(ds_dir)
            repo_id = ds_dir
            meta = LeRobotDatasetMetadata(
                repo_id=repo_id,
                root=ds_root,
                skip_episode_stats=lazy_parquet_loading,
            )
            metas.append(meta)

        fps_list = [m.fps for m in metas]
        fps = fps_list[0]

        self.global_sample_stride = global_sample_stride

        self.val_set_proportion = val_set_proportion
        self.is_training_set = is_training_set

        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]

        delta_timestamps = {}
        for meta in self.image_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"observation.images.{key}" if key != "default" else "observation.images"
            delta_timestamps[meta["lerobot_key"]] = [
                (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
            ]

        for meta in self.state_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"observation.state.{key}" if key != "default" else "observation.state"
            delta_timestamps[meta["lerobot_key"]] = [
                (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
            ]

        for meta in self.action_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"action.{key}" if key != "default" else "action"
            delta_timestamps[meta["lerobot_key"]] = [
                (t * global_sample_stride + action_time_offset_frames) / fps
                for t in range(-past_action_size, -past_action_size + action_size)
            ]

        # Optional per-frame camera extrinsic (ego cam-delta): auto-detect the
        # `observation.cam_extrinsic` column and query it over the OBS window, so the
        # training-time anchor-camera transform can read cam_extrinsic[0] as anchor.
        # Datasets without the column (robotwin joint/eef) are unaffected.
        self.cam_extrinsic_key = None
        _feat0 = getattr(metas[0], "features", None) or metas[0].info.get("features", {})
        if "observation.cam_extrinsic" in _feat0:
            self.cam_extrinsic_key = "observation.cam_extrinsic"
            delta_timestamps[self.cam_extrinsic_key] = [
                (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
            ]

        # Resolve the candidate episode pool per dataset. By default this is every
        # episode; an optional `episode_subset` whitelist restricts it (e.g. RoboTwin
        # clean-only). The whitelist is applied BEFORE the train/val split so both
        # splits draw only from the selected episodes.
        if episode_subset is not None:
            if isinstance(episode_subset, str):
                with open(episode_subset) as f:
                    episode_subset = json.load(f)
            episode_subset = [int(e) for e in episode_subset]
            total = metas[0].total_episodes
            pools = {metas[0].repo_id: list(episode_subset)}
            logger.info(
                f"Using episode_subset: {len(episode_subset)}/{total} episodes for {metas[0].repo_id}"
            )
        else:
            pools = {meta.repo_id: list(range(meta.total_episodes)) for meta in metas}

        pools = _apply_episode_exclusion_manifest(
            metas,
            pools,
            episode_exclusion_manifest,
        )

        (
            segment_manifest_path,
            segment_manifest_roots,
            segment_manifest_document,
        ) = load_episode_segment_manifest(
            episode_segment_manifest,
            include_document=True,
        )
        if segment_manifest_path is not None:
            for meta, resolved_root in zip(metas, resolved_dataset_dirs, strict=True):
                root_segments = segment_manifest_roots[resolved_root]["segments"]
                if root_segments is None:
                    logger.info(
                        "Logical segment manifest passes through all candidate "
                        "physical episodes under %s",
                        resolved_root,
                    )
                    continue
                referenced = {
                    int(segment["episode_index"])
                    for segment in root_segments
                }
                pools[meta.repo_id] = [
                    episode_index
                    for episode_index in pools[meta.repo_id]
                    if episode_index in referenced
                ]
                logger.info(
                    "Logical segment view selects %d physical episodes under %s",
                    len(pools[meta.repo_id]),
                    resolved_root,
                )

        episodes = {}
        if val_set_proportion < 1e-6:
            for meta in metas:
                episodes.update({meta.repo_id: list(pools[meta.repo_id])})
        else:
            for meta in metas:
                pool = pools[meta.repo_id]
                split_idx = int(len(pool) * (1 - val_set_proportion))
                # random shuffle episode indices before splitting
                episode_indices = list(pool)
                rng = np.random.default_rng(seed)
                rng.shuffle(episode_indices)
                if self.is_training_set:
                    episodes.update({meta.repo_id: [episode_indices[i] for i in range(split_idx)]})
                else:
                    episodes.update({meta.repo_id: [episode_indices[i] for i in range(split_idx, len(pool))]})

        self.temporal_resampling = _resolve_temporal_resampling_by_root(
            self.dataset_dirs,
            temporal_resampling,
        )
        self.multi_dataset = MultiLeRobotDataset(
            dataset_dirs=self.dataset_dirs,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            tolerances_s={root: float(video_timestamp_tolerance_s) for root in self.dataset_dirs},
            video_backend=video_backend,
            lazy_parquet_loading=lazy_parquet_loading,
            metadatas=metas,
            temporal_resampling=self.temporal_resampling,
            codebook_registry=(
                segment_manifest_document.get("codebooks")
                if segment_manifest_document is not None
                else None
            ),
            file_cache=self.content_file_cache,
            episode_segments=(
                [
                    segment_manifest_roots[root]["segments"]
                    for root in resolved_dataset_dirs
                ]
                if segment_manifest_path is not None
                else None
            ),
        )
        self.episode_segment_manifest = (
            str(segment_manifest_path) if segment_manifest_path is not None else None
        )
        self.codebook_schema_version = (
            segment_manifest_document.get("codebook_schema_version")
            if segment_manifest_document is not None
            else None
        )
        self.codebook_registry = (
            segment_manifest_document.get("codebooks")
            if segment_manifest_document is not None
            else None
        )
        self.codebook_annotation_provenance = (
            segment_manifest_document.get("codebook_annotation_provenance")
            if segment_manifest_document is not None
            else None
        )

        # HACK: lerobot 3.0 will fix this
        episode_data_index = []
        end_index = 0
        for dataset in self.multi_dataset._datasets:
            multi_episode_data_index = {
                "from": dataset.episode_data_index["from"] + end_index,
                "to": dataset.episode_data_index["to"] + end_index,
            }
            episode_data_index.append(multi_episode_data_index)
            end_index = multi_episode_data_index["to"][-1]

        self.episode_data_index = {
            "from": torch.cat([dataset["from"] for dataset in episode_data_index]),
            "to": torch.cat([dataset["to"] for dataset in episode_data_index]),
        }

        self.environment_sampling = self._build_environment_sampling(
            segment_manifest_document
        )

        # per-hand supervision.  Metadata already retains source_hand_tag;
        # keep only a corpus-level presence flag instead of duplicating 1.3M entries
        # into another Python dict on every training rank.
        self._hand_map = self._build_hand_map()

    def _build_environment_sampling(
        self, manifest_document: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """Build compact global frame spans for an env-uniform segment manifest."""

        if manifest_document is None:
            return None
        contract = manifest_document.get("environment_sampling")
        if contract is None:
            return None
        raw_order = contract.get("environment_order")
        environment_order = tuple(str(value).strip() for value in raw_order)

        spans: dict[str, list[tuple[int, int]]] = {
            environment_id: [] for environment_id in environment_order
        }
        root_offset = 0
        for dataset in self.multi_dataset._datasets:
            tags = getattr(dataset, "_logical_segment_environment_ids", None)
            starts = dataset.episode_data_index["from"].tolist()
            ends = dataset.episode_data_index["to"].tolist()
            for environment_id, start, end in zip(tags, starts, ends, strict=True):
                global_start = root_offset + int(start)
                global_end = root_offset + int(end)
                existing = spans[environment_id]
                if existing and existing[-1][1] == global_start:
                    existing[-1] = (existing[-1][0], global_end)
                else:
                    existing.append((global_start, global_end))
            root_offset += int(dataset.num_frames)
        capacities = tuple(
            sum(end - start for start, end in spans[environment_id])
            for environment_id in environment_order
        )

        training_total_samples = int(contract.get("training_total_samples", 0))
        samples_per_environment = training_total_samples // len(environment_order)

        result = {
            "mode": "uniform",
            "environment_ids": environment_order,
            "spans": tuple(tuple(spans[value]) for value in environment_order),
            "capacities": capacities,
            "training_total_samples": training_total_samples,
            "samples_per_environment": samples_per_environment,
        }
        active_axis = contract.get("active_axis")
        if active_axis is not None:
            provenance = manifest_document.get("codebook_annotation_provenance") or {}
            result.update(
                {
                    "active_axis": str(active_axis),
                    "environment_ids_sha256": contract["environment_ids_sha256"],
                    "source_annotation_manifest_content_sha256": provenance[
                        "source_annotation_manifest_content_sha256"
                    ],
                    "registry_sha256": provenance["registry_sha256"],
                    "assignment_sha256": provenance["assignment_sha256"],
                    "projected_population_sha256": provenance[
                        "projected_population_sha256"
                    ],
                    "projection_policy_sha256": provenance[
                        "projection_policy_sha256"
                    ],
                    "legacy_projection_before_sha256": provenance[
                        "legacy_projection_before_sha256"
                    ],
                    "legacy_projection_after_sha256": provenance[
                        "legacy_projection_after_sha256"
                    ],
                }
            )
        return result

    def _build_hand_map(self):
        for dataset in self.multi_dataset._datasets:
            logical_tags = getattr(
                dataset, "_logical_segment_source_hand_tags", ()
            )
            if any(tag in ("left", "right", "both") for tag in logical_tags):
                return True
            if any(
                ep.get("source_hand_tag") in ("left", "right", "both")
                for ep in dataset.meta.episodes.values()
            ):
                return True
        return None

    def _get_source_hand_tag(self, dataset_idx: int, source_episode_idx: int):
        if self._hand_map is None:
            return None
        episode = self.multi_dataset._datasets[dataset_idx].meta.episodes.get(
            int(source_episode_idx), {}
        )
        tag = episode.get("source_hand_tag")
        return tag if tag in ("left", "right", "both") else None

    def _get_action(self, meta, lerobot_sample) -> torch.Tensor:
        lerobot_key = meta["lerobot_key"]
        action: torch.Tensor = lerobot_sample[lerobot_key] # [T, action_dim]
        if action.ndim == 1: # for shape of 1, like gripper
            action = action.unsqueeze(-1)
        return action

    def _get_state(self, meta, lerobot_sample) -> torch.Tensor:
        lerobot_key = meta["lerobot_key"]
        state: torch.Tensor = lerobot_sample[lerobot_key]
        if state.ndim == 1: # for shape of 1, like gripper
            state = state.unsqueeze(-1)
        return state

    def _get_image(self, meta, lerobot_sample) -> torch.Tensor:
        lerobot_key = meta["lerobot_key"]
        image: torch.Tensor = lerobot_sample[lerobot_key]
        if image.ndim == 3: # time dim will lost when obs_size is 1
            image = image.unsqueeze(0)
        # Both the float-cache and the float[0,1] decode path already yield
        # float in [0,1]; pass through. (A stray uint8 path is normalized to
        # [0,1].) Downstream ToTensor is a no-op on float -> no round-trips.
        if image.dtype == torch.uint8:
            image = image.float().div_(255.0)
        return image

    def _split_lerobot_sample(self, lerobot_sample) -> Dict[str, Any]:
        return lerobot_sample

    def _locate_episode(self, episode_idx):
        """Map a concatenated episode position to dataset/local/source indices."""
        original_idx = int(episode_idx)
        remaining = original_idx
        for dataset_idx, dataset in enumerate(self.multi_dataset._datasets):
            if remaining < dataset.num_episodes:
                source_episode_idx = dataset._source_episode_index(remaining)
                return dataset_idx, remaining, source_episode_idx
            remaining -= dataset.num_episodes

    def _get_episode_environment_id(self, episode_idx: int) -> Optional[str]:
        """Return the logical segment's env ID for env-uniform condition stats."""

        remaining = int(episode_idx)
        for dataset in self.multi_dataset._datasets:
            if remaining < dataset.num_episodes:
                tags = getattr(dataset, "_logical_segment_environment_ids", None)
                if tags is None:
                    return None
                return tags[remaining]
            remaining -= dataset.num_episodes

    def _get_episode_data(
        self,
        episode_idx,
        action_stride: int = 1,
        include_action_is_pad: bool = False,
    ):
        dataset_idx, _, source_episode_idx = self._locate_episode(episode_idx)
        lerobot_sample = self.multi_dataset.get_episode_data(episode_idx)
        lerobot_sample = self._split_lerobot_sample(lerobot_sample)
        state, action = {}, {}
        for meta in self.state_meta:
            s = self._get_state(meta, lerobot_sample)
            state[meta["key"]] = s.unsqueeze(1).float()
        action_is_pad = None
        for meta in self.action_meta:
            a = self._get_action(meta, lerobot_sample)
            window, cur_action_is_pad = sliding_window_with_replication(
                a,
                self.action_size,
                stride=action_stride,
                return_is_pad=True,
            )
            action[meta["key"]] = window.float()
            if action_is_pad is None:
                action_is_pad = cur_action_is_pad
        batch = {"action": action, "state": state}
        if include_action_is_pad:
            batch["action_is_pad"] = action_is_pad
        # cam_extrinsic anchor per frame (N,1,7): each row's own frame = its window anchor,
        # matching how action sliding-windows anchor at their start (for delta-space stats).
        if self.cam_extrinsic_key is not None and self.cam_extrinsic_key in lerobot_sample:
            ce = lerobot_sample[self.cam_extrinsic_key].float()  # (N,7)
            batch["cam_extrinsic"] = ce.unsqueeze(1)  # (N,1,7)
            # Per-source wrist-C: the stats pass must route through the same constants
            # as training.
            batch["wrist_source_path"] = str(self.dataset_dirs[dataset_idx])

        # Preserve the effective supervision tag for normalization. A logical segment
        # may override a physical left/right tag with `both` after independent
        # validation of both hands.
        if self._hand_map is not None:
            source_hand_tag = lerobot_sample.get("logical_source_hand_tag")
            if source_hand_tag is None:
                source_hand_tag = self._get_source_hand_tag(
                    dataset_idx, source_episode_idx
                )
            if source_hand_tag is not None:
                batch["source_hand_tag"] = source_hand_tag
        return batch

    def _set_return_images(self, flag: bool):
        self.return_images = flag
        self.multi_dataset.set_during_training(flag)

    def __len__(self):
        return self.multi_dataset.num_frames

    def _get_additional_data(self, sample, lerobot_sample):
        return sample


    def _runtime_episode_key(self, sample_idx: int):
        """Map a global frame index to (dataset index, source episode index)."""
        remaining = int(sample_idx)
        for dataset_idx, dataset in enumerate(self.multi_dataset._datasets):
            if remaining < dataset.num_frames:
                locate = getattr(dataset, "_locate_virtual_index", None)
                if not callable(locate):
                    return None
                _, source_episode_idx, _ = locate(remaining)
                return dataset_idx, int(source_episode_idx)
            remaining -= dataset.num_frames
        return None

    def __getitem__(self, idx):

        sample_idx = idx
        lerobot_sample = self._split_lerobot_sample(self.multi_dataset[sample_idx])

        # Get data from lerobot, organized in nested dict
        sample = {
            "idx": sample_idx,
            "task": lerobot_sample["task"],
            "action": {},
            "state": {},
            "images": {},
        }
        for meta in self.state_meta:
            sample["state"][meta["key"]] = self._get_state(meta, lerobot_sample)

        for meta in self.action_meta:
            sample["action"][meta["key"]] = self._get_action(meta, lerobot_sample)

        for meta in self.image_meta:
            sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)

        sample["action_is_pad"] = lerobot_sample[f"{self.action_meta[0]['lerobot_key']}_is_pad"]
        sample["state_is_pad"] = lerobot_sample[f"{self.state_meta[0]['lerobot_key']}_is_pad"]
        sample["image_is_pad"] = lerobot_sample[f"{self.image_meta[0]['lerobot_key']}_is_pad"]

        # per-frame camera extrinsic over the obs window (ego cam-delta anchor input)
        if self.cam_extrinsic_key is not None and self.cam_extrinsic_key in lerobot_sample:
            ce = lerobot_sample[self.cam_extrinsic_key]
            if ce.ndim == 1:
                ce = ce.unsqueeze(0)
            sample["cam_extrinsic"] = ce.float()  # [obs_size, 7]
            # Reuse the observation query only when it covers every action
            # timestamp exactly. The motion transform fails closed otherwise.
            # Do not use cam[1:] blindly: stride and action offset are independent.
            camera_offsets = (
                torch.arange(self.action_size) * self.global_sample_stride
                + self.action_time_offset_frames
                + self.past_obs_size * self.global_sample_stride
            )
            if (bool((camera_offsets % self.global_sample_stride == 0).all())
                    and int(camera_offsets.min()) >= 0
                    and int(camera_offsets.max()) < ce.shape[0] * self.global_sample_stride):
                sample["action_cam_extrinsic"] = ce[
                    camera_offsets // self.global_sample_stride
                ].float()

        sample = self._get_additional_data(sample, lerobot_sample)

        for key in lerobot_sample:
            if key not in sample and "observation" not in key and "action" not in key:
                sample[key] = lerobot_sample[key]

        # Effective per-sample supervision tag. A logical paired segment can override
        # its physical left/right metadata with `both`.
        logical_source_hand_tag = sample.pop("logical_source_hand_tag", None)
        if self._hand_map is not None and "episode_index" in sample:
            di = int(sample["dataset_index"]) if "dataset_index" in sample else 0
            ei = int(sample["episode_index"])
            sample["operating_hand"] = (
                logical_source_hand_tag
                or self._get_source_hand_tag(di, ei)
                or "both"
            )

        # Per-source wrist-C (joint ego pretrain): stamp the source dataset root so the
        # wrist-canonicalize transform can pick the right C per sample. Ego-only (gated
        # on cam_extrinsic); robotwin/other datasets are unaffected.
        if self.cam_extrinsic_key is not None:
            di = int(sample["dataset_index"]) if "dataset_index" in sample else 0
            sample["wrist_source_path"] = str(self.dataset_dirs[di])

        # Preprocess the sample using the processor
        # for quick data loading
        if self.processor is not None:
            sample = self.processor.preprocess(sample)

        # Processors rebuild their output mapping. Keep the exact fetched logical
        # segment label (including after retry) until RobotVideoDataset masks loss.
        if "logical_bad_reason" in lerobot_sample:
            sample["logical_bad_reason"] = lerobot_sample["logical_bad_reason"]

        return sample

    def set_processor(self, processor: BaseProcessor):
        """Set processor instance from external initialization."""
        self.processor = processor
        if self.is_training_set:
            self.processor.train()
        else:
            self.processor.eval()
        return self

    def get_dataset_stats(
        self,
        preprocessor: BaseProcessor,
        max_workers: Optional[int] = None,
        max_pending_futures: Optional[int] = None,
        norm_stats_weighting: str = "legacy",
        full_horizon_query_deltas: Optional[tuple[int, int]] = None,
        quantile_reservoir_size: int = 0,
        quantile_seed: int = 0,
    ):
        """Compute legacy-compatible normalization stats with bounded memory.

        Concurrency can be set per call or through ``FASTWAM_STATS_MAX_WORKERS`` and
        ``FASTWAM_STATS_MAX_PENDING_FUTURES``.  Only a small bounded set of episodes is
        in flight; completed episode summaries are immediately folded into online
        accumulators instead of being retained for the full corpus. ``legacy`` retains
        episode-equal moments over replicated action padding. ``loss`` matches action
        loss element weights and uses frame/anchor-equal state moments.
        """
        norm_stats_weighting = str(norm_stats_weighting).lower()
        loss_weighted = norm_stats_weighting == "loss"
        quantile_reservoir_size = int(quantile_reservoir_size)
        quantile_seed = int(quantile_seed)
        if full_horizon_query_deltas is not None:
            min_delta, max_delta = map(int, full_horizon_query_deltas)
        environment_anchor_weights = None
        if self.environment_sampling is not None:
            environment_anchor_weights = {
                environment_id: 1.0 / capacity
                for environment_id, capacity in zip(
                    self.environment_sampling["environment_ids"],
                    self.environment_sampling["capacities"],
                    strict=True,
                )
            }
        episodes_num = self.multi_dataset.num_episodes

        cpu_count = os.cpu_count() or 1
        max_workers = _resolve_stats_limit(
            max_workers,
            "FASTWAM_STATS_MAX_WORKERS",
            min(DEFAULT_STATS_MAX_WORKERS, cpu_count),
            min(MAX_STATS_MAX_WORKERS, cpu_count),
        )
        max_workers = min(max_workers, episodes_num)
        max_pending_futures = _resolve_stats_limit(
            max_pending_futures,
            "FASTWAM_STATS_MAX_PENDING_FUTURES",
            2 * max_workers,
            MAX_STATS_PENDING_FUTURES,
        )
        # A lower pending limit would silently leave configured workers unused.
        max_pending_futures = min(episodes_num, max(max_workers, max_pending_futures))
        logger.info(
            f"Computing {norm_stats_weighting}-weighted normalization stats with "
            f"{max_workers} workers and at most {max_pending_futures} pending episodes"
        )

        accumulator_type = (
            _StreamingLossWeightedTensorStats if loss_weighted else _StreamingTensorStats
        )
        accumulator_kwargs = (
            {"quantile_reservoir_size": quantile_reservoir_size, "quantile_seed": quantile_seed}
            if loss_weighted
            else {}
        )
        state_stats = {
            meta["key"]: accumulator_type(**accumulator_kwargs) for meta in self.state_meta
        }
        action_stats = {
            meta["key"]: accumulator_type(**accumulator_kwargs) for meta in self.action_meta
        }

        def process_episode(episode_idx):
            batch = self._get_episode_data(
                episode_idx,
                action_stride=self.global_sample_stride if loss_weighted else 1,
                include_action_is_pad=loss_weighted,
            )
            num_anchors = next(iter(batch['state'].values())).shape[0]
            if full_horizon_query_deltas is not None:
                start = max(0, -min_delta)
                end = num_anchors - max(0, max_delta)
                if end <= start:
                    return {"num_anchors": 0, "state": {}, "action": {}}
                def select(value):
                    if isinstance(value, dict):
                        return {key: select(item) for key, item in value.items()}
                    if torch.is_tensor(value) and value.ndim and value.shape[0] == num_anchors:
                        return value[start:end]
                    return value
                batch = select(batch)
                num_anchors = end - start
            batch = preprocessor.action_state_transform(batch)
            source_hand_tag = batch.get("source_hand_tag")
            anchor_weight = 1.0
            if environment_anchor_weights is not None:
                environment_id = self._get_episode_environment_id(episode_idx)
                anchor_weight = environment_anchor_weights[environment_id]

            if loss_weighted:
                # The training loss first averages over valid action channels, then
                # over non-padding horizon tokens.  Thus every anchor has total action
                # element weight one, split as 1/(valid_horizons * valid_channels).
                valid_horizons = (~batch["action_is_pad"]).to(torch.float64)
                valid_horizon_count = valid_horizons.sum(1, keepdim=True).clamp_min(1.0)
                action_token_weights = (
                    valid_horizons / valid_horizon_count * anchor_weight
                )
                return {
                    "num_anchors": num_anchors,
                    "state": {
                        meta["key"]: _summarize_loss_weighted_tensor(
                            batch["state"][meta["key"]],
                            torch.full(
                                batch["state"][meta["key"]].shape[:2],
                                anchor_weight,
                                dtype=torch.float64,
                            ),
                            return_values=quantile_reservoir_size > 0,
                        )
                        for meta in self.state_meta
                    },
                    "action": {
                        meta["key"]: _summarize_loss_weighted_tensor(
                            batch["action"][meta["key"]],
                            action_token_weights,
                            valid_dims=_dual_arm_action_valid_dims(
                                batch["action"][meta["key"]], source_hand_tag
                            ),
                            normalize_dims=True,
                            return_values=quantile_reservoir_size > 0,
                        )
                        for meta in self.action_meta
                    },
                }

            action_summaries = {}
            for meta in self.action_meta:
                key = meta["key"]
                action = batch["action"][key]
                action_summaries[key] = {
                    "stats": _summarize_episode_tensor(action),
                    "valid_dims": _dual_arm_action_valid_dims(action, source_hand_tag),
                }
            return {
                "state": {
                    meta["key"]: _summarize_episode_tensor(batch["state"][meta["key"]])
                    for meta in self.state_meta
                },
                "action": action_summaries,
            }

        next_episode_idx = 0
        effective_anchors = 0
        pending = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while next_episode_idx < max_pending_futures:
                future = executor.submit(process_episode, next_episode_idx)
                pending[future] = next_episode_idx
                next_episode_idx += 1

            with tqdm(total=episodes_num, desc="Iterating dataset to get normalization") as progress:
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        episode_idx = pending.pop(future)
                        try:
                            episode_summary = future.result()
                            effective_anchors += episode_summary.get("num_anchors", 0)
                            for key, summary in episode_summary["state"].items():
                                state_stats[key].update(summary)
                            if loss_weighted:
                                for key, summary in episode_summary["action"].items():
                                    action_stats[key].update(summary)
                            else:
                                for key, action_summary in episode_summary["action"].items():
                                    action_stats[key].update(
                                        action_summary["stats"],
                                        valid_dims=action_summary["valid_dims"],
                                    )
                        except Exception as exc:
                            for pending_future in pending:
                                pending_future.cancel()
                            logger.error(f"Error processing episode {episode_idx}: {exc}")
                            print(traceback.format_exc())
                            raise

                        progress.update(1)
                        if next_episode_idx < episodes_num:
                            replacement = executor.submit(process_episode, next_episode_idx)
                            pending[replacement] = next_episode_idx
                            next_episode_idx += 1

        stats = {
            "state": DefaultDict(dict),
            "action": DefaultDict(dict),
            "num_episodes": episodes_num,
            "num_transition": (effective_anchors if full_horizon_query_deltas is not None
                               else self.multi_dataset.num_frames),
        }
        for key, accumulator in state_stats.items():
            stats["state"][key].update(accumulator.finalize())
        for key, accumulator in action_stats.items():
            stats["action"][key].update(accumulator.finalize())
        return stats


def sliding_window_with_replication(
    x: torch.Tensor,
    window_size: int,
    stride: int = 1,
    return_is_pad: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Construct a sliding-window tensor from the input tensor x (shape: [N, D]).
    The output shape is [N, window_size, D].

    For each starting index i:
        out[i, j, :] =
            x[i + j*stride, :] if i + j*stride < N
            x[-1, :]         otherwise (replicate the last row when out of bounds)

    Args:
        x (torch.Tensor): Input tensor of shape [N, D]
        window_size (int): Size of the sliding window
        stride (int): Input-frame stride between adjacent window steps.
        return_is_pad (bool): Also return a ``[N, window_size]`` padding mask.

    Returns:
        torch.Tensor: Tensor of shape [N, window_size, D]
    """

    N, D = x.shape

    # shape [N, window_size]
    # indices[i, j] = i + j
    i_indices = torch.arange(N).unsqueeze(1)            # [N, 1]
    j_indices = torch.arange(window_size).unsqueeze(0)  # [1, window_size]
    indices = i_indices + j_indices * stride            # [N, window_size]
    is_pad = indices >= N

    # N-1
    # torch.clamp  [0, N-1]
    clamped_indices = torch.clamp(indices, min=0, max=N - 1)

    # clamped_indices [N, window_size]，x [N, D]
    # out[i, j, :] = x[clamped_indices[i, j], :]
    out = x[clamped_indices]  # [N, window_size, D]

    if return_is_pad:
        return out, is_pad
    return out
