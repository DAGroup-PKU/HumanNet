# Modified by Open-HumanScale contributors: package namespace renamed to humanwam.
from abc import ABC
from typing import Dict, Any, Optional, List

import torch
from copy import deepcopy
from .task_instruction import task_instruction
from ..utils.normalizer import LinearNormalizer, NormMode
from humanwam.utils.pytorch_utils import dict_apply


class BaseProcessor(ABC):
    def __init__(
        self,
        # keys
        shape_meta: Dict[str, Any],
        num_obs_steps: int,
        num_output_cameras: int,
        action_output_dim: int,
        proprio_output_dim: int,

        action_state_transforms: Optional[List[Any]],

        # action & state normalization
        use_stepwise_action_norm: bool,
        norm_default_mode: NormMode,
        norm_exception_mode: Dict[str, Dict[str, NormMode]],

        action_state_merger,

        # image transform
        train_transforms: Dict[str, List[Any]] | None,
        val_transforms: Dict[str, List[Any]] | None,

        # instruction transform
        drop_high_level_prob: float,
        use_zh_instruction: bool,

        tokenizer: Any,
        post_normalization_clips: Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]] = None,
    ):
        self.shape_meta = shape_meta
        self.num_obs_steps = num_obs_steps
        self.num_output_cameras = num_output_cameras
        self.action_output_dim = action_output_dim
        self.proprio_output_dim = proprio_output_dim

        self.drop_high_level_prob = drop_high_level_prob
        self.use_zh_instruction = use_zh_instruction

        # image
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        self._is_train = None

        self.action_state_transforms = action_state_transforms
        self.action_state_merger = action_state_merger
        self.action_state_merger.set_shape_meta(self.shape_meta)

        self.use_stepwise_action_norm = use_stepwise_action_norm
        self.norm_default_mode = norm_default_mode
        self.norm_exception_mode = norm_exception_mode
        self.post_normalization_clips = post_normalization_clips
        self._normalizer = None

        self.tokenizer = tokenizer

    @property
    def is_train(self):
        return self._is_train

    @property
    def normalizer(self) -> LinearNormalizer:
        return self._normalizer

    def train(self):
        self._is_train = True
        return self

    def eval(self):
        self._is_train = False
        return self

    def set_normalizer_from_stats(self, dataset_stats: Dict[str, Any] = None):
        self._normalizer = LinearNormalizer(
            use_stepwise_action_norm=self.use_stepwise_action_norm,
            shape_meta=self.shape_meta,
            default_mode=self.norm_default_mode,
            exception_mode=self.norm_exception_mode,
            stats=dataset_stats,
            post_normalization_clips=self.post_normalization_clips,
        )

    def augment_instruction(self, data: Dict[str, Any]) -> str:
        # Retain the method name for processor API compatibility.
        return task_instruction(data)

    def action_state_transform(self, batch):
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                _, _ = meta["key"], meta["raw_shape"]

        for meta in self.shape_meta["state"]:
            _, _ = meta["key"], meta["raw_shape"]

        if self.action_state_transforms is not None:
            for trans in self.action_state_transforms:
                batch = trans.forward(batch)


        return batch

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the data for the policy model.

        Args:
            Data: Dict[str, Any], lerobot sample in raw mcap obtained from dataset __getitem__:
                - "action": Optional, Dict[str, torch.Tensor] -> [action_horizon, action_dim]
                - "state": Dict[str, torch.Tensor] -> [num_obs_steps, state_dim]
                - "images": Dict[str, torch.Tensor] -> [num_obs_steps, C, H, W]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "state_is_pad": torch.Tensor -> [num_obs_steps,]
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                - "idx": int, sample index

        Returns:
            Sample: Dict[str, Any], which can collated:
                - "input_ids": torch.Tensor -> [max_image_text_tokens,]
                - "attention_mask": torch.Tensor -> [max_image_text_tokens,]
                - "pixel_values": torch.Tensor -> [num_input_cameras, C, H, W]
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                - "proprio": torch.Tensor -> [num_obs_steps, proprio_dim]
                - "state_is_pad": torch.Tensor -> [num_obs_steps,]
                - "action": Optional, torch.Tensor -> [action_horizon, action_dim]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "gt_action: Optional, deepcopy of input action for open loop eval, which is left untouched
                - "idx": int, sample index
        """
        sample = {}
        # 1. instruction
        sample["instruction"] = self.augment_instruction(data)
        sample["image_is_pad"] = data["image_is_pad"]

        # 2. image
        processed_images = []
        for meta in self.shape_meta["images"]:
            key = meta["key"]
            image = data["images"][key]  # [num_obs_steps, C, H, W]

            # Apply transforms efficiently on the merged batch
            transforms = self.train_transforms if self.is_train else self.val_transforms
            for trans in transforms[key]:
                image = trans(image)

            processed_images.append(image)

        pixel_values = torch.cat(processed_images, dim=0) # [num_input_cameras, C, H, W]
        if self.num_output_cameras > pixel_values.shape[0]:
            out = torch.zeros((self.num_output_cameras,) + pixel_values.shape[1:], device=pixel_values.device, dtype=pixel_values.dtype)
            out[0: pixel_values.shape[0]] = pixel_values
            sample["pixel_values"] = out
        else:
            sample["pixel_values"] = pixel_values

        # Copy action before transform for open-loop evaluation,
        # disabled for training dataset as it may cause collating key problem.
        if not self.is_train and "action" in data:
            sample["gt_action"] = deepcopy(data["action"])

        # 3. action & state
        data = self.action_state_transform(data)
        data = self.normalizer.forward(data)
        data = self.action_state_merger.forward(data)

        if "action" in data:
            sample["action"] = data["action"] # [action_horizon, action_dim]
            sample["action_is_pad"] = data["action_is_pad"] # [action_horizon,]
            sample["action_dim_is_pad"] = data["action_dim_is_pad"] # [action_dim,]

        # TODO: rename all "state" into "proprio"
        sample["proprio"] = data["state"] # [num_obs_steps, proprio_dim]
        sample["proprio_is_pad"] = data["state_is_pad"] # [num_obs_steps,]
        sample["proprio_dim_is_pad"] = data["state_dim_is_pad"] # [proprio_dim,]

        sample["idx"] = data["idx"]

        sample = self.tokenizer(sample)

        return sample

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess the data for the policy model.

        Args:
            data: Dict[str, Any], lerobot sample in raw mcap

        Returns:
            data: Dict[str, Any], processed data including unnormalized action
        """
        data["state"] = data.pop("proprio")
        data = self.action_state_merger.backward(data)
        data = self.normalizer.backward(data)
        if self.action_state_transforms is not None:
            for trans in reversed(self.action_state_transforms):
                data = trans.backward(data)

        start_obs_step = self.num_obs_steps - 1
        data["action"] = dict_apply(data["action"], lambda x: x[:, start_obs_step:, :])
        return data
