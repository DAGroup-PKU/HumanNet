"""Action supervision follows logical annotations, never dataset path names."""
import torch


BAD_SEGMENT_REASONS = frozenset({"action_idle", "body_locomotion", "action_instability"})


def bad_segment_action_mask(action, reason, existing_mask=None):
    mask = torch.ones(action.shape[-1], dtype=torch.bool, device=action.device)
    if existing_mask is not None:
        mask &= existing_mask.to(device=action.device, dtype=torch.bool)
    if reason:
        mask.zero_()
    return mask
