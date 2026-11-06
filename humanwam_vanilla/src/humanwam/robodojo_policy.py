"""RoboDojo W0 policy: world-frame link6 EEF16 and FP32 compact RGB canvas.

Use configs/robodojo.yaml and its full35 statistics. This is a policy adapter;
it does not by itself launch or validate an Isaac simulator rollout.
"""
import numpy as np
import torch
from torchvision.transforms import functional as TF

from humanwam.robotwin_policy import RoboTwinPolicy


def prepare_images(cameras, *, device, dtype):
    views = []
    for name in ('head_camera', 'left_camera', 'right_camera'):
        rgb = np.asarray(cameras[name])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f'{name} must be uint8 HWC RGB')
        image = torch.from_numpy(np.array(rgb, copy=True, order='C')).permute(2, 0, 1).float().div(255)
        views.append(TF.resize(image, [256, 352], antialias=True).mul(2).sub(1))
    wrists = [torch.nn.functional.interpolate(v.unsqueeze(0), size=(128, 176),
              mode='bilinear', align_corners=False, antialias=True)[0] for v in views[1:]]
    canvas = torch.cat([views[0], torch.cat(wrists, dim=-1)], dim=-2)
    # Training cache quantizes once, after both FP32 resizes. No MP4 round trip.
    pixels = ((canvas.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return pixels.float().div(127.5).sub(1).unsqueeze(0).to(device=device, dtype=dtype)


class RoboDojoPolicy(RoboTwinPolicy):
    def __init__(self, cfg, checkpoint, stats, **kwargs):
        targets = [str(t['_target_']) for t in cfg.data.train.processor.action_state_transforms]
        if not any(t.endswith('.DualArmWristCanonicalizeTransform') for t in targets):
            raise ValueError('RoboDojo requires its wrist-canonicalized config, not RoboTwin unalign')
        super().__init__(cfg, checkpoint, stats, **kwargs)

    def prepare_images(self, cameras):
        return prepare_images(cameras, device=self.device, dtype=self.dtype)


class RoboDojoDeployment:
    """Single-environment XPolicyLab observation/action interface.

    The simulator's joint-space dimension table is 6+1 per arm. EEF control
    instead needs 7+1; explicitly bind pose fields rather than using that table.
    """
    STATE_KEYS = ('left_ee_pose', 'left_ee_joint_state',
                  'right_ee_pose', 'right_ee_joint_state')

    def __init__(self, policy, replan_steps=24):
        if not 1 <= replan_steps <= 32:
            raise ValueError('replan_steps must be between 1 and 32')
        self.policy, self.replan_steps, self.obs = policy, replan_steps, None

    def reset(self):
        self.obs = None
        self.policy.reset()

    def update_obs(self, obs):
        self.obs = obs

    def get_action(self):
        if self.obs is None:
            raise ValueError('Call update_obs before get_action')
        state, vision = self.obs['state'], self.obs['vision']
        parts = [np.asarray(state[k], dtype=np.float32).reshape(-1) for k in self.STATE_KEYS]
        if [len(x) for x in parts] != [7, 1, 7, 1]:
            raise ValueError('Expected world-frame link6 poses (7) and grippers (1) per arm')
        cameras = {key: vision[name]['color'] for key, name in zip(
            ('head_camera', 'left_camera', 'right_camera'),
            ('cam_head', 'cam_left_wrist', 'cam_right_wrist'))}
        prompt = self.obs.get('task_instruction', self.obs.get('instruction'))
        if isinstance(prompt, (tuple, list)):
            prompt = prompt[0] if prompt else None
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('Missing task instruction')
        actions = self.policy.predict(cameras, np.concatenate(parts), prompt)[:self.replan_steps]
        keys = self.STATE_KEYS
        return [{keys[0]:x[:7], keys[1]:x[7:8], keys[2]:x[8:15], keys[3]:x[15:]}
                for x in actions]
