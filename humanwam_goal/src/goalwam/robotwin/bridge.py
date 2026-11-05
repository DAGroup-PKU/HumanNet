"""RoboTwin 2.0 Aloha-AgileX observations <-> GoalWAM 49-D state and joint commands.

RoboTwin reports SAPIEN world poses and gripper *commands* (openness, 0 closed
and 1 open). GoalWAM conditions on end-effector poses in the head-camera frame
(camera NWU axes, Astribot tool axes) and predicts absolute 49-D commands.
"""

import numpy as np
from scipy.spatial.transform import Rotation

ROBOT = "robotwin_aloha_agilex"
CAMERAS = {"head": "head_camera", "left": "left_camera", "right": "right_camera"}
WIRE_CAMERAS = {
    "head_camera": "cam_head",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}
CAMERA_NWU_TO_GL = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float64)
ASTRIBOT_TO_RAW_TOOL = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
# Six arm joints and one gripper per arm, plus both EEF poses and grippers.
ACTION_MASK = np.zeros(49, dtype=bool)
ACTION_MASK[np.r_[0:6, 7, 8:14, 15, 26:42]] = True
STATE_MASK = ACTION_MASK.copy()
STATE_MASK[42:49] = True  # head-camera pose: the identity in the head-camera frame
JOINT_COMMAND_DIMS = np.r_[0:6, 7, 8:14, 15]
GRIPPER_REPRESENTATIONS = ("closedness_100", "command_openness")


def native_observation(obs):
    """Restore RoboTwin field names from one XPolicyLab v1.0 observation."""
    if obs["data_format_version"] != "v1.0" or obs["env_idx"] != 0:
        raise ValueError("Expected one RoboTwin v1.0 environment (env_idx=0)")
    cameras = {}
    for native, wire in WIRE_CAMERAS.items():
        camera = obs["vision"][wire]
        extrinsic = np.asarray(camera["extrinsics_matrix"], dtype=np.float64)
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])
        if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
            raise ValueError("Expected finite world-to-camera CV extrinsics")
        # SAPIEN CV axes are right/down/forward; GL is right/up/backward.
        cameras[native] = dict(
            rgb=camera["color"], cam2world_gl=np.linalg.inv(extrinsic) @ np.diag([1, -1, -1, 1])
        )
    state, joints, endpose = obs["state"], {}, {}
    for side in ("left", "right"):
        grip = np.asarray(state[f"{side}_ee_joint_state"]).reshape(-1)
        if grip.shape != (1,):
            raise ValueError("Expected one gripper openness value per arm")
        joints[f"{side}_arm"] = state[f"{side}_arm_joint_state"]
        joints[f"{side}_gripper"] = endpose[f"{side}_gripper"] = float(grip[0])
        endpose[f"{side}_endpose"] = state[f"{side}_ee_pose"]
    return dict(observation=cameras, joint_action=joints, endpose=endpose)


def camera_transform(cam2world_gl):
    matrix = np.asarray(cam2world_gl, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite 4x4 head cam2world_gl")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
    ):
        raise ValueError("Camera extrinsics must be a rigid camera-to-world transform")
    return rotation @ CAMERA_NWU_TO_GL, matrix[:3, 3]


def world_pose_to_canonical(pose, cam2world_gl):
    """SAPIEN world xyz + wxyz tool pose -> head-camera xyz + xyzw Astribot-tool pose."""
    pose = np.asarray(pose, dtype=np.float64)
    if (
        pose.shape != (7,)
        or not np.isfinite(pose).all()
        or abs(np.linalg.norm(pose[3:]) - 1) > 1e-3
    ):
        raise ValueError("RoboTwin EEF must be xyz + unit wxyz world/tool quaternion")
    rotation, origin = camera_transform(cam2world_gl)
    xyz = rotation.T @ (pose[:3] - origin)
    orientation = Rotation.from_matrix(
        rotation.T @ Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix() @ ASTRIBOT_TO_RAW_TOOL
    ).as_quat()
    return np.r_[xyz, orientation].astype(np.float32)


def extract_images(observation):
    result = {}
    for name, key in CAMERAS.items():
        value = np.asarray(observation["observation"][key]["rgb"])
        if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise ValueError(f"{key} must contain decoded HWC uint8 RGB")
        result[name] = np.ascontiguousarray(value)
    return result


def encode_observation(observation, *, gripper_representation="command_openness"):
    """49-D state in the head-camera frame, RGB images and the head camera pose.

    The model uses ``command_openness``. ``closedness_100`` ((1 - openness) x 100)
    is the unit of the released goal cache's initial states.
    """
    if gripper_representation not in GRIPPER_REPRESENTATIONS:
        raise ValueError(f"gripper_representation must be one of {GRIPPER_REPRESENTATIONS}")
    matrix = np.asarray(observation["observation"]["head_camera"]["cam2world_gl"])
    camera_transform(matrix)
    state = np.zeros(49, dtype=np.float32)
    for side, joint, grip, pose in [("left", 0, 7, 26), ("right", 8, 15, 34)]:
        joints = np.asarray(observation["joint_action"][f"{side}_arm"], dtype=np.float32)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("This bridge supports six-joint aloha-agilex arms")
        state[joint : joint + 6] = joints
        for source, target in [("joint_action", grip), ("endpose", pose + 7)]:
            openness = float(observation[source][f"{side}_gripper"])
            if not np.isfinite(openness) or not -1e-4 <= openness <= 1.0001:
                raise ValueError("RoboTwin gripper openness must be in [0,1]")
            state[target] = (
                np.clip(openness, 0, 1)
                if gripper_representation == "command_openness"
                else (1 - np.clip(openness, 0, 1)) * 100
            )
        state[pose : pose + 7] = world_pose_to_canonical(
            observation["endpose"][f"{side}_endpose"], matrix
        )
    state[48] = 1  # The common frame is the head camera itself: identity pose.
    return dict(state=state, images=extract_images(observation), cam2world_gl=matrix.copy())


def joint_commands(absolute, valid):
    """Absolute 49-D commands -> RoboTwin qpos rows (6 joints + openness per arm)."""
    absolute = np.asarray(absolute, dtype=np.float32)
    valid = np.asarray(valid)
    if (
        absolute.ndim != 2
        or absolute.shape[0] < 1
        or absolute.shape[1] != 49
        or valid.shape != absolute.shape
        or valid.dtype != bool
    ):
        raise ValueError("Expected a nonempty Tx49 chunk and boolean mask")
    if not np.isfinite(absolute[valid]).all():
        raise ValueError("Nonfinite valid command")
    if not valid[:, JOINT_COMMAND_DIMS].all():
        raise ValueError("Joint control requires joint supervision")
    result = absolute[:, JOINT_COMMAND_DIMS].copy()
    result[:, [6, 13]] = np.clip(result[:, [6, 13]], 0, 1)
    return result
