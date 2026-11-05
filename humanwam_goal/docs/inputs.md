# Observation format

`GoalWAM.predict()` accepts one observation dictionary. The CLI uses the same
fields in JSON, with image filenames relative to the JSON file. The HTTP endpoint
uses base64 image contents instead.

| Field | Value |
| --- | --- |
| `instruction` | Nonempty text instruction |
| `images` | Current RGB images keyed by `head`, and optionally `left` and `right` |
| `goal_images` | Goal RGB image keyed by `head` |
| `observation_timestamp` | Capture time in seconds |
| `history` | Oldest-first list of `{"timestamp": ..., "images": {...}}`; use `[]` initially |
| `seed` | Optional noise seed; defaults to `0` |

Python images are HWC `uint8` NumPy arrays in RGB order. Head images are required;
missing wrist views retain empty cells. The checkpoint determines resizing and
which historical captures to select. All history timestamps must precede the
current observation.

## Robot coordinates

For physical inputs, add `state`, a vector of 49 values. The checkpoint's
`robot.json` supplies the robot's validity masks, coordinate conventions, gripper
calibration, and normalization. Only valid channels are used.

Indices below are zero-based; ranges use Python's exclusive upper bound.

| Channels | Meaning |
| --- | --- |
| `0:23` | Robot-specific joint channels, including joint grippers at `7` and `15` |
| `23:25`, `25` | Chassis position (metres) and heading (radians) |
| `26:29`, `29:33`, `33` | Left end-effector position, xyzw quaternion, gripper |
| `34:37`, `37:41`, `41` | Right end-effector position, xyzw quaternion, gripper |
| `42:45`, `45:49` | Head-camera pose used for coordinate conversion; excluded from predicted actions |

End-effector and camera positions use metres. Physical grippers use closedness
from `0` (open) to `100` (closed), unless `robot.json` declares
`"gripper_space": "command_openness"`: then `state`, `prev_chunk` and the
returned actions keep the robot's own dimensionless gripper commands (RoboTwin:
openness from `0` closed to `1` open), normalized only by the checkpoint bounds.
Valid quaternions must have unit length. The
robot configuration specifies whether end-effector poses use base or head-camera
coordinates. Preserve the measured head-camera pose when conversion requires it.

For RTC, send the remaining previous absolute commands as `prev_chunk`, an
`[N, 49]` array. Optional integer `s` limits how many rows are available. The
engine uses at most the checkpoint's configured prefix length, transforms the
prefix relative to the **current** state, and keeps it fixed during denoising.
Omit `prev_chunk` when no previous commands remain.

Physical predictions return:

- `normalized_actions`: `[1, 48, 49]`, in model coordinates.
- `relative_actions`: `[48, 49]`, relative to the current robot state.
- `actions`: `[48, 49]`, absolute robot commands in the configured coordinates.
- `action_valid_mask`: `[48, 49]` booleans; `False` channels are zero in both action arrays.
- `prefix_length`: number of leading RTC rows included in the output chunk.

Invalid channels are zero. A controller must account for the returned prefix
when scheduling the new chunk.

## Already-normalized inputs

To provide model coordinates directly, replace `state` with `normalized_state`
and supply `state_valid_mask`, `action_valid_mask`, and `domain_id`. Masks are
boolean vectors and must disable channels `42:49`. Joint channels require the
robot domain ID associated with the checkpoint; with all joint channels disabled,
use `-1`.

Optional `normalized_prefix` supplies the RTC prefix in the same coordinates.
Do not combine normalized inputs with physical `prev_chunk` commands. This mode
returns only `normalized_actions` and `prefix_length`; converting them into
executable robot commands remains the caller's responsibility.

`examples/make_sample.py` generates a complete synthetic normalized example for
checking the installation. See [the Python API example](../examples/infer.py)
for loading an observation, running either backend, and saving the predicted actions.
