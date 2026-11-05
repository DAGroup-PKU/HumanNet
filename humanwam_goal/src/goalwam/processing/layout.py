"""Model action-vector layout."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionLayout:
    joint: slice
    eef: slice
    left_arm: slice
    right_arm: slice
    chassis_xy: slice
    chassis_angle: int
    left_eef_pos: slice
    right_eef_pos: slice
    gripper_dims: tuple[int, ...]

    @property
    def chassis_x(self) -> int:
        return self.chassis_xy.start

    @property
    def chassis_y(self) -> int:
        return self.chassis_xy.start + 1

    @property
    def chassis_se2(self) -> slice:
        return slice(self.chassis_xy.start, self.chassis_angle + 1)

    @property
    def left_eef_quat(self) -> slice:
        return slice(self.left_eef_pos.stop, self.left_eef_pos.stop + 4)

    @property
    def right_eef_quat(self) -> slice:
        return slice(self.right_eef_pos.stop, self.right_eef_pos.stop + 4)

    @property
    def head_eef_pos(self) -> slice | None:
        start = self.right_eef_quat.stop + 1
        return slice(start, start + 3) if self.eef.stop >= start + 7 else None

    @property
    def head_eef_quat(self) -> slice | None:
        pos = self.head_eef_pos
        return None if pos is None else slice(pos.stop, pos.stop + 4)


SHARED_EEF_LAYOUT = ActionLayout(
    joint=slice(0, 23),
    eef=slice(26, 49),
    left_arm=slice(0, 7),
    right_arm=slice(8, 15),
    chassis_xy=slice(23, 25),
    chassis_angle=25,
    left_eef_pos=slice(26, 29),
    right_eef_pos=slice(34, 37),
    gripper_dims=(7, 15, 33, 41),
)


def resolve_action_layout(layout: ActionLayout | None = None) -> ActionLayout:
    if layout is None:
        return SHARED_EEF_LAYOUT
    if not isinstance(layout, ActionLayout):
        raise TypeError("Expected an ActionLayout")
    return layout
