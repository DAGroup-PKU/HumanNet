"""One RoboTwin 2.0 evaluation session on top of ``goalwam.GoalWAM``.

The session receives XPolicyLab observations, conditions the model on the
released goal for the exact native case, and returns 32 joint commands per
query (one query every 32 steps of the 30-Hz control grid). Query ``q`` samples
with seed ``generation_seed + q``. With a goal cache, the step-zero RGB, state
and head-camera pose must equal the cached initial observation before any
action is returned.
"""

import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from goalwam.processing.images import select_goalwam_history

from .bridge import ACTION_MASK, STATE_MASK, encode_observation, joint_commands, native_observation

FREQUENCY = 30.0
REPLAN_STEPS = 32
DEBUG_GENERATION_SEED = 9000


class HistoryBuffer:
    """Episode-local captures on simulator control time, bounded by the oldest offset."""

    def __init__(self, offsets, tolerance):
        self.offsets, self.tolerance = tuple(offsets), float(tolerance)
        self.captures = deque()

    def append(self, timestamp, images):
        if self.captures and timestamp <= self.captures[-1]["timestamp"]:
            raise ValueError("History captures must advance in simulator time")
        self.captures.append(
            dict(timestamp=timestamp, images={key: value.copy() for key, value in images.items()})
        )
        oldest = timestamp - max(self.offsets, default=0.0) - self.tolerance
        while self.captures and self.captures[0]["timestamp"] < oldest:
            self.captures.popleft()

    def select(self, captures, timestamp):
        return select_goalwam_history(captures, timestamp, self.offsets, self.tolerance)

    def request(self, timestamp):
        # The current capture is history for later queries, never for its own.
        past = [capture for capture in self.captures if capture["timestamp"] < timestamp]
        chosen = {id(capture) for capture, _ in self.select(past, timestamp)}
        return dict(observation_timestamp=timestamp, history=[c for c in past if id(c) in chosen])


class RoboTwinSession:
    def __init__(self, model, output, goals=None):
        """``goals`` is an ``OfficialGoalCache``; ``None`` serves goal-free debug observations."""
        self.model, self.output, self.goals = model, Path(output), goals
        if goals is not None:
            goals.validate_model(model)
            self.generation_seed = goals.config["generation_seed"]
        else:
            self.generation_seed = DEBUG_GENERATION_SEED
        self.execution_indices = np.arange(REPLAN_STEPS)
        self.output.mkdir(parents=True, exist_ok=True)
        self.case, self.goal_case = {}, None
        self.requests = 0
        self.reset()

    def _record(self, event, **fields):
        record = dict(
            event=event, wall_time=time.time(), case_id=self.case.get("action_case_id"), **fields
        )
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def _history(self):
        c = self.model.config
        return HistoryBuffer(c.history_offsets_seconds, c.history_tolerance_seconds)

    def prepare_case(self, metadata):
        self.case = metadata
        self.goal_case = None  # A failed prepare must never retain the preceding goal.
        self.current, self.initial_checked = None, False
        if self.goals is None:
            self._record("prepare_case", metadata=metadata)
            return None
        prepared = self.goals.prepare(metadata)
        config = prepared["config"]
        if config["replan_steps"] != REPLAN_STEPS or config["action_type"] != "qpos":
            raise ValueError("Control mode/prefix differs from the released evaluation profile")
        self.goal_case = prepared
        self._record(
            "prepare_case",
            metadata=metadata,
            task_config=config["task_config"],
            **prepared["provenance"],
        )
        return self.goals.acknowledgement(prepared, "prepare_case", False)

    def validate_cases(self, request):
        if self.goals is None:
            raise ValueError("Case preflight requires the goal cache")
        return self.goals.validate_cases(request, "joint", REPLAN_STEPS)

    def reset(self):
        self.current = None
        self.observations = self.queries = 0
        self.frequency = None
        self.capture_plan = None
        self.initial_checked = False
        self.history = self._history()
        self._record("reset")

    def update_obs(self, obs):
        try:
            return self._update_obs(obs)
        except Exception:
            if self.goals is not None:
                # A failed check must not leave a ready goal for a client that continues.
                self.current = self.goal_case = None
                self.initial_checked = False
            raise

    def _check_initial(self, native, obs):
        if obs["additional_info"].get("capture_step", 0) != 0:
            raise ValueError("The initial observation must be captured at step zero")
        # The cache stores initial states in its closedness unit, (1 - openness) x 100.
        current = encode_observation(native, gripper_representation="closedness_100")
        initial = self.goal_case["reference"]["initial"]
        np.testing.assert_allclose(current["state"], initial["state"], rtol=0, atol=1e-5)
        np.testing.assert_allclose(
            current["cam2world_gl"], initial["cam2world_gl"], rtol=0, atol=1e-6
        )
        differences = {}
        for camera, expected in initial["images"].items():
            actual = current["images"][camera]
            if actual.shape != expected.shape:
                raise ValueError("Initial camera dimensions differ from the cached scene")
            delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
            differences[camera] = dict(
                exact=bool(not delta.any()), max_abs=int(delta.max()), mean_abs=float(delta.mean())
            )
        exact = all(item["exact"] for item in differences.values())
        self._record(
            "initial_scene_check", state_and_camera_match=True, rgb=differences, exact_rgb=exact
        )
        if not exact:
            raise ValueError("Initial RGB differs from the cached scene")
        self.initial_checked = True

    def _update_obs(self, obs):
        frequency = float(obs["additional_info"]["frequency"])
        if frequency != FREQUENCY or (self.frequency is not None and frequency != self.frequency):
            raise ValueError("This RoboTwin checkpoint uses the 30-Hz action grid")
        self.frequency = frequency
        native = native_observation(obs)
        self.instruction = obs["instruction"]
        if self.goals is not None:
            if self.goal_case is None:
                raise ValueError("A cached goal case must be prepared before observations")
            if self.instruction != self.goal_case["policy_instruction"]:
                raise ValueError("Observation instruction differs from the prepared case")
            if not self.initial_checked:
                self._check_initial(native, obs)
        self.current = encode_observation(native)
        # Sparse captures carry explicit control steps; dense clients send every step.
        step = obs["additional_info"].get("capture_step", self.observations)
        if type(step) is not int or step < 0:
            raise ValueError("capture_step must be a nonnegative integer")
        self.timestamp = step / frequency
        self.history.append(self.timestamp, self.current["images"])
        self.observations += 1
        if self.observations == 1:
            self._record(
                "first_observation",
                instruction=self.instruction,
                cameras={key: list(value.shape) for key, value in self.current["images"].items()},
            )
        if self.goals is not None:
            return self.goals.acknowledgement(self.goal_case, "update_obs", self.initial_checked)
        return None

    def get_observation_plan(self, request):
        """Which control steps to capture: each query step and its history steps."""
        if request.get("frequency") != FREQUENCY or self.queries:
            raise ValueError("Plan a fresh episode on the 30-Hz action grid")
        limit = request["step_limit"]
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise ValueError("step_limit must be an integer in [1,10000]")
        buffer = self._history()
        steps = [dict(timestamp=step / FREQUENCY, step=step) for step in range(limit)]
        queries, captures = [], set()
        for step in range(0, limit, REPLAN_STEPS):
            selected = buffer.select(steps[:step], step / FREQUENCY)
            past = sorted(capture["step"] for capture, _ in selected)
            queries.append(dict(step=step, history_steps=past))
            captures.update(past + [step])
        self.capture_plan = dict(
            clock="action_step",
            frequency=30,
            step_limit=limit,
            query_every_steps=REPLAN_STEPS,
            queries=queries,
            capture_steps=sorted(captures),
            history_offsets_seconds=list(buffer.offsets),
            cameras=["cam_head", "cam_left_wrist", "cam_right_wrist"],
        )
        self._record("observation_plan", plan=self.capture_plan)
        return self.capture_plan

    def get_action(self, obs=None):
        if obs is not None:
            if self.capture_plan is None:
                raise ValueError("get_observation_plan is required before scheduled captures")
            if self.queries >= len(self.capture_plan["queries"]):
                raise ValueError("No planned query remains in this episode")
            query = self.capture_plan["queries"][self.queries]
            captures = obs["captures"]
            steps = [frame["additional_info"]["capture_step"] for frame in captures]
            if steps != query["history_steps"] + [query["step"]]:
                raise ValueError("Observation window does not match the planned query")
            self.history = self._history()
            for frame in captures:
                self.update_obs(frame)
        if self.current is None:
            raise ValueError("update_obs is required before get_action")
        if self.goals is not None:
            if self.goal_case is None or not self.initial_checked:
                raise ValueError(
                    "The cached goal and initial scene must be verified before inference"
                )
            goal = self.goal_case["goal"]
        else:
            goal = self.current["images"]["head"]  # Debug observations match no cached scene.
        seed = self.generation_seed + self.queries
        observation = dict(
            instruction=self.instruction,
            images=self.current["images"],
            goal_images={"head": goal},
            state=self.current["state"],
            state_valid_mask=STATE_MASK,
            action_valid_mask=ACTION_MASK,
            seed=seed,
            **self.history.request(self.timestamp),
        )
        started = time.perf_counter()
        try:
            result = self.model.predict(observation)
            model_seconds = time.perf_counter() - started
            commands = joint_commands(result["actions"], result["action_valid_mask"])
        except Exception as error:
            self._record("inference_error", query=self.queries, error=repr(error))
            raise
        np.savez_compressed(
            self.output / f"commands_{self.requests:06d}.npz",
            commands=commands,
            execution_indices=self.execution_indices,
            anchor_state=self.current["state"],
            cam2world_gl=self.current["cam2world_gl"],
        )
        self._record(
            "action",
            query=self.queries,
            request=self.requests,
            generation_seed=seed,
            observations=self.observations,
            timestamp=self.timestamp,
            history_timestamps=[x["timestamp"] for x in observation["history"]],
            seconds=time.perf_counter() - started,
            model_seconds=model_seconds,
            shape=list(commands.shape),
            returned_steps=len(self.execution_indices),
        )
        self.queries += 1
        self.requests += 1
        return commands[self.execution_indices]

    def on_trial_end(self, result):
        self._record(
            "trial_end", result=result, observations=self.observations, queries=self.queries
        )
