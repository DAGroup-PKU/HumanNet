"""Public inference API shared by native and PhyAI execution."""

import threading
from pathlib import Path

import torch

from .config import SamplingConfig, read_config
from .processing.observation import ObservationProcessor, ProcessorConfig
from .processing.text import InstructionTokenizer


class GoalWAM:
    def __init__(self, checkpoint, *, backend="native", tp=1, cuda_graph=True, device="cuda:0"):
        checkpoint = Path(checkpoint).expanduser().resolve()
        if backend not in ("native", "phyai") or type(tp) is not int or tp not in (1, 2, 4):
            raise ValueError("Choose backend native/phyai and TP=1/2/4")
        self.config = read_config(checkpoint / "processor.json", ProcessorConfig)
        self.sampling = read_config(checkpoint / "sampling.json", SamplingConfig)
        if self.config.prefix_length != self.sampling.prefix_length:
            raise ValueError("Processor and sampler RTC prefix lengths disagree")
        self.processor = ObservationProcessor(self.config)
        self.tokenizer = InstructionTokenizer(checkpoint / "text_tokenizer")
        self.instruction = None
        self.lock = threading.RLock()
        self.robot = None
        if (checkpoint / "robot.json").is_file():
            from .processing.robot import RobotProcessor

            self.robot = RobotProcessor(checkpoint / "robot.json")
        if backend == "native":
            from .backends.native.engine import NativeBackend

            factory = NativeBackend
        else:
            from .backends.phyai import PhyAIBackend

            factory = PhyAIBackend
        self.backend = factory(checkpoint, tp=tp, cuda_graph=cuda_graph, device=device)

    @torch.inference_mode()
    def predict(self, observation):
        """Return normalized actions and, for physical state input, absolute actions."""
        with self.lock:
            if self.backend is None:
                raise RuntimeError("This GoalWAM engine is closed")
            if "state" in observation and "normalized_state" in observation:
                raise ValueError("Supply state or normalized_state, not both")
            source, context = observation, None
            if "state" not in observation and observation.get("prev_chunk") is not None:
                raise ValueError(
                    "prev_chunk requires physical state; normalized inputs use normalized_prefix"
                )
            if "state" in observation:
                if self.robot is None:
                    raise ValueError("Physical state inputs require robot.json in the checkpoint")
                source, context = self.robot.preprocess(observation, self.sampling.prefix_length)
            instruction = source.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("instruction must be a nonempty string")
            prepared = self.processor(source)
            # Both backends use the same unquantized BF16 arithmetic policy.
            previous = torch.backends.cuda.matmul.allow_tf32
            previous_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            try:
                if instruction != self.instruction:
                    self.backend.set_instruction(*self.tokenizer(instruction))
                    self.instruction = instruction
                actions = self.backend.predict(prepared)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                    previous_reduction
                )
            prefix = int((~prepared["action_noisy_mask"]).sum()) - 1
            result = {"normalized_actions": actions.numpy(), "prefix_length": prefix}
            if context is not None:
                result.update(self.robot.postprocess(actions[0], context))
            return result

    def close(self):
        with self.lock:
            if self.backend is not None:
                self.backend.close()
                self.backend = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
