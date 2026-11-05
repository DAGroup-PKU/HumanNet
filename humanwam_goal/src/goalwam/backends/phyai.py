"""GoalWAM entrypoint for the optional PhyAI engine."""


class PhyAIBackend:
    def __init__(self, checkpoint, *, tp=1, cuda_graph=True, device="cuda:0"):
        try:
            from phyai import Engine, EngineArgs
            from phyai.engine_config import (
                AttentionParallelConfig,
                DenseParallelConfig,
                DeviceConfig,
                EngineConfig,
                ParallelConfig,
                RuntimeConfig,
            )
            from phyai.models.goalwam.main_goalwam import GoalWAMArgs
            from phyai.models.goalwam.requests_goalwam import (
                GoalWAMObservationRequest,
                GoalWAMTextRequest,
            )
        except ImportError as error:
            raise ImportError(
                "Install the GoalWAM phyai extra with PhyAI 0.2.0rc2 to use this backend"
            ) from error
        self.observation_type = GoalWAMObservationRequest
        self.text_type = GoalWAMTextRequest
        self.engine = Engine(
            EngineArgs(
                plugin="goalwam",
                plugin_args=GoalWAMArgs(checkpoint_dir=checkpoint),
                config=EngineConfig(
                    device=DeviceConfig(target=device),
                    parallel=ParallelConfig(
                        dense=DenseParallelConfig(tp_size=tp),
                        attention=AttentionParallelConfig(tp_size=tp),
                    ),
                    runtime=RuntimeConfig(use_cuda_graph=cuda_graph, allow_cudnn_tf32=False),
                ),
            )
        )

    def set_instruction(self, ids, mask):
        self.engine.step(self.text_type(ids, mask))

    def predict(self, observation):
        result = self.engine.step(self.observation_type(**observation))
        actions = result["action"][:, 1:, :49].detach().float().cpu().clone()
        del result
        return actions

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None
