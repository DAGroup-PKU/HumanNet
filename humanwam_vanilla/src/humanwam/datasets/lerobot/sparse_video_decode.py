"""Select model-consumed RGB timestamps before decode and spatial transforms."""
from dataclasses import dataclass


@dataclass
class SparseVideoQuery:
    query: object
    indices: tuple
    observation_frames: int

    def __call__(self, timestamps, episode):
        selected = {key: [values[i] for i in self.indices] for key, values in timestamps.items()}
        result = self.query(selected, episode)
        result['_fastwam_video_sample_indices'] = self.indices
        return result


def install_sparse_video_decode(dataset):
    for inner in dataset.lerobot_dataset.multi_dataset._datasets:
        inner._query_videos = SparseVideoQuery(
            inner._query_videos, tuple(dataset.video_sample_indices), dataset.num_frames)
