#!/usr/bin/env python
# Modified by Open-HumanScale contributors: removed unused release runtime code.

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Includes FastWAM/OpenHumanScale adaptations; original upstream notices are retained.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import logging
import os
import warnings
import atexit
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import av
import pyarrow as pa
import torch
import torchvision
from datasets.features.features import register_feature


_PYAV_DECODE_THREADS_ENV = "FASTWAM_PYAV_DECODE_THREADS"
_PYAV_CONTAINERS = OrderedDict()


def _close_pyav_containers():
    while _PYAV_CONTAINERS:
        _, container = _PYAV_CONTAINERS.popitem(last=False)
        container.close()


atexit.register(_close_pyav_containers)
if hasattr(os, "register_at_fork"):
    # Never inherit active FFmpeg decoder threads into DataLoader children.
    os.register_at_fork(before=_close_pyav_containers)


@contextmanager
def _selected_rgb_container(video_path):
    capacity = int(os.environ.get("FASTWAM_PYAV_CONTAINER_CACHE_SIZE", "0"))
    threads = _resolve_pyav_decode_threads()
    thread_type = os.environ.get("FASTWAM_PYAV_THREAD_TYPE", "SLICE")
    if not capacity:
        _close_pyav_containers()
        with av.open(str(video_path), "r") as container:
            container.streams.video[0].codec_context.thread_count = threads
            container.streams.video[0].codec_context.thread_type = thread_type
            yield container
        return
    key = (os.fspath(video_path), threads, thread_type)
    container = _PYAV_CONTAINERS.pop(key, None)
    reused = container is not None
    if container is None:
        container = av.open(str(video_path), "r")
    try:
        if not reused:
            container.streams.video[0].codec_context.thread_count = threads
            container.streams.video[0].codec_context.thread_type = thread_type
        # The caller always seeks, which flushes the decoder after a prior read.
        yield container
    except BaseException:
        container.close()
        raise
    else:
        previous = _PYAV_CONTAINERS.pop(key, None)
        if previous is not None:
            previous.close()
        _PYAV_CONTAINERS[key] = container
        while len(_PYAV_CONTAINERS) > capacity:
            _, evicted = _PYAV_CONTAINERS.popitem(last=False)
            evicted.close()


def _resolve_pyav_decode_threads() -> int:
    """Return the bounded FFmpeg thread count used by PyAV video readers.

    PyAV otherwise leaves ``AVCodecContext.thread_count`` at zero, which asks
    FFmpeg to auto-size the decoder pool from all CPUs visible to the process.
    On a 192-CPU Arena node, one reader can therefore create hundreds of native
    threads.  Multiplied by ranks and DataLoader workers, that can leave every
    distributed rank waiting for data while thousands of decoder threads
    contend for the same CPUs.
    """

    raw_value = os.environ.get(_PYAV_DECODE_THREADS_ENV, "1").strip()
    thread_count = int(raw_value)
    return thread_count


def _configure_pyav_reader_threads(reader: Any, thread_count: int) -> None:
    """Apply a concrete thread limit before the lazy PyAV decoder is consumed."""

    stream_id = int(getattr(reader, "pyav_stream", {}).get("video", 0))
    video_streams = reader.container.streams.video
    video_streams[stream_id].codec_context.thread_count = int(thread_count)


def get_safe_default_codec():
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning(
            "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder"
        )
        return "pyav"


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    """
    Decodes video frames using the specified backend.

    Args:
        video_path (Path): Path to the video file.
        timestamps (list[float]): List of timestamps to extract frames.
        tolerance_s (float): Allowed deviation in seconds for frame retrieval.
        backend (str, optional): Backend to use for decoding. Defaults to "torchcodec" when available in the platform; otherwise, defaults to "pyav"..

    Returns:
        torch.Tensor: Decoded frames.

    Currently supports torchcodec on cpu and pyav.
    """
    if backend is None:
        backend = get_safe_default_codec()
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s)
    return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)


def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        if not hasattr(torchvision.io, "VideoReader") or os.environ.get('FASTWAM_PYAV_SELECTED_RGB') == '1':
            return decode_video_frames_pyav_direct(
                video_path,
                timestamps,
                tolerance_s,
                log_loaded_timestamps=log_loaded_timestamps,
            )
        keyframes_only = True  # pyav doesn't support accurate seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")
    if backend == "pyav":
        # TorchVision's VideoReader ``num_threads`` argument is ignored by its
        # PyAV backend. Set AVCodecContext directly before first iteration.
        _configure_pyav_reader_threads(reader, _resolve_pyav_decode_threads())

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usually smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    # Use float32 for timestamp distance computation (torch.cdist doesn't support bfloat16).
    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts = torch.tensor(loaded_ts, dtype=torch.float32)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)


    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255
    return closest_frames


def decode_video_frames_pyav_direct(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Decode timestamped frames with PyAV when torchvision removed VideoReader."""

    if os.environ.get('FASTWAM_PYAV_SELECTED_RGB', '0') == '1':
        return _decode_pyav_selected_rgb(video_path, timestamps, tolerance_s, log_loaded_timestamps)

    first_ts = min(timestamps)
    last_ts = max(timestamps)
    loaded_frames = []
    loaded_ts = []
    with av.open(str(video_path), "r") as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = _resolve_pyav_decode_threads()
        seek_pts = max(0, int(first_ts / float(stream.time_base)))
        container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current_ts = float(frame.pts * stream.time_base)
            if log_loaded_timestamps:
                logging.info("frame loaded at timestamp=%.4f", current_ts)
            rgb = frame.to_ndarray(format="rgb24")
            loaded_frames.append(
                torch.from_numpy(rgb.copy()).permute(2, 0, 1).contiguous()
            )
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break
    query_ts = torch.tensor(timestamps, dtype=torch.float64)
    loaded_ts_tensor = torch.tensor(loaded_ts, dtype=torch.float64)
    distances = torch.cdist(
        query_ts[:, None],
        loaded_ts_tensor[:, None],
        p=1,
    )
    minimum, indices = distances.min(1)
    closest_frames = torch.stack([loaded_frames[index] for index in indices])
    return closest_frames.to(dtype=torch.float32).div_(255)


def _decode_pyav_selected_rgb(video_path, timestamps, tolerance_s, log_loaded_timestamps=False):
    """Keep nearest AVFrames and convert only requested frames to RGB.

    Timestamp comparisons use float64: float32 spacing at long-video offsets
    exceeds the timestamp tolerance. No resize, seek or codec change.
    RGB memory/conversion scales with requested frames, not the keyframe interval.
    """
    # Python floats retain float64 precision without allocating several torch
    # tensors for every decoded frame (including the unrequested GOP prefix).
    query = [float(t) for t in timestamps]
    best_distance = [float('inf')] * len(query)
    selected = [None] * len(timestamps)
    first_ts, last_ts = min(query), max(query)
    with _selected_rgb_container(video_path) as container:
        stream = container.streams.video[0]
        container.seek(max(0,int(first_ts/float(stream.time_base))),
                       stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current = float(frame.pts * stream.time_base)
            if log_loaded_timestamps:
                logging.info('frame loaded at timestamp=%.4f',current)
            for i, timestamp in enumerate(query):
                distance = abs(timestamp - current)
                if distance < best_distance[i]:
                    best_distance[i] = distance
                    selected[i] = frame
            if current >= last_ts:
                break
        converted = {}
        result=[]
        for frame in selected:
            key=id(frame)
            if key not in converted:
                rgb=frame.to_ndarray(format='rgb24')
                converted[key]=torch.from_numpy(rgb.copy()).permute(2,0,1).contiguous()
            result.append(converted[key])
    return torch.stack(result).to(dtype=torch.float32).div_(255)


def decode_video_frames_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    device: str = "cpu",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated with the requested timestamps of a video using torchcodec.

    Note: Setting device="cuda" outside the main process, e.g. in data loader workers, will lead to CUDA initialization errors.

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """

    from torchcodec.decoders import VideoDecoder

    # initialize video decoder
    decoder = VideoDecoder(video_path, device=device, seek_mode="approximate")
    loaded_frames = []
    loaded_ts = []
    # get metadata for frame information
    metadata = decoder.metadata
    average_fps = metadata.average_fps

    # convert timestamps to frame indices
    frame_indices = [round(ts * average_fps) for ts in timestamps]

    # retrieve frames based on indices
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=False):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            logging.info(f"Frame loaded at timestamp={pts:.4f}")

    # Use float32 for timestamp distance computation (torch.cdist doesn't support bfloat16).
    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts = torch.tensor(loaded_ts, dtype=torch.float32)

    # compute distances between each query timestamp and loaded timestamps
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)


    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to float32 in [0,1] range (channel first)
    closest_frames = closest_frames.type(torch.float32) / 255
    return closest_frames


# GPU speed up video encoding


@dataclass
class VideoFrame:
    # TODO(rcadene, lhoestq): move to Hugging Face `datasets` repo
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    # to make VideoFrame available in HuggingFace `datasets`
    register_feature(VideoFrame, "VideoFrame")
