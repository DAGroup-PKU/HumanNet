"""Exercise the serialized cache formats consumed by training workers."""
import io
import json
import pickle
import sqlite3

import torch

from humanwam.datasets.lerobot.text_embedding_pack import TextEmbeddingPack
from humanwam.datasets.lerobot.video_latent_cache import VideoLatentCache


def test_text_embedding_pack_roundtrip_and_worker_reopen(tmp_path):
    expected = {"context": torch.arange(12).reshape(3, 4).float(),
                "context_mask": torch.tensor([True, True, False])}
    stream = io.BytesIO()
    torch.save(expected, stream)
    with sqlite3.connect(tmp_path / "a.sqlite") as connection:
        connection.execute("CREATE TABLE embeddings (key TEXT PRIMARY KEY, payload BLOB)")
        connection.execute("INSERT INTO embeddings VALUES (?, ?)", ("a.pt", stream.getvalue()))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "shards": {"a": {"file": "a.sqlite"}}}))
    pack = TextEmbeddingPack(tmp_path, context_len=3)
    clone = None
    try:
        loaded = pack.load("a.pt")
        clone = pickle.loads(pickle.dumps(pack))
        reopened = clone.load("a.pt")
        for key in expected:
            torch.testing.assert_close(loaded[key], expected[key])
            torch.testing.assert_close(reopened[key], expected[key])
    finally:
        pack.close()
        if clone is not None:
            clone.close()


def test_video_latent_cache_reads_shard_boundaries_and_returns_copies(tmp_path):
    expected = torch.arange(24).reshape(4, 2, 3).to(torch.bfloat16)
    shards = []
    for start, stop in [(0, 2), (2, 4)]:
        filename = f"{start}.bin"
        expected[start:stop].view(torch.uint16).numpy().tofile(tmp_path / filename)
        shards.append({"start": start, "stop": stop, "file": filename})
    (tmp_path / "CACHE_MANIFEST.json").write_text(json.dumps({
        "samples": 4, "video_frames": 5, "latent_shape": [2, 3], "shards": shards}))
    cache = VideoLatentCache(tmp_path)
    for index in range(4):
        torch.testing.assert_close(cache.lookup(index), expected[index])
    cache.lookup(2).zero_()
    torch.testing.assert_close(cache.lookup(2), expected[2])
