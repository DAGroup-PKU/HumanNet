# Modified by Open-HumanScale contributors: package namespace renamed to humanwam.
"""Read-only sharded SQLite packs for FastWAM text embeddings.

The legacy cache stores one torch payload per prompt.  On object-storage mounts
that turns every sample into metadata lookup + open + read operations.  A pack
keeps the exact serialized payload bytes but indexes them in a small number of
SQLite files.  Connections are persistent and process/thread local so DataLoader
fork/spawn workers never share SQLite state.
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote

import torch


TEXT_EMBEDDING_PACK_SCHEMA = "humanwam.text-embedding-pack.v1"
TEXT_EMBEDDING_PACK_USER_VERSION = 1


class TextEmbeddingPack:
    """Read-only lookup over prefix-sharded SQLite files."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        context_len: int,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.context_len = int(context_len)
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_shards = manifest.get("shards")

        self._shards: dict[str, Path] = {}
        self._shard_records: dict[str, dict[str, Any]] = {}
        for prefix, record in raw_shards.items():
            filename = record.get("file")
            shard_path = self.root / filename
            self._shards[prefix] = shard_path
            self._shard_records[prefix] = record

        self.manifest = manifest
        self._reset_connections()

    def _reset_connections(self) -> None:
        self._connections: dict[tuple[int, int, str], sqlite3.Connection] = {}
        self._connections_pid = os.getpid()
        self._connections_lock = threading.Lock()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_connections"] = {}
        state["_connections_pid"] = None
        state["_connections_lock"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._reset_connections()

    def close(self) -> None:
        connections = list(getattr(self, "_connections", {}).values())
        self._connections = {}
        for connection in connections:
            connection.close()

    def _connection(self, prefix: str) -> sqlite3.Connection:
        if getattr(self, "_connections_pid", None) != os.getpid():
            self.close()
            self._reset_connections()
        key = (os.getpid(), threading.get_ident(), prefix)
        connection = self._connections.get(key)
        if connection is not None:
            return connection
        with self._connections_lock:
            connection = self._connections.get(key)
            if connection is not None:
                return connection
            shard_path = self._shards[prefix]
            uri = f"file:{quote(str(shard_path), safe='/')}?mode=ro"
            connection = sqlite3.connect(
                uri,
                uri=True,
                timeout=30.0,
                check_same_thread=False,
                isolation_level=None,
            )
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA temp_store=MEMORY")
            self._connections[key] = connection
            return connection

    def load(self, legacy_cache_filename: str) -> dict[str, Any]:
        """Load one exact legacy torch payload from the indexed pack."""

        filename = str(legacy_cache_filename)
        prefix = filename[:1].lower()
        row = self._connection(prefix).execute(
            "SELECT payload "
            "FROM embeddings WHERE key = ?",
            (filename,),
        ).fetchone()
        (payload,) = row
        payload = bytes(payload)
        loaded = torch.load(io.BytesIO(payload), map_location="cpu")
        return loaded
