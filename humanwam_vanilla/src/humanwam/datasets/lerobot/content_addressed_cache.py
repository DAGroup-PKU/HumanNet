# Modified by Open-HumanScale contributors: package namespace renamed to humanwam.
"""Process-safe content-addressed staging for remote LeRobot files."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import quote


CONTENT_CACHE_MANIFEST_SCHEMA = "humanwam.content-cache-manifest.v1"


def _normalize_source(path: str | os.PathLike[str]) -> str:
    # Deliberately avoid Path.resolve(): on a manifest-backed cache hit even a
    # readlink/stat on the BOS source would violate the "fully bypass BOS" contract.
    return os.path.normpath(os.path.abspath(os.path.expanduser(os.fspath(path))))


class ContentAddressedFileCache:
    """Atomic, bounded PFS cache for immutable BOS/PFS source files.

    With a local manifest containing source path, size, and sha256, a valid cache
    hit never touches the source filesystem.  Without a manifest, source size and
    mtime are checked on every lookup to prevent stale reuse.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        manifest_path: str | os.PathLike[str] | None = None,
        max_bytes: int = 0,
        cleanup_interval: int = 128,
        touch_interval_s: float = 60.0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects_dir = self.root / "objects"
        self.locators_dir = self.root / "locators"
        self.locks_dir = self.root / "locks"
        self.tmp_dir = self.root / "tmp"
        for directory in (
            self.objects_dir,
            self.locators_dir,
            self.locks_dir,
            self.tmp_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.cleanup_interval = int(cleanup_interval)
        self.touch_interval_s = float(touch_interval_s)

        self.manifest_path = (
            Path(manifest_path).expanduser().resolve()
            if manifest_path is not None
            else None
        )
        self._manifest_entries: dict[str, dict] = {}
        self._manifest_is_sqlite = False
        if self.manifest_path is not None:
            if self.manifest_path.suffix in {".sqlite", ".sqlite3", ".db"}:
                self._manifest_is_sqlite = True
            else:
                self._manifest_entries = self._load_manifest(self.manifest_path)
        self._reset_process_state()

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, dict]:
        document = json.loads(path.read_text(encoding="utf-8"))
        entries = document.get("entries")
        validated = {}
        for source, entry in entries.items():
            sha256 = str(entry.get("sha256", "")).lower()
            size = int(entry.get("size", -1))
            validated[source] = {"sha256": sha256, "size": size}
        return validated

    def _reset_process_state(self) -> None:
        self._pid = os.getpid()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "source_bytes": 0,
            "fills": 0,
            "evictions": 0,
        }
        self._fills_since_cleanup = 0
        self._last_touched: dict[str, float] = {}
        self._manifest_connections: dict[
            tuple[int, int], sqlite3.Connection
        ] = {}
        self._manifest_connections_lock = threading.Lock()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_pid"] = None
        state["_stats"] = {}
        state["_fills_since_cleanup"] = 0
        state["_last_touched"] = {}
        state["_manifest_connections"] = {}
        state["_manifest_connections_lock"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._reset_process_state()

    def _ensure_process_state(self) -> None:
        if getattr(self, "_pid", None) != os.getpid():
            for connection in getattr(
                self, "_manifest_connections", {}
            ).values():
                connection.close()
            self._reset_process_state()

    def _manifest_entry(self, source: str) -> dict | None:
        if not self._manifest_is_sqlite:
            return self._manifest_entries.get(source)
        self._ensure_process_state()
        key = (os.getpid(), threading.get_ident())
        connection = self._manifest_connections.get(key)
        if connection is None:
            with self._manifest_connections_lock:
                connection = self._manifest_connections.get(key)
                if connection is None:
                    uri = (
                        f"file:{quote(str(self.manifest_path), safe='/')}"
                        "?mode=ro"
                    )
                    connection = sqlite3.connect(
                        uri,
                        uri=True,
                        timeout=30.0,
                        check_same_thread=False,
                        isolation_level=None,
                    )
                    connection.execute("PRAGMA query_only=ON")
                    self._manifest_connections[key] = connection
        row = connection.execute(
            "SELECT size, sha256 FROM entries WHERE source = ?",
            (source,),
        ).fetchone()
        if row is None:
            return None
        size, sha256 = int(row[0]), str(row[1]).lower()
        return {"size": size, "sha256": sha256}

    def stats(self) -> dict[str, int]:
        self._ensure_process_state()
        return dict(self._stats)

    def _object_path(self, sha256: str) -> Path:
        return self.objects_dir / sha256[:2] / f"{sha256}.blob"

    def _locator_path(self, source: str) -> Path:
        key = hashlib.sha256(source.encode("utf-8")).hexdigest()
        return self.locators_dir / f"{key}.json"

    @contextmanager
    def _lock(self, key: str, *, blocking: bool = True) -> Iterator[bool]:
        path = self.locks_dir / f"{key}.lock"
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        acquired = False
        try:
            operation = fcntl.LOCK_EX
            if not blocking:
                operation |= fcntl.LOCK_NB
            try:
                fcntl.flock(fd, operation)
                acquired = True
            except BlockingIOError:
                acquired = False
            yield acquired
        finally:
            if acquired:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _object_is_cached(self, path: Path, entry: dict) -> bool:
        try:
            observed_size = path.stat().st_size
        except FileNotFoundError:
            return False
        if observed_size != int(entry["size"]):
            return False
        return True


    def _touch(self, path: Path, sha256: str) -> None:
        now = time.monotonic()
        previous = self._last_touched.get(sha256, 0.0)
        if now - previous < self.touch_interval_s:
            return
        try:
            os.utime(path, None)
        except FileNotFoundError:
            return
        self._last_touched[sha256] = now

    def _record_hit(self, path: Path, sha256: str) -> Path:
        self._stats["hits"] += 1
        self._touch(path, sha256)
        return path

    def _read_locator(self, source: str) -> dict | None:
        path = self._locator_path(source)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if (
            document.get("source") != source
            or int(document.get("size", -1)) < 0
            or int(document.get("mtime_ns", -1)) < 0
        ):
            return None
        return document

    def _write_locator(self, source: str, entry: dict) -> None:
        path = self._locator_path(source)
        payload = {
            "source": source,
            "sha256": entry["sha256"],
            "size": int(entry["size"]),
            "mtime_ns": int(entry["mtime_ns"]),
        }
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, path)

    def resolved_entry(self, source_path: str | os.PathLike[str]) -> dict | None:
        """Return locally recorded source metadata without touching the source."""

        source = _normalize_source(source_path)
        manifest_entry = self._manifest_entry(source)
        if manifest_entry is not None:
            return dict(manifest_entry)
        locator = self._read_locator(source)
        if locator is None:
            return None
        return {
            "sha256": locator["sha256"],
            "size": int(locator["size"]),
            "mtime_ns": int(locator["mtime_ns"]),
        }

    def stage(self, source_path: str | os.PathLike[str]) -> Path:
        """Return a validated local object, filling it atomically on a miss."""

        self._ensure_process_state()
        source = _normalize_source(source_path)
        manifest_entry = self._manifest_entry(source)
        if manifest_entry is not None:
            object_path = self._object_path(manifest_entry["sha256"])
            if self._object_is_cached(object_path, manifest_entry):
                # No source stat/read occurs on this path.
                return self._record_hit(object_path, manifest_entry["sha256"])
            lock_key = manifest_entry["sha256"]
            expected = dict(manifest_entry)
            expected["manifest_backed"] = True
        else:
            source_stat = os.stat(source)
            locator = self._read_locator(source)
            if (
                locator is not None
                and int(locator["size"]) == source_stat.st_size
                and int(locator["mtime_ns"]) == source_stat.st_mtime_ns
            ):
                object_path = self._object_path(locator["sha256"])
                if self._object_is_cached(object_path, locator):
                    return self._record_hit(object_path, locator["sha256"])
            lock_key = hashlib.sha256(source.encode("utf-8")).hexdigest()
            expected = {
                "size": source_stat.st_size,
                "mtime_ns": source_stat.st_mtime_ns,
                "manifest_backed": False,
            }

        self._stats["misses"] += 1
        with self._lock(lock_key):
            if manifest_entry is not None:
                object_path = self._object_path(manifest_entry["sha256"])
                if self._object_is_cached(object_path, manifest_entry):
                    return self._record_hit(
                        object_path, manifest_entry["sha256"]
                    )
            else:
                source_stat = os.stat(source)
                locator = self._read_locator(source)
                if (
                    locator is not None
                    and int(locator["size"]) == source_stat.st_size
                    and int(locator["mtime_ns"]) == source_stat.st_mtime_ns
                ):
                    object_path = self._object_path(locator["sha256"])
                    if self._object_is_cached(object_path, locator):
                        return self._record_hit(object_path, locator["sha256"])
                expected["size"] = source_stat.st_size
                expected["mtime_ns"] = source_stat.st_mtime_ns

            temporary = self.tmp_dir / f"fill-{lock_key}.{os.getpid()}.tmp"
            digest = hashlib.sha256()
            copied = 0
            try:
                with open(source, "rb", buffering=0) as src, temporary.open(
                    "wb", buffering=0
                ) as dst:
                    while True:
                        payload = src.read(8 * 1024 * 1024)
                        if not payload:
                            break
                        dst.write(payload)
                        digest.update(payload)
                        copied += len(payload)
                    dst.flush()
                    os.fsync(dst.fileno())
                after = os.stat(source)
                observed_sha = digest.hexdigest()
                object_path = self._object_path(observed_sha)
                object_path.parent.mkdir(parents=True, exist_ok=True)
                if self._object_is_cached(
                    object_path, {"size": copied, "sha256": observed_sha}
                ):
                    temporary.unlink(missing_ok=True)
                else:
                    os.replace(temporary, object_path)
                locator_entry = {
                    "sha256": observed_sha,
                    "size": copied,
                    "mtime_ns": after.st_mtime_ns,
                }
                self._write_locator(source, locator_entry)
            finally:
                temporary.unlink(missing_ok=True)

        self._stats["source_bytes"] += copied
        self._stats["fills"] += 1
        self._fills_since_cleanup += 1
        self._touch(object_path, observed_sha)
        if (
            self.max_bytes > 0
            and self._fills_since_cleanup >= self.cleanup_interval
        ):
            self.cleanup(protected_sha256=observed_sha)
            self._fills_since_cleanup = 0
        return object_path

    def cleanup(self, *, protected_sha256: str | None = None) -> int:
        """Evict oldest objects until the configured byte bound is met."""

        self._ensure_process_state()
        if self.max_bytes <= 0:
            return 0
        objects = []
        total = 0
        for path in self.objects_dir.glob("*/*.blob"):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            sha256 = path.stem
            total += stat.st_size
            objects.append((stat.st_atime_ns, stat.st_mtime_ns, sha256, stat.st_size, path))
        if total <= self.max_bytes:
            return 0
        evictions = 0
        for _, _, sha256, size, path in sorted(objects):
            if total <= self.max_bytes:
                break
            if sha256 == protected_sha256:
                continue
            with self._lock(sha256, blocking=False) as acquired:
                if not acquired:
                    continue
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
            total -= size
            evictions += 1
        self._stats["evictions"] += evictions
        return evictions
