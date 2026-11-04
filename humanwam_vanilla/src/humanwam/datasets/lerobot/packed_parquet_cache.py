"""Opt-in, byte-bounded cache for immutable file-indexed episode tables.

One cache is shared by all dataset roots in a worker. This avoids multiplying
the memory limit by the number of roots. Episode filtering remains in LeRobot.
"""

from collections import OrderedDict
import os
from pathlib import Path
import threading

import pyarrow.parquet as pq


class PackedParquetTableCache:
    def __init__(self):
        self._reset()

    def _reset(self):
        self.pid = os.getpid()
        self.lock = threading.RLock()
        self.entries = OrderedDict()
        self.resident_bytes = 0
        self.hits = 0
        self.misses = 0

    def _ensure_process(self):
        # Check before acquiring a lock which another parent thread might have
        # held at fork. Spawn imports a fresh module/cache automatically.
        if self.pid != os.getpid():
            self._reset()

    @staticmethod
    def _identity(path):
        st = path.stat()
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    def _discard(self, path):
        _, _, size = self.entries.pop(path)
        self.resident_bytes -= size

    def read(self, source, max_bytes):
        self._ensure_process()
        path = Path(source).absolute()
        with self.lock:
            while self.entries and self.resident_bytes > max_bytes:
                self._discard(next(iter(self.entries)))
            if max_bytes <= 0:
                return pq.read_table(source)
            identity = self._identity(path)
            entry = self.entries.get(path)
            if entry is not None:
                if entry[0] == identity:
                    self.entries.move_to_end(path)
                    self.hits += 1
                    return entry[1]
                self._discard(path)
            # Serialize misses in a worker to avoid duplicate large reads.
            self.misses += 1
            table = pq.read_table(source)
            size = table.get_total_buffer_size()
            if size <= max_bytes:
                while self.entries and self.resident_bytes + size > max_bytes:
                    self._discard(next(iter(self.entries)))
                self.entries[path] = (identity, table, size)
                self.resident_bytes += size
            return table


PACKED_PARQUET_TABLE_CACHE = PackedParquetTableCache()


def read_packed_parquet_table(path):
    """Cache only when explicitly enabled; zero preserves the original reader.

    The budget bounds retained Arrow buffers, excluding the current read and
    downstream episode tensors. Training inputs must remain immutable, as with
    the existing episode LRU; the file cache checks identity on each file lookup.
    """
    max_bytes = int(os.environ.get("FASTWAM_PACKED_PARQUET_CACHE_BYTES", "0"))
    return PACKED_PARQUET_TABLE_CACHE.read(path, max_bytes)
