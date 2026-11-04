"""Opt-in GC policies for the training process, after data workers are started.

The freeze policy collects unreachable setup objects before freezing live setup
objects. Activations created by training remain in the ordinary GC generations.
Newly forked data workers restore the inherited original settings.
"""
import atexit
import gc
import json
import os
import resource
import time
from pathlib import Path


class TrainingGC:
    def __init__(self, mode, directory, rank=0):
        self.pid = os.getpid()
        self.rank = rank
        self.mode = mode
        self.step = -1
        self.phase = 'setup'
        self.original_threshold = gc.get_threshold()
        self.original_enabled = gc.isenabled()
        self.original_frozen = gc.get_freeze_count()
        self.did_freeze = False
        self.closed = False
        self.starts = {}
        self.counts = [0, 0, 0]
        self.duration_ms = [0., 0., 0.]
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(directory/f'rank{rank}.gc.jsonl', os.O_CREAT|os.O_APPEND|os.O_WRONLY, 0o600)
        gc.callbacks.append(self.on_gc)
        self.emit('configure_begin', threshold=self.original_threshold,
                  enabled=self.original_enabled, frozen=self.original_frozen)
        before = time.monotonic()
        collected = None
        if mode == 'freeze10':
            collected = gc.collect(2)
            gc.freeze()
            self.did_freeze = True
        if mode in {'threshold10', 'freeze10'}:
            t0, t1, t2 = self.original_threshold
            # Reduce trigger frequency while preserving young-generation cleanup
            # and leaving Python's relative promotion policy unchanged.
            gc.set_threshold(t0*10, t1, t2)
        self.emit('configure_end', threshold=gc.get_threshold(),
                  frozen=gc.get_freeze_count(), collected=collected,
                  setup_ms=(time.monotonic()-before)*1000)
        if hasattr(os, 'register_at_fork'):
            os.register_at_fork(after_in_child=self.after_fork)
        atexit.register(self.close)

    def emit(self, kind, **fields):
        if self.closed or os.getpid()!=self.pid:
            return
        row=dict(kind=kind, pid=self.pid, rank=self.rank, mode=self.mode,
                 step=self.step, phase=self.phase, time=time.time(), **fields)
        os.write(self.fd, (json.dumps(row)+'\n').encode())

    def on_gc(self, phase, info):
        if self.closed or os.getpid()!=self.pid:
            return
        generation=info['generation']
        if phase=='start':
            self.starts[generation]=(time.monotonic(),time.thread_time())
            return
        started=self.starts.pop(generation,None)
        if started is None:
            return
        elapsed=(time.monotonic()-started[0])*1000
        self.counts[generation]+=1
        self.duration_ms[generation]+=elapsed
        if generation>=1 or elapsed>=5:
            self.emit('collection', generation=generation, wall_ms=elapsed,
                      cpu_thread_ms=(time.thread_time()-started[1])*1000,
                      collected=info['collected'], uncollectable=info['uncollectable'])

    def tick(self, step, phase):
        if self.step==step:
            return
        self.step, self.phase=int(step),phase
        if step%10:
            return
        # RSS reads are cheap; avoid repeatedly walking smaps or enumerating the
        # millions of live Python objects just to measure memory.
        status={}
        for line in Path('/proc/self/status').read_text().splitlines():
            key,_,value=line.partition(':')
            if key in {'VmRSS','VmHWM','RssAnon','RssFile','VmSwap'}:
                status[key]=int(value.split()[0])*1024
        self.emit('sample', memory_bytes=status, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  threshold=gc.get_threshold(), count=gc.get_count(),
                  gc_counts=self.counts.copy(), gc_wall_ms=self.duration_ms.copy(),
                  frozen=gc.get_freeze_count())

    def after_fork(self):
        if os.getpid()==self.pid or self.closed:
            return
        if self.on_gc in gc.callbacks:
            gc.callbacks.remove(self.on_gc)
        gc.set_threshold(*self.original_threshold)
        if self.did_freeze:
            gc.unfreeze()
        (gc.enable if self.original_enabled else gc.disable)()
        os.close(self.fd)
        self.closed=True

    def close(self):
        if self.closed or os.getpid()!=self.pid:
            return
        self.emit('close', gc_counts=self.counts, gc_wall_ms=self.duration_ms,
                  threshold=gc.get_threshold(), frozen=gc.get_freeze_count())
        if self.on_gc in gc.callbacks:
            gc.callbacks.remove(self.on_gc)
        gc.set_threshold(*self.original_threshold)
        if self.did_freeze:
            gc.unfreeze()
        (gc.enable if self.original_enabled else gc.disable)()
        self.closed=True
        os.close(self.fd)


def configure_from_env(rank):
    mode=os.environ.get('FASTWAM_TRAIN_GC_MODE','').strip()
    if not mode:
        return None
    return TrainingGC(mode,os.environ['FASTWAM_GC_LOCAL'],rank)
