from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, TypeVar

import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def cuda_memory_snapshot(device=None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    if device is None:
        device = torch.cuda.current_device()
    return {
        "device": torch.cuda.get_device_name(device),
        "allocated_mb": round(torch.cuda.memory_allocated(device) / 2**20, 1),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / 2**20, 1),
        "peak_allocated_mb": round(
            torch.cuda.max_memory_allocated(device) / 2**20, 1
        ),
        "peak_reserved_mb": round(
            torch.cuda.max_memory_reserved(device) / 2**20, 1
        ),
    }


class RuntimeProfiler:
    """Low-overhead stage timings with optional synchronized CUDA measurements."""

    def __init__(self) -> None:
        self.enabled = _env_bool("REALVIDEO_PROFILE", True)
        self.cuda_sync = _env_bool("REALVIDEO_PROFILE_CUDA_SYNC", False)
        self.log_every = max(
            1, int(os.getenv("REALVIDEO_PROFILE_LOG_EVERY", "10"))
        )
        self.trace_block = int(os.getenv("REALVIDEO_TORCH_PROFILE_BLOCK", "0"))
        self.trace_with_stack = _env_bool(
            "REALVIDEO_TORCH_PROFILE_WITH_STACK", False
        )
        self.output_dir = Path(
            os.getenv("REALVIDEO_PROFILE_DIR", "profiles")
        ).resolve()
        self._lock = threading.Lock()
        self._counts: dict[str, int] = defaultdict(int)
        self._total_ms: dict[str, float] = defaultdict(float)
        self._recent_ms: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self._trace_lock = threading.Lock()
        self._traced_blocks: set[int] = set()
        self._event_stream = None
        self._event_path = None
        self._write_failed = False
        atexit.register(self.close)

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        cuda: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        cuda_timing = cuda and torch.cuda.is_available() and self.cuda_sync
        start_event = end_event = None
        if cuda_timing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        started = time.perf_counter()
        record_context = torch.profiler.record_function(f"realvideo::{name}")
        try:
            with record_context:
                yield
        finally:
            wall_ms = (time.perf_counter() - started) * 1000
            cuda_ms = None
            if cuda_timing and start_event is not None and end_event is not None:
                end_event.record()
                end_event.synchronize()
                cuda_ms = start_event.elapsed_time(end_event)
            self._record(name, wall_ms, cuda_ms, metadata)

    def call(
        self,
        name: str,
        fn: Callable[[], T],
        *,
        cuda: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> T:
        with self.stage(name, cuda=cuda, metadata=metadata):
            return fn()

    @contextmanager
    def operator_trace(self, block_index: int) -> Iterator[None]:
        should_trace = (
            self.enabled
            and self.trace_block > 0
            and block_index == self.trace_block
            and torch.cuda.is_available()
        )
        if not should_trace:
            with nullcontext():
                yield
            return

        with self._trace_lock:
            already_traced = block_index in self._traced_blocks
            if not already_traced:
                self._traced_blocks.add(block_index)
        if already_traced:
            yield
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        rank = os.getenv("RANK", "0")
        trace_path = (
            self.output_dir / f"torch-rank-{rank}-block-{block_index}.json"
        )
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        logger.info("Capturing PyTorch operator trace for block %d", block_index)
        try:
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=self.trace_with_stack,
            ) as profile:
                yield
            profile.export_chrome_trace(str(trace_path))
            logger.info("PyTorch operator trace written to %s", trace_path)
        except Exception:
            with self._trace_lock:
                self._traced_blocks.discard(block_index)
            logger.exception("Failed to capture PyTorch operator trace")
            raise

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stages = {}
            for name, count in self._counts.items():
                recent = list(self._recent_ms[name])
                stages[name] = {
                    "count": count,
                    "mean_ms": round(self._total_ms[name] / count, 3),
                    "last_ms": round(recent[-1], 3),
                    "recent_max_ms": round(max(recent), 3),
                }
        snapshot = {
            "enabled": self.enabled,
            "cuda_sync": self.cuda_sync,
            "operator_trace_block": self.trace_block,
            "output_dir": str(self.output_dir),
            "stages": stages,
        }
        if torch.cuda.is_available():
            snapshot["cuda_memory"] = cuda_memory_snapshot()
        return snapshot

    def close(self) -> None:
        with self._lock:
            if self._event_stream is not None:
                self._event_stream.close()
                self._event_stream = None

    def _record(
        self,
        name: str,
        wall_ms: float,
        cuda_ms: Optional[float],
        metadata: Optional[Mapping[str, Any]],
    ) -> None:
        with self._lock:
            self._counts[name] += 1
            count = self._counts[name]
            self._total_ms[name] += wall_ms
            self._recent_ms[name].append(wall_ms)
            mean_ms = self._total_ms[name] / count

        event = {
            "timestamp": time.time(),
            "pid": os.getpid(),
            "stage": name,
            "wall_ms": round(wall_ms, 3),
            "cuda_ms": round(cuda_ms, 3) if cuda_ms is not None else None,
            "count": count,
            "mean_ms": round(mean_ms, 3),
            "metadata": dict(metadata or {}),
        }
        self._write_event(event)
        if count % self.log_every == 0:
            cuda_text = (
                f", cuda={cuda_ms:.2f} ms" if cuda_ms is not None else ""
            )
            logger.info(
                "PROFILE %s: wall=%.2f ms%s, mean=%.2f ms, count=%d",
                name,
                wall_ms,
                cuda_text,
                mean_ms,
                count,
            )

    def _write_event(self, event: Mapping[str, Any]) -> None:
        if self._write_failed:
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            rank = os.getenv("RANK", "0")
            path = self.output_dir / f"metrics-rank-{rank}.jsonl"
            with self._lock:
                if self._event_stream is None or self._event_path != path:
                    if self._event_stream is not None:
                        self._event_stream.close()
                    self._event_stream = path.open(
                        "a", encoding="utf-8", buffering=1
                    )
                    self._event_path = path
                self._event_stream.write(json.dumps(event, default=str) + "\n")
        except OSError:
            self._write_failed = True
            logger.exception("Unable to write profiling event")


runtime_profiler = RuntimeProfiler()
