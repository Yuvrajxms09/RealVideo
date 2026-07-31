from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Optional, TypeVar

import torch
from einops import rearrange

from config.config import config as service_config
from core.profiler import cuda_memory_snapshot, runtime_profiler

if TYPE_CHECKING:
    from core.dit_service import StreamCausalInferencePipeline

logger = logging.getLogger(__name__)

T = TypeVar("T")
GpuRunner = Callable[
    [str, Callable[[], T], Optional[dict[str, Any]]], Awaitable[T]
]
OutputHandler = Callable[[torch.Tensor, str], Awaitable[None]]
ErrorHandler = Callable[[Exception], Awaitable[None]]


class SingleGPUInferenceEngine:
    """In-process equivalent of the distributed DiT service state machine."""

    def __init__(
        self,
        pipeline: StreamCausalInferencePipeline,
        gpu_runner: GpuRunner,
        output_handler: OutputHandler,
        error_handler: Optional[ErrorHandler] = None,
    ) -> None:
        self.pipeline = pipeline
        self._run_gpu = gpu_runner
        self._output_handler = output_handler
        self._error_handler = error_handler
        self._commands: asyncio.Queue[
            tuple[str, dict[str, Any], int, float]
        ] = asyncio.Queue(maxsize=128)
        self._worker_task: Optional[asyncio.Task] = None
        self._stop_requested = asyncio.Event()
        self._stop_waiters: list[asyncio.Future] = []

        self._running = False
        self._conditional_dict: dict[str, Any] = {}
        self._sp_dim: Optional[str] = None
        self._audio_ptr = 0
        self._current_audio_length = 0
        self._audio_finished = True
        self._must_generate_once = False
        self.last_error: Optional[str] = None
        self._command_sequence = 0
        self._active_command_sequence = 0
        self._session_id = ""
        self._flow_log_every = max(
            1, int(os.getenv("REALVIDEO_FLOW_LOG_EVERY", "1"))
        )

    @property
    def running(self) -> bool:
        return self._running

    def request_stop(self) -> None:
        """Request a block-boundary stop without waiting for the worker."""
        self._stop_requested.set()

    def status(self) -> dict[str, Any]:
        return {
            "mode": "single_gpu",
            "running": self._running,
            "queued_commands": self._commands.qsize(),
            "audio_ptr": self._audio_ptr,
            "audio_length": self._current_audio_length,
            "generated_blocks": self.pipeline.generated_block_count,
            "active_command_sequence": self._active_command_sequence,
            "session_id": self._session_id,
            "last_error": self.last_error,
        }

    async def start_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(
                self._worker(), name="single-gpu-dit-worker"
            )
            logger.info("event=engine_worker_started mode=single_gpu")

    async def submit(self, signal: str, conditional_dict: dict[str, Any]) -> None:
        if signal not in {"start", "update", "stop"}:
            raise ValueError(f"Unsupported single-GPU signal: {signal}")
        await self.start_worker()
        self._command_sequence += 1
        command_sequence = self._command_sequence
        enqueued_at = time.perf_counter()
        if signal == "stop":
            self._stop_requested.set()
            completion = asyncio.get_running_loop().create_future()
            self._stop_waiters.append(completion)
        await self._commands.put(
            (signal, conditional_dict, command_sequence, enqueued_at)
        )
        logger.info(
            "event=engine_command_enqueued signal=%s command=%d session_id=%s "
            "condition_id=%s queue_depth=%d",
            signal,
            command_sequence,
            conditional_dict.get("session_id", self._session_id),
            conditional_dict.get("id", ""),
            self._commands.qsize(),
        )
        if signal == "stop":
            await completion

    async def close(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None
        self._reset_state()
        self._complete_stop_waiters()

    async def _worker(self) -> None:
        while True:
            try:
                (
                    signal,
                    conditional_dict,
                    command_sequence,
                    enqueued_at,
                ) = await self._commands.get()
                self._active_command_sequence = command_sequence
                self._log_command_dequeued(
                    signal, conditional_dict, command_sequence, enqueued_at
                )
                if signal == "stop":
                    self._handle_stop()
                    continue
                if signal != "start":
                    logger.warning("Ignoring %s while single-GPU engine is idle", signal)
                    continue

                await self._handle_start(conditional_dict)
                await self._generate_session()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.exception(
                    "event=engine_failure session_id=%s command=%d "
                    "error_type=%s cuda_memory=%s",
                    self._session_id,
                    self._active_command_sequence,
                    type(exc).__name__,
                    cuda_memory_snapshot(),
                )
                self._reset_state()
                self._complete_stop_waiters()
                if self._error_handler is not None:
                    await self._error_handler(exc)

    async def _generate_session(self) -> None:
        while self._running:
            if self._stop_requested.is_set():
                self._handle_stop()
                return

            if self._audio_finished and not self._must_generate_once:
                (
                    signal,
                    conditional_dict,
                    command_sequence,
                    enqueued_at,
                ) = await self._commands.get()
                self._active_command_sequence = command_sequence
                self._log_command_dequeued(
                    signal, conditional_dict, command_sequence, enqueued_at
                )
                if signal == "stop":
                    self._handle_stop()
                    return
                if signal == "start":
                    logger.warning("Restarting an active single-GPU session")
                    await self._handle_start(conditional_dict)
                else:
                    await self._handle_update(conditional_dict)

            if not self._running or self._stop_requested.is_set():
                continue

            await self._generate_one_block()
            self._must_generate_once = False

    async def _handle_start(self, conditional_dict: dict[str, Any]) -> None:
        if self._stop_requested.is_set():
            logger.info(
                "event=engine_start_skipped reason=stop_pending command=%d",
                self._active_command_sequence,
            )
            return
        self.last_error = None
        self._conditional_dict = dict(conditional_dict)
        self._conditional_dict["motion_latents"] = None
        self._sp_dim = self._conditional_dict.get("sp_dim")
        self._session_id = self._conditional_dict.get("session_id", "")

        started = time.perf_counter()
        await self._run_gpu(
            "dit.inference_init",
            lambda: self.pipeline.inference_init(
                conditional_dict=self._conditional_dict, sp_dim=self._sp_dim
            ),
            None,
        )
        if "prompt_embeds" in conditional_dict:
            await self._run_gpu(
                "dit.cross_attention_reset",
                self.pipeline.reset_crossattn_cache,
                None,
            )

        self._running = True
        self._set_audio_state(conditional_dict, silence_is_finished=False)
        self._must_generate_once = "audio_input" not in conditional_dict
        logger.info(
            "event=engine_state_changed from=idle to=generating session_id=%s "
            "command=%d init_ms=%.2f audio_length=%d sp_dim=%s",
            self._session_id,
            self._active_command_sequence,
            (time.perf_counter() - started) * 1000,
            self._current_audio_length,
            self._sp_dim,
        )

    async def _handle_update(self, conditional_dict: dict[str, Any]) -> None:
        self._conditional_dict.update(conditional_dict)
        self._sp_dim = self._conditional_dict.get("sp_dim", self._sp_dim)
        if "prompt_embeds" in conditional_dict:
            await self._run_gpu(
                "dit.cross_attention_reset",
                self.pipeline.reset_crossattn_cache,
                None,
            )
        self._set_audio_state(
            conditional_dict,
            silence_is_finished=conditional_dict.get("silence", False),
        )
        # The distributed implementation emits one block even for a silence
        # update, whose audio is marked complete immediately.
        self._must_generate_once = True
        logger.info(
            "event=engine_condition_applied session_id=%s command=%d "
            "request_id=%s condition_id=%s silence=%s audio_length=%d",
            self._session_id,
            self._active_command_sequence,
            conditional_dict.get("request_id", ""),
            conditional_dict.get("id", ""),
            conditional_dict.get("silence", False),
            self._current_audio_length,
        )

    def _set_audio_state(
        self, conditional_dict: dict[str, Any], *, silence_is_finished: bool
    ) -> None:
        if "audio_input" not in conditional_dict:
            self._audio_ptr = 0
            self._current_audio_length = 0
            self._audio_finished = True
            return
        self._audio_ptr = 0
        self._current_audio_length = conditional_dict["audio_input"].shape[-1] // 4
        self._audio_finished = silence_is_finished

    async def _generate_one_block(self) -> None:
        if (
            not service_config.lip_sync.no_refresh_inference
            and self.pipeline.current_start_frame
            >= service_config.lip_sync.s2v_video_refresh_interval
        ):
            self._conditional_dict["motion_latents"] = rearrange(
                torch.cat(self.pipeline.output_latent_queue, dim=1)[
                    :, -self._conditional_dict["motion_frames"][1] :, ...
                ],
                "b t c h w -> b c t h w",
            )
            await self._run_gpu(
                "dit.refresh",
                lambda: self.pipeline.inference_init(
                    conditional_dict=self._conditional_dict, sp_dim=self._sp_dim
                ),
                None,
            )

        block_index = self.pipeline.generated_block_count + 1
        audio_ptr = max(
            0,
            min(
                self._audio_ptr,
                self._current_audio_length - self.pipeline.num_frame_per_block,
            ),
        )

        def infer() -> torch.Tensor:
            with runtime_profiler.operator_trace(block_index):
                return self.pipeline.inference_one_block(
                    conditional_dict=self._conditional_dict,
                    sp_dim=self._sp_dim,
                    audio_ptr=audio_ptr,
                )

        cycle_started = time.perf_counter()
        output_block = await self._run_gpu(
            "dit.block",
            infer,
            {
                "block": block_index,
                "denoising_steps": len(self.pipeline.denoising_step_list),
                "session_id": self._session_id,
                "command": self._active_command_sequence,
                "request_id": self._conditional_dict.get("request_id", ""),
            },
        )
        block_dispatch_ms = (time.perf_counter() - cycle_started) * 1000

        if not service_config.lip_sync.no_refresh_inference:
            self.pipeline.output_latent_queue.append(output_block)

        if not self._audio_finished:
            self._audio_ptr += self.pipeline.num_frame_per_block
            self._audio_finished = self._audio_ptr >= self._current_audio_length

        with runtime_profiler.stage("single_gpu.output_dispatch"):
            await self._output_handler(
                output_block, self._conditional_dict.get("id", "")
            )
        cycle_ms = (time.perf_counter() - cycle_started) * 1000
        if block_index % self._flow_log_every == 0:
            logger.info(
                "event=engine_block_completed session_id=%s command=%d "
                "request_id=%s condition_id=%s block=%d block_dispatch_ms=%.2f "
                "generation_to_send_ms=%.2f audio_ptr=%d audio_length=%d "
                "audio_finished=%s output_shape=%s",
                self._session_id,
                self._active_command_sequence,
                self._conditional_dict.get("request_id", ""),
                self._conditional_dict.get("id", ""),
                block_index,
                block_dispatch_ms,
                cycle_ms,
                self._audio_ptr,
                self._current_audio_length,
                self._audio_finished,
                tuple(output_block.shape),
            )

    def _handle_stop(self) -> None:
        logger.info(
            "event=engine_state_changed from=generating to=idle "
            "session_id=%s command=%d queued_commands=%d",
            self._session_id,
            self._active_command_sequence,
            self._commands.qsize(),
        )
        self._reset_state()
        while not self._commands.empty():
            try:
                self._commands.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._complete_stop_waiters()

    def _complete_stop_waiters(self) -> None:
        for waiter in self._stop_waiters:
            if not waiter.done():
                waiter.set_result(None)
        self._stop_waiters.clear()

    def _reset_state(self) -> None:
        self._running = False
        self._stop_requested.clear()
        self._conditional_dict = {}
        self._sp_dim = None
        self._audio_ptr = 0
        self._current_audio_length = 0
        self._audio_finished = True
        self._must_generate_once = False

    def _log_command_dequeued(
        self,
        signal: str,
        conditional_dict: dict[str, Any],
        command_sequence: int,
        enqueued_at: float,
    ) -> None:
        logger.info(
            "event=engine_command_dequeued signal=%s command=%d session_id=%s "
            "condition_id=%s queue_wait_ms=%.2f queue_depth=%d",
            signal,
            command_sequence,
            conditional_dict.get("session_id", self._session_id),
            conditional_dict.get("id", ""),
            (time.perf_counter() - enqueued_at) * 1000,
            self._commands.qsize(),
        )
