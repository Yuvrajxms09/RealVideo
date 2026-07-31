import asyncio
import base64
import io
import logging
import time
import traceback
import uuid
from typing import Any, Optional

import torch
import torchaudio

from config.config import config as service_config
from core.lip_sync import LipSyncManager
from core.profiler import runtime_profiler
from core.tts_pipeline import TTSPipeline
from core.utils import encode_audio_to_base64

logger = logging.getLogger(__name__)


class ModelHandler:
    def __init__(self, single_gpu: bool = False):
        self.vae_idle_event = asyncio.Event()
        self.vae_idle_event.set()

        self.lip_sync_manager = LipSyncManager(
            vae_idle_event=self.vae_idle_event, single_gpu=single_gpu
        )
        self.tts_pipeline = TTSPipeline(vae_idle_event=self.vae_idle_event)

        self.audio_count = 0

        self.audio_chunk_length = 1000 * service_config.lip_sync.audio_segment_length
        self.text_input_queue = asyncio.Queue(16)
        self.audio_chunk_queue = asyncio.Queue(64)  # audio output chunk

        self.audio_process_task = None
        self.audio_watermark_length = 0.6
        self.websocket = None

        self.audio_chunk_sizes = [1, 1, 3, 5]
        self._session_lock = asyncio.Lock()

    def runtime_status(self):
        return {
            "inference": self.lip_sync_manager.runtime_status(),
            "queues": {
                "text_input": self.text_input_queue.qsize(),
                "audio_chunks": self.audio_chunk_queue.qsize(),
            },
            "workers": {
                "audio": self._task_status(self.audio_process_task),
                "llm": self._task_status(self.tts_pipeline.llm_task),
                "tts": self._task_status(self.tts_pipeline.tts_task),
            },
            "tts_available": self.tts_pipeline.available,
        }

    @staticmethod
    def _task_status(task):
        if task is None:
            return "not_started"
        if task.cancelled():
            return "cancelled"
        if task.done():
            return "failed" if task.exception() is not None else "completed"
        return "running"

    async def close(self):
        await self.end_session()
        await self.lip_sync_manager.close()
        logger.info("event=model_handler_stopped")

    async def end_session(self):
        async with self._session_lock:
            if (
                self.audio_process_task is not None
                and not self.audio_process_task.done()
            ):
                self.audio_process_task.cancel()
                await asyncio.gather(
                    self.audio_process_task, return_exceptions=True
                )
            self.audio_process_task = None
            await self.tts_pipeline.stop_async_tasks()
            self._drain_queue(self.text_input_queue)
            self._drain_queue(self.audio_chunk_queue)
            self.tts_pipeline.reset_status()
            await self.lip_sync_manager.disconnect_websocket()
            self.websocket = None
            logger.info("event=model_session_ended")

    @staticmethod
    def _drain_queue(queue):
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    async def start_jobs(self, websocket):
        self.websocket = websocket
        self.tts_pipeline.start_async_tasks(
            text_input_queue=self.text_input_queue, output_queue=self.audio_chunk_queue
        )
        if self.audio_process_task is None:
            ready_event = asyncio.Event()
            ready_event.clear()
            self.audio_process_task = asyncio.create_task(
                self.process_audio(ready_event)
            )
            await ready_event.wait()
            logger.info("Audio processing task started")

    async def process_message(
        self,
        audio_base64: Optional[str] = None,
        sample_rate: Optional[int] = None,
        profile_content: Optional[str] = None,
        text_content: Optional[str] = None,
        voice_id: Optional[str] = None,
        websocket: Optional[Any] = None,
    ) -> None:
        request_id = str(uuid.uuid4())
        if audio_base64 and len(audio_base64) < 100:
            raise ValueError(
                "Audio input is too short to contain a valid audio chunk"
            )
        if audio_base64 is not None and (
            sample_rate is None or sample_rate <= 0
        ):
            raise ValueError(
                "A positive sample_rate is required for audio input"
            )
        if text_content is not None and not self.tts_pipeline.available:
            raise RuntimeError(
                "Text input requires ZAI_API_KEY, but direct audio input "
                "remains available"
            )
        if text_content is not None and not text_content.strip():
            raise ValueError("Text input cannot be empty")
        self.websocket = websocket

        try:
            if text_content is not None:
                await self.text_input_queue.put(
                    {
                        "request_id": request_id,
                        "profile": profile_content,
                        "text": text_content,
                        "voice_id": voice_id,
                    }
                )
            elif audio_base64 is not None:
                await self.audio_chunk_queue.put(
                    {
                        "request_id": request_id,
                        "audio_base64": audio_base64,
                        "sample_rate": sample_rate,
                        "chunk_id": 1,
                        "time": time.time(),
                    }
                )
            elif audio_base64 is None and sample_rate is None:
                await self.audio_chunk_queue.put(None)

            logger.info(
                "event=input_enqueued request_id=%s kind=%s "
                "text_queue_depth=%d audio_queue_depth=%d",
                request_id,
                "text" if text_content is not None else "audio",
                self.text_input_queue.qsize(),
                self.audio_chunk_queue.qsize(),
            )
        except Exception as e:
            logger.exception(
                "event=input_enqueue_failed request_id=%s error=%s",
                request_id,
                e,
            )
            raise

    async def process_audio(self, ready_event):
        logger.info("Starting audio processing task")
        ready_event.set()
        while True:
            audio = None
            audio_list = []
            audio_segment_id = 0
            chunk_count = 0
            request_id = ""
            try:
                while True:
                    await asyncio.sleep(0)
                    chunk = await self.audio_chunk_queue.get()

                    if chunk is None:
                        break
                    if chunk.get("end", False):
                        request_id = chunk.get("request_id", request_id)
                        break

                    await asyncio.sleep(0)
                    with runtime_profiler.stage("audio.base64_decode"):
                        current_audio_bytes = base64.b64decode(
                            chunk["audio_base64"]
                        )
                    sample_rate = chunk["sample_rate"]
                    chunk_id = chunk["chunk_id"]
                    request_id = chunk.get("request_id", request_id)
                    logger.info(
                        "event=audio_chunk_received request_id=%s chunk=%d "
                        "sample_rate=%d encoded_bytes=%d",
                        request_id,
                        chunk_id,
                        sample_rate,
                        len(chunk["audio_base64"]),
                    )

                    await asyncio.sleep(0)
                    with runtime_profiler.stage("audio.waveform_decode"):
                        current_audio, sr = torchaudio.load(
                            io.BytesIO(current_audio_bytes), format="s16le"
                        )
                    logger.debug(
                        "event=audio_waveform_decoded request_id=%s chunk=%d "
                        "shape=%s decoder_sample_rate=%d",
                        request_id,
                        chunk_id,
                        tuple(current_audio.shape),
                        sr,
                    )

                    await asyncio.sleep(0)
                    with runtime_profiler.stage("audio.resample"):
                        current_audio_16k = torchaudio.functional.resample(
                            current_audio,
                            orig_freq=sample_rate,
                            new_freq=service_config.audio.sample_rate,
                        )

                    if chunk_id == 0:
                        audio_list.append(
                            current_audio_16k[
                                ...,
                                int(
                                    self.audio_watermark_length
                                    * service_config.audio.sample_rate
                                ) :,
                            ]
                        )
                    else:
                        audio_list.append(current_audio_16k)

                    while sum([x.shape[-1] for x in audio_list]) >= (
                        self.audio_chunk_sizes[
                            min(audio_segment_id, len(self.audio_chunk_sizes) - 1)
                        ]
                        * service_config.audio_samples_per_video_block
                    ):
                        await asyncio.sleep(0)
                        audio_segment_id += 1
                        with runtime_profiler.stage("audio.segment"):
                            tmp_audio = torch.cat(audio_list, dim=-1)
                            chunk_audio = tmp_audio[..., : self.audio_chunk_length]
                            audio = tmp_audio[..., self.audio_chunk_length :]
                            audio_list = [audio]

                        await asyncio.sleep(0)
                        with runtime_profiler.stage("audio.wav_encode"):
                            chunk_audio_base64 = encode_audio_to_base64(
                                chunk_audio
                            )

                        await self._process_audio_chunk(
                            {
                                "request_id": request_id,
                                "audio_base64": chunk_audio_base64,
                                "decoded_audio": chunk_audio,
                            }
                        )
                        logger.info(
                            "event=audio_segment_emitted request_id=%s "
                            "segment=%d samples=%d",
                            request_id,
                            audio_segment_id,
                            chunk_audio.shape[-1],
                        )

                    chunk_count += 1  # audio chunk

                if sum([x.shape[-1] for x in audio_list]) > 0:
                    await asyncio.sleep(0)
                    with runtime_profiler.stage("audio.final_segment"):
                        tmp_audio = torch.cat(audio_list, dim=-1)
                        audio_base64 = encode_audio_to_base64(tmp_audio)

                    await self._process_audio_chunk(
                        {
                            "request_id": request_id,
                            "audio_base64": audio_base64,
                            "decoded_audio": tmp_audio,
                        }
                    )

                logger.info(
                    "event=audio_stream_completed request_id=%s chunks=%d",
                    request_id,
                    chunk_count,
                )
                await self._process_audio_chunk(None, request_id=request_id)

            except Exception as error:
                logger.exception(
                    "event=audio_processing_failed request_id=%s error=%s",
                    request_id,
                    error,
                )

    async def _process_audio_chunk(self, audio_data, request_id=""):
        await asyncio.sleep(0)
        try:
            if audio_data is None:
                audio_base64 = None
                await self.lip_sync_manager.process_audio_chunk(
                    None, None, request_id=request_id
                )
            else:
                request_id = audio_data.get("request_id", request_id)
                audio_base64 = audio_data.get("audio_base64", "")
                decoded_audio = audio_data.get("decoded_audio", None)  # tensor
                if not audio_base64:
                    return
                if decoded_audio is None or len(decoded_audio) == 0:
                    return

                await self.lip_sync_manager.process_audio_chunk(
                    audio_base64, decoded_audio, request_id=request_id
                )

        except Exception as e:
            logger.exception(f"Failed to process audio chunk: {e}")
            raise
