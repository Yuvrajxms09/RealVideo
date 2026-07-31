import asyncio
import datetime
import json
import logging
import math
import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import starlette
import torch
import torchvision.transforms as TT
import websockets
from einops import rearrange
from PIL import Image

from config.config import config as service_config
from core import comm_utils
from core.dit_service import load_inference_pipeline
from core.distributed import send_dict
from core.profiler import cuda_memory_snapshot, runtime_profiler
from core.single_gpu_engine import SingleGPUInferenceEngine
from core.utils import encode_image_async, encode_image_to_base64
from self_forcing.utils import parallel_state as mpu
from self_forcing.utils.wan_wrapper import WanTextEncoder, WanVAEWrapper
from self_forcing.wan.modules.audio_encoder import AudioEncoder

logger = logging.getLogger(__name__)


async def send_cond_worker_async(
    cond_queue: asyncio.Queue,
    ready_signal_queue: asyncio.Queue,
    vae_idle_event: asyncio.Event,
    profile=False,
):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    client_socket = None
    while True:
        try:
            logger.debug("Rank %d send worker: waiting for cond_queue" % mpu.get_rank())
            signal, conditional_dict = await cond_queue.get()
            logger.debug(
                "Rank %d send worker: cond fetched, waiting vae idle with signal: %s"
                % (mpu.get_rank(), signal)
            )

            await asyncio.sleep(0)
            if not conditional_dict.get("silence", False):
                await vae_idle_event.wait()
            conditional_dict["uuid"] = str(uuid.uuid4())
            logger.debug(
                "Rank %d send worker: vae idle passed, sending signal %s"
                % (mpu.get_rank(), signal)
            )

            client_socket = comm_utils.socket_send(
                data={"signal": signal},
                port=service_config.server.diffusion_socket_port,
                client_socket=client_socket,
            )
            if signal == "stop":
                continue
            logger.debug(
                "Rank %d: socket send done, waiting for target ready" % mpu.get_rank()
            )

            await ready_signal_queue.get()
            logger.debug(
                "Rank %d: ready signal received, sending conditional dict, uuid: %s"
                % (mpu.get_rank(), conditional_dict["uuid"])
            )

            send_dict(conditional_dict, dst=1, profile=profile)
            logger.debug(
                "Rank %d: conditional dict sent, uuid: %s"
                % (mpu.get_rank(), conditional_dict["uuid"])
            )
            await asyncio.sleep(0.01)

        except Exception as e:
            logger.exception(f"Exception in send_cond_worker: {e}")
            logger.exception(traceback.format_exc())


def remap_image(img: torch.Tensor):
    img = (
        (torch.clamp((img + 1) / 2, min=0, max=1) * 255).to(torch.uint8).flip(dims=[-1])
    )
    img = img.cpu().numpy().astype(np.uint8)
    return img


def nearest_multiple_of_64(n):
    lower_multiple = (n // 64) * 64
    upper_multiple = (n // 64 + 1) * 64
    if abs(n - lower_multiple) < abs(n - upper_multiple):
        return lower_multiple
    else:
        return upper_multiple


def get_closest_aspect_ratio(aspect_ratio):
    target_ratios = [16 / 9, 4 / 3, 1.0]
    distances = [abs(aspect_ratio - ratio) for ratio in target_ratios]
    closest_idx = distances.index(min(distances))
    return target_ratios[closest_idx]


def read_image(image_path, image_size=None, max_image_area=262144):
    image = Image.open(image_path).convert("RGB")
    img_W, img_H = image.size
    area = img_H * img_W
    if image_size is None:
        resize_ratio = math.sqrt(max_image_area / area)
        img_H = round(img_H * resize_ratio)
        img_W = round(img_W * resize_ratio)

    if img_H < img_W:
        target_ratio = get_closest_aspect_ratio(img_W / img_H)
        if image_size is None:
            target_H = int(nearest_multiple_of_64(img_H))
        else:
            target_H = image_size
        target_W = int(nearest_multiple_of_64(target_H * target_ratio))
        if img_W / img_H > target_ratio:
            resize_H = target_H
            resize_W = int(img_W / img_H * resize_H)
        else:
            resize_W = target_W
            resize_H = int(img_H / img_W * resize_W)
    else:
        target_ratio = get_closest_aspect_ratio(img_H / img_W)
        if image_size is None:
            target_W = int(nearest_multiple_of_64(img_W))
        else:
            target_W = image_size
        target_H = int(nearest_multiple_of_64(target_W * target_ratio))
        if img_H / img_W > target_ratio:
            resize_W = target_W
            resize_H = int(img_H / img_W * resize_W)
        else:
            resize_H = target_H
            resize_W = int(img_W / img_H * resize_H)

    chained_trainsforms = []
    chained_trainsforms.append(TT.Resize(size=[resize_H, resize_W], interpolation=3))
    chained_trainsforms.append(TT.CenterCrop(size=[target_H, target_W]))
    chained_trainsforms.append(TT.ToTensor())
    transform = TT.Compose(chained_trainsforms)
    image = transform(image).unsqueeze(0)  # chw
    image = image * 2.0 - 1.0

    if image.shape[-2] < image.shape[-1]:
        sp_dim = "h"
    else:
        sp_dim = "w"
    return image, sp_dim


class LipSyncManager:
    """
    VAE and image/audio encoder. Rank 0 of the inference service.
    The process of SelfForcingLipSync including:
        1. Handling user interaction. It receives user input and get audio response from a LLM+TTS or voice LLM.
        2. Sending Audio to AR DiT service.
        3. Receiving generated latent blocks from AR DiT service.
        4. Decoding latent blocks by local VAE. Sending decoded frames to frontend.
    """

    def __init__(self, vae_idle_event: asyncio.Event, single_gpu: bool = False):
        self.fps = service_config.lip_sync.fps
        self.predefined_frames = []
        self.vae_idle_event = vae_idle_event
        self.single_gpu = single_gpu
        self._gpu_lock = asyncio.Lock()
        self._gpu_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="realvideo-gpu")
            if single_gpu
            else None
        )

        self.device = torch.cuda.current_device()
        logger.info("event=model_loading name=vae device=%s", self.device)
        with runtime_profiler.stage("models.vae_load", cuda=True):
            self.vae = WanVAEWrapper().to(
                dtype=torch.bfloat16, device=self.device
            )
        logger.info("event=model_loaded name=vae")
        logger.info("event=model_loading name=text_encoder device=%s", self.device)
        with runtime_profiler.stage("models.text_encoder_load", cuda=True):
            self.text_encoder = WanTextEncoder().to(
                dtype=torch.bfloat16, device=self.device
            )
        logger.info("event=model_loaded name=text_encoder")
        logger.info("event=model_loading name=audio_encoder device=%s", self.device)
        with runtime_profiler.stage("models.audio_encoder_load", cuda=True):
            self.s2v_audio_encoder = AudioEncoder(device=self.device)
        logger.info("event=model_loaded name=audio_encoder")

        self.signal_queue = asyncio.Queue(8)  # For receiving control signals
        self.ready_signal_queue = asyncio.Queue(2)  # For receiving ready signals
        self.cond_queue = asyncio.Queue(128)  # For sending conditional dict

        self.socket_server_task = None
        self.ready_socket_server_task = None
        self.sender_task = None
        self.vae_task = None

        self.do_decode_control_queue = asyncio.Queue(1)

        self.conditional_dict = {}
        self.audio_storage = {}  # For return audio_base64
        self.websocket = None
        self.session_id = ""
        self._background_stop_task = None

        self.silence_audio = (
            torch.randn(
                (1, service_config.lip_sync.audio_min_length * 1000),
                dtype=torch.float32,
            )
            * 1e-2
        )

        self.silence_prompt = service_config.video.silence_prompt
        self.speaking_prompt = service_config.video.speaking_prompt

        self.service_running = asyncio.Event()
        self.service_running.clear()

        self.frame_count = 0
        self.profile = service_config.lip_sync.profile

        self.ready_socket = None

        self.single_gpu_engine = None
        if self.single_gpu:
            pipeline = load_inference_pipeline(self.device)
            self.single_gpu_engine = SingleGPUInferenceEngine(
                pipeline=pipeline,
                gpu_runner=self._run_gpu,
                output_handler=self._handle_single_gpu_output,
                error_handler=self._handle_single_gpu_error,
            )
            # Model transfers are issued on the startup thread. Complete them
            # before handing all subsequent CUDA work to the dedicated worker.
            torch.cuda.synchronize(self.device)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        logger.info(
            "event=lip_sync_initialized rank=%d mode=%s cuda_memory=%s",
            mpu.get_rank(),
            "single_gpu" if self.single_gpu else "distributed",
            cuda_memory_snapshot(self.device),
        )

    async def _run_gpu(self, stage, fn, metadata=None):
        queued_at = time.perf_counter()
        async with self._gpu_lock:
            lock_wait_ms = (time.perf_counter() - queued_at) * 1000
            started = time.perf_counter()
            self.vae_idle_event.clear()
            try:
                if self._gpu_executor is None:
                    result = runtime_profiler.call(
                        stage, fn, cuda=True, metadata=metadata
                    )
                else:
                    def execute():
                        torch.cuda.set_device(self.device)
                        return runtime_profiler.call(
                            stage, fn, cuda=True, metadata=metadata
                        )

                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        self._gpu_executor, execute
                    )
                logger.debug(
                    "event=gpu_stage_completed session_id=%s stage=%s "
                    "lock_wait_ms=%.2f dispatch_ms=%.2f",
                    self.session_id,
                    stage,
                    lock_wait_ms,
                    (time.perf_counter() - started) * 1000,
                )
                return result
            except Exception as error:
                logger.exception(
                    "event=gpu_stage_failed session_id=%s stage=%s "
                    "lock_wait_ms=%.2f error_type=%s cuda_memory=%s",
                    self.session_id,
                    stage,
                    lock_wait_ms,
                    type(error).__name__,
                    cuda_memory_snapshot(self.device),
                )
                raise
            finally:
                self.vae_idle_event.set()

    async def _submit_condition(self, signal, conditional_dict):
        conditional_dict = dict(conditional_dict)
        conditional_dict.setdefault("session_id", self.session_id)
        logger.info(
            "event=condition_submitted session_id=%s signal=%s "
            "condition_id=%s silence=%s summary=%s",
            self.session_id,
            signal,
            conditional_dict.get("id", ""),
            conditional_dict.get("silence", False),
            self._condition_summary(conditional_dict),
        )
        if self.single_gpu_engine is not None:
            await self.single_gpu_engine.submit(signal, conditional_dict)
        else:
            await self.cond_queue.put((signal, conditional_dict))

    async def _handle_single_gpu_output(self, output_block, output_id):
        if not self.service_running.is_set():
            return
        decode_wait_started = time.perf_counter()
        while self.service_running.is_set():
            try:
                await asyncio.wait_for(
                    self.do_decode_control_queue.get(), timeout=5
                )
                break
            except asyncio.TimeoutError:
                logger.warning(
                    "event=decode_backpressure_waiting session_id=%s "
                    "condition_id=%s waited_ms=%.2f",
                    self.session_id,
                    output_id,
                    (time.perf_counter() - decode_wait_started) * 1000,
                )
        decode_wait_ms = (time.perf_counter() - decode_wait_started) * 1000
        if not self.service_running.is_set():
            return
        output_started = time.perf_counter()

        def decode_and_remap():
            frame_block = self.vae.decode_to_pixel(output_block, use_cache=True)
            frame_block = rearrange(
                frame_block, "b t c h w -> b t h w c"
            ).squeeze(0)
            return remap_image(frame_block)

        frames = await self._run_gpu("vae.decode_and_remap", decode_and_remap)
        with runtime_profiler.stage(
            "frames.jpeg_encode", metadata={"frames": len(frames)}
        ):
            frame_dicts = await asyncio.gather(
                *[
                    asyncio.to_thread(
                        lambda frame=frame: {
                            "image_base64": encode_image_to_base64(frame)
                        }
                    )
                    for frame in frames
                ]
            )

        request_id = ""
        audio_to_output_ms = None
        if output_id in self.audio_storage:
            audio_data = self.audio_storage.pop(output_id)
            request_id = audio_data.get("request_id", "")
            audio_to_output_ms = (
                time.time() - audio_data.get("time", time.time())
            ) * 1000
            frame_dicts[0].update(
                {
                    "audio_base64": audio_data.get("data", ""),
                    "audio_length": audio_data.get("length", 5),
                }
            )

        with runtime_profiler.stage(
            "websocket.frame_batch_send", metadata={"frames": len(frame_dicts)}
        ):
            sent_frames = 0
            for frame_data in frame_dicts:
                sent = await self.send_frame(frame_data, self.frame_count)
                if not sent:
                    break
                self.frame_count += 1
                sent_frames += 1
        logger.info(
            "event=output_block_delivered session_id=%s request_id=%s "
            "condition_id=%s frames=%d decode_backpressure_ms=%.2f "
            "output_pipeline_ms=%.2f audio_to_output_ms=%s",
            self.session_id,
            request_id,
            output_id,
            sent_frames,
            decode_wait_ms,
            (time.perf_counter() - output_started) * 1000,
            (
                f"{audio_to_output_ms:.2f}"
                if audio_to_output_ms is not None
                else "n/a"
            ),
        )

    async def _handle_single_gpu_error(self, error):
        self.service_running.clear()
        logger.error(
            "event=single_gpu_service_failed session_id=%s error_type=%s "
            "error=%s",
            self.session_id,
            type(error).__name__,
            error,
        )
        if self.websocket is not None:
            try:
                await self.websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": f"Video inference failed: {error}",
                            "timestamp": time.time(),
                        }
                    )
                )
            except Exception:
                logger.exception("Unable to report single-GPU inference error")

    async def _recover_from_send_failure(self):
        if self.single_gpu_engine is not None:
            # This can run inside the engine's output callback. Waiting for the
            # engine to stop here would deadlock that callback.
            self.service_running.clear()
            self.single_gpu_engine.request_stop()
            if (
                self._background_stop_task is None
                or self._background_stop_task.done()
            ):
                self._background_stop_task = asyncio.create_task(
                    self._stop_after_send_failure(),
                    name="single-gpu-send-failure-stop",
                )
            return
        await self.disconnect_websocket()

    async def _stop_after_send_failure(self):
        try:
            await self.stop()
        except Exception:
            logger.exception(
                "event=send_failure_cleanup_failed session_id=%s",
                self.session_id,
            )
        finally:
            self.websocket = None

    def runtime_status(self):
        if self.single_gpu_engine is None:
            return {"mode": "distributed"}
        return self.single_gpu_engine.status()

    async def close(self):
        if self._background_stop_task is not None:
            await self._background_stop_task
        if self.single_gpu_engine is not None:
            await self.single_gpu_engine.close()
        if self._gpu_executor is not None:
            self._gpu_executor.shutdown(wait=True, cancel_futures=True)
            self._gpu_executor = None

    @staticmethod
    def _condition_summary(conditional_dict):
        summary = {}
        for key, value in conditional_dict.items():
            if isinstance(value, torch.Tensor):
                summary[key] = {
                    "shape": tuple(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }
            elif key not in {"text"}:
                summary[key] = value
        return summary

    async def process_control_message(self, message):
        if message["type"] == "control":
            if message["text"] == "stop decode":
                while True:
                    try:
                        self.do_decode_control_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            elif message["text"] == "do decode":
                try:
                    self.do_decode_control_queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

        elif message["type"] == "image_config":
            logger.info(
                "event=image_config_received session_id=%s", self.session_id
            )
            image_path = message.get("image_path", "")

            cond = await self.update_condition(
                image_path=image_path,
                text=self.silence_prompt,
                audio=self.silence_audio,
                silence=True,
            )
            await self.start(cond)

    async def connect_websocket(self, websocket):
        self.websocket = websocket
        self.session_id = str(uuid.uuid4())

        if self.single_gpu_engine is not None:
            await self.single_gpu_engine.start_worker()
        else:
            if self.socket_server_task is None:
                server_ready_event = asyncio.Event()
                self.socket_server_task = asyncio.create_task(
                    comm_utils.run_socket_server_async(
                        self.signal_queue,
                        service_config.server.app_socket_port,
                        server_ready_event,
                    )
                )
                await server_ready_event.wait()

            if self.ready_socket_server_task is None:
                server_ready_event = asyncio.Event()
                self.ready_socket_server_task = asyncio.create_task(
                    comm_utils.run_socket_server_async(
                        self.ready_signal_queue,
                        service_config.server.app_ready_socket_port,
                        server_ready_event,
                    )
                )
                await server_ready_event.wait()

            if self.sender_task is None:
                self.sender_task = asyncio.create_task(
                    send_cond_worker_async(
                        cond_queue=self.cond_queue,
                        ready_signal_queue=self.ready_signal_queue,
                        vae_idle_event=self.vae_idle_event,
                        profile=self.profile,
                    )
                )

            if self.vae_task is None:
                self.vae_task = asyncio.create_task(
                    self.vae_decode(
                        signal_queue=self.signal_queue,
                        vae_idle_event=self.vae_idle_event,
                    )
                )

        try:
            self.do_decode_control_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

        self.vae_idle_event.set()

        logger.info(
            "event=websocket_connected session_id=%s mode=%s",
            self.session_id,
            "single_gpu" if self.single_gpu else "distributed",
        )

    async def disconnect_websocket(self):
        disconnect_session_id = self.session_id
        if self.websocket is not None:
            try:
                await self.websocket.close()

            except Exception as e:
                logger.exception(f"Exception in disconnect_websocket: {e}")
                logger.exception(traceback.format_exc())

            try:
                self.do_decode_control_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
            await self.stop()
            self.websocket = None

        logger.info(
            "event=websocket_disconnected session_id=%s",
            disconnect_session_id,
        )

    async def send_frame(self, frame_data, frame_idx):
        if self.websocket:
            try:
                data = {
                    "type": "audio_image",
                    "audio": frame_data.get("audio_base64", ""),
                    "image": frame_data.get("image_base64", ""),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "frame_index": frame_idx,
                    "audio_finish_frame": math.ceil(
                        frame_idx
                        + frame_data.get("audio_length", 0) * service_config.video.fps
                    ),
                    "total_frames": frame_idx + 1,
                }

                await self.websocket.send_text(json.dumps(data))
                logger.info(
                    "event=frame_sent session_id=%s frame=%d has_audio=%s "
                    "image_bytes=%d",
                    self.session_id,
                    frame_idx,
                    bool(frame_data.get("audio_base64")),
                    len(frame_data.get("image_base64", "")),
                )
                return True

            except (
                websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosed,
                starlette.websockets.WebSocketDisconnect,
            ) as e:
                logger.exception(f"Websocket closed in send_frame, {e}")
                await self._recover_from_send_failure()
                return False

            except RuntimeError as e:
                logger.exception(f"Websocket closed in runtime error: {e}")
                await self._recover_from_send_failure()
                return False

            except Exception as e:
                logger.exception(f"Failed to send frame: {e}")
                await self._recover_from_send_failure()
                return False

        else:
            logger.warning(
                "event=frame_dropped reason=no_websocket session_id=%s "
                "frame=%d",
                self.session_id,
                frame_idx,
            )
            return False

    async def update_condition(
        self, image_path=None, text=None, audio=None, id=None, silence=False
    ):
        conditional_dict = {}
        if text is not None:
            text_condition = await self._run_gpu(
                "condition.text_encode",
                lambda: self.text_encoder(text_prompts=[text]),
            )
            conditional_dict.update(text_condition)
            conditional_dict["text"] = text

        if id is not None:
            conditional_dict["id"] = id

        if image_path is not None:
            with runtime_profiler.stage("condition.image_read"):
                image, sp_dim = await asyncio.to_thread(
                    read_image, image_path=image_path
                )
            original_image = rearrange(image.squeeze(0), "c h w -> h w c")
            logger.info("Image size: %s", original_image.shape)

            self.predefined_frames = [original_image]

            def encode_image():
                image_gpu = image.to(self.device).unsqueeze(2)
                return self.vae.encode_to_latent(
                    image_gpu.to(torch.bfloat16)
                ).permute(0, 2, 1, 3, 4)

            image_latent = await self._run_gpu(
                "condition.image_vae_encode", encode_image
            )
            conditional_dict["ref_latents"] = image_latent.to(torch.bfloat16)
            conditional_dict["sp_dim"] = sp_dim

        if audio is not None:
            def encode_audio():
                z = self.s2v_audio_encoder.extract_audio_feat(
                    audio_input=audio.squeeze(0), return_all_layers=True
                )
                audio_embed_bucket, num_repeat = (
                    self.s2v_audio_encoder.get_audio_embed_bucket_fps(
                        z,
                        fps=service_config.video.fps,
                        batch_frames=(audio.shape[1] // 1000),
                        m=0,
                    )
                )
                audio_embed_bucket = audio_embed_bucket.to(
                    self.device, dtype=torch.bfloat16
                ).unsqueeze(0)
                if audio_embed_bucket.ndim == 3:
                    audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
                elif audio_embed_bucket.ndim == 4:
                    audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
                return audio_embed_bucket, num_repeat

            audio_embed_bucket, num_repeat = await self._run_gpu(
                "condition.audio_encode",
                encode_audio,
                {"silence": silence},
            )

            conditional_dict["audio_input"] = audio_embed_bucket
            conditional_dict["num_repeat"] = num_repeat
            conditional_dict["motion_frames"] = [73, 19]

        conditional_dict.update({"cond_finish_time": time.time()})
        conditional_dict["silence"] = silence
        conditional_dict["session_id"] = self.session_id

        logger.info(
            "event=condition_ready session_id=%s condition_id=%s "
            "silence=%s summary=%s",
            self.session_id,
            id or "",
            silence,
            self._condition_summary(conditional_dict),
        )

        return conditional_dict

    async def start(self, conditional_dict):
        try:
            if not self.predefined_frames:
                raise RuntimeError(
                    "Cannot start video generation before an image is configured"
                )
            self.frame_count = 1
            self.service_running.set()
            logger.info(
                "event=lip_sync_starting session_id=%s condition_id=%s",
                self.session_id,
                conditional_dict.get("id", ""),
            )

            if self.single_gpu:
                await self._run_gpu("vae.cache_reset", self.vae.model.clear_cache)

            await self.send_frame(
                frame_data={
                    "image_base64": encode_image_to_base64(
                        remap_image(self.predefined_frames[0])
                    ),
                    "time": time.time(),
                },
                frame_idx=0,
            )
            if not self.service_running.is_set():
                return
            await self._submit_condition("start", conditional_dict)

            await asyncio.sleep(0)
        except Exception as e:
            self.service_running.clear()
            logger.exception(
                "event=lip_sync_start_failed session_id=%s error=%s",
                self.session_id,
                e,
            )
            raise

    async def stop(self):
        if (
            not self.service_running.is_set()
            and self.single_gpu_engine is None
        ):
            self.audio_storage = {}
            logger.info(
                "event=lip_sync_stop_skipped reason=already_idle session_id=%s",
                self.session_id,
            )
            return
        self.service_running.clear()
        logger.info("event=lip_sync_stopping session_id=%s", self.session_id)
        await self._submit_condition(
            "stop", {"cond_finish_time": time.time()}
        )
        self.audio_storage = {}
        logger.info("event=lip_sync_stopped session_id=%s", self.session_id)

    async def process_audio_chunk(
        self,
        audio_base64: Optional[str],
        decoded_audio: Optional[torch.Tensor],
        request_id: str = "",
    ) -> None:
        if audio_base64 is None and decoded_audio is None:  # silence
            id = str(uuid.uuid4())
            logger.info(
                "event=audio_condition_started session_id=%s condition_id=%s "
                "request_id=%s kind=silence",
                self.session_id,
                id,
                request_id,
            )
            cond_diff = await self.update_condition(
                audio=self.silence_audio, id=id, text=self.silence_prompt, silence=True
            )
            cond_diff["request_id"] = request_id
            self.audio_storage[id] = {
                "request_id": request_id,
                "data": "",
                "time": time.time(),
                "length": self.silence_audio.shape[-1]
                / service_config.audio.sample_rate,
            }
            await self._submit_condition("update", cond_diff)

        else:
            audio_data = decoded_audio
            id = str(uuid.uuid4())
            logger.info(
                "event=audio_condition_started session_id=%s condition_id=%s "
                "request_id=%s kind=speech samples=%d",
                self.session_id,
                id,
                request_id,
                audio_data.shape[-1],
            )
            self.audio_storage[id] = {
                "request_id": request_id,
                "data": audio_base64,
                "time": time.time(),
                "length": audio_data.shape[-1] / service_config.audio.sample_rate,
            }

            audio_data_16k = audio_data
            target_length = (
                math.ceil(
                    (
                        audio_data_16k.shape[1]
                        - (1000 * service_config.lip_sync.audio_padding_rem)
                    )
                    / (1000 * service_config.lip_sync.audio_padding_div)
                )
                * 1000
                * service_config.lip_sync.audio_padding_div
                + 1000 * service_config.lip_sync.audio_padding_rem
            )
            pad_length = target_length - audio_data_16k.shape[1]

            audio_data_16k = torch.cat(
                [
                    audio_data_16k,
                    torch.zeros(
                        size=(1, pad_length),
                        dtype=audio_data_16k.dtype,
                        device=audio_data_16k.device,
                    ),
                ],
                dim=1,
            )

            cond_diff = await self.update_condition(
                audio=audio_data_16k, id=id, text=self.speaking_prompt, silence=False
            )
            cond_diff["request_id"] = request_id

            await self._submit_condition("update", cond_diff)

    async def vae_decode(
        self, signal_queue: asyncio.Queue, vae_idle_event: asyncio.Event
    ):
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        while True:
            try:
                logger.info(
                    "Rank %d VAE: waiting do decode control event" % mpu.get_rank()
                )

                if self.service_running.is_set():
                    await self.do_decode_control_queue.get()
                    logger.info(
                        "Rank %d VAE: do decode control event passed, waiting for signal_queue"
                        % mpu.get_rank()
                    )
                else:
                    logger.info(
                        "Rank %d VAE: service not running, skipping do decode control event"
                        % mpu.get_rank()
                    )

                data_dict = await signal_queue.get()

                vae_idle_event.clear()
                logger.info("Rank %d Setting vae lock" % mpu.get_rank())

                if data_dict["signal"] == "output":
                    id = data_dict["id"]
                    shape = data_dict["shape"]

                    logger.info(
                        "Rank %d: trying to receive output block" % mpu.get_rank()
                    )
                    output_block = torch.empty(
                        shape, dtype=torch.bfloat16, device=torch.cuda.current_device()
                    )

                    self.ready_socket = comm_utils.socket_send(
                        data={"signal": "ready"},
                        port=service_config.server.diffusion_ready_socket_port,
                        client_socket=self.ready_socket,
                    )
                    logger.info(
                        "Rank %d: ready to receive output block" % mpu.get_rank()
                    )

                    torch.distributed.recv(output_block, src=1)
                    logger.info(
                        f"Rank {mpu.get_rank()}: output block received, {output_block.shape}"
                    )

                    output_block_dict = {
                        "output_block": output_block,
                        "id": id,
                        "shape": shape,
                    }

                else:
                    logger.info(
                        "Rank %d Releasing vae lock in incorrect signal"
                        % mpu.get_rank()
                    )
                    vae_idle_event.set()
                    continue

                if not self.service_running.is_set():
                    logger.info(
                        "Rank %d flushing output block in vae_decode" % mpu.get_rank()
                    )
                    vae_idle_event.set()
                    continue

                if self.profile:
                    vae_start = torch.cuda.Event(enable_timing=True)
                    vae_end = torch.cuda.Event(enable_timing=True)
                    vae_start.record()

                output_block = output_block_dict["output_block"]
                id = output_block_dict.get("id", None)

                frame_block = self.vae.decode_to_pixel(
                    output_block, use_cache=True
                )  # btchw
                frame_block = rearrange(frame_block, "b t c h w -> b t h w c").squeeze(
                    0
                )
                await asyncio.sleep(0)

                logger.info(
                    f"Rank {mpu.get_rank()}: vae decoded frame_block: {frame_block.shape}"
                )
                frames = remap_image(frame_block)
                await asyncio.sleep(0)

                if self.profile:
                    vae_end.record()
                    torch.cuda.synchronize()
                    vae_time = vae_start.elapsed_time(vae_end)
                    logger.info(f"  - VAE decode time: {vae_time:.2f} ms")
                    base64_start = time.time()

                frame_dicts = [
                    {"image_base64": await encode_image_async(frame)}
                    for frame in frames
                ]
                if self.profile:
                    base64_time = (time.time() - base64_start) * 1000
                    logger.info(f"  - Base64 encode time: {base64_time:.3f} ms")
                await asyncio.sleep(0)

                if id in self.audio_storage.keys():
                    audio_data = self.audio_storage.get(id, dict())
                    audio_base64 = audio_data.get("data", "")
                    audio_length = audio_data.get("length", 5)
                    frame_dicts[0].update(
                        {"audio_base64": audio_base64, "audio_length": audio_length}
                    )
                    self.audio_storage.pop(id)

                for f in frame_dicts:
                    # send
                    await self.send_frame(frame_data=f, frame_idx=self.frame_count)
                    self.frame_count += 1

                logger.info("Rank %d Releasing vae lock" % mpu.get_rank())
                vae_idle_event.set()

            except Exception as e:
                logger.exception(
                    f"Rank {mpu.get_rank()}, Releasing vae lock in exception {e}"
                )
                vae_idle_event.set()
