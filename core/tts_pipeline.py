import asyncio
import logging
import os
import re
import time

import aiohttp
import orjson

from core.profiler import runtime_profiler

logger = logging.getLogger(__name__)


class TTSPipeline:
    def __init__(
        self,
        vae_idle_event: asyncio.Event,
        model_name_llm="glm-4.5-airx",
        model_name_tts="glm-tts",
    ):
        self.vae_idle_event = vae_idle_event

        self.model_name_llm = model_name_llm
        self.model_name_tts = model_name_tts
        self.stc_split_pattern = r"([。？！?!\n]”?|[.?!]\s?)"
        self.substc_split_pattern = "(，|, )"
        self.stc_min_length = 10
        self.stc_max_length = 50
        self.chat_history = []
        self.async_tasks_started = False
        self.proxy = os.environ.get("HTTP_PROXY", None) or os.environ.get(
            "http_proxy", None
        )
        self.llm_task = None
        self.tts_task = None
        self.api_key = os.environ.get("ZAI_API_KEY")

    def reset_status(self):
        self.chat_history = []

    @property
    def available(self):
        return bool(self.api_key)

    def start_async_tasks(
        self, text_input_queue: asyncio.Queue, output_queue: asyncio.Queue
    ):
        if not self.api_key:
            logger.warning(
                "event=llm_tts_workers_disabled reason=missing_ZAI_API_KEY"
            )
            return
        if not self.async_tasks_started:
            sentence_queue = asyncio.Queue(32)
            self.llm_task = asyncio.create_task(
                self.llm_worker_async(text_input_queue, sentence_queue)
            )
            self.tts_task = asyncio.create_task(
                self.tts_worker_async(sentence_queue, output_queue)
            )
            self.async_tasks_started = True
            logger.info("event=llm_tts_workers_started")

    async def stop_async_tasks(self):
        tasks = [
            task
            for task in (self.llm_task, self.tts_task)
            if task is not None and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.llm_task = None
        self.tts_task = None
        self.async_tasks_started = False
        logger.info("event=llm_tts_workers_stopped workers=%d", len(tasks))

    async def llm_worker_async(self, text_input_queue: asyncio.Queue, sentence_queue):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        llm_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        body_template = {
            "model": self.model_name_llm,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,
            "thinking": {"type": "disabled"},
        }

        timeout = aiohttp.ClientTimeout(
            total=None, connect=10, sock_connect=10, sock_read=30
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                request_id = ""
                try:
                    text_item = await text_input_queue.get()
                    request_id = text_item.get("request_id", "")
                    profile = text_item.get("profile") or ""
                    text_input = text_item["text"]
                    voice_id = text_item.get("voice_id", None)
                    logger.info(
                        "event=llm_request_started request_id=%s text_chars=%d "
                        "profile_chars=%d",
                        request_id,
                        len(text_input),
                        len(profile or ""),
                    )

                    body = {
                        "messages": [{"role": "system", "content": profile}]
                        + self.chat_history
                        + [{"role": "user", "content": text_input}]
                    }
                    body.update(body_template)

                    buffer = b""
                    chunk_resp = b""
                    text_buffer = ""
                    finished = False
                    text_response = ""
                    current_sentence = ""

                    start = time.time()

                    with runtime_profiler.stage("llm.request_to_headers"):
                        response_context = session.post(
                            llm_url,
                            headers=headers,
                            json=body,
                            proxy=self.proxy,
                        )
                        response = await response_context.__aenter__()
                    try:
                        response.raise_for_status()
                        logger.info(
                            "event=llm_response_started request_id=%s "
                            "headers_ms=%.2f",
                            request_id,
                            1000 * (time.time() - start),
                        )

                        while True:
                            await asyncio.sleep(0)
                            chunk_resp = await response.content.readline()

                            buffer += chunk_resp
                            if not buffer:
                                break

                            pos = buffer.find(b"\n")
                            if pos == -1 and not chunk_resp:
                                pos = len(buffer)

                            while pos > -1:
                                await asyncio.sleep(0)

                                bline = buffer[: pos + 1]
                                buffer = buffer[pos + 1 :]
                                pos = buffer.find(b"\n")

                                if not bline:
                                    break

                                bline = bline.strip()
                                if not bline or not bline.startswith(b"data:"):
                                    continue

                                if bline.startswith(b"data: [DONE]"):
                                    break

                                await asyncio.sleep(0)
                                chunk = orjson.loads(
                                    bline[6:].strip()
                                )  # remove 'data: '

                                finished = chunk["choices"][0].get("finish_reason", "")
                                if finished == "stop" or finished == "stop_sequence":
                                    break

                                if chunk["choices"][0]["delta"]["content"]:
                                    text_chunk = chunk["choices"][0]["delta"]["content"]
                                    text_buffer += text_chunk

                                    while True:
                                        await asyncio.sleep(0)
                                        m = re.search(
                                            self.stc_split_pattern, text_buffer
                                        )
                                        if m is not None:
                                            if m.end() > self.stc_max_length:
                                                pass

                                            current_sentence = text_buffer[
                                                : m.end()
                                            ].strip()
                                            text_buffer = text_buffer[m.end() :]
                                            if current_sentence:
                                                await sentence_queue.put(
                                                    {
                                                        "request_id": request_id,
                                                        "sentence": current_sentence,
                                                        "voice_id": voice_id,
                                                    }
                                                )
                                                logger.info(
                                                    "event=llm_sentence_emitted "
                                                    "request_id=%s chars=%d "
                                                    "elapsed_ms=%.2f",
                                                    request_id,
                                                    len(current_sentence),
                                                    (time.time() - start) * 1000,
                                                )
                                                text_response += current_sentence

                                        else:
                                            break

                    finally:
                        await response_context.__aexit__(None, None, None)

                    text_buffer = text_buffer.strip()
                    if text_buffer:
                        await sentence_queue.put(
                            {
                                "request_id": request_id,
                                "sentence": text_buffer,
                                "voice_id": voice_id,
                            }
                        )
                        text_response += text_buffer

                    await sentence_queue.put(
                        {"request_id": request_id, "end": True}
                    )
                    self.chat_history.append(
                        {"role": "assistant", "content": text_response}
                    )
                    logger.info(
                        "event=llm_request_completed request_id=%s "
                        "response_chars=%d elapsed_ms=%.2f",
                        request_id,
                        len(text_response),
                        (time.time() - start) * 1000,
                    )

                    await asyncio.sleep(0)
                except Exception as e:
                    logger.exception(
                        "event=llm_request_failed request_id=%s error=%s",
                        request_id,
                        e,
                    )
                    await sentence_queue.put(
                        {"request_id": request_id, "end": True, "failed": True}
                    )

    async def tts_worker_async(
        self, sentence_queue: asyncio.Queue, output_queue: asyncio.Queue
    ):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(
            total=None, connect=10, sock_connect=10, sock_read=30
        )

        async with aiohttp.ClientSession(timeout=timeout) as session:
            failed_request_ids = set()
            while True:
                request_id = ""
                try:
                    sentence_item = await sentence_queue.get()
                    request_id = sentence_item.get("request_id", "")
                    if request_id in failed_request_ids:
                        if sentence_item.get("end", False):
                            failed_request_ids.discard(request_id)
                        continue
                    if sentence_item.get("end", False):
                        logger.info(
                            "event=tts_request_completed request_id=%s "
                            "llm_failed=%s",
                            request_id,
                            sentence_item.get("failed", False),
                        )
                        await output_queue.put(
                            {"request_id": request_id, "end": True}
                        )
                        continue

                    sentence = sentence_item.get("sentence", None)
                    voice_id = sentence_item.get("voice_id", None)
                    logger.info(
                        "event=tts_sentence_started request_id=%s chars=%d",
                        request_id,
                        len(sentence or ""),
                    )

                    body = {
                        "input": sentence,
                        "stream": True,
                        "model": self.model_name_tts,
                        "voice": voice_id,
                        "response_format": "pcm",
                        "speed": 1.0,
                        "volume": 1.0,
                    }
                    tts_url = "https://open.bigmodel.cn/api/paas/v4/audio/speech"

                    buffer = b""
                    chunk_id = -1
                    finished = False
                    chunk_resp_list = []
                    chunk_resp_list_length = 0

                    start = time.time()
                    async with session.post(
                        tts_url, headers=headers, json=body, proxy=self.proxy
                    ) as response:
                        response.raise_for_status()
                        logger.info(
                            "event=tts_response_started request_id=%s "
                            "headers_ms=%.2f",
                            request_id,
                            1000 * (time.time() - start),
                        )

                        while True:
                            await asyncio.sleep(0)
                            chunk_resp = await response.content.read(1024)

                            await asyncio.sleep(0)

                            pos = chunk_resp.find(b"\n")
                            if pos > -1:
                                pos += chunk_resp_list_length

                            chunk_resp_list.append(chunk_resp)
                            chunk_resp_list_length += len(chunk_resp)
                            if chunk_resp_list_length == 0:
                                break

                            if pos == -1 and not chunk_resp:
                                pos = chunk_resp_list_length

                            while pos > -1:
                                await asyncio.sleep(0)
                                buffer = b"".join(chunk_resp_list)

                                bline = buffer[: pos + 1]
                                buffer = buffer[pos + 1 :]
                                chunk_resp_list = [buffer]
                                chunk_resp_list_length = len(buffer)
                                pos = buffer.find(b"\n")
                                chunk_id += 1

                                logger.debug(
                                    "event=tts_chunk_parsing request_id=%s "
                                    "chunk=%d",
                                    request_id,
                                    chunk_id,
                                )

                                await asyncio.sleep(0)
                                bline = bline.strip()
                                if not bline:
                                    break

                                if not bline or not bline.startswith(b"data:"):
                                    continue

                                await asyncio.sleep(0)
                                chunk = orjson.loads(bline[5:])  # remove 'data:'

                                choice = chunk["choices"][0]
                                index = choice["index"]
                                is_finished = choice.get("finish_reason", "")
                                if is_finished == "stop":
                                    finished = True
                                    break
                                audio_delta = choice["delta"]["content"]
                                sr = choice["delta"]["return_sample_rate"]

                                logger.info(
                                    "event=tts_audio_chunk request_id=%s "
                                    "chunk=%d index=%s encoded_bytes=%d "
                                    "sample_rate=%d",
                                    request_id,
                                    chunk_id,
                                    index,
                                    len(audio_delta),
                                    sr,
                                )
                                await output_queue.put(
                                    {
                                        "request_id": request_id,
                                        "audio_base64": audio_delta,
                                        "sample_rate": sr,
                                        "chunk_id": chunk_id,
                                        "time": time.time(),
                                    }
                                )

                            if finished:
                                break

                except Exception as e:
                    failed_request_ids.add(request_id)
                    logger.exception(
                        "event=tts_request_failed request_id=%s error=%s",
                        request_id,
                        e,
                    )
                    await output_queue.put(
                        {"request_id": request_id, "end": True}
                    )
