# RealVideo

RealVideo is a WebSocket-based video calling system that supports text input. It leverages **GLM-4.5-AirX** and 
**GLM-TTS** models to generate audio responses and utilizes autoregressive diffusion to generate corresponding 
video frames. The system features a modular design with full functionality and a clean code structure.
Visit [blog](https://z.ai/blog/realvideo) here!

## Example Video


<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/4353a47f-32db-4f07-af68-c7cf4eb9b7ec" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/13a674d7-9d2b-4979-be00-3ba37664252d" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/e8e02325-5e63-4bfe-8ffc-c319cea5fe21" width="100%" controls autoplay loop></video>
      </td>
  </tr>
</table>

## Features

- **Text Input**: Supports text message input.
- **AI Voice Response**: Integrates GLM-4.5-AirX and GLM-TTS models to generate voice responses.
- **Lip Sync**: Generates real-time conversational video based on any input image and audio.
- **Real-time Communication**: WebSocket-based real-time bidirectional communication.

## Download

| Model                        | Download Links                                                                                                                                                       |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|    RealVideo          | [🤗 Hugging Face](https://huggingface.co/zai-org/RealVideo)<br>[🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/RealVideo)                           |

## Quick Start

### 1. Requirements

- Python 3.10 - 3.12
- pip3
- Modern browser (supporting WebSocket and Web Audio API)

### 2. Install Dependencies

```bash
pip3 install -r requirements.txt
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir-use-symlinks False --local-dir wan_models/Wan2.2-S2V-14B
```

### 3. Configure API Key

Before using, please set the ZAI API key:

```bash
export ZAI_API_KEY="your_actual_api_key_here"
```

and change `config/config.py` line:

```python
PATH_TO_YOUR_MODEL = "zai-org/RealVideo/model.pt"  # Replace with your model path
```

### 4. Start the Service

Set the checkpoint path and choose the visible GPUs. With one visible GPU, the
service automatically uses the in-process single-GPU engine:

```bash
export REALVIDEO_CHECKPOINT_PATH=/absolute/path/to/model.pt
CUDA_VISIBLE_DEVICES=0 bash ./scripts/run_app.sh
```

The single-GPU engine serializes DiT, VAE, text, and audio encoder work on one
CUDA worker. A 96 GB GPU is recommended for the 14B checkpoint. Because DiT and
VAE cannot overlap on one device, its throughput is not directly comparable to
the `sp_size=1` numbers below, which use a separate GPU for the VAE.

For distributed execution:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/run_app.sh
```

One GPU will be used for the VAE service, while the remaining GPUs will be automatically allocated for parallel
computation of the DiT service.

The table below shows reference times (in ms) for DiT to generate one block. If the time is within **500ms**, smooth
real-time generation can be achieved. Numbers in parentheses indicate the time taken with compilation enabled.

| DiT sp size / Denoising steps | 2                         | 4                     |
|-------------------------------|---------------------------|-----------------------|
| 1                             | 563.84 ms (**442.61 ms**) | 943.13 ms (723.06 ms) |
| 2                             | **384.86 ms**             | 655.92 ms (527.11 ms) |
| 4                             | **306.39 ms**             | 513.72 ms (**480.68 ms**) |

### 5. Profiling

Stage profiling writes JSON Lines metrics to `profiles/metrics-rank-0.jsonl`
and exposes rolling summaries at `/api/status`.

```bash
REALVIDEO_PROFILE=1 \
CUDA_VISIBLE_DEVICES=0 \
bash ./scripts/run_app.sh
```

The default profiler is low overhead. Use synchronized CUDA timings only for a
short diagnostic run because synchronization changes pipeline throughput:

```bash
REALVIDEO_PROFILE=1 \
REALVIDEO_PROFILE_CUDA_SYNC=1 \
CUDA_VISIBLE_DEVICES=0 \
bash ./scripts/run_app.sh
```

For kernel/operator analysis, capture one warmed-up generated block and open
the resulting Chrome trace in Perfetto or `chrome://tracing`:

```bash
REALVIDEO_PROFILE=1 \
REALVIDEO_TORCH_PROFILE_BLOCK=3 \
CUDA_VISIBLE_DEVICES=0 \
bash ./scripts/run_app.sh
```

Traces are written to `profiles/torch-rank-0-block-3.json`. Change
`REALVIDEO_PROFILE_DIR` to select another output directory.

Structured lifecycle logs are written to stdout. To retain rotating log files:

```bash
REALVIDEO_LOG_FILE=logs/realvideo.log \
LOG_LEVEL=INFO \
CUDA_VISIBLE_DEVICES=0 \
bash ./scripts/run_app.sh
```

The logs carry `session_id`, `request_id`, condition ID, command sequence,
queue-wait latency, block dispatch latency, decode backpressure, output
latency, and CUDA memory on failures. `REALVIDEO_FLOW_LOG_EVERY=N` controls
how often completed blocks are logged. Use `LOG_LEVEL=WARNING` and
`REALVIDEO_PROFILE=0` for final uninstrumented throughput measurements.

### 6. Access the Application

- **Main Page**: http://localhost:8003

The browser client uses the page's origin for HTTP and WebSocket traffic. It
therefore works unchanged when accessed directly over localhost or through an
HTTPS reverse proxy such as Cloudflare Tunnel.

#### Google Colab

Use [`RealVideo_Colab_Profiling.ipynb`](RealVideo_Colab_Profiling.ipynb) for a
single-GPU setup with model downloads, direct image-and-audio inference, three
repeatable profiling trials, result aggregation, and optional Cloudflare UI
access. Direct-audio profiling does not require a Z.ai API key.

Processes running in the same Colab runtime can access the service directly at
`http://127.0.0.1:8003`. Automated profiling clients should use this local
address so external network latency does not affect their measurements.

To access the UI from your browser, expose the same service with a Cloudflare
Quick Tunnel after `cloudflared` is installed in the Colab runtime:

```bash
cloudflared tunnel --url http://127.0.0.1:8003
```

Open the generated `https://*.trycloudflare.com` URL. The UI automatically uses
`wss://` through the tunnel and `ws://` when opened over localhost.

RealVideo intentionally allows one active WebSocket session at a time. Stop or
disconnect the automated profiling client before clicking **Connect** in the
browser UI, and disconnect the browser UI before starting another profiling
run. Cloudflare Quick Tunnel URLs are public development endpoints; do not
share the URL or use it as a production deployment without authentication.

## Usage Instructions

1. **Set Avatar and Voice**: Use the file upload button to upload an image to set the avatar, or upload a speech audio
   file longer than 3 seconds for voice cloning.
2. **Connect WebSocket**: Click the "Connect" button to establish the WebSocket connection.
3. **Text Input**: Enter a message in the text box and press Enter or click "Send" to send the message.
4. **Real-time Response**: The real-time generated video response will be displayed on the left.

## Technical Highlights

- **Model Integration**: Allows for convenient and quick voice cloning, taking text input to generate audio output.
- **Modular Design**: Clear code structure, easy to maintain and extend.
- **Real-time Performance**: Optimized audio processing and real-time video generation algorithms.

## Acknowledgements

This project utilizes the following open-source libraries:

- [self forcing](https://github.com/guandeh17/Self-Forcing)
- [Wan2.2-S2V](https://github.com/Wan-Video/Wan2.2)
