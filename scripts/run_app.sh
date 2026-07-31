#! /bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

function get_gpu_count() {
    if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --list-gpus | wc -l
        else
            echo "0"
        fi
    elif [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        echo "0"
    else
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l
    fi
}

if [ -n "${MLP_SOCKET_IFNAME:-}" ]; then
    export GLOO_SOCKET_IFNAME="$MLP_SOCKET_IFNAME"
    export NCCL_SOCKET_IFNAME="$MLP_SOCKET_IFNAME"
fi
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=./.inductor_cache
export TORCHDYNAMO_VERBOSE="${TORCHDYNAMO_VERBOSE:-0}"
GPU_COUNT=$(get_gpu_count)

export NCCL_PXN_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=4
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_TIMEOUT=25
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=VERSION
export NCCL_IB_TC=106

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

if ! [[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "No CUDA GPU is visible. Set CUDA_VISIBLE_DEVICES and retry." >&2
    exit 1
fi

if [ "$GPU_COUNT" -eq 1 ]; then
    echo "Starting RealVideo in single-GPU mode..."
else
    echo "Starting RealVideo in distributed mode on $GPU_COUNT GPUs..."
fi

exec torchrun \
    --standalone \
    --nproc_per_node="$GPU_COUNT" \
    app.py
