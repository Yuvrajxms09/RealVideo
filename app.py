import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

import torch

from config.config import config as service_config
from self_forcing.wan.modules import inference_utils

inference_utils.COMPILE = service_config.lip_sync.compile
inference_utils.NO_REFRESH_INFERENCE = service_config.lip_sync.no_refresh_inference

from core.app_interface import main as interface_main
from core.distributed import launch_distributed_job
from core.dit_service import main as dit_main
from core.profiler import cuda_memory_snapshot, runtime_profiler
from self_forcing.utils import parallel_state as mpu

log_handlers = [logging.StreamHandler()]
log_file = os.getenv("REALVIDEO_LOG_FILE")
if log_file:
    log_path = Path(log_file).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handlers.append(
        RotatingFileHandler(
            log_path,
            maxBytes=100 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
    )
logging.basicConfig(
    level=logging.DEBUG,
    format=(
        "%(asctime)s level=%(levelname)s process=%(process)d "
        "logger=%(name)s %(message)s"
    ),
    handlers=log_handlers,
)
logging.root.setLevel(service_config.log_level)

if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
    os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]


def main():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    sp_size = max(0, world_size - 1)
    single_gpu = world_size == 1
    logger = logging.getLogger(__name__)

    # Initialize distributed inference
    with runtime_profiler.stage("service.distributed_initialize"):
        launch_distributed_job()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    torch.set_grad_enabled(False)

    logger.info(
        "event=service_starting rank=%d mode=%s sp_size=%d "
        "denoising_steps=%s compile=%s profile=%s cuda_memory=%s",
        local_rank,
        "single-GPU" if single_gpu else "distributed",
        sp_size,
        list(service_config.lip_sync.dit_config.denoising_step_list),
        service_config.lip_sync.compile,
        runtime_profiler.enabled,
        cuda_memory_snapshot(local_rank),
    )

    try:
        if local_rank == 0:
            interface_main(single_gpu=single_gpu)
        else:
            dit_main()
        torch.distributed.barrier()
    except Exception:
        logger.exception(
            "event=service_failed rank=%d mode=%s cuda_memory=%s",
            local_rank,
            "single_gpu" if single_gpu else "distributed",
            cuda_memory_snapshot(local_rank),
        )
        raise
    finally:
        mpu.destroy_parallel_groups()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("event=service_stopped rank=%d", local_rank)


if __name__ == "__main__":
    main()
