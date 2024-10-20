"""
Util functions for setting up distributed training.
Credit: https://github.com/mlfoundations/open_clip/blob/main/src/training/distributed.py
"""

import os
import torch

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_using_horovod():
    # NOTE w/ horovod run, OMPI vars should be set, but w/ SLURM PMI vars will be set
    # Differentiating between horovod and DDP use via SLURM may not be possible, so horovod arg still required...
    ompi_vars = ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]
    pmi_vars = ["PMI_RANK", "PMI_SIZE"]
    if all([var in os.environ for var in ompi_vars]) or all(
        [var in os.environ for var in pmi_vars]
    ):
        return True
    else:
        return False


def is_using_distributed():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device():
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    world_size = 1
    rank = 0  # global rank
    local_rank = 0
    dist_backend = "nccl"
    dist_url = "env://"
    distributed = False
    if is_using_distributed():
        if "SLURM_PROCID" in os.environ:
            # DDP via SLURM
            local_rank, rank, world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
                world_size=world_size,
                rank=rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            local_rank, rank, world_size = world_info_from_env()
            torch.distributed.init_process_group(
                backend=dist_backend, init_method=dist_url
            )
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        distributed = True
    else:
        # needed to run on single gpu
        torch.distributed.init_process_group(
            backend=dist_backend,
            init_method=dist_url,
            world_size=1,
            rank=0,
        )

    if torch.cuda.is_available():
        if distributed:
            device = "cuda:%d" % local_rank
        else:
            device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    device = torch.device(device)
    return device, rank
