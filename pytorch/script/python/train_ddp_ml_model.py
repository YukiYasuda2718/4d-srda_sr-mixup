import argparse
import copy
import datetime
import os
import pathlib
import sys
import time
import traceback
from logging import INFO, FileHandler, StreamHandler, getLogger

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from src.dataloader import (
    make_dataloaders_vorticity_making_observation_inside_time_series_splitted,
)
from src.loss_maker import make_loss
from src.model_maker import make_model
from src.optim_helper import test_ddp, train_ddp
from src.utils import set_seeds
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic
set_seeds(42, use_deterministic=False)
# In PyTorch 1.11.0, `upsample_trilinear3d_backward_out_cuda` does not have a deterministic implementation.
# Thus, `torch.use_deterministic_algorithms` cannot be used.

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--world_size", type=int, required=True)

ROOT_DIR = str((pathlib.Path(os.environ["PYTHONPATH"]) / "..").resolve())


def check_hr_data_paths_in_dataloader(dataloader):
    for path in dataloader.dataset.hr_paths:
        logger.info(os.path.basename(path))


def setup(rank: int, world_size: int, backend: str = "nccl"):
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_and_validate(
    rank: int,
    world_size: int,
    config: dict,
    weight_path: str,
    learning_history_path: str,
):

    setup(rank, world_size)
    set_seeds(config["train"]["seed"])

    if rank == 0:
        logger.info("\n###############################")
        logger.info("Make dataloaders and samplers")
        logger.info("################################\n")

    if config["data"]["data_dir_name"] == "jet02":
        (
            dataloaders,
            samplers,
        ) = make_dataloaders_vorticity_making_observation_inside_time_series_splitted(
            ROOT_DIR, config, world_size=world_size, rank=rank
        )
    else:
        raise Exception(
            f'datadir name ({config["data"]["data_dir_name"]}) is not supported.'
        )

    if rank == 0:
        # logger.info("\nCheck train_loader")
        # check_hr_data_paths_in_dataloader(dataloaders["train"])
        # logger.info("\nCheck valid_loader")
        # check_hr_data_paths_in_dataloader(dataloaders["valid"])

        logger.info("\n###############################")
        logger.info("Make model and optimizer")
        logger.info("###############################\n")

    model = make_model(config)
    model = DDP(model.to(rank), device_ids=[rank])
    loss_fn = make_loss(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])

    if rank == 0:
        logger.info("\n###############################")
        logger.info("Train model")
        logger.info("###############################\n")

    all_scores = []
    best_epoch = 0
    best_loss = np.inf
    best_weights = copy.deepcopy(model.module.state_dict())

    for epoch in range(config["train"]["num_epochs"]):
        _time = time.time()
        dist.barrier()
        loss = train_ddp(
            dataloader=dataloaders["train"],
            sampler=samplers["train"],
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
        )
        dist.barrier()
        val_loss = test_ddp(
            dataloader=dataloaders["valid"],
            sampler=samplers["valid"],
            model=model,
            loss_fn=loss_fn,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
        )
        dist.barrier()

        all_scores.append({"loss": loss, "val_loss": val_loss})

        if rank == 0:
            logger.info(
                f"Epoch: {epoch + 1}, loss = {loss:.8f}, val_loss = {val_loss:.8f}"
            )

            if val_loss <= best_loss:
                best_epoch = epoch + 1
                best_loss = val_loss
                best_weights = copy.deepcopy(model.module.state_dict())

                torch.save(best_weights, weight_path)
                logger.info("Best loss is updated.")

            if epoch % 10 == 0:
                pd.DataFrame(all_scores).to_csv(learning_history_path, index=False)

            logger.info(f"Elapsed time = {time.time() - _time} sec")
            logger.info("-----")

    if rank == 0:
        torch.save(best_weights, weight_path)
        pd.DataFrame(all_scores).to_csv(learning_history_path, index=False)
        logger.info(f"Best epoch: {best_epoch}, best_loss: {best_loss:.8f}")

    cleanup()


if __name__ == "__main__":
    try:

        os.environ["MASTER_ADDR"] = "localhost"

        # Port is arbitrary, but set random value to avoid collision
        np.random.seed(datetime.datetime.now().microsecond)
        port = str(np.random.randint(12000, 65535))
        os.environ["MASTER_PORT"] = port

        world_size = parser.parse_args().world_size
        config_path = parser.parse_args().config_path

        with open(config_path) as file:
            config = yaml.safe_load(file)

        experiment_name = config_path.split("/")[-2]
        config_name = os.path.basename(config_path).split(".")[0]

        _dir = f"{ROOT_DIR}/data/pytorch/DL_results/{experiment_name}/{config_name}"
        os.makedirs(_dir, exist_ok=False)

        weight_path = f"{_dir}/weights.pth"
        learning_history_path = f"{_dir}/learning_history.csv"
        logger.addHandler(FileHandler(f"{_dir}/log.txt"))

        logger.info("\n*********************************************************")
        logger.info(f"Start DDP: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

        logger.info(f"experiment name = {experiment_name}")
        logger.info(f"config name = {config_name}")
        logger.info(f"config path = {config_path}")

        if not torch.cuda.is_available():
            logger.error("No GPU.")
            raise Exception("No GPU.")

        logger.info(f"Num available GPUs = {torch.cuda.device_count()}")
        logger.info(f"Names of GPUs = {torch.cuda.get_device_name()}")
        logger.info(f"Device capability = {torch.cuda.get_device_capability()}")
        logger.info(f"World size = {world_size}")

        start_time = time.time()

        mp.spawn(
            train_and_validate,
            args=(world_size, config, weight_path, learning_history_path),
            nprocs=world_size,
            join=True,
        )

        end_time = time.time()

        logger.info(f"Total elapsed time = {(end_time - start_time) / 60.} min")

        logger.info("\n*********************************************************")
        logger.info(f"End DDP: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

    except Exception as e:
        logger.info("\n*********************************************************")
        logger.info("Error")
        logger.info("*********************************************************\n")
        logger.error(e)
        logger.error(traceback.format_exc())