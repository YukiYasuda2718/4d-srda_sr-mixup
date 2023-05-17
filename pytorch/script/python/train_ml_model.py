import argparse
import copy
import datetime
import gc
import os
import pathlib
import random
import sys
import time
import traceback
from logging import INFO, FileHandler, StreamHandler, getLogger

import numpy as np
import pandas as pd
import torch
import yaml
from src.dataloader import (
    make_dataloaders_vorticity_making_observation_inside_time_series_splitted,
)
from src.loss_maker import make_loss
from src.model_maker import make_model
from src.optim_helper import test, train
from src.utils import set_seeds

os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic
set_seeds(42, use_deterministic=False)
# In PyTorch 1.11.0, `upsample_trilinear3d_backward_out_cuda` does not have a deterministic implementation.
# Thus, `torch.use_deterministic_algorithms` cannot be used.

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)

ROOT_DIR = str((pathlib.Path(os.environ["PYTHONPATH"]) / "..").resolve())


def check_hr_data_paths_in_dataloader(dataloader):
    for path in dataloader.dataset.hr_paths:
        logger.info(os.path.basename(path))


if __name__ == "__main__":
    try:
        config_path = parser.parse_args().config_path

        with open(config_path) as file:
            config = yaml.safe_load(file)

        experiment_name = config_path.split("/")[-2]
        config_name = os.path.basename(config_path).split(".")[0]

        _dir = f"{ROOT_DIR}/data/pytorch/DL_results/{experiment_name}/{config_name}"
        os.makedirs(_dir, exist_ok=False)

        WEIGHT_PATH = f"{_dir}/weights.pth"
        LEARNING_HISTORY_PATH = f"{_dir}/learning_history.csv"
        logger.addHandler(FileHandler(f"{_dir}/log.txt"))

        logger.info("\n*********************************************************")
        logger.info(f"Start: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

        logger.info(f"experiment name = {experiment_name}")
        logger.info(f"config name = {config_name}")
        logger.info(f"config path = {config_path}")

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if DEVICE == "cuda":
            logger.info("GPU is used.")
        else:
            logger.error("No GPU. CPU is used.")
            raise Exception("No GPU. CPU is used.")

        logger.info(f"Num available GPUs = {torch.cuda.device_count()}")
        logger.info(f"Names of GPUs = {torch.cuda.get_device_name()}")
        logger.info(f"Device capability = {torch.cuda.get_device_capability()}")

        logger.info("\n*********************************************************")
        logger.info("Make dataloadars")
        logger.info("*********************************************************\n")

        if (
            config["data"]["data_dir_name"] == "jet02"
            or config["data"]["data_dir_name"] == "jet04"
        ):
            (
                dataloaders,
                _,
            ) = make_dataloaders_vorticity_making_observation_inside_time_series_splitted(
                ROOT_DIR, config
            )
        else:
            raise Exception(
                f'datadir name ({config["data"]["data_dir_name"]}) is not supported.'
            )
        train_loader = dataloaders["train"]
        valid_loader = dataloaders["valid"]

        # logger.info("\nCheck train_loader")
        # check_hr_data_paths_in_dataloader(train_loader)
        # logger.info("\nCheck valid_loader")
        # check_hr_data_paths_in_dataloader(valid_loader)

        logger.info("\n*********************************************************")
        logger.info("Make model and optimizer")
        logger.info("*********************************************************\n")

        set_seeds(config["train"]["seed"])
        model = make_model(config).to(DEVICE)
        loss_fn = make_loss(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])

        logger.info("\n*********************************************************")
        logger.info("Train model")
        logger.info("*********************************************************\n")

        start_time = time.time()
        logger.info(f"Train start: {datetime.datetime.utcnow().isoformat()} UTC")

        all_scores = []
        best_epoch = 0
        best_loss = np.inf
        best_weights = copy.deepcopy(model.state_dict())

        for epoch in range(config["train"]["num_epochs"]):
            _time = time.time()
            logger.info(f"Epoch: {epoch + 1} / {config['train']['num_epochs']}")

            random.seed(epoch)
            np.random.seed(epoch)

            loss = train(
                dataloader=train_loader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=DEVICE,
                epoch=epoch,
                hide_progress_bar=True,
            )
            val_loss = test(
                dataloader=valid_loader,
                model=model,
                loss_fn=loss_fn,
                device=DEVICE,
                epoch=epoch,
                hide_progress_bar=True,
            )

            all_scores.append({"loss": loss, "val_loss": val_loss})

            if val_loss <= best_loss:
                best_epoch = epoch + 1
                best_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())

                torch.save(best_weights, WEIGHT_PATH)
                logger.info("Best loss is updated.")

            if epoch % 10 == 0:
                pd.DataFrame(all_scores).to_csv(LEARNING_HISTORY_PATH, index=False)

            logger.info(f"Elapsed time = {time.time() - _time} sec")
            logger.info("-----")

        torch.save(best_weights, WEIGHT_PATH)
        pd.DataFrame(all_scores).to_csv(LEARNING_HISTORY_PATH, index=False)

        end_time = time.time()

        logger.info(f"Best epoch: {best_epoch}, best_loss: {best_loss:.8f}")
        logger.info(f"Train end: {datetime.datetime.utcnow().isoformat()} UTC")
        logger.info(f"Total elapsed time = {(end_time - start_time) / 60.} min")

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("\n*********************************************************")
        logger.info("Evaluate model")
        logger.info("*********************************************************\n")

        test_loader = dataloaders["test"]
        # logger.info("\nCheck valid_loader")
        # check_hr_data_paths_in_dataloader(test_loader)

        # Re-load best weights
        model = make_model(config).to(DEVICE)
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
        model.eval()

        test_loss = test(
            dataloader=test_loader,
            model=model,
            loss_fn=loss_fn,
            device=DEVICE,
            epoch=0,  # arbitrary, this is used to set seed in test method.
            hide_progress_bar=True,
        )

        logger.info(f"Test loss = {test_loss:.6f}")

        logger.info("\n*********************************************************")
        logger.info(f"End: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

    except Exception as e:
        logger.info("\n*********************************************************")
        logger.info("Error")
        logger.info("*********************************************************\n")
        logger.error(e)
        logger.error(traceback.format_exc())