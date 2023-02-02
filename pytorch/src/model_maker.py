from logging import getLogger

import torch
from ml_model.conv2d_transformer import ConvTransformerSrDaNet

logger = getLogger()


def make_model(config: dict) -> torch.nn.Module:

    if config["model"]["model_name"] == "ConvTransformerSrDaNet":
        logger.info("ConvTransformerSrDaNet is created")

        if "input_sampling_interval" in config["data"]:
            _interval = config["data"]["input_sampling_interval"]
        elif "lr_time_interval" in config["data"]:
            _interval = config["data"]["lr_time_interval"]
        else:
            _interval = config["data"]["lr_input_sampling_interval"]

        return ConvTransformerSrDaNet(
            input_sampling_interval=_interval, **config["model"]
        )
    else:
        raise NotImplementedError(f"{config['model']['model_name']} is not supported")