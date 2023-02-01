from logging import getLogger

from torch import nn

logger = getLogger()


def make_loss(config: dict) -> nn.Module:

    if config["train"]["loss"]["name"] == "L1":
        logger.info("L1 loss is created.")
        return nn.L1Loss(reduction="mean")

    elif config["train"]["loss"]["name"] == "MSE":
        logger.info("MSE loss is created.")
        return nn.MSELoss(reduction="mean")

    else:
        raise NotImplementedError(
            f'{config["train"]["loss"]["name"]} is not supported.'
        )
