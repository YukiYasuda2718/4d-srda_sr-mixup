import os
import pickle
import random
import typing
from logging import getLogger

import numpy as np
import torch

logger = getLogger()


def read_pickle(file_path: str) -> typing.Any:
    with open(str(file_path), "rb") as p:
        return pickle.load(p)


def write_pickle(data: typing.Any, file_path: str) -> None:
    with open(str(file_path), "wb") as p:
        pickle.dump(data, p)


class AverageMeter(object):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seeds(seed: int = 42, use_deterministic: bool = False) -> None:
    """
    Do not forget to run `torch.use_deterministic_algorithms(True)`
    just after importing torch in your main program.

    # How to reproduce the same results using pytorch.
    # https://pytorch.org/docs/stable/notes/randomness.html
    """
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        if use_deterministic:
            torch.use_deterministic_algorithms(True)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as e:
        logger.error(e)


def seed_worker(worker_id: int):
    """
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def get_torch_generator(seed: int = 42) -> torch.Generator:
    """
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def count_model_params(model: torch.nn.Module) -> int:
    num_params, total_num_params = 0, 0
    for p in model.parameters():
        total_num_params += p.numel()
        if p.requires_grad:
            num_params += p.numel()

    assert num_params == total_num_params

    return num_params