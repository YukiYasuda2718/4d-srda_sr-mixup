import random
import sys
from logging import getLogger

import numpy as np
import torch
import torch.distributed as dist
from src.utils import AverageMeter
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Sampler

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


logger = getLogger()


def train(
    *,
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.functional,
    optimizer: Optimizer,
    device: str,
    epoch: int,
    hide_progress_bar: bool = True,
) -> float:

    train_loss = AverageMeter()
    random.seed(epoch)
    np.random.seed(epoch)
    model.train()

    with tqdm(total=len(dataloader), disable=hide_progress_bar) as t:
        for Xs, obs, ys in dataloader:
            Xs, obs, ys = Xs.to(device), obs.to(device), ys.to(device)

            preds = model(Xs, obs)
            loss = loss_fn(preds, ys)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), n=len(Xs))
            t.update(1)

    logger.info(f"Train error: avg loss = {train_loss.avg:.8f}")

    return train_loss.avg


def test(
    *,
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.functional,
    device: str,
    epoch: int,
    hide_progress_bar: bool = True,
) -> float:

    val_loss = AverageMeter()
    random.seed(epoch)
    np.random.seed(epoch)
    model.eval()

    with tqdm(total=len(dataloader), disable=hide_progress_bar) as t:
        with torch.no_grad():
            for Xs, obs, ys in dataloader:
                Xs, obs, ys = Xs.to(device), obs.to(device), ys.to(device)

                preds = model(Xs, obs)
                val_loss.update(loss_fn(preds, ys).item(), n=len(Xs))
                t.update(1)

    logger.info(f"Valid error: avg loss = {val_loss.avg:.8f}")

    return val_loss.avg


def train_ddp(
    *,
    dataloader: DataLoader,
    sampler: Sampler,
    model: nn.Module,
    loss_fn: nn.functional,
    optimizer: Optimizer,
    epoch: int,
    rank: int,
    world_size: int,
) -> float:

    mean_loss, cnt = 0.0, 0

    random.seed(epoch)
    np.random.seed(epoch)
    sampler.set_epoch(epoch)
    model.train()

    for Xs, obs, ys in dataloader:
        Xs, obs, ys = Xs.to(rank), obs.to(rank), ys.to(rank)

        preds = model(Xs, obs)
        loss = loss_fn(preds, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss * Xs.shape[0]
        cnt += Xs.shape[0]
    mean_loss /= cnt

    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)

    return mean_loss.item() / world_size


def test_ddp(
    *,
    dataloader: DataLoader,
    sampler: Sampler,
    model: nn.Module,
    loss_fn: nn.functional,
    epoch: int,
    rank: int,
    world_size: int,
) -> float:

    mean_loss, cnt = 0.0, 0

    random.seed(epoch)
    np.random.seed(epoch)
    sampler.set_epoch(epoch)
    model.eval()

    with torch.no_grad():
        for Xs, obs, ys in dataloader:
            Xs, obs, ys = Xs.to(rank), obs.to(rank), ys.to(rank)

            preds = model(Xs, obs)
            loss = loss_fn(preds, ys)

            mean_loss += loss * Xs.shape[0]
            cnt += Xs.shape[0]
    mean_loss /= cnt

    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)

    return mean_loss.item() / world_size