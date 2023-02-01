import functools
import glob
import os
import typing
from logging import getLogger

from sklearn.model_selection import train_test_split
from src.dataset import DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling
from src.utils import get_torch_generator, seed_worker
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

logger = getLogger()


def split_file_paths(
    paths: typing.List[str], train_valid_test_ratios: typing.List[float]
) -> tuple:

    assert len(train_valid_test_ratios) == 3  # train, valid, test, three ratios

    test_size = train_valid_test_ratios[-1]
    _paths, test_paths = train_test_split(paths, test_size=test_size, shuffle=False)

    valid_size = train_valid_test_ratios[1] / (
        train_valid_test_ratios[0] + train_valid_test_ratios[1]
    )
    train_paths, valid_paths = train_test_split(
        _paths, test_size=valid_size, shuffle=False
    )

    assert set(train_paths).isdisjoint(set(valid_paths))
    assert set(train_paths).isdisjoint(set(test_paths))
    assert set(valid_paths).isdisjoint(set(test_paths))

    return train_paths, valid_paths, test_paths


def _make_dataloaders_vorticity_making_observation_inside_time_series_splitted_with_mixup(
    *,
    dict_dir_paths: dict,
    lr_kind_names: typing.List[str],
    lr_time_interval: int,
    obs_time_interval: int,
    obs_grid_interval: int,
    obs_noise_std: float,
    use_observation: bool,
    vorticity_bias: float,
    vorticity_scale: float,
    use_mixup: bool,
    use_mixup_init_time: bool,
    beta_dist_alpha: float,
    beta_dist_beta: float,
    batch_size: int,
    use_lr_forecast: bool = True,
    num_workers: int = 2,
    seed: int = 42,
    world_size: int = None,
    rank: int = None,
    train_valid_test_kinds: typing.List[str] = ["train", "valid", "test"],
    **kwargs,
):
    if world_size is not None:
        assert batch_size % world_size == 0

    logger.info(
        f"Batch size = {batch_size}, world_size = {world_size}, rank = {rank}\n"
    )

    dict_dataloaders, dict_samplers = {}, {}

    for kind in train_valid_test_kinds:

        dataset = DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling(
            data_dirs=dict_dir_paths[kind],
            lr_kind_names=lr_kind_names,
            lr_time_interval=lr_time_interval,
            obs_time_interval=obs_time_interval,
            obs_grid_interval=obs_grid_interval,
            obs_noise_std=obs_noise_std,
            use_observation=use_observation,
            vorticity_bias=vorticity_bias,
            vorticity_scale=vorticity_scale,
            use_ground_truth_clamping=False if kind == "test" else True,
            seed=seed,
            use_mixup=use_mixup,
            use_mixup_init_time=use_mixup_init_time,
            use_lr_forecast=use_lr_forecast,
            beta_dist_alpha=beta_dist_alpha,
            beta_dist_beta=beta_dist_beta,
        )

        if world_size is None or rank is None:
            dict_dataloaders[kind] = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=True if kind == "train" else False,
                shuffle=True if kind == "train" else False,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
            )
            logger.info(
                f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
            )

        else:
            dict_samplers[kind] = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                shuffle=True if kind == "train" else False,
                drop_last=True if kind == "train" else False,
            )

            dict_dataloaders[kind] = DataLoader(
                dataset,
                sampler=dict_samplers[kind],
                batch_size=batch_size // world_size,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
                drop_last=True if kind == "train" else False,
            )

            if rank == 0:
                logger.info(
                    f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
                )
    return dict_dataloaders, dict_samplers


def make_dataloaders_vorticity_making_observation_inside_time_series_splitted(
    root_dir: str,
    config: dict,
    *,
    train_valid_test_kinds: typing.List[str] = ["train", "valid", "test"],
    world_size: int = None,
    rank: int = None,
):
    cfd_dir_path = f"{root_dir}/data/pytorch/CFD/{config['data']['data_dir_name']}"
    logger.info(f"CFD dir path = {cfd_dir_path}")

    data_dirs = sorted([p for p in glob.glob(f"{cfd_dir_path}/*") if os.path.isdir(p)])

    train_dirs, valid_dirs, test_dirs = split_file_paths(
        data_dirs, config["data"]["train_valid_test_ratios"]
    )

    dict_data_dirs = {"train": train_dirs, "valid": valid_dirs, "test": test_dirs}

    logger.info("Dataloader with mixup is used.")

    return _make_dataloaders_vorticity_making_observation_inside_time_series_splitted_with_mixup(
        dict_dir_paths=dict_data_dirs,
        world_size=world_size,
        rank=rank,
        train_valid_test_kinds=train_valid_test_kinds,
        **config["data"],
    )


def _set_hr_file_paths(data_dirs: typing.List[str]):
    lst_hr_file_paths = [
        glob.glob(f"{dir_path}/*_hr_omega.npy") for dir_path in data_dirs
    ]
    hr_file_paths = functools.reduce(lambda l1, l2: l1 + l2, lst_hr_file_paths, [])

    extracted_paths = []
    for path in hr_file_paths:
        if "start12" in path:
            continue
        if "start08" in path:
            continue
        if "start04" in path:
            continue
        if "start00" in path:
            continue
        extracted_paths.append(path)

    return extracted_paths


def get_hr_file_paths(
    root_dir: str, cfd_dir_name: str, train_valid_test_ratios: typing.List[str]
):
    cfd_dir_path = f"{root_dir}/data/pytorch/CFD/{cfd_dir_name}"
    logger.info(f"CFD dir path = {cfd_dir_path}")

    data_dirs = sorted([p for p in glob.glob(f"{cfd_dir_path}/*") if os.path.isdir(p)])

    train_dirs, valid_dirs, test_dirs = split_file_paths(
        data_dirs, train_valid_test_ratios
    )

    return {
        "train": _set_hr_file_paths(train_dirs),
        "valid": _set_hr_file_paths(valid_dirs),
        "test": _set_hr_file_paths(test_dirs),
    }