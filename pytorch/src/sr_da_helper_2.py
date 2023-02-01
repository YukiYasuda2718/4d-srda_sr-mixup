import itertools
import os
import random
import sys
import typing
from logging import INFO, WARNING, getLogger

import numpy as np
import torch
from cfd_model.cfd.periodic_channel_domain import TorchSpectralModel2D
from cfd_model.interpolator.torch_interpolator import interpolate
from src.dataloader import (
    make_dataloaders_vorticity_making_observation_inside_time_series_splitted,
)
from src.dataset import DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling
from src.model_maker import make_model

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


logger = getLogger()


def get_testdataset(
    root_dir: str, config: dict
) -> DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling:
    logger.setLevel(WARNING)
    (
        dataloaders,
        _,
    ) = make_dataloaders_vorticity_making_observation_inside_time_series_splitted(
        root_dir=root_dir, config=config, train_valid_test_kinds=["test"]
    )
    logger.setLevel(INFO)

    return dataloaders["test"].dataset


def get_testdataloader(
    root_dir: str, config: dict
) -> torch.utils.data.dataloader.DataLoader:
    logger.setLevel(WARNING)
    (
        dataloaders,
        _,
    ) = make_dataloaders_vorticity_making_observation_inside_time_series_splitted(
        root_dir=root_dir, config=config, train_valid_test_kinds=["test"]
    )
    logger.setLevel(INFO)

    return dataloaders["test"]


def make_models(config: dict, weight_path: str, cfd_config: dict):
    logger.setLevel(WARNING)

    sr_model = make_model(config).to(cfd_config["device"])
    sr_model.load_state_dict(torch.load(weight_path, map_location=cfd_config["device"]))
    _ = sr_model.eval()

    lr_model = TorchSpectralModel2D(**cfd_config)

    srda_model = TorchSpectralModel2D(**cfd_config)

    logger.setLevel(INFO)

    return sr_model, lr_model, srda_model


def initialize_models(
    t0: float,
    hr_omega0: torch.Tensor,
    lr_forcing,
    lr_model,
    srda_model,
    *,
    n_ens,
    hr_nx,
    hr_ny,
    lr_nx,
    lr_ny,
    **kwargs,
):

    assert hr_omega0.shape == (n_ens, hr_nx, hr_ny)
    omega0 = interpolate(hr_omega0, lr_nx, lr_ny, "bicubic")

    lr_ens_forcing = torch.broadcast_to(lr_forcing, (n_ens, lr_nx, lr_ny)).clone()
    assert omega0.shape == lr_ens_forcing.shape == (n_ens, lr_nx, lr_ny)

    lr_model.initialize(t0=t0, omega0=omega0, forcing=lr_ens_forcing)
    lr_model.calc_grid_data()

    srda_model.initialize(t0=t0, omega0=omega0, forcing=lr_ens_forcing)
    srda_model.calc_grid_data()


def read_all_hr_omegas(
    test_dataset: DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling,
):
    paths = test_dataset.hr_file_paths
    return _read_all_hr_omegas(paths)


def read_all_hr_omegas_with_combining(
    hr_file_paths: typing.List[str], max_ensenbles: int = 20
):
    key_func = lambda p: os.path.basename(os.path.dirname(p))
    dict_hr_paths = {
        k: sorted(g) for k, g in itertools.groupby(hr_file_paths, key_func)
    }

    all_hr_omegas = []

    for key, paths in tqdm(dict_hr_paths.items(), total=len(dict_hr_paths)):
        hr_omegas = []

        for i, p in enumerate(paths):
            all_data = []
            for j in range(max_ensenbles):
                data = np.load(p.replace("_00.npy", f"_{j:02}.npy"))
                if i > 0:
                    # skip the first time index, except for the first dataset.
                    data = data[1:]
                all_data.append(data)
            # Stack along a new dim, i.e., batch dim
            hr_omegas.append(np.stack(all_data, axis=0))

        # Concat along time dim
        hr_omegas = np.concatenate(hr_omegas, axis=1)
        all_hr_omegas.append(hr_omegas)

    # Concat along batch dim
    return torch.from_numpy(np.concatenate(all_hr_omegas, axis=0))


def _read_all_hr_omegas(paths: list):
    key_func = lambda p: os.path.basename(os.path.dirname(p))
    dict_hr_paths = {k: sorted(g) for k, g in itertools.groupby(paths, key_func)}

    all_hr_omegas = []
    for key, paths in tqdm(dict_hr_paths.items(), total=len(dict_hr_paths)):
        hr_omegas = []
        for i, p in enumerate(paths):
            data = np.load(p)
            if i > 0:
                # skip the first time index, except for the first dataset.
                data = data[:, 1:]
            hr_omegas.append(data)

        # Concat along time dim
        hr_omegas = np.concatenate(hr_omegas, axis=1)
        all_hr_omegas.append(hr_omegas)

    # Concat along batch dim
    return torch.from_numpy(np.concatenate(all_hr_omegas, axis=0))


def read_all_lr_omegas(
    test_dataset: DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling,
    lr_kind: str = "lr_omega_no-noise",
):
    paths = test_dataset.hr_file_paths
    key_func = lambda p: os.path.basename(os.path.dirname(p))
    dict_hr_paths = {k: sorted(g) for k, g in itertools.groupby(paths, key_func)}

    all_lr_omegas = []
    for key, paths in tqdm(dict_hr_paths.items(), total=len(dict_hr_paths)):
        lr_omegas = []
        for i, p in enumerate(paths):
            p = p.replace("hr_omega", lr_kind)
            assert lr_kind in p

            data = np.load(p)
            # skip the last time index, except for the last dataset.
            if i != len(paths) - 1:
                data = data[:, :-1]
                assert data.shape[1] == 4
            lr_omegas.append(data)

        # Concat along time dim
        lr_omegas = np.concatenate(lr_omegas, axis=1)
        all_lr_omegas.append(lr_omegas)

    # Concat along batch dim
    return torch.from_numpy(np.concatenate(all_lr_omegas, axis=0))


def get_observation_with_noise(
    hr_omega: torch.Tensor,
    test_dataset: DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling,
    *,
    n_ens,
    hr_nx,
    hr_ny,
    lr_nx,
    lr_ny,
    **kwargs,
) -> torch.Tensor:

    assert hr_omega.ndim == 4  # ens, time, x, y dims
    assert hr_omega.shape[0] == n_ens
    assert hr_omega.shape[2] == hr_nx
    assert hr_omega.shape[3] == hr_ny

    is_obses = []
    for _ in range(n_ens):
        i = random.randint(0, len(test_dataset.is_obses) - 1)
        _is_obs = test_dataset.is_obses[i]
        assert _is_obs.shape == (hr_nx, hr_ny)

        # to the same dims of time, x, and y
        _is_obs = torch.broadcast_to(_is_obs, size=hr_omega.shape[1:])
        is_obses.append(_is_obs)

    is_obses = torch.stack(is_obses, dim=0)
    assert is_obses.shape == hr_omega.shape

    hr_obsrv = torch.full_like(hr_omega, np.nan)
    hr_obsrv = torch.where(
        is_obses > 0,
        hr_omega,
        hr_obsrv,
    )

    if test_dataset.obs_noise_std <= 0:
        logger.info("No observation noise.")
        return hr_obsrv

    noise = np.random.normal(
        loc=0, scale=test_dataset.obs_noise_std, size=hr_obsrv.shape
    )
    logger.info(f"Observation noise std = {test_dataset.obs_noise_std}")

    return hr_obsrv + torch.from_numpy(noise)


def append_zeros(data: torch.Tensor):
    assert data.ndim == 4  # ens, time, x, and y

    zs = torch.zeros(data.shape[0:3], dtype=data.dtype)[..., None]

    appended = torch.cat([data, zs], dim=-1)

    # Check the last index of y has zero values
    assert torch.max(torch.abs(appended[..., -1])).item() == 0.0

    return appended


def _preprocess(
    data: torch.Tensor,
    biases: torch.Tensor,
    scales: torch.Tensor,
    clamp_min: float,
    clamp_max: float,
):
    ret = (data - biases) / scales
    return torch.clamp(ret, min=clamp_min, max=clamp_max)


def preprocess(
    *,
    data: torch.Tensor,
    biases: torch.Tensor,
    scales: torch.Tensor,
    clamp_min: float,
    clamp_max: float,
    n_ens: int,
    assimilation_period: int,
    ny: int,
    nx: int,
    device: str,
    dtype: torch.dtype = torch.float32,
):
    # time, ens, x, y --> ens, time, y, x
    data = data[..., :-1].permute(1, 0, 3, 2).contiguous()
    data = data.unsqueeze(2)  # Add channel dim
    assert data.shape[0] == n_ens
    assert data.ndim == 5  # batch, time, channel, y, x

    data = _preprocess(data, biases, scales, clamp_min, clamp_max)
    data = data.to(dtype).to(device)
    assert data.shape[0] == n_ens
    assert data.ndim == 5  # batch, time, channel, y, x

    return data


def inv_preprocess(data: torch.Tensor, biases: torch.Tensor, scales: torch.Tensor):
    return data * scales + biases


def make_preprocessed_lr(
    lr_forecast: torch.Tensor,
    last_omega0: torch.Tensor,
    test_dataset: DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling,
    *,
    assimilation_period,
    n_ens,
    lr_nx,
    lr_ny,
    device,
    **kwargs,
):

    lr = torch.stack(lr_forecast[-(assimilation_period + 1) :], dim=0)
    if last_omega0 is not None:
        lr[0, ...] = last_omega0
        logger.debug("last omega is added.")
    lr = lr[:: test_dataset.lr_time_interval]

    return preprocess(
        data=lr,
        biases=test_dataset.vorticity_bias,
        scales=test_dataset.vorticity_scale,
        clamp_max=test_dataset.clamp_max,
        clamp_min=test_dataset.clamp_min,
        n_ens=n_ens,
        assimilation_period=None,  # not used
        ny=lr_ny,
        nx=lr_nx,
        device=device,
    )


def make_preprocessed_obs(
    hr_obs: torch.Tensor,
    test_dataset: DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling,
    *,
    assimilation_period,
    n_ens,
    lr_nx,
    lr_ny,
    device,
    **kwargs,
):

    obs = torch.stack(hr_obs[-(assimilation_period + 1) :], dim=0)

    obs = preprocess(
        data=obs,
        biases=test_dataset.vorticity_bias,
        scales=test_dataset.vorticity_scale,
        clamp_max=test_dataset.clamp_max,
        clamp_min=test_dataset.clamp_min,
        n_ens=n_ens,
        assimilation_period=None,  # not used
        ny=lr_ny,
        nx=lr_nx,
        device=device,
    )

    return torch.nan_to_num(obs, nan=test_dataset.missing_value)


def make_invprocessed_sr(
    preds: torch.Tensor,
    test_dataset: DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling,
    *,
    assimilation_period,
    n_ens,
    hr_nx,
    hr_ny,
    **kwargs,
):
    sr = inv_preprocess(
        preds, test_dataset.vorticity_bias, test_dataset.vorticity_scale
    )

    # Delete channel dim
    sr = sr.squeeze(2)

    # ens, time, y, x -> time, ens, x, y
    sr = append_zeros(sr.permute(1, 0, 3, 2))

    assert sr.shape == (
        assimilation_period + 1,
        n_ens,
        hr_nx,
        hr_ny,
    )

    return sr