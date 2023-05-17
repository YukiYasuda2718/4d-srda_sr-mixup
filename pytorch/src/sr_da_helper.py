import os
import sys
import time
from logging import getLogger

import numpy as np
import torch
from cfd_model.cfd.periodic_channel_domain import TorchSpectralModel2D
from cfd_model.enkf.observation_matrix import HrObservationMatrixGenerator
from cfd_model.enkf.sr_enkf import assimilate_with_existing_data
from cfd_model.interpolator.torch_interpolator import interpolate
from src.utils import read_pickle

logger = getLogger()

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


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


def extract_observation(
    *,
    obs_matrix_generator: HrObservationMatrixGenerator,
    hr_omega: torch.Tensor,
    obs_interval: int,
    hr_nx: int,
    hr_ny: int,
):
    proj_matrix = obs_matrix_generator.generate_projection_matrix()

    _ones = torch.ones(proj_matrix.shape[1], dtype=proj_matrix.dtype)
    is_obs = proj_matrix.mm(_ones[:, None]).reshape(hr_nx, hr_ny)
    is_obs = is_obs.to(hr_omega.device)

    assert is_obs.shape == hr_omega.shape

    return torch.where(is_obs > 0, hr_omega, torch.full_like(hr_omega, torch.nan))


def perform_srda_and_enkf(
    *,
    sr_model: torch.nn.Module,
    lr_model: TorchSpectralModel2D,
    srda_model: TorchSpectralModel2D,
    lr_ens_model: TorchSpectralModel2D,
    n_cycles: int,
    assimilation_period: int,
    offset: int,
    hr_omega: torch.Tensor,
    obs_matrix_generator: HrObservationMatrixGenerator,
    torch_rand_generator: torch.Generator,
    localization_matrix: torch.Tensor,
    sys_noise_generator: torch.distributions.distribution.Distribution,
    test_dataset: torch.utils.data.Dataset,
    hr_nx: int,
    hr_ny: int,
    lr_ne: int,
    lr_nx: int,
    lr_ny: int,
    lr_nt: int,
    lr_dt: float,
    obs_interval: int,
    inflation: float,
    device: str,
    hide_progressbar: bool = True,
    sr_interpolator: torch.nn.Module = None,
):
    # Add ensemble dim
    biases, scales = test_dataset.biases[None, ...], test_dataset.scales[None, ...]

    clamp_min, clamp_max = test_dataset.clamp_min, test_dataset.clamp_max
    missing_value = test_dataset.missing_value
    obs_noise_std = test_dataset.obs_noise_std
    np_rng = test_dataset.np_rng

    ts, hr_obs, lr_omega, lr_forecast, sr_analysis, lr_enkf = [], [], [], [], [], []

    last_omega0 = None

    for i_cycle in tqdm(range(n_cycles + 1 - offset), disable=hide_progressbar):
        logger.debug(f"\nStart: i_cycle = {i_cycle}, t = {lr_model.t:.2f}")

        # Make observations
        hr_gt = hr_omega[0, i_cycle]
        obs_matrix = obs_matrix_generator.generate_obs_matrix(
            nx=hr_nx, ny=hr_ny, obs_interval=obs_interval
        )
        obs = extract_observation(
            obs_matrix_generator=obs_matrix_generator,
            hr_omega=hr_gt,
            obs_interval=obs_interval,
            hr_nx=hr_nx,
            hr_ny=hr_ny,
        )
        obs = obs[None, ...]  # Add batch dim

        ts.append(lr_model.t)
        lr_omega.append(lr_model.omega.cpu().clone())
        lr_forecast.append(srda_model.omega.cpu().clone())

        if i_cycle % assimilation_period == 0:
            p = 100 * torch.sum(obs_matrix).item() / (hr_nx * hr_ny)
            logger.debug(f"Observation grid ratio = {p} %")
            hr_obs.append(obs)
        else:
            hr_obs.append(torch.full_like(obs, torch.nan))

        if i_cycle > 0 and i_cycle % assimilation_period == 0:
            _ = assimilate_with_existing_data(
                hr_omega=hr_gt.to(torch.float64).to(device),
                lr_ens_model=lr_ens_model,
                obs_matrix=obs_matrix.to(torch.float64).to(device),
                obs_noise_std=obs_noise_std,
                inflation=inflation,
                rand_generator=torch_rand_generator,
                localization_matrix=localization_matrix,
                interpolator=sr_interpolator,
                bias=biases.squeeze().item(),
                scale=scales.squeeze().item(),
            )
            logger.debug(f"EnKF Assimilation at i = {i_cycle}")

        lr_enkf.append(lr_ens_model.omega.cpu().clone())

        if i_cycle % assimilation_period == 0:
            if i_cycle > 0 and inflation != 1.0:
                continue
            _noise = sys_noise_generator.sample([lr_ne]).reshape(lr_ne, lr_nx, lr_ny)
            _noise = _noise - torch.mean(_noise, dim=0, keepdims=True)
            _omega = lr_ens_model.omega + _noise.to(device)
            lr_ens_model.initialize(t0=lr_ens_model.t, omega0=_omega)
            lr_ens_model.calc_grid_data()
            logger.debug(f"Add system nosize at i_cycle = {i_cycle}")

        if i_cycle > 0 and i_cycle % assimilation_period == 0:

            lr = torch.stack(lr_forecast[-(assimilation_period + 1) :], dim=0)
            if last_omega0 is not None:
                lr[0, ...] = last_omega0

            x = preprocess(
                data=lr,
                biases=biases,
                scales=scales,
                clamp_max=clamp_max,
                clamp_min=clamp_min,
                n_ens=1,
                assimilation_period=assimilation_period,
                ny=lr_ny,
                nx=lr_nx,
                device=device,
            )

            obs = torch.stack(hr_obs[-(assimilation_period + 1) :], dim=0)
            noise = np_rng.normal(loc=0, scale=obs_noise_std, size=obs.shape)
            obs = obs + noise

            o = preprocess(
                data=obs,
                biases=biases,
                scales=scales,
                clamp_max=clamp_max,
                clamp_min=clamp_min,
                n_ens=1,
                assimilation_period=assimilation_period,
                ny=hr_ny,
                nx=hr_nx,
                device=device,
            )
            o = torch.nan_to_num(o, nan=missing_value)

            start_time = time.time()
            with torch.no_grad():
                sr = sr_model(x, o).detach().cpu().clone()
                sr = inv_preprocess(sr, biases, scales)
            logger.debug(f"Elapsed time = {time.time() - start_time}")

            # Delete channel dim
            sr = sr.squeeze(2)

            # ens, time, y, x -> time, ens, x, y
            sr = append_zeros(sr.permute(1, 0, 3, 2))
            assert sr.shape == (assimilation_period + 1, 1, hr_nx, hr_ny)

            last_omega0 = interpolate(sr[-1, :], nx=lr_nx, ny=lr_ny)
            srda_model.initialize(t0=srda_model.t, omega0=last_omega0)

            i_start = 0 if len(sr_analysis) == 0 else 1
            for it in range(i_start, sr.shape[0]):
                sr_analysis.append(sr[it])

            logger.debug(f"SR Assimilation at i = {i_cycle}")

        lr_model.time_integrate(dt=lr_dt, nt=lr_nt, hide_progress_bar=True)
        lr_model.calc_grid_data()

        srda_model.time_integrate(dt=lr_dt, nt=lr_nt, hide_progress_bar=True)
        srda_model.calc_grid_data()

        lr_ens_model.time_integrate(dt=lr_dt, nt=lr_nt, hide_progress_bar=True)
        lr_ens_model.calc_grid_data()

    # Stack along time dim
    lr_omega = torch.stack(lr_omega, dim=1)
    lr_enkf = torch.stack(lr_enkf, dim=1)
    hr_obsrv = torch.stack(hr_obs, dim=1)
    sr_analysis = torch.stack(sr_analysis, dim=1)

    return lr_omega, lr_enkf, hr_obsrv, sr_analysis