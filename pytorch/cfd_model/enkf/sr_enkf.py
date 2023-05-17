import typing
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from cfd_model.cfd.abstract_cfd_model import AbstractCfdModel
from cfd_model.enkf.observation_matrix import HrObservationMatrixGenerator
from cfd_model.interpolator.torch_interpolator import interpolate
from torch.distributions.distribution import Distribution
from torch.distributions.multivariate_normal import MultivariateNormal

logger = getLogger()


def gaspari_cohn_99(r: np.ndarray) -> np.ndarray:
    r5 = r**5
    r4 = r**4
    r3 = r**3
    r2 = r**2
    zeros = np.zeros_like(r)

    ret = np.where(
        r <= 1.0,
        1 - 1 / 4 * r5 + 1 / 2 * r4 + 5 / 8 * r3 - 5 / 3 * r2,
        zeros,
    )

    with np.errstate(divide="ignore"):
        ret = np.where(
            r > 1,
            1 / 12 * r5 - 1 / 2 * r4 + 5 / 8 * r3 + 5 / 3 * r2 - 5 * r + 4 - 2 / 3 / r,
            ret,
        )

    ret = np.where(r > 2, zeros, ret)

    return ret


def calc_localization_matrix(*, nx: int, ny: int, d_x: float, d_y: float) -> np.ndarray:

    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    y = np.linspace(0, np.pi, ny, endpoint=True)
    x, y = np.meshgrid(x, y, indexing="ij")
    x, y = x.reshape(-1), y.reshape(-1)

    pos = np.stack([x, y], axis=-1)

    r = pos[None, ...] - pos[:, None, :]
    assert r.shape == (nx * ny, nx * ny, 2)

    r = np.sqrt((r[:, :, 0] / d_x) ** 2 + (r[:, :, 1] / d_y) ** 2)
    localization = gaspari_cohn_99(r)

    return torch.tensor(localization, dtype=torch.float64)


def _calc_cov(
    *, nx: int, ny: int, sigma: float, d_x: float, d_y: float, eps=1e-14
) -> np.ndarray:

    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    y = np.linspace(0, np.pi, ny, endpoint=True)
    x, y = np.meshgrid(x, y, indexing="ij")
    x, y = x.reshape(-1), y.reshape(-1)

    pos = np.stack([x, y], axis=-1)

    cov = pos[None, ...] - pos[:, None, :]
    assert cov.shape == (nx * ny, nx * ny, 2)

    cov = (cov[:, :, 0] / d_x) ** 2 + (cov[:, :, 1] / d_y) ** 2
    conv = (sigma**2) * np.exp(-0.5 * cov)
    conv = torch.tensor(conv, dtype=torch.float64)

    # Assure conv is symmetric.
    conv = (conv + conv.T) / 2.0

    # Assure positive definiteness
    conv += torch.diag(torch.full(size=(nx * ny,), fill_value=eps))

    return conv


def get_multivariate_normal_sampler(
    *, nx: int, ny: int, sigma: float, d_x: float, d_y: float
) -> Distribution:
    cov = _calc_cov(nx=nx, ny=ny, sigma=sigma, d_x=d_x, d_y=d_y)
    mean = torch.zeros(cov.shape[0], dtype=torch.float64)
    return MultivariateNormal(loc=mean, covariance_matrix=cov)


def _add_noise_and_calc_obs_covariance(
    observation: torch.Tensor,
    obs_std: float,
    n_ens: int,
    rand_generator: torch.Generator = None,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:

    obs = observation.reshape(-1)
    n_obs = obs.shape[0]

    noise = obs_std * torch.randn(n_ens, n_obs, generator=rand_generator).to(obs.device)
    noise = noise - torch.mean(noise, dim=0, keepdim=True)
    assert noise.shape == (n_ens, n_obs)

    cov = noise.t().mm(noise) / (n_ens - 1)

    obs = obs[None, :]
    assert obs.shape[1] == noise.shape[1] == n_obs
    obs = obs + noise

    return obs, cov


def _calc_forecast_stats(state: torch.Tensor):

    assert state.dim() >= 2

    # num of ensemble members
    ne = state.shape[0]
    assert ne > 1

    forecast_all = state.reshape(ne, -1)

    forecast_mean = torch.mean(forecast_all, dim=0, keepdim=True)
    forecast_anomaly = forecast_all - forecast_mean
    forecast_covariance = forecast_anomaly.t().mm(forecast_anomaly) / (ne - 1)

    return forecast_mean, forecast_anomaly, forecast_all, forecast_covariance


def _calc_kalman_gain(
    forecast_cov: torch.Tensor, obs_cov: torch.Tensor, obs_matrix: torch.Tensor
):
    assert forecast_cov.ndim == obs_cov.ndim == obs_matrix.ndim == 2

    _cov = obs_matrix.mm(forecast_cov)
    _cov = _cov.mm(obs_matrix.t())

    _inv = torch.linalg.inv(_cov + obs_cov)

    kalman_gain = _inv.mm(obs_matrix).mm(forecast_cov)

    return kalman_gain


def _assimilate(
    *,
    observation: torch.Tensor,
    model_state: torch.Tensor,
    obs_noise_std: float,
    obs_matrix: torch.Tensor,
    inflation: float = 1.0,
    rand_generator: torch.Generator = None,
    localization_matrix: torch.Tensor = None,
):

    assert model_state.dim() >= 2
    assert obs_matrix.dim() == 2
    assert np.cumprod(observation.shape)[-1] == obs_matrix.shape[0]
    assert np.cumprod(model_state.shape[1:])[-1] == obs_matrix.shape[1]

    if inflation == 1.0:
        inflated_model_state = model_state
        logger.debug("No inflation.")
    else:
        mean_model_state = torch.mean(model_state, dim=0, keepdims=True)
        inflated_model_state = (model_state - mean_model_state) * inflation
        inflated_model_state = inflated_model_state + mean_model_state

    _, _, forecast_all, forecast_cov = _calc_forecast_stats(inflated_model_state)

    if localization_matrix is not None:
        assert localization_matrix.shape == forecast_cov.shape
        forecast_cov = forecast_cov * localization_matrix
        logger.debug("Localized forecast cov matrix.")

    logger.debug(
        f"max diag forecast cov = {torch.max(torch.diagonal(forecast_cov, dim1 = 0, dim2 = 1))}"
    )
    logger.debug(
        f"ave diag forecast cov = {torch.mean(torch.diagonal(forecast_cov, dim1 = 0, dim2 = 1))}"
    )
    logger.debug(f"obs noise std = {obs_noise_std}")

    obs, obs_cov = _add_noise_and_calc_obs_covariance(
        observation=observation,
        obs_std=obs_noise_std,
        n_ens=inflated_model_state.shape[0],
        rand_generator=rand_generator,
    )

    kalman_gain = _calc_kalman_gain(forecast_cov, obs_cov, obs_matrix)

    innovation = obs - forecast_all.mm(obs_matrix.t())
    analysis_all = forecast_all + innovation.mm(kalman_gain)

    return analysis_all, forecast_cov


def assimilate(
    *,
    hr_model: AbstractCfdModel,
    lr_ens_model: AbstractCfdModel,
    obs_matrix_generator: HrObservationMatrixGenerator,
    obs_noise_std: float,
    inflation: float,
    rand_generator: torch.Generator,
    device: str,
):
    _, hr_nx, hr_ny = hr_model.state_size
    lr_ne, lr_nx, lr_ny = lr_ens_model.state_size

    # Map lr model state to hr space
    lr_state = interpolate(lr_ens_model.omega, nx=hr_nx, ny=hr_ny).reshape(lr_ne, -1)

    obs_matrix = obs_matrix_generator.generate_obs_matrix(
        nx=hr_nx, ny=hr_ny, device=device
    )
    obs = obs_matrix.mm(hr_model.omega.reshape(-1, 1))

    analysis_all, _ = _assimilate(
        observation=obs,
        model_state=lr_state,
        obs_noise_std=obs_noise_std,
        obs_matrix=obs_matrix,
        inflation=inflation,
        rand_generator=rand_generator,
    )

    analysis_all = analysis_all.reshape(lr_ne, hr_nx, hr_ny)
    omega_all = interpolate(analysis_all, nx=lr_nx, ny=lr_ny)

    t = lr_ens_model.t
    lr_ens_model.initialize(t0=t, omega0=omega_all)
    lr_ens_model.calc_grid_data()


def assimilate_with_existing_data(
    *,
    hr_omega: torch.Tensor,
    lr_ens_model: AbstractCfdModel,
    obs_matrix: torch.Tensor,
    obs_noise_std: float,
    inflation: float,
    rand_generator: torch.Generator,
    localization_matrix: torch.Tensor = None,
    interpolator: torch.nn.Module = None,
    bias: float = None,
    scale: float = None,
    return_hr_analysis: bool = False,
):
    assert hr_omega.ndim == 2
    assert obs_matrix.ndim == 2

    hr_nx, hr_ny = hr_omega.shape
    assert obs_matrix.shape[1] == hr_nx * hr_ny

    lr_ne, lr_nx, lr_ny = lr_ens_model.state_size

    # Map lr model state to hr space
    if interpolator is None:
        lr_state = interpolate(
            lr_ens_model.omega, nx=hr_nx, ny=hr_ny, mode="bicubic"
        ).reshape(lr_ne, -1)
    else:
        # Add channel dim and drop last index along y
        X = lr_ens_model.omega[:, None, :, :-1].to(torch.float32)
        X = X.permute(0, 1, 3, 2)  # Exchange x and y dims
        X = (X - bias) / scale

        with torch.no_grad():
            y = interpolator(X, None)
        y = y.permute(0, 1, 3, 2)
        y = y * scale + bias

        # Append zeros to the last of y indices
        zs = torch.zeros(y.shape[0:3], dtype=y.dtype, device=y.device)[..., None]
        y = torch.cat([y, zs], dim=-1)

        y = y.to(torch.float64)
        lr_state = y.reshape(lr_ne, -1)

    obs = obs_matrix.mm(hr_omega.reshape(-1, 1))

    analysis_all, forecast_cov = _assimilate(
        observation=obs,
        model_state=lr_state,
        obs_noise_std=obs_noise_std,
        obs_matrix=obs_matrix,
        inflation=inflation,
        rand_generator=rand_generator,
        localization_matrix=localization_matrix,
    )

    analysis_all = analysis_all.reshape(lr_ne, hr_nx, hr_ny)
    omega_all = interpolate(analysis_all, nx=lr_nx, ny=lr_ny)

    t = lr_ens_model.t
    lr_ens_model.initialize(t0=t, omega0=omega_all)
    lr_ens_model.calc_grid_data()

    if return_hr_analysis:
        return analysis_all, forecast_cov
    else:
        return forecast_cov


def hr_assimilate_with_existing_data(
    *,
    hr_omega: torch.Tensor,
    hr_ens_model: AbstractCfdModel,
    obs_matrix: torch.Tensor,
    obs_noise_std: float,
    inflation: float,
    rand_generator: torch.Generator,
    localization_matrix: torch.Tensor = None,
    return_hr_analysis: bool = False,
):
    assert hr_omega.ndim == 2
    assert obs_matrix.ndim == 2

    hr_nx, hr_ny = hr_omega.shape
    assert obs_matrix.shape[1] == hr_nx * hr_ny

    hr_ne, _, _ = hr_ens_model.state_size
    hr_state = hr_ens_model.omega.reshape(hr_ne, -1)

    obs = obs_matrix.mm(hr_omega.reshape(-1, 1))

    analysis_all, forecast_cov = _assimilate(
        observation=obs,
        model_state=hr_state,
        obs_noise_std=obs_noise_std,
        obs_matrix=obs_matrix,
        inflation=inflation,
        rand_generator=rand_generator,
        localization_matrix=localization_matrix,
    )

    analysis_all = analysis_all.reshape(hr_ne, hr_nx, hr_ny)

    t = hr_ens_model.t
    hr_ens_model.initialize(t0=t, omega0=analysis_all)
    hr_ens_model.calc_grid_data()

    if return_hr_analysis:
        return analysis_all, forecast_cov
    else:
        return forecast_cov