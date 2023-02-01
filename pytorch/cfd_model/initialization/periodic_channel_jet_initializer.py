import sys
import typing
from logging import getLogger

import numpy as np
import torch
from cfd_model.fft.periodic_channel_domain import TorchFftCalculator

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger()


def calc_jet_shape(
    y: np.ndarray, y0: float, sigma: float, u0: float = 1.0
) -> np.ndarray:

    exponent = (y - y0) / sigma
    jet = u0 / (np.cosh(exponent)) ** 2

    return jet - np.mean(jet)


def calc_jet_forcing(
    *,
    nx: int,
    ny: int,
    ne: int,
    y0: float,
    sigma: float,
    tau0: float,
    dtype: torch.dtype = torch.float64,
    **kwargs,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:

    logger.info(f"y0 = {y0}, sigma = {sigma}, tau0 = {tau0}")
    assert y0 > 0 and sigma > 0 and tau0 > 0

    fft_calculator = TorchFftCalculator(nx=nx, ny=ny)

    xs = np.linspace(0, 2 * np.pi, num=nx, endpoint=False)
    ys = np.linspace(0, np.pi, num=ny, endpoint=True)
    _, y = np.meshgrid(xs, ys, indexing="ij")

    jet = calc_jet_shape(y, y0=y0, sigma=sigma)
    assert jet.shape == (nx, ny)
    jet = np.broadcast_to(jet, (ne, nx, ny))

    forcing = torch.tensor(tau0 * jet, dtype=dtype)

    # The above forcing is for u, so it is converted to the forcing for omega (v is set to be zero)
    forcing = fft_calculator.calculate_omega_from_uv(
        u=forcing, v=torch.zeros_like(forcing)
    )

    assert forcing.shape == jet.shape

    return jet, forcing


def calc_init_perturbation_omegas(
    *,
    hr_nx: int,
    hr_ny: int,
    lr_nx: int,
    lr_ny: int,
    noise_amp: float,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:

    x = np.linspace(0, 2 * np.pi, num=hr_nx, endpoint=False)
    y = np.linspace(0, np.pi, num=hr_ny, endpoint=True)
    x, y = np.meshgrid(x, y, indexing="ij")
    hr_x = x[None, :, :]  # add emsemble channel
    hr_y = y[None, :, :]

    x = np.linspace(0, 2 * np.pi, num=lr_nx, endpoint=False)
    y = np.linspace(0, np.pi, num=lr_ny, endpoint=True)
    x, y = np.meshgrid(x, y, indexing="ij")
    lr_x = x[None, :, :]  # add emsemble channel
    lr_y = y[None, :, :]

    hr_max_kx = hr_nx // 3
    hr_max_ky = hr_ny // 3
    lr_max_kx = lr_nx // 3
    lr_max_ky = lr_ny // 3

    logger.info(f"hr_max: kx = {hr_max_kx}, ky = {hr_max_ky}")
    logger.info(f"lr_max: kx = {lr_max_kx}, ky = {lr_max_ky}")

    np.random.seed(seed)
    lst_amp = np.random.randn(1, 2 * hr_max_kx + 1, 2 * hr_max_ky + 1) * noise_amp
    lst_phs = np.random.randn(1, 2 * hr_max_kx + 1, 2 * hr_max_ky + 1) * np.pi

    hr_omega = np.zeros((1, hr_nx, hr_ny), dtype=np.float64)
    lr_omega = np.zeros((1, lr_nx, lr_ny), dtype=np.float64)

    for i, kx in enumerate(range(-hr_max_kx, hr_max_kx + 1)):
        if kx == 0:
            continue
        for j, ky in enumerate(range(-hr_max_ky, hr_max_ky + 1)):
            if ky == 0:
                continue
            amp = lst_amp[:, i, j]
            phs = lst_phs[:, i, j]
            amp = amp[:, None, None]  # add x and y dim
            phs = phs[:, None, None]

            hr_omega += amp * np.sin(kx * hr_x + ky * hr_y + phs)

            if abs(kx) <= lr_max_kx and abs(ky) <= lr_max_ky:
                lr_omega += amp * np.sin(kx * lr_x + ky * lr_y + phs)

    assert hr_omega.shape == (1, hr_nx, hr_ny)
    assert lr_omega.shape == (1, lr_nx, lr_ny)

    return torch.tensor(hr_omega, dtype=dtype), torch.tensor(lr_omega, dtype=dtype)


def calc_init_perturbation_hr_omegas(
    *,
    nx: int,
    ny: int,
    ne: int,
    noise_amp: float,
    seed: int = 42,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:

    logger.info(f"Seed in making init perturbation = {seed}")

    x = np.linspace(0, 2 * np.pi, num=nx, endpoint=False)
    y = np.linspace(0, np.pi, num=ny, endpoint=True)
    x, y = np.meshgrid(x, y, indexing="ij")

    x = np.broadcast_to(x, (ne, nx, ny))
    y = np.broadcast_to(y, (ne, nx, ny))

    max_kx = nx // 3
    max_ky = ny // 3
    logger.info(f"max: kx = {max_kx}, ky = {max_ky}")

    np.random.seed(seed)
    lst_amp = np.random.randn(ne, 2 * max_kx + 1, 2 * max_ky + 1) * noise_amp
    lst_phs = np.random.randn(ne, 2 * max_kx + 1, 2 * max_ky + 1) * np.pi

    omega = np.zeros((ne, nx, ny), dtype=np.float64)

    for i, kx in tqdm(enumerate(range(-max_kx, max_kx + 1)), total=2 * max_kx + 1):
        if kx == 0:
            continue
        for j, ky in enumerate(range(-max_ky, max_ky + 1)):
            if ky == 0:
                continue
            amp = lst_amp[:, i, j]
            phs = lst_phs[:, i, j]
            amp = amp[:, None, None]  # add x and y dim
            phs = phs[:, None, None]

            omega += amp * np.sin(kx * x + ky * y + phs)

    assert omega.shape == (ne, nx, ny)

    return torch.tensor(omega, dtype=dtype)


def calc_ens_perturbation_omega(
    *,
    max_kx: int,
    max_ky: int,
    nx: int,
    ny: int,
    ne: int,
    noise_amp: float,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:

    x = np.linspace(0, 2 * np.pi, num=nx, endpoint=False)
    y = np.linspace(0, np.pi, num=ny, endpoint=True)
    x, y = np.meshgrid(x, y, indexing="ij")
    x = x[None, :, :]  # add emsemble channel
    y = y[None, :, :]

    np.random.seed(seed)
    lst_amp = np.random.randn(ne, max_kx, max_ky) * noise_amp
    lst_phs = np.random.randn(ne, max_kx, max_ky) * np.pi

    omega = np.zeros((ne, nx, ny), dtype=np.float64)

    for i, kx in enumerate(range(1, max_kx + 1)):
        for j, ky in enumerate(range(1, max_ky + 1)):
            amp = lst_amp[:, i, j]
            phs = lst_phs[:, i, j]
            amp = amp[:, None, None]  # add x and y dim
            phs = phs[:, None, None]

            omega += amp * np.sin(kx * x + ky * y + phs)

    assert omega.shape == (ne, nx, ny)

    return torch.tensor(omega, dtype=dtype)


def calc_init_omega(
    *,
    perturb_omega: torch.Tensor,
    jet: torch.Tensor,
    u0: float,
    dtype: torch.dtype = torch.float64,
):
    logger.info(f"u0 = {u0}")
    assert u0 > 0

    ne, nx, ny = jet.shape

    U = torch.tensor(u0 * jet, dtype=dtype)

    fft_calculator = TorchFftCalculator(nx=nx, ny=ny)
    u, v = fft_calculator.calculate_uv_from_omega(perturb_omega)
    u += U

    omega = fft_calculator.calculate_omega_from_uv(u, v)
    assert omega.shape == (ne, nx, ny)

    return omega