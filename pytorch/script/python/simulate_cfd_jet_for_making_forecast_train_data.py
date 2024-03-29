import argparse
import os
import pathlib
import sys
import time
import traceback
from logging import INFO, WARNING, FileHandler, StreamHandler, getLogger
from typing import Callable

import numpy as np
import pandas as pd
import torch
from cfd_model.cfd.periodic_channel_domain import TorchSpectralModel2D
from cfd_model.filter.low_pass_periodic_channel_domain import LowPassFilter
from cfd_model.initialization.periodic_channel_jet_initializer import (
    calc_init_omega,
    calc_init_perturbation_hr_omegas,
    calc_jet_forcing,
)
from scipy.ndimage import sobel
from src.utils import set_seeds

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

LR_NX = 32
LR_NY = 17
LR_DT = 5e-4
LR_NT = 500

HR_NX = 128
HR_NY = 65
HR_DT = LR_DT / 4.0
HR_NT = LR_NT * 4

assert HR_DT * HR_NT == LR_DT * LR_NT

N_CYCLES = 140
ASSIM_PERIOD = 4
N_ENSEMBLES = 20

assert N_CYCLES % ASSIM_PERIOD == 0

Y0 = np.pi / 2.0
SIGMA = 0.4
U0 = 3.0
TAU0 = 0.3
PERTUB_NOISE = 0.0025

BETA = 0.1
COEFF_LINEAR_DRAG = 1e-2
ORDER_DIFFUSION = 2
HR_COEFF_DIFFUSION = 1e-5
LR_COEFF_DIFFUSION = 1e-5

ROOT_DIR = str((pathlib.Path(os.environ["PYTHONPATH"]) / "..").resolve())
DEVICE = "cuda"

DF_SEEDS = pd.read_csv(f"{ROOT_DIR}/pytorch/config/cfd_seeds/seeds01.csv").set_index(
    "SimulationNumber"
)

EXPERIMENT_DIR = f"{ROOT_DIR}/data/pytorch/CFD/jet03"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

low_pass_filter = LowPassFilter(
    nx_lr=LR_NX, ny_lr=LR_NY, nx_hr=HR_NX, ny_hr=HR_NY, device=DEVICE
)


def make_and_initialize_hr_model(n_ensembles: int, seed: int, t0: float = 0.0):
    logger.setLevel(WARNING)

    hr_jet, hr_forcing = calc_jet_forcing(
        nx=HR_NX,
        ny=HR_NY,
        ne=n_ensembles,
        y0=Y0,
        sigma=SIGMA,
        tau0=TAU0,
    )

    hr_perturb = calc_init_perturbation_hr_omegas(
        nx=HR_NX, ny=HR_NY, ne=n_ensembles, noise_amp=PERTUB_NOISE, seed=seed
    )

    hr_omega0 = calc_init_omega(
        perturb_omega=hr_perturb,
        jet=hr_jet,
        u0=U0,
    )

    hr_model = TorchSpectralModel2D(
        nx=HR_NX,
        ny=HR_NY,
        coeff_linear_drag=COEFF_LINEAR_DRAG,
        coeff_diffusion=HR_COEFF_DIFFUSION,
        order_diffusion=ORDER_DIFFUSION,
        beta=BETA,
        device=DEVICE,
    )
    hr_model.initialize(t0=t0, omega0=hr_omega0, forcing=hr_forcing)
    hr_model.calc_grid_data()

    logger.setLevel(INFO)

    return hr_model


def make_and_initialize_lr_model(
    t0: float,
    hr_omega: torch.Tensor,
    noise_generator: Callable[[torch.Tensor], torch.Tensor] = None,
):
    logger.setLevel(WARNING)

    _, lr_forcing = calc_jet_forcing(
        nx=LR_NX,
        ny=LR_NY,
        ne=hr_omega.shape[0],  # n_ensembles
        y0=Y0,
        sigma=SIGMA,
        tau0=TAU0,
    )

    lr_omega0 = None
    if noise_generator is None:
        lr_omega0 = low_pass_filter.apply(hr_omega)
    else:
        noise = noise_generator(hr_omega)
        lr_omega0 = low_pass_filter.apply(hr_omega + noise)

    lr_model = TorchSpectralModel2D(
        nx=LR_NX,
        ny=LR_NY,
        coeff_linear_drag=COEFF_LINEAR_DRAG,
        coeff_diffusion=LR_COEFF_DIFFUSION,
        order_diffusion=ORDER_DIFFUSION,
        beta=BETA,
        device=DEVICE,
    )
    lr_model.initialize(t0=t0, omega0=lr_omega0, forcing=lr_forcing)
    lr_model.calc_grid_data()

    logger.setLevel(INFO)

    return lr_model


def write_out(*, hr_omegas, dict_lr_omegas, i_seed, i_start, i_end, result_dir):
    hr_omegas = torch.stack(hr_omegas, dim=1)  # stack along time dim.
    assert hr_omegas.shape == (N_ENSEMBLES, 2 * ASSIM_PERIOD + 1, HR_NX, HR_NY)

    file_path = (
        f"{result_dir}/seed{i_seed:05}_start{i_start:02}_end{i_end:02}_hr_omega.npy"
    )
    np.save(file_path, hr_omegas.cpu().numpy())

    for name in dict_lr_omegas.keys():
        lr_omegas = torch.stack(dict_lr_omegas[name], dim=1)  # stack along time dim.
        assert lr_omegas.shape == (N_ENSEMBLES, 2 * ASSIM_PERIOD + 1, LR_NX, LR_NY)

        file_path = f"{result_dir}/seed{i_seed:05}_start{i_start:02}_end{i_end:02}_lr_omega_{name}.npy"
        np.save(file_path, lr_omegas.cpu().numpy())


def store_data(*, hr_omegas, hr_model, dict_lr_omegas, dict_lr_models):
    hr_omegas.append(hr_model.omega.cpu().clone().to(torch.float32))
    for name in dict_lr_omegas.keys():
        dict_lr_omegas[name].append(
            dict_lr_models[name].omega.cpu().clone().to(torch.float32)
        )


class GaussianNoiseGenerator:
    def __init__(self, noise_amplitude: float):
        self.noise_amplitude = noise_amplitude
        logger.info(f"noise_amplitude = {self.noise_amplitude}")

    def __call__(self, hr_omega: torch.Tensor):
        noise = torch.randn_like(hr_omega)
        noise *= self.noise_amplitude
        return noise


class SobelNoiseGenerator:
    def __init__(self, axis: int, noise_amplitude: float):
        self.axis = axis
        self.noise_amplitude = noise_amplitude
        logger.info(f"axis = {self.axis}, noise amp = {self.noise_amplitude}")

    def __call__(self, hr_omega: torch.Tensor):
        noise = sobel(hr_omega.cpu().numpy(), axis=self.axis)
        noise = noise / np.mean(np.abs(noise), axis=(-2, -1), keepdims=True)
        noise *= self.noise_amplitude
        return torch.from_numpy(noise)


LR_NOISE_GENERATORS = {
    "no-noise": None,
}

parser = argparse.ArgumentParser()
parser.add_argument("--i_seed_start", type=int, required=True)
parser.add_argument("--i_seed_end", type=int, required=True)

if __name__ == "__main__":

    start_time = time.time()
    i_seed_start = parser.parse_args().i_seed_start
    i_seed_end = parser.parse_args().i_seed_end
    logger.addHandler(
        FileHandler(
            f"{EXPERIMENT_DIR}/log_start{i_seed_start:05}_end{i_seed_end:05}.txt"
        )
    )

    for i_seed in range(i_seed_start, i_seed_end + 1):
        try:
            seed = DF_SEEDS.loc[i_seed, "Seed0"]
            set_seeds(seed, use_deterministic=True)
            logger.info("\n*********************************************************")
            logger.info(f"i_seed = {i_seed}, seed = {seed}")
            logger.info("*********************************************************\n")

            start_time_per_loop = time.time()

            result_dir = f"{EXPERIMENT_DIR}/seed{i_seed:05}"
            os.makedirs(result_dir, exist_ok=True)

            hr_model = make_and_initialize_hr_model(seed=seed, n_ensembles=N_ENSEMBLES)

            all_hr_omegas = []
            dict_lr_init_infos = {}

            for i_cycle in tqdm(range(N_CYCLES + 1), total=N_CYCLES + 1):
                if i_cycle % ASSIM_PERIOD == 0:
                    i_start = i_cycle
                    i_end = i_start + 2 * ASSIM_PERIOD
                    logger.info(
                        f"LR model register: i_start = {i_start}, i_end = {i_end} (i_cycle = {i_cycle})"
                    )

                    _dict = {}
                    _t0 = hr_model.t
                    _hr_omega = hr_model.omega.cpu().clone()
                    for name, gen in LR_NOISE_GENERATORS.items():
                        _dict[name] = make_and_initialize_lr_model(_t0, _hr_omega, gen)

                    dict_lr_init_infos[i_end] = {
                        "i_start": i_start,
                        "i_end": i_end,
                        "t0": _t0,
                        "lr_models": _dict,
                    }

                all_hr_omegas.append(hr_model.omega.cpu().clone().to(torch.float32))

                hr_model.time_integrate(dt=HR_DT, nt=HR_NT, hide_progress_bar=True)
                hr_model.calc_grid_data()

                if i_cycle in dict_lr_init_infos.keys():

                    i_start = dict_lr_init_infos[i_cycle]["i_start"]
                    i_end = dict_lr_init_infos[i_cycle]["i_end"]

                    logger.info(
                        f"\nLR model integration: i_start = {i_start}, i_end = {i_end} (i_cycle = {i_cycle})"
                    )

                    hr_omegas = all_hr_omegas[i_start : i_end + 1]
                    logger.info(
                        f"hr omega len = {len(hr_omegas)}, all hr omega len = {len(all_hr_omegas)}"
                    )

                    lr_models = dict_lr_init_infos[i_cycle]["lr_models"]

                    dict_lr_omegas = {
                        key: [model.omega.cpu().clone().to(torch.float32)]
                        for key, model in lr_models.items()
                    }

                    for key, model in lr_models.items():
                        for _ in range(i_end - i_start):
                            model.time_integrate(
                                dt=LR_DT, nt=LR_NT, hide_progress_bar=True
                            )
                            model.calc_grid_data()
                            dict_lr_omegas[key].append(
                                model.omega.cpu().clone().to(torch.float32)
                            )
                        logger.info(
                            f"lr model [{key}] len = {len(dict_lr_omegas[key])}"
                        )

                    write_out(
                        hr_omegas=hr_omegas,
                        dict_lr_omegas=dict_lr_omegas,
                        i_seed=i_seed,
                        i_start=i_start,
                        i_end=i_end,
                        result_dir=result_dir,
                    )

                    logger.info("all data were written out.\n")
                    del dict_lr_init_infos[i_cycle], lr_models, dict_lr_omegas

            end_time_per_loop = time.time()
            logger.info(
                f"Elapsed time per loop = {end_time_per_loop - start_time_per_loop} sec"
            )

        except Exception as e:
            logger.info("\n*********************************************************")
            logger.info("Error")
            logger.info("*********************************************************\n")
            logger.error(e)
            logger.error(traceback.format_exc())

    end_time = time.time()
    logger.info(f"Total elapsed time = {end_time - start_time} sec")