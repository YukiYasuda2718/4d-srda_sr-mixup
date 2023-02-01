import argparse
import json
import os
import pathlib
import random
import sys
import time
import traceback
from logging import INFO, FileHandler, StreamHandler, getLogger

import numpy as np
import torch
import yaml
from cfd_model.cfd.periodic_channel_domain import TorchSpectralModel2D
from cfd_model.enkf.sr_enkf import (
    assimilate_with_existing_data,
    calc_localization_matrix,
    get_multivariate_normal_sampler,
)
from cfd_model.initialization.periodic_channel_jet_initializer import calc_jet_forcing
from cfd_model.interpolator.torch_interpolator import interpolate
from src.dataloader import get_hr_file_paths
from src.dataset import generate_is_obs_and_obs_matrix
from src.sr_da_helper import _read_all_hr_omegas
from src.utils import set_seeds
from tqdm import tqdm

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic
set_seeds(42, use_deterministic=True)

ROOT_DIR = str((pathlib.Path(os.environ["PYTHONPATH"]) / "..").resolve())
DEVICE = "cuda"

CFD_DIR_NAME = "jet11"
TRAIN_VALID_TEST_RATIOS = [0.7, 0.2, 0.1]

OBS_NOISE_STD = 0.1
OBS_GRID_INTERVALS = [4, 6, 8, 10, 12]
SEED = 42

ASSIMILATION_PERIOD = 4
START_TIME_INDEX = 16

LR_NX = 32
LR_NY = 17
LR_DT = 5e-4
LR_NT = 500

HR_NX = 128
HR_NY = 65

Y0 = np.pi / 2.0
SIGMA = 0.4
TAU0 = 0.3

BETA = 0.1
COEFF_LINEAR_DRAG = 1e-2
ORDER_DIFFUSION = 2
LR_COEFF_DIFFUSION = 5e-5

DT = LR_DT * LR_NT
T0 = START_TIME_INDEX * LR_DT * LR_NT

LR_CFD_CONFIG = {
    "nx": LR_NX,
    "ny": LR_NY,
    "ne": None,
    "lr_nx": LR_NX,
    "lr_ny": LR_NY,
    "hr_nx": HR_NX,
    "hr_ny": HR_NY,
    "n_ens": None,
    "assimilation_period": ASSIMILATION_PERIOD,
    "coeff_linear_drag": COEFF_LINEAR_DRAG,
    "coeff_diffusion": LR_COEFF_DIFFUSION,
    "order_diffusion": ORDER_DIFFUSION,
    "beta": BETA,
    "device": DEVICE,
    "y0": Y0,
    "sigma": SIGMA,
    "tau0": TAU0,
    "t0": T0,
}


def initialize_lr_model(
    *,
    t0: float,
    hr_omega0: torch.Tensor,
    lr_forcing: torch.Tensor,
    lr_model: TorchSpectralModel2D,
    n_ens: int,
    hr_nx: int,
    hr_ny: int,
    lr_nx: int,
    lr_ny: int,
    **kwargs,
):

    assert hr_omega0.shape == (hr_nx, hr_ny)
    omega0 = interpolate(hr_omega0[None, ...], lr_nx, lr_ny, "bicubic")
    omega0 = torch.broadcast_to(omega0, (n_ens, lr_nx, lr_ny))

    lr_model.initialize(t0=t0, omega0=omega0, forcing=lr_forcing)
    lr_model.calc_grid_data()


def get_obs_matrices(obs_grid_interval: int):
    obs_matrices = []

    for init_x in tqdm(range(obs_grid_interval)):
        for init_y in range(obs_grid_interval):
            _, obs_mat = generate_is_obs_and_obs_matrix(
                nx=HR_NX,
                ny=HR_NY,
                init_index_x=init_x,
                init_index_y=init_y,
                interval_x=obs_grid_interval,
                interval_y=obs_grid_interval,
                dtype=torch.float64,
            )
            obs_matrices.append(obs_mat)

    return obs_matrices


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)


if __name__ == "__main__":
    config_path = parser.parse_args().config_path
    with open(config_path) as file:
        config = yaml.safe_load(file)

    experiment_name = config_path.split("/")[-2]
    config_name = os.path.basename(config_path).split(".")[0]

    result_dir = f"{ROOT_DIR}/data/pytorch/EnKF_results/{experiment_name}/{config_name}"
    os.makedirs(result_dir, exist_ok=False)
    logger.addHandler(FileHandler(f"{result_dir}/log.txt"))

    logger.info(f"config = {json.dumps(config, indent=2)}")
    USED_DATA_KIND = config["USED_DATA_KIND"]
    NUM_SIMULATIONS = config["NUM_SIMULATIONS"]
    N_ENS = config["N_ENS"]
    SYS_NOISE_STD = config["SYS_NOISE_STD"]
    SYS_NOISE_DX = config["SYS_NOISE_DX"]
    LOCALIZE_DX = config["LOCALIZE_DX"]
    SYS_NOISE_DY = SYS_NOISE_DX
    LOCALIZE_DY = LOCALIZE_DX
    LR_CFD_CONFIG["ne"] = N_ENS
    LR_CFD_CONFIG["n_ens"] = N_ENS

    hr_file_paths = get_hr_file_paths(ROOT_DIR, CFD_DIR_NAME, TRAIN_VALID_TEST_RATIOS)
    hr_file_paths = hr_file_paths[USED_DATA_KIND]
    all_hr_omegas = _read_all_hr_omegas(hr_file_paths)

    torch_rand_generator = torch.Generator().manual_seed(SEED)

    sys_noise_generator = get_multivariate_normal_sampler(
        nx=LR_NX, ny=LR_NY, sigma=SYS_NOISE_STD, d_x=SYS_NOISE_DX, d_y=SYS_NOISE_DY
    )

    localization_matrix = calc_localization_matrix(
        nx=HR_NX, ny=HR_NY, d_x=LOCALIZE_DX, d_y=LOCALIZE_DY
    ).to(DEVICE)

    max_time_index = all_hr_omegas.shape[1]

    for obs_grid_interval in OBS_GRID_INTERVALS:
        try:
            logger.info("\n--------------------------------------------------")
            logger.info(f"obs_grid_interval = {obs_grid_interval}")
            logger.info("--------------------------------------------------")
            obs_matrices = get_obs_matrices(obs_grid_interval)

            all_enkf_means = []

            for i_ens in range(NUM_SIMULATIONS):
                logger.info(f"i_ens = {i_ens}")
                start_time = time.time()

                lr_model = TorchSpectralModel2D(**LR_CFD_CONFIG)
                _, lr_forcing = calc_jet_forcing(**LR_CFD_CONFIG)
                initialize_lr_model(
                    hr_omega0=all_hr_omegas[i_ens, 0],  # initial time
                    lr_forcing=lr_forcing,
                    lr_model=lr_model,
                    **LR_CFD_CONFIG,
                )

                i = random.randint(0, len(obs_matrices) - 1)
                obs_matrix = obs_matrices[i]
                logger.info(f"Obs matrix index = {i}")

                lr_enkfs = []

                for i_cycle in tqdm(range(max_time_index)):
                    if i_cycle > 0 and i_cycle % ASSIMILATION_PERIOD == 0:
                        _ = assimilate_with_existing_data(
                            hr_omega=all_hr_omegas[i_ens, i_cycle]
                            .to(torch.float64)
                            .to(DEVICE),
                            lr_ens_model=lr_model,
                            obs_matrix=obs_matrix.clone().to(DEVICE),
                            obs_noise_std=OBS_NOISE_STD,
                            inflation=1.0,
                            rand_generator=torch_rand_generator,
                            localization_matrix=localization_matrix,
                        )

                    lr_enkfs.append(lr_model.omega.cpu().clone())

                    if i_cycle % ASSIMILATION_PERIOD == 0:
                        noise = sys_noise_generator.sample([N_ENS]).reshape(
                            N_ENS, LR_NX, LR_NY
                        )
                        noise = noise - torch.mean(noise, dim=0, keepdims=True)
                        omega = lr_model.omega + noise.to(DEVICE)
                        lr_model.initialize(t0=lr_model.t, omega0=omega)
                        lr_model.calc_grid_data()

                    lr_model.time_integrate(dt=LR_DT, nt=LR_NT, hide_progress_bar=True)
                    lr_model.calc_grid_data()

                # Stack along time dim
                lr_enkfs = torch.stack(lr_enkfs, dim=1)

                # Calc ensemble mean
                all_enkf_means.append(torch.mean(lr_enkfs, dim=0))

                end_time = time.time()
                logger.info(f"Elapsed Time = {end_time - start_time} [sec]\n")

            all_enkf_means = (
                torch.stack(all_enkf_means, dim=0).to(torch.float32).numpy()
            )
            np.save(
                f"{result_dir}/{config_name}_obs_grid_interval_{obs_grid_interval:02}.npy",
                all_enkf_means,
            )
        except Exception as e:
            logger.info("\n*********************************************************")
            logger.info("Error")
            logger.info("*********************************************************\n")
            logger.error(e)
            logger.error(traceback.format_exc())