import argparse
import glob
import json
import os
import pathlib
import random
import sys
import time
import traceback
import typing
from logging import INFO, FileHandler, StreamHandler, getLogger

import numpy as np
import pandas as pd
import torch
import yaml
from cfd_model.cfd.periodic_channel_domain import TorchSpectralModel2D
from cfd_model.enkf.sr_enkf import (
    calc_localization_matrix,
    hr_assimilate_with_existing_data,
)
from cfd_model.initialization.periodic_channel_jet_initializer import (
    calc_init_omega,
    calc_init_perturbation_hr_omegas,
    calc_jet_forcing,
)
from cfd_model.interpolator.torch_interpolator import interpolate_time_series
from src.dataloader import split_file_paths
from src.dataset import generate_is_obs_and_obs_matrix
from src.ssim import SSIM
from src.utils import read_pickle, set_seeds
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic
set_seeds(42, use_deterministic=True)

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

ROOT_DIR = str((pathlib.Path(os.environ["PYTHONPATH"]) / "..").resolve())
DEVICE = "cuda"

CFD_DIR_NAME = "jet12"
TRAIN_VALID_TEST_RATIOS = [0.7, 0.2, 0.1]

OBS_GRID_INTERVALS = [6, 8, 10, 12]
SEED = 31415
INFLATION = 1.0

ASSIMILATION_PERIOD = 4
OBS_NOISE_STD = 0.1
INITIAL_TIME_INDEX = 0

LR_NX = 32
LR_NY = 17
LR_DT = 5e-4
LR_NT = 500

HR_NX = 128
HR_NY = 65
HR_DT = LR_DT / 4.0
HR_NT = LR_NT * 4

Y0 = np.pi / 2.0
SIGMA = 0.4
U0 = 3.0
TAU0 = 0.3
PERTUB_NOISE = 0.0025

BETA = 0.1
COEFF_LINEAR_DRAG = 1e-2
ORDER_DIFFUSION = 2
HR_COEFF_DIFFUSION = 1e-5

DT = LR_DT * LR_NT

HR_CFD_CONFIG = {
    "nx": HR_NX,
    "ny": HR_NY,
    "hr_nx": HR_NX,
    "hr_ny": HR_NY,
    "assimilation_period": ASSIMILATION_PERIOD,
    "coeff_linear_drag": COEFF_LINEAR_DRAG,
    "coeff_diffusion": HR_COEFF_DIFFUSION,
    "order_diffusion": ORDER_DIFFUSION,
    "beta": BETA,
    "device": DEVICE,
    "y0": Y0,
    "sigma": SIGMA,
    "tau0": TAU0,
    "t0": 0.0,
}


def get_initial_hr_omega():
    hr_jet, _ = calc_jet_forcing(
        nx=HR_NX,
        ny=HR_NY,
        ne=NUM_SIMULATIONS,
        y0=Y0,
        sigma=SIGMA,
        tau0=TAU0,
    )

    hr_perturb = calc_init_perturbation_hr_omegas(
        nx=HR_NX, ny=HR_NY, ne=NUM_SIMULATIONS, noise_amp=PERTUB_NOISE, seed=SEED
    )

    hr_omega0 = calc_init_omega(
        perturb_omega=hr_perturb,
        jet=hr_jet,
        u0=U0,
    )

    return hr_omega0


def create_initialized_hr_model(n_ens: int, hr_omega0: torch.Tensor, t0: float = 0.0):
    assert hr_omega0.shape == (HR_NX, HR_NY)

    omega0 = torch.broadcast_to(hr_omega0, (n_ens, HR_NX, HR_NY))

    hr_model = TorchSpectralModel2D(**HR_CFD_CONFIG)
    _, hr_forcing = calc_jet_forcing(**HR_CFD_CONFIG)

    hr_model.initialize(t0=t0, omega0=omega0, forcing=hr_forcing)
    hr_model.calc_grid_data()

    return hr_model


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


def load_hr_data(
    root_dir: str,
    cfd_dir_name: str,
    train_valid_test_ratios: typing.List[str],
    kind: str,
    num_hr_omega_sets: int,
    max_ens_index: int = 20,
) -> torch.Tensor:

    cfd_dir_path = f"{root_dir}/data/pytorch/CFD/{cfd_dir_name}"
    logger.info(f"CFD dir path = {cfd_dir_path}")

    data_dirs = sorted([p for p in glob.glob(f"{cfd_dir_path}/*") if os.path.isdir(p)])

    train_dirs, valid_dirs, test_dirs = split_file_paths(
        data_dirs, train_valid_test_ratios
    )

    if kind == "train":
        target_dirs = train_dirs
    elif kind == "valid":
        target_dirs = valid_dirs
    elif kind == "test":
        target_dirs = test_dirs
    else:
        raise Exception(f"{kind} is not supported.")

    logger.info(f"Kind = {kind}, Num of dirs = {len(target_dirs)}")

    all_hr_omegas = []
    for dir_path in sorted(target_dirs):
        for i in range(max_ens_index):

            hr_omegas = []
            for file_path in sorted(glob.glob(f"{dir_path}/*_hr_omega_{i:02}.npy")):
                data = np.load(file_path)

                # This is to avoid overlapping at the start/end point
                if len(hr_omegas) > 0:
                    data = data[1:]
                hr_omegas.append(data)

            # Concat along time axis
            all_hr_omegas.append(np.concatenate(hr_omegas, axis=0))

            if len(all_hr_omegas) == num_hr_omega_sets:
                # Concat along batch axis
                ret = np.stack(all_hr_omegas, axis=0)
                return torch.from_numpy(ret).to(torch.float64)

    ret = np.stack(all_hr_omegas, axis=0)
    return torch.from_numpy(ret).to(torch.float64)


class ObsMatrixSampler:
    def __init__(self, obs_grid_interval: int):
        self.obs_matrices = get_obs_matrices(obs_grid_interval)

    def __call__(self):
        i = random.randint(0, len(self.obs_matrices) - 1)
        return self.obs_matrices[i].clone().to(DEVICE)


def calc_errors(all_gt: torch.Tensor, all_enkf: torch.Tensor) -> pd.DataFrame:
    ssim_func = SSIM(size_average=False, use_gauss=True)
    df_errors = pd.DataFrame()

    for kind in ["LR", "HR"]:

        if kind == "HR":
            gt = all_gt
        else:
            gt = interpolate_time_series(all_gt, LR_NX, LR_NY, "bicubic")

        if kind == "HR":
            pred = all_enkf
        else:
            pred = interpolate_time_series(all_enkf, LR_NX, LR_NY, "bicubic")

        # batch, time, x, and y dims
        assert gt.ndim == 4
        assert gt.shape == pred.shape

        mae = torch.mean(torch.abs(gt - pred), dim=(-2, -1))  # mean over x and y
        nrms = torch.mean(torch.abs(gt), dim=(-2, -1))
        maer = torch.mean(mae / nrms, dim=0)  # mean over batch dim

        ssim = ssim_func(
            img1=gt.to(DEVICE),
            img2=pred.to(DEVICE),
        )
        ssim = torch.mean(ssim, dim=(0, -2, -1))  # mean over batch, x, and y
        ssim = 1.0 - ssim

        assert ssim.shape == maer.shape

        df_errors[f"{kind}_MAER"] = maer.numpy()
        df_errors[f"{kind}_SSIMLoss"] = ssim.cpu().numpy()

    return df_errors


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)


if __name__ == "__main__":
    config_path = parser.parse_args().config_path
    with open(config_path) as file:
        config = yaml.safe_load(file)

    experiment_name = config_path.split("/")[-2]
    config_name = os.path.basename(config_path).split(".")[0]

    result_dir = f"{ROOT_DIR}/data/pytorch/EnKF/{experiment_name}/{config_name}"
    os.makedirs(result_dir, exist_ok=True)
    logger.addHandler(FileHandler(f"{result_dir}/log.txt"))

    logger.info(f"config = {json.dumps(config, indent=2)}")

    USED_DATA_KIND = config["USED_DATA_KIND"]
    NUM_SIMULATIONS = config["NUM_SIMULATIONS"]
    N_ENS = config["N_ENS"]
    LOCALIZE_DX = config["LOCALIZE_DX"]
    SYS_NOISE_FACTOR = config["SYS_NOISE_FACTOR"]
    INIT_SYS_NOISE_FACTOR = config["INIT_SYS_NOISE_FACTOR"]
    OBS_PERTURB_STD = config["OBS_PERTURB_STD"]

    LOCALIZE_DY = LOCALIZE_DX
    HR_CFD_CONFIG["ne"] = N_ENS
    HR_CFD_CONFIG["n_ens"] = N_ENS

    set_seeds(SEED, use_deterministic=True)

    cov_file_path = f"{ROOT_DIR}/data/pytorch/hr_sys_noise_covs.pickle"
    all_covs = read_pickle(cov_file_path)
    loc = torch.zeros(all_covs.shape[-1], dtype=torch.float64)
    sys_noise_generators = [MultivariateNormal(loc, cov) for cov in all_covs]
    logger.info(f"len of noise generator = {len(sys_noise_generators)}")

    all_hr_omegas = load_hr_data(
        root_dir=ROOT_DIR,
        cfd_dir_name=CFD_DIR_NAME,
        train_valid_test_ratios=TRAIN_VALID_TEST_RATIOS,
        kind=USED_DATA_KIND,
        num_hr_omega_sets=NUM_SIMULATIONS,
    )
    assert (
        len(sys_noise_generators)
        == int(all_hr_omegas.shape[1] / ASSIMILATION_PERIOD) + 1
    )

    all_hr_omegas = all_hr_omegas[:, INITIAL_TIME_INDEX:]
    sys_noise_generators = sys_noise_generators[
        INITIAL_TIME_INDEX // ASSIMILATION_PERIOD :
    ]

    logger.info(f"All hr omega shape == {all_hr_omegas.shape}")

    torch_rand_generator = torch.Generator().manual_seed(SEED)

    localization_matrix = calc_localization_matrix(
        nx=HR_NX, ny=HR_NY, d_x=LOCALIZE_DX, d_y=LOCALIZE_DY
    ).to(DEVICE)

    all_init_hr_omegas = get_initial_hr_omega()
    assert all_init_hr_omegas.shape == (NUM_SIMULATIONS, HR_NX, HR_NY)

    max_time_index = all_hr_omegas.shape[1]

    for obs_grid_interval in OBS_GRID_INTERVALS:
        try:
            logger.info("\n--------------------------------------------------")
            logger.info(f"obs_grid_interval = {obs_grid_interval}")
            logger.info("--------------------------------------------------")

            obs_matrix_sampler = ObsMatrixSampler(obs_grid_interval)

            all_enkf_means = []

            for i_sim in range(NUM_SIMULATIONS):
                logger.info(f"i_ens = {i_sim}")
                start_time = time.time()

                # The index 0 means the initial time
                hr_model = create_initialized_hr_model(
                    N_ENS, all_init_hr_omegas[i_sim].clone()
                )

                hr_enkfs = []
                cnt_assim = 0

                for i_cycle in tqdm(range(max_time_index)):

                    # Data assimilation
                    if i_cycle > 0 and i_cycle % ASSIMILATION_PERIOD == 0:
                        obs = all_hr_omegas[i_sim, i_cycle]
                        noise = np.random.normal(
                            loc=0.0, scale=OBS_NOISE_STD, size=obs.shape
                        )
                        obs = obs + torch.from_numpy(noise).to(torch.float64)

                        # This method returns forecast conv
                        hr_assimilate_with_existing_data(
                            hr_omega=obs.to(DEVICE),
                            hr_ens_model=hr_model,
                            obs_matrix=obs_matrix_sampler(),
                            obs_noise_std=OBS_PERTURB_STD,
                            inflation=INFLATION,
                            rand_generator=torch_rand_generator,
                            localization_matrix=localization_matrix,
                        )

                    # Store data "after" assimilation
                    hr_enkfs.append(hr_model.omega.cpu().clone())

                    # Add additive system noise
                    if i_cycle == 0 or (
                        INFLATION == 1.0 and i_cycle % ASSIMILATION_PERIOD == 0
                    ):
                        noise = sys_noise_generators[cnt_assim].sample([N_ENS])
                        noise = noise.reshape(N_ENS, HR_NX, HR_NY)
                        noise = noise - torch.mean(noise, dim=0, keepdims=True)
                        assert noise.shape == hr_model.omega.shape

                        factor = (
                            INIT_SYS_NOISE_FACTOR if i_cycle == 0 else SYS_NOISE_FACTOR
                        )
                        omega = hr_model.omega + factor * noise.to(DEVICE)

                        hr_model.initialize(t0=hr_model.t, omega0=omega)
                        hr_model.calc_grid_data()
                        cnt_assim += 1

                    hr_model.time_integrate(dt=HR_DT, nt=HR_NT, hide_progress_bar=True)
                    hr_model.calc_grid_data()

                # Stack along time dim
                hr_enkfs = torch.stack(hr_enkfs, dim=1)

                # Add an ensemble mean
                all_enkf_means.append(torch.mean(hr_enkfs, dim=0))

                end_time = time.time()
                logger.info(f"Elapsed Time = {end_time - start_time} [sec]\n")

            # Stack along the simulation dim
            all_enkf_means = torch.stack(all_enkf_means, dim=0)
            np.save(f"{result_dir}/og{obs_grid_interval:02}.npy", all_enkf_means)

            df_errors = calc_errors(all_gt=all_hr_omegas, all_enkf=all_enkf_means)
            df_errors.to_csv(f"{result_dir}/og{obs_grid_interval:02}.csv", index=False)

        except Exception as e:
            logger.info("\n*********************************************************")
            logger.info("Error")
            logger.info("*********************************************************\n")
            logger.error(e)
            logger.error(traceback.format_exc())