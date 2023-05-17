import argparse
import datetime
import os
import pathlib
import sys
from logging import INFO, WARNING, FileHandler, StreamHandler, getLogger

import numpy as np
import pandas as pd
import torch
from cfd_model.cfd.periodic_channel_domain import TorchSpectralModel2D
from cfd_model.initialization.periodic_channel_jet_initializer import (
    calc_init_omega,
    calc_init_perturbation_hr_omegas_for_only_low_wavenumber,
    calc_jet_forcing,
)
from src.utils import set_seeds
from tqdm.notebook import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)


LR_NX = 32
LR_NY = 17
LR_DT = 5e-4
LR_NT = 500

HR_NX = 1024
HR_NY = 513
HR_DT = LR_DT / 32.0
HR_NT = LR_NT * 32

LR_KX_CUTOFF = 256
LR_KY_CUTOFF = 128

assert HR_DT * HR_NT == LR_DT * LR_NT

N_CYCLES = 100
N_ENSEMBLES = 1

Y0 = np.pi / 2.0
SIGMA = 0.4
U0 = 3.0
TAU0 = 0.3
PERTUB_NOISE = 0.0025

BETA = 0.1
COEFF_LINEAR_DRAG = 1e-2
ORDER_DIFFUSION = 2
HR_COEFF_DIFFUSION = 2e-6


ROOT_DIR = str((pathlib.Path(os.environ["PYTHONPATH"]) / "..").resolve())

DF_SEEDS = pd.read_csv(f"{ROOT_DIR}/pytorch/config/cfd_seeds/seeds01.csv").set_index(
    "SimulationNumber"
)

EXPERIMENT_DIR = f"{ROOT_DIR}/data/pytorch/CFD/jet09"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)


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

    hr_perturb = calc_init_perturbation_hr_omegas_for_only_low_wavenumber(
        nx=HR_NX,
        ny=HR_NY,
        ne=n_ensembles,
        noise_amp=PERTUB_NOISE,
        seed=seed,
        lr_kx_cutoff=LR_KX_CUTOFF,
        lr_ky_cutoff=LR_KY_CUTOFF,
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


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, required=True)

if __name__ == "__main__":

    DEVICE = parser.parse_args().device
    logger.addHandler(
        FileHandler(
            f"{EXPERIMENT_DIR}/log_{datetime.datetime.utcnow():%Y%m%dT%H%M%S}.txt"
        )
    )
    for i_seed in range(9999, 9949, -1):
        try:
            seed = DF_SEEDS.loc[i_seed, "Seed0"]
            set_seeds(seed, use_deterministic=True)
            logger.info(f"i_seed = {i_seed}, seed = {seed}")

            result_dir = f"{EXPERIMENT_DIR}/seed{i_seed:05}"
            if os.path.exists(result_dir):
                logger.info("Result dir already exist. So skip.")
                continue

            os.makedirs(result_dir, exist_ok=False)

            hr_model = make_and_initialize_hr_model(seed=seed, n_ensembles=N_ENSEMBLES)

            for i_cycle in tqdm(range(N_CYCLES + 1), total=N_CYCLES + 1):
                output_omega = hr_model.omega.cpu().to(torch.float32).numpy()

                t = str(np.round(hr_model.t, 2)).replace(".", "p")
                output_path = f"{result_dir}/omega_i{i_cycle:03}_t{t}.npy"

                assert not os.path.exists(output_path)
                np.save(output_path, output_omega)
                logger.info(f"ouput was made: {output_path}")

                hr_model.time_integrate(dt=HR_DT, nt=HR_NT, hide_progress_bar=False)
                hr_model.calc_grid_data()
        except Exception as e:
            logger.error(e)