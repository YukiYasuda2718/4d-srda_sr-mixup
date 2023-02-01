import random
from logging import getLogger

import numpy as np
import torch

logger = getLogger()


class HrObservationMatrixGenerator:
    def __init__(self, seed: int = 42):
        logger.info(f"obs matrix generator got seed = {seed}")
        self.rnd = random.Random(seed)
        self.obs_matrix = None

    def generate_obs_matrix(
        self,
        *,
        nx: int,
        ny: int,
        obs_interval: int = 34,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        obs_init_index = self.rnd.randint(0, obs_interval - 1)

        obs_indices = np.zeros(nx * ny)
        obs_indices[obs_init_index::obs_interval] = 1.0
        obs_indices = np.where(obs_indices == 1.0)[0]

        num_obs = len(obs_indices)

        obs_matrix = torch.zeros(num_obs, nx * ny, dtype=dtype)

        for i, j in enumerate(obs_indices):
            obs_matrix[i, j] = 1.0

        p = 100 * torch.sum(obs_matrix).item() / (nx * ny)
        logger.debug(f"observatio prob = {p} [%]")

        self.obs_matrix = obs_matrix.to(device)

        return self.obs_matrix

    def generate_projection_matrix(self):

        if self.obs_matrix is None:
            raise Exception()

        return self.obs_matrix.t().mm(self.obs_matrix)