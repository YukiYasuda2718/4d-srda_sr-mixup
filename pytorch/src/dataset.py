import copy
import functools
import glob
import itertools
import os
import random
import re
import sys
import typing
from collections import OrderedDict
from logging import getLogger

import numpy as np
import torch
from torch.utils.data import Dataset

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


logger = getLogger()


def generate_is_obs_and_obs_matrix(
    *,
    nx: int,
    ny: int,
    init_index_x: int,
    init_index_y: int,
    interval_x: int,
    interval_y: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    assert 0 <= init_index_x <= interval_x - 1
    assert 0 <= init_index_y <= interval_y - 1

    is_obs = torch.zeros(nx, ny, dtype=dtype)
    is_obs[init_index_x::interval_x, init_index_y::interval_y] = 1.0

    obs_indices = is_obs.reshape(-1)
    obs_indices = torch.where(obs_indices == 1.0)[0]

    num_obs = len(obs_indices)

    obs_matrix = torch.zeros(num_obs, nx * ny, dtype=dtype)

    for i, j in enumerate(obs_indices):
        obs_matrix[i, j] = 1.0

    p = 100 * torch.sum(obs_matrix).item() / (nx * ny)
    logger.debug(f"observatio prob = {p} [%]")

    return is_obs.to(device), obs_matrix.to(device)


class DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling(Dataset):
    def __init__(
        self,
        *,
        data_dirs: typing.List[str],
        lr_kind_names: typing.List[str],
        lr_time_interval: int,
        obs_time_interval: int,
        obs_grid_interval: int,
        obs_noise_std: float,
        use_observation: bool,
        vorticity_bias: float,
        vorticity_scale: float,
        use_ground_truth_clamping: bool,
        beta_dist_alpha: float,
        beta_dist_beta: float,
        use_mixup: bool,
        use_mixup_init_time: bool,
        use_lr_forecast: bool = True,
        missing_value: float = 0.0,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
        nx: int = 128,
        ny: int = 65,
        max_ensemble: int = 20,
        is_output_only_last: bool = False,
        is_last_obs_missing: bool = False,
        min_start_time_index: int = 12,
        max_start_time_index: int = 96,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        logger.info(
            "Dataset is DatasetMakingObsInsideTimeseriesSplittedWithMixupRandomSampling"
        )

        if obs_grid_interval <= 0:
            assert not use_observation
        else:
            assert use_observation

        if use_mixup_init_time:
            assert use_mixup

        self.dtype = dtype
        self.lr_time_interval = lr_time_interval
        self.obs_time_interval = obs_time_interval
        self.obs_grid_interval = obs_grid_interval if obs_grid_interval > 0 else 8
        self.obs_noise_std = obs_noise_std
        self.use_obs = use_observation
        self.vorticity_bias = vorticity_bias
        self.vorticity_scale = vorticity_scale
        self.beta_dist_alpha = beta_dist_alpha
        self.beta_dist_beta = beta_dist_beta
        self.use_mixup = use_mixup
        self.use_mixup_init_time = use_mixup_init_time
        self.use_lr_forecast = use_lr_forecast
        self.missing_value = missing_value
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.use_gt_clamp = use_ground_truth_clamping
        self.is_output_only_last = is_output_only_last
        self.is_last_obs_missing = is_last_obs_missing
        self.min_start_time_index = min_start_time_index
        self.max_start_time_index = max_start_time_index

        logger.info(f"LR time interval = {self.lr_time_interval}")
        if obs_grid_interval > 0:
            logger.info("Obs grid interval is not replaced with 8.")
        logger.info(f"Obs time interval = {self.obs_time_interval}")
        logger.info(f"Obs grid interval = {self.obs_grid_interval}")
        logger.info(f"Obs noise std = {self.obs_noise_std}")
        logger.info(f"Use observation = {self.use_obs}")
        logger.info(f"Bias = {self.vorticity_bias}, Scale = {self.vorticity_scale}")
        logger.info(f"beta_dist_alpha = {self.beta_dist_alpha}")
        logger.info(f"beta_dist_beta = {self.beta_dist_beta}")
        logger.info(f"use_mixup = {self.use_mixup}")
        logger.info(f"use_mixup_init_time = {self.use_mixup_init_time}")
        logger.info(f"use_lr_forecast = {self.use_lr_forecast}")
        logger.info(f"Use clamp for ground truth = {self.use_gt_clamp}")
        logger.info(f"Clamp: min = {self.clamp_min}, max = {self.clamp_max}")
        logger.info(f"missing value = {self.missing_value}")
        logger.info(f"is_output_only_last = {self.is_output_only_last}")
        logger.info(f"is_last_obs_missing = {self.is_last_obs_missing}")
        logger.info(f"min_start_time_index = {self.min_start_time_index}")
        logger.info(f"max_start_time_index = {self.max_start_time_index}")

        self._set_hr_file_paths(data_dirs)
        self.str_seeds = list(
            set(map(lambda p: os.path.basename(p).split("_")[0], self.hr_file_paths))
        )

        self.lr_kind_names = copy.deepcopy(lr_kind_names)
        _lst = "\n  ".join(self.lr_kind_names)
        logger.info(f"lr_kind_names = {_lst}")

        self.max_ensemble = max_ensemble
        logger.info(f"Max ensemble = {self.max_ensemble}")

        self._load_all_lr_data_at_init_time()

        self.is_obses = []
        self.obs_matrices = []
        ratio_mean = []

        for init_x in tqdm(range(self.obs_grid_interval)):
            for init_y in range(self.obs_grid_interval):
                is_obs, obs_mat = generate_is_obs_and_obs_matrix(
                    nx=nx,
                    ny=ny,
                    init_index_x=init_x,
                    init_index_y=init_y,
                    interval_x=self.obs_grid_interval,
                    interval_y=self.obs_grid_interval,
                    dtype=self.dtype,
                )
                self.is_obses.append(is_obs)
                self.obs_matrices.append(obs_mat)
                ratio_mean.append(torch.mean(is_obs).item())
        ratio_mean = sum(ratio_mean) / len(ratio_mean)
        logger.warning(
            f"Observation interval = {self.obs_grid_interval}, Observation grid ratio = {ratio_mean}"
        )

    def _set_hr_file_paths(self, data_dirs: typing.List[str]):
        lst_hr_file_paths = [
            glob.glob(f"{dir_path}/*_hr_omega_*.npy") for dir_path in data_dirs
        ]
        hr_file_paths = functools.reduce(lambda l1, l2: l1 + l2, lst_hr_file_paths, [])

        extracted_paths = []
        for path in hr_file_paths:
            if not path.endswith("_00.npy"):
                continue

            grps = re.match(
                r"seed(\d+)_start(\d+)_end", os.path.basename(path)
            ).groups()

            start_idx = int(grps[1])
            if start_idx >= self.max_start_time_index:
                continue

            is_contained = True
            for _idx in range(self.min_start_time_index+1):
                if f"start{_idx:02}_" in path:
                    is_contained = False
                    break
            if is_contained:
                extracted_paths.append(path)

        self.hr_file_paths = extracted_paths

    def __len__(self) -> int:
        return len(self.hr_file_paths)

    def _load_all_lr_data_at_init_time(self):
        self.dict_all_lr_data_at_init_time = {}

        for lr_kind in self.lr_kind_names:
            self.dict_all_lr_data_at_init_time[lr_kind] = {}
            for hr_path in tqdm(self.hr_file_paths):

                for ens_idx in range(self.max_ensemble):
                    lr_path = hr_path.replace("hr_omega", lr_kind).replace(
                        "_00.npy", f"_{ens_idx:02}.npy"
                    )

                    key = re.search(
                        r"start\d+_end\d+", os.path.basename(lr_path)
                    ).group()
                    if key not in self.dict_all_lr_data_at_init_time[lr_kind]:
                        self.dict_all_lr_data_at_init_time[lr_kind][key] = []

                    lr = np.load(lr_path)
                    self.dict_all_lr_data_at_init_time[lr_kind][key].append(
                        {"data": lr[0], "path": lr_path}
                    )
                    # `0` means the initial time

    def _load_np_data(
        self, path_idx: int, ens_idx: int = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        hr_path = self.hr_file_paths[path_idx]
        if ens_idx is None:
            ens_idx = random.randint(0, self.max_ensemble - 1)
            logger.debug(f"Random ens_idx = {ens_idx}")
        hr_path = hr_path.replace("_00.npy", f"_{ens_idx:02}.npy")
        hr = np.load(hr_path)

        lr_kind = np.random.choice(self.lr_kind_names, size=1, replace=False)[0]
        target_lr_path = hr_path.replace("hr_omega", lr_kind)
        target_lr = np.load(target_lr_path)

        key = re.search(r"start\d+_end\d+", os.path.basename(target_lr_path)).group()
        # Pass target_lr at the initial time, which has the index of `0`
        source_lr_path = self._get_similar_source_lr_path(lr_kind, key, target_lr[0])
        source_lr = np.load(source_lr_path)

        hr = torch.from_numpy(hr).to(self.dtype)
        target_lr = torch.from_numpy(target_lr).to(self.dtype)
        source_lr = torch.from_numpy(source_lr).to(self.dtype)

        logger.debug(f"Selected lr kind = {lr_kind}")
        logger.debug(f"hr = {hr_path}")
        logger.debug(f"target_lr={target_lr_path}")
        logger.debug(f"lr key = {key}")
        logger.debug(f"source_lr={source_lr_path}")

        return target_lr, source_lr, hr

    def _get_similar_source_lr_path(
        self, lr_kind: str, key: str, target_lr: np.ndarray
    ):
        all_lrs = self.dict_all_lr_data_at_init_time[lr_kind][key]

        min_path, min_norm = None, np.inf

        for i in list(
            set(np.random.randint(0, len(all_lrs), size=2 * self.max_ensemble))
        )[: self.max_ensemble]:
            data = all_lrs[i]["data"]
            path = all_lrs[i]["path"]
            assert data.shape == target_lr.shape

            norm = np.mean((data - target_lr) ** 2)
            if 0 < norm < min_norm:
                min_norm = norm
                min_path = path
                logger.debug(f"norm = {norm}, path = {min_path}")

        return min_path

    def _extract_observation_without_noise(self, hr_omega: torch.Tensor):

        i = random.randint(0, len(self.is_obses) - 1)
        is_obs = self.is_obses[i]
        assert is_obs.shape == hr_omega.shape[1:]

        is_obs = torch.broadcast_to(is_obs, hr_omega.shape)
        logger.debug(f"index of is_obs = {i}")

        obs = torch.full_like(hr_omega, torch.nan)
        _tmp = torch.where(is_obs > 0, hr_omega, obs)
        obs[:: self.obs_time_interval] = _tmp[:: self.obs_time_interval]

        return obs

    def _preprocess(
        self, data: torch.Tensor, use_clipping: bool = False
    ) -> torch.Tensor:

        # Add channel dim, drop the last index along y, and standardize
        ret = (data[:, None, :, :-1] - self.vorticity_bias) / self.vorticity_scale

        if use_clipping:
            ret = torch.clamp(ret, min=self.clamp_min, max=self.clamp_max)

        # time, channel, x, y --> time, channel, y, x dims
        return ret.permute(0, 1, 3, 2)

    def get_pair_of_lr(
        self, path_idx: int, ens_idx: int = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        target_lr, source_lr, _ = self._load_np_data(path_idx, ens_idx)
        return target_lr, source_lr

    def _getitem(
        self, path_idx: int, ens_idx: int = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        target_lr, source_lr, gt = self._load_np_data(path_idx, ens_idx)
        obs = self._extract_observation_without_noise(gt)

        if self.obs_noise_std > 0:
            noise = np.random.normal(loc=0, scale=self.obs_noise_std, size=obs.shape)
            obs = obs + torch.from_numpy(noise).to(self.dtype)

        target_lr = self._preprocess(target_lr, use_clipping=True)
        source_lr = self._preprocess(source_lr, use_clipping=True)
        obs = self._preprocess(obs, use_clipping=True)
        gt = self._preprocess(gt, use_clipping=self.use_gt_clamp)

        if self.use_obs:
            is_obs = ~torch.isnan(obs)
            obs = torch.nan_to_num(obs, nan=self.missing_value)
        else:
            obs = torch.full_like(obs, fill_value=self.missing_value)

        if self.use_mixup:
            source_prob = np.random.beta(
                a=self.beta_dist_alpha, b=self.beta_dist_beta, size=1
            )[0]
            logger.debug(f"similar source_prob = {source_prob}")

            if self.use_mixup_init_time:
                lr = target_lr
                lr[0] = source_prob * source_lr[0] + (1 - source_prob) * lr[0]
            else:
                lr = source_prob * source_lr + (1 - source_prob) * target_lr

        else:
            lr = target_lr

        if self.is_last_obs_missing:
            obs[-1] = self.missing_value

        lr = lr[:: self.lr_time_interval]
        if not self.use_lr_forecast:
            lr = torch.full_like(lr, fill_value=self.missing_value)

        if self.is_output_only_last:
            return lr[-1], obs[-1], is_obs[-1], gt[-1]

        return lr, obs, gt

    def __getitem__(
        self, idx: int
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._getitem(path_idx=idx)

    def get_specified_item(
        self,
        i_batch: int,
        i_cycle: int,
        start_time_index: int = 16,
        is_specified_end: bool = True,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        dict_hr_paths = OrderedDict(
            [
                (k, sorted(g))
                for k, g in itertools.groupby(
                    self.hr_file_paths,
                    lambda p: os.path.basename(os.path.dirname(p)),
                )
            ]
        )

        target_paths = None
        for i, ps in enumerate(dict_hr_paths.values()):
            # Each file contains 20 random members
            if i == i_batch // 20:
                target_paths = ps
                break

        # ensemble index in target 20 members.
        i_ensemble = i_batch - i_batch // 20 * 20

        path = None
        for p in target_paths:
            if is_specified_end and f"end{i_cycle + start_time_index:02}" in p:
                path = p
                break
            if (not is_specified_end) and f"start{i_cycle + start_time_index:02}" in p:
                path = p
                break
        print(path)

        i_path = self.hr_file_paths.index(path)

        logger.info(f"Target path = {path}")
        logger.info(f"i_path = {i_path}, i_ensemble = {i_ensemble}")

        return self._getitem(i_path, i_ensemble)