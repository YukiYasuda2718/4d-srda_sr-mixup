import numpy as np
import torch
from cfd_model.fft.periodic_channel_domain import (
    TorchFftCalculator,
    get_wavenumber,
    get_grid_index,
)


class LowPassFilter:
    def __init__(self, *, nx_lr: int, ny_lr: int, nx_hr: int, ny_hr: int, device: str):
        assert nx_lr < nx_hr
        assert ny_lr < ny_hr

        self.device = device

        self.nx_lr = nx_lr
        self.ny_lr = ny_lr
        self.full_ny_lr = (ny_lr - 1) * 2
        self.half_ny_lr = self.full_ny_lr // 2 + 1
        self.fft_lr = TorchFftCalculator(
            nx=self.nx_lr, ny=self.ny_lr, device=self.device, norm="ortho"
        )

        self.nx_hr = nx_hr
        self.ny_hr = ny_hr
        self.full_ny_hr = (ny_hr - 1) * 2
        self.half_ny_hr = self.full_ny_hr // 2 + 1
        self.fft_hr = TorchFftCalculator(
            nx=self.nx_hr, ny=self.ny_hr, device=self.device, norm="ortho"
        )

        self.rescale_factor = np.sqrt(
            (self.nx_lr * self.full_ny_lr) / (self.nx_hr * self.full_ny_hr)
        )

    def _truncate(self, hr_spec_data: torch.Tensor) -> torch.Tensor:
        size = hr_spec_data.shape[:-2] + (self.nx_lr, self.ny_lr)
        lr_spec_data = torch.zeros(
            size=size, dtype=hr_spec_data.dtype, device=self.device
        )

        for i in range(self.nx_hr):
            kx = get_wavenumber(i, self.nx_hr)
            for j in range(self.half_ny_hr):
                ky = get_wavenumber(j, self.full_ny_hr)
                if abs(kx) >= self.nx_lr / 2 or abs(ky) >= self.full_ny_lr / 2:
                    continue
                _i = get_grid_index(kx, self.nx_lr)
                _j = get_grid_index(ky, self.full_ny_lr)

                lr_spec_data[..., _i, _j] = hr_spec_data[..., i, j]

        return lr_spec_data * self.rescale_factor

    def apply(self, hr_gird_data: torch.Tensor) -> torch.Tensor:
        assert hr_gird_data.ndim >= 2  # the last two dims = x and y
        assert hr_gird_data.shape[-2] == self.nx_hr
        assert hr_gird_data.shape[-1] == self.ny_hr

        hr_spec_data = self.fft_hr.apply_fft2(hr_gird_data.to(self.device))

        lr_spec_data = self._truncate(hr_spec_data)

        return self.fft_lr.apply_ifft2(lr_spec_data)