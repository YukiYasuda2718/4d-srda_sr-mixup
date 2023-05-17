import typing

import torch
from cfd_model.fft.abstract_fft_calculator import AbstractFftCalculator


def reflect(z: torch.Tensor, is_odd: bool = True) -> torch.Tensor:
    reflected = z.flip(dims=(-1,))[..., 1:-1]
    if is_odd:
        reflected *= -1
    return torch.cat([z, reflected], dim=-1)


def get_wavenumber(idx: int, total_num: int) -> int:
    if idx <= total_num // 2:
        return idx
    return idx - total_num


def get_grid_index(wavenumber: int, total_num: int) -> int:
    if wavenumber >= 0:
        return wavenumber
    return wavenumber + total_num


class TorchFftCalculator(AbstractFftCalculator):
    def __init__(
        self,
        nx: int,
        ny: int,
        norm: typing.Literal["backward", "ortho", "forward"] = "ortho",
        dtype: torch.dtype = torch.complex128,
        device: str = "cpu",
        beta: float = 0.0,
    ):
        assert ny % 2 == 1

        self.nx = nx
        self.ny = ny

        full_ny = (ny - 1) * 2
        self.full_ny = full_ny

        half_ny = full_ny // 2 + 1
        self.half_ny = half_ny

        self.norm = norm
        self.dtype = dtype
        self.beta = beta

        self.jkx = torch.zeros((1, nx, half_ny), dtype=dtype, device=device)
        self.jky = torch.zeros((1, nx, half_ny), dtype=dtype, device=device)
        self.k2 = torch.zeros((1, nx, half_ny), dtype=dtype, device=device)

        self.coef_u = torch.zeros((1, nx, half_ny), dtype=dtype, device=device)
        self.coef_v = torch.zeros((1, nx, half_ny), dtype=dtype, device=device)

        # filter to remove aliasing errors
        self.filter = torch.ones((1, nx, half_ny), dtype=dtype, device=device)

        for i in range(nx):
            kx = get_wavenumber(i, nx)
            for j in range(half_ny):
                ky = get_wavenumber(j, full_ny)

                self.jkx[0, i, j] = kx * 1j
                self.jky[0, i, j] = ky * 1j

                k2 = kx**2 + ky**2
                self.k2[0, i, j] = k2

                if k2 != 0:
                    self.coef_u[0, i, j] = ky * 1j / k2
                    self.coef_v[0, i, j] = -kx * 1j / k2

                if abs(kx) > nx // 3 or abs(ky) > full_ny // 3:
                    self.filter[0, i, j] = 0.0

    def _apply_fft2(self, grid_data: torch.Tensor) -> torch.Tensor:
        assert grid_data.ndim == 3  # batch, x, and y dims
        assert grid_data.shape[-2] == self.nx
        assert grid_data.shape[-1] == self.full_ny  # assuming reflected data

        spec = torch.fft.rfft2(grid_data, dim=(-2, -1), norm=self.norm)

        # Filter is applied to remove aliasing errors.
        return spec * self.filter

    def apply_fft2(self, grid_data: torch.Tensor, is_odd: bool = True) -> torch.Tensor:
        reflected = reflect(grid_data, is_odd)
        return self._apply_fft2(reflected)

    def _apply_ifft2(self, spec_data: torch.Tensor) -> torch.Tensor:
        assert spec_data.ndim == 3  # batch, x, and y dims
        assert spec_data.shape[-2] == self.nx
        assert spec_data.shape[-1] == self.half_ny

        return torch.fft.irfft2(spec_data, dim=(-2, -1), norm=self.norm)

    def apply_ifft2(self, spec_data: torch.Tensor) -> torch.Tensor:
        grid_data = self._apply_ifft2(spec_data)
        return grid_data[..., : self.ny]  # returning half because of reflection

    def calculate_uv_from_omega(
        self, grid_omega: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        assert grid_omega.ndim == 3  # batch, x, and y dims
        assert grid_omega.shape[-2] == self.nx
        assert grid_omega.shape[-1] == self.ny

        reflected = reflect(grid_omega)

        spec = self._apply_fft2(reflected)
        u = self._apply_ifft2(spec * self.coef_u)
        v = self._apply_ifft2(spec * self.coef_v)

        return u[..., : self.ny], v[..., : self.ny]

    def calculate_omega_from_uv(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        reflected_u = reflect(u, is_odd=False)
        reflected_v = reflect(v, is_odd=True)

        spec_u = self._apply_fft2(reflected_u)
        spec_v = self._apply_fft2(reflected_v)

        spec = self.jkx * spec_v - self.jky * spec_u

        return self.apply_ifft2(spec)

    def calculate_advection_from_spec_omega(
        self, spec_omega: torch.Tensor, apply_fft: bool = True
    ) -> torch.Tensor:

        d_omega_dx = self._apply_ifft2(spec_omega * self.jkx)
        d_omega_dy = self._apply_ifft2(spec_omega * self.jky)

        u = self._apply_ifft2(spec_omega * self.coef_u)
        v = self._apply_ifft2(spec_omega * self.coef_v)

        adv = u * d_omega_dx + v * d_omega_dy

        if self.beta != 0.0:
            adv += self.beta * v

        adv = adv[..., : self.ny]

        if not apply_fft:
            return adv

        adv = reflect(adv)

        return self._apply_fft2(adv)

    def calculate_advection_from_grid_omega(
        self, grid_omega: torch.Tensor, apply_fft: bool = True
    ) -> torch.Tensor:

        assert grid_omega.ndim == 3  # batch, x, and y dims
        assert grid_omega.shape[-2] == self.nx
        assert grid_omega.shape[-1] == self.ny

        reflected = reflect(grid_omega)
        spec = self._apply_fft2(reflected)
        return self.calculate_advection_from_spec_omega(spec, apply_fft)