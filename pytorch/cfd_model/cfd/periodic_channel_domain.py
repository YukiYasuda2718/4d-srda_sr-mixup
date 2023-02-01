import sys
import typing
from logging import getLogger

import torch
from cfd_model.fft.periodic_channel_domain import TorchFftCalculator
from cfd_model.time_integration.runge_kutta import runge_kutta_2nd_order
from cfd_model.cfd.abstract_cfd_model import AbstractCfdModel

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger()


class TorchSpectralModel2D(AbstractCfdModel):
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        beta: float,
        coeff_linear_drag: float,
        coeff_diffusion: float,
        order_diffusion: int,
        device: str,
        norm: typing.Literal["backward", "ortho", "forward"] = "ortho",
        dtype: torch.dtype = torch.complex128,
        **kwargs,
    ):
        assert ny % 2 == 1
        assert coeff_linear_drag > 0
        assert coeff_diffusion > 0
        assert isinstance(order_diffusion, int) and order_diffusion > 0

        self.nx = nx
        self.ny = ny
        self.device = device

        logger.info("Periodic channel model parameters:")
        logger.info(f"nx = {nx}, ny = {ny}")
        logger.info(f"coeff_linear_drag = {coeff_linear_drag}")
        logger.info(f"coeff_diffusion = {coeff_diffusion}")
        logger.info(f"order_diffusion = {order_diffusion}")
        logger.info(f"beta = {beta}")
        logger.info(f"norm = {norm}")

        self.__fft = TorchFftCalculator(
            nx=nx, ny=ny, norm=norm, dtype=dtype, device=device, beta=beta
        )
        self.__time_integrator = runge_kutta_2nd_order

        self.__diffusion_operator = coeff_linear_drag + coeff_diffusion * torch.pow(
            self.__fft.k2, order_diffusion
        )

        self.ne = 0
        self.t = None
        self.spec_omega = None
        self.omega = None
        self.u = None
        self.v = None
        self.forcing = None

    @property
    def time(self):
        return self.t

    @property
    def vorticity(self):
        return self.omega

    @property
    def state_size(self):
        return self.ne, self.nx, self.ny

    def _time_derivative(self, t: float, spec_omega: torch.Tensor) -> torch.Tensor:
        spec_adv = self.__fft.calculate_advection_from_spec_omega(
            spec_omega, apply_fft=True
        )

        if self.forcing is None:
            return -spec_adv - self.__diffusion_operator * spec_omega
        else:
            return -spec_adv - self.__diffusion_operator * spec_omega + self.forcing

    def calc_grid_data(self):
        self.omega = self.__fft.apply_ifft2(self.spec_omega)
        self.u, self.v = self.__fft.calculate_uv_from_omega(self.omega)

    def initialize(self, t0: float, omega0: torch.Tensor, forcing: torch.Tensor = None):
        assert isinstance(t0, float)
        assert omega0.ndim == 3  # batch, x, y dims
        assert omega0.shape[-2] == self.nx
        assert omega0.shape[-1] == self.ny

        self.ne = omega0.shape[0]
        self.t = t0
        self.spec_omega = self.__fft.apply_fft2(omega0.to(self.device))

        if forcing is not None:
            self.forcing = self.__fft.apply_fft2(forcing.to(self.device))

    def get_forcing(self):
        if self.forcing is None:
            return None
        return self.__fft.apply_ifft2(self.forcing)

    def time_integrate(self, dt: float, nt: int, hide_progress_bar: bool = False):
        for _ in tqdm(range(nt), disable=hide_progress_bar):
            self.spec_omega = self.__time_integrator(
                dt=dt, t=self.t, x=self.spec_omega, dxdt=self._time_derivative
            )
            self.t += dt

            has_nan = torch.max(torch.isnan(self.spec_omega)).item()
            if has_nan:
                raise Exception(f"Nan is found at t = {self.t:.4f}")