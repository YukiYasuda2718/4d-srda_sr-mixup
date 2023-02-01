import typing

import numpy as np
import torch


def runge_kutta_2nd_order(
    *,
    dt: float,
    t: float,
    x: typing.Union[np.ndarray, torch.Tensor],
    dxdt: typing.Callable[
        [float, typing.Union[np.ndarray, torch.Tensor]], typing.Union[np.ndarray, torch.Tensor]
    ]
) -> typing.Union[np.ndarray, torch.Tensor]:

    k1 = dxdt(t, x)
    k2 = dxdt(t + dt / 2, x + k1 * dt / 2)

    return x + k2 * dt