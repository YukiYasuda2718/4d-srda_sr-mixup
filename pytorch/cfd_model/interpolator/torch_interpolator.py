import torch
import torch.nn.functional as F


def interpolate(
    data: torch.Tensor, nx: int, ny: int, mode: str = "bicubic"
) -> torch.Tensor:
    assert data.ndim == 3

    return F.interpolate(
        data.unsqueeze(1),  # add channel dim
        size=(nx, ny),
        mode=mode,
        align_corners=None if mode == "nearest" else False,
    ).squeeze(1)


def interpolate_time_series(
    data: torch.Tensor, nx: int, ny: int, mode: str = "bicubic"
) -> torch.Tensor:
    assert data.ndim == 4  # ens, time, x, y, dims or time, ens, x, y

    return F.interpolate(
        data,
        size=(nx, ny),
        mode=mode,
        align_corners=None if mode == "nearest" else False,
    )