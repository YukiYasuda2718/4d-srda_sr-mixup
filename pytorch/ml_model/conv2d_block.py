import typing
from logging import getLogger

import torch
import torch.nn as nn

logger = getLogger()


class EncoderBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int = 3,
        bias: bool = False,
        type_down_sample: typing.Literal["conv"] = "conv",
        num_layers: int = 2,
    ):
        super(EncoderBlock, self).__init__()

        assert num_layers >= 2

        p = kernel_size // 2

        if type_down_sample == "conv":
            self.down = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=p,
                    bias=bias,
                ),
                nn.ReLU(),
            )
        else:
            raise Exception(f"Downsampling type of {type_down_sample} is not supported")

        convs = []
        for _ in range(num_layers - 1):
            convs.append(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=p,
                    bias=bias,
                )
            )
            convs.append(nn.ReLU())

        self.convs = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        return self.convs(y)


class PixelShuffleBlock(nn.Module):
    def __init__(self, *, in_channels: int, kernel_size: int = 3, bias: bool = False):
        super(PixelShuffleBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            4 * in_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

        self.act = nn.LeakyReLU()

        self.upsample = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.act(y)
        return self.upsample(y)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = False,
        type_up_sample: typing.Literal["pixel_shuffle"] = "pixel_shuffle",
        num_layers: int = 2,
    ):
        super(DecoderBlock, self).__init__()

        assert num_layers >= 2

        if type_up_sample == "pixel_shuffle":
            self.up = PixelShuffleBlock(
                in_channels=in_channels, kernel_size=kernel_size, bias=bias
            )
        else:
            raise Exception(f"Upsampling type of {type_up_sample} is not supported")

        convs = []
        for i in range(num_layers - 1):
            convs.append(
                nn.Conv2d(
                    (in_channels if i == 0 else out_channels),
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                    bias=bias,
                )
            )
            convs.append(nn.LeakyReLU())

        self.convs = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.up(x)
        return self.convs(y)