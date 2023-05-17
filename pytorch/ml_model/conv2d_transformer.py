import math
import typing
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_model.conv2d_block import DecoderBlock, EncoderBlock

logger = getLogger()


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        x_feat_channels_0: int,
        x_feat_channels_1: int,
        x_feat_channels_2: int,
        x_feat_channels_3: int,
        o_feat_channels_0: int,
        o_feat_channels_1: int,
        o_feat_channels_2: int,
        o_feat_channels_3: int,
        out_channels: int,
        kernel_size: int = 3,
        num_layers_x_encoder: int = 2,
        num_layers_o_encoder: int = 2,
        bias_x_encoder: bool = False,
        bias_o_encoder: bool = False,
        scale_factor: int = 4,
    ):
        super(Encoder, self).__init__()
        self.scale_factor = scale_factor

        self.x_encoder = nn.Sequential(
            EncoderBlock(
                in_channels=(x_feat_channels_0 + o_feat_channels_0),
                out_channels=x_feat_channels_1,
                kernel_size=kernel_size,
                stride=2,
                bias=bias_x_encoder,
                num_layers=num_layers_x_encoder,
            ),
            EncoderBlock(
                in_channels=x_feat_channels_1,
                out_channels=x_feat_channels_2,
                kernel_size=kernel_size,
                stride=2,
                bias=bias_x_encoder,
                num_layers=num_layers_x_encoder,
            ),
            EncoderBlock(
                in_channels=x_feat_channels_2,
                out_channels=o_feat_channels_3,
                kernel_size=kernel_size,
                stride=2,
                bias=bias_x_encoder,
                num_layers=num_layers_x_encoder,
            ),
            EncoderBlock(
                in_channels=o_feat_channels_3,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                bias=bias_x_encoder,
                num_layers=num_layers_x_encoder,
            ),
        )

        self.o_encoder = nn.Sequential(
            EncoderBlock(
                in_channels=o_feat_channels_0,
                out_channels=o_feat_channels_1,
                kernel_size=kernel_size,
                stride=2,
                bias=bias_o_encoder,
                num_layers=num_layers_o_encoder,
            ),
            EncoderBlock(
                in_channels=o_feat_channels_1,
                out_channels=o_feat_channels_2,
                kernel_size=kernel_size,
                stride=2,
                bias=bias_o_encoder,
                num_layers=num_layers_o_encoder,
            ),
            EncoderBlock(
                in_channels=o_feat_channels_2,
                out_channels=o_feat_channels_3,
                kernel_size=kernel_size,
                stride=2,
                bias=bias_o_encoder,
                num_layers=num_layers_o_encoder,
            ),
            EncoderBlock(
                in_channels=o_feat_channels_3,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                bias=bias_o_encoder,
                num_layers=num_layers_o_encoder,
            ),
        )

    def forward(
        self, x: torch.Tensor, obs: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        y = torch.cat([x, obs], dim=1)  # concat along channel dims
        y = self.x_encoder(y)

        z = self.o_encoder(obs)

        return y, z


class Decoder(nn.Module):
    def __init__(
        self,
        feat_channels_0: int,
        feat_channels_1: int,
        feat_channels_2: int,
        feat_channels_3: int,
        out_channels: int,
        kernel_size: int = 3,
        num_layers: int = 2,
        bias: bool = False,
    ):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            DecoderBlock(
                in_channels=feat_channels_0,
                out_channels=feat_channels_1,
                kernel_size=kernel_size,
                bias=bias,
                num_layers=num_layers,
            ),
            DecoderBlock(
                in_channels=feat_channels_1,
                out_channels=feat_channels_2,
                kernel_size=kernel_size,
                bias=bias,
                num_layers=num_layers,
            ),
            DecoderBlock(
                in_channels=feat_channels_2,
                out_channels=feat_channels_3,
                kernel_size=kernel_size,
                bias=bias,
                num_layers=num_layers,
            ),
            DecoderBlock(
                in_channels=feat_channels_3,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
                num_layers=num_layers,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class TransformerTimeSeriesMappingBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        sequence_length: int,
        bias: bool,
    ):
        super(TransformerTimeSeriesMappingBlock, self).__init__()

        self.input_size = d_model
        self.sequence_length = sequence_length

        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )

        pos = torch.linspace(0, 1, steps=self.sequence_length, dtype=torch.float32)
        pos = pos[None, :, None]  # add batch and channel dims
        self.positions = nn.Parameter(pos, requires_grad=False)

        self.linear = nn.Linear(2 * d_model, d_model - 1, bias=bias)

    def forward(self, xs: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        y = torch.cat([xs, obs], dim=-1)  # cocat along channel
        y = self.linear(y)

        pos = torch.broadcast_to(
            self.positions, size=(y.shape[0], self.sequence_length, 1)
        )
        y = torch.cat([pos, y], dim=-1)  # cocat along channel

        return self.transformer(y)


class ConvTransformerSrDaNet(nn.Module):
    def __init__(
        self,
        *,
        feat_channels_0: int,
        feat_channels_1: int,
        feat_channels_2: int,
        feat_channels_3: int,
        latent_channels: int,
        n_multi_attention_heads: int,
        sequence_length: int,
        input_sampling_interval: int,
        n_transformer_blocks: int,
        use_global_skip_connection: bool,
        in_channels: int = 1,
        out_channels: int = 1,
        scale_factor: int = 4,
        lr_x_size: int = 32,
        lr_y_size: int = 16,
        kernel_size: int = 3,
        bias: bool = False,
        **kwargs,
    ):
        super(ConvTransformerSrDaNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sequence_length = sequence_length
        self.input_sampling_interval = input_sampling_interval
        self.input_sequence_length = math.ceil(
            sequence_length / input_sampling_interval
        )
        logger.info(f"Sequence length = {self.sequence_length}")
        logger.info(f"Input sampling interval = {self.input_sampling_interval}")
        logger.info(f"Input sequence length = {self.input_sequence_length}")
        logger.info(f"bias: {bias}")

        self.scale_factor = scale_factor
        self.lr_x_size = lr_x_size
        self.lr_y_size = lr_y_size
        self.hr_x_size = lr_x_size * self.scale_factor
        self.hr_y_size = lr_y_size * self.scale_factor

        # 16 == 2**4 (encoder has 4 blocks to down sample)
        self.latent_x_size = self.hr_x_size // 16
        self.latent_y_size = self.hr_y_size // 16
        self.latent_channels = latent_channels
        self.latent_dim = self.latent_x_size * self.latent_y_size * self.latent_channels
        self.use_gsc = use_global_skip_connection

        logger.info(f"LR size y = {self.lr_y_size}, x = {self.lr_x_size}")
        logger.info(f"HR size y = {self.hr_y_size}, x = {self.hr_x_size}")
        logger.info(f"Latent size y = {self.latent_y_size}, x = {self.latent_x_size}")
        logger.info(f"latent: dim= {self.latent_dim}, channels= {self.latent_channels}")
        logger.info(f"bias = {bias}")
        logger.info(f"use global skip connection = {self.use_gsc}")

        p = kernel_size // 2

        self.x_feat_extractor = nn.Conv2d(
            in_channels, feat_channels_0, kernel_size, padding=p, bias=bias
        )
        self.o_feat_extractor = nn.Conv2d(
            in_channels, feat_channels_0, kernel_size, padding=p, bias=bias
        )

        self.transformers = []
        for _ in range(n_transformer_blocks):
            self.transformers.append(
                TransformerTimeSeriesMappingBlock(
                    d_model=self.latent_dim,
                    nhead=n_multi_attention_heads,
                    dim_feedforward=self.latent_dim,
                    sequence_length=self.sequence_length,
                    bias=bias,
                )
            )
        self.transformers = nn.ModuleList(self.transformers)

        self.encoder = Encoder(
            x_feat_channels_0=feat_channels_0,
            x_feat_channels_1=feat_channels_1,
            x_feat_channels_2=feat_channels_2,
            x_feat_channels_3=feat_channels_3,
            o_feat_channels_0=feat_channels_0,
            o_feat_channels_1=feat_channels_1,
            o_feat_channels_2=feat_channels_2,
            o_feat_channels_3=feat_channels_3,
            out_channels=latent_channels,
            bias_x_encoder=bias,
            bias_o_encoder=bias,
        )

        self.decoder = Decoder(
            feat_channels_0=latent_channels,
            feat_channels_1=feat_channels_3,
            feat_channels_2=feat_channels_2,
            feat_channels_3=feat_channels_1,
            out_channels=feat_channels_0,
            bias=bias,
        )

        p = kernel_size // 2

        if self.use_gsc:
            self.reconstructor = nn.Sequential(
                nn.Conv2d(
                    3 * feat_channels_0,
                    feat_channels_0,
                    kernel_size,
                    padding=p,
                    bias=bias,
                ),
                nn.LeakyReLU(),
                nn.Conv2d(
                    feat_channels_0, out_channels, kernel_size, padding=p, bias=True
                ),
            )
        else:
            self.reconstructor = nn.Sequential(
                nn.Conv2d(
                    feat_channels_0, feat_channels_0, kernel_size, padding=p, bias=bias
                ),
                nn.LeakyReLU(),
                nn.Conv2d(
                    feat_channels_0, out_channels, kernel_size, padding=p, bias=True
                ),
            )

    def _interpolate_features_along_time(self, feat: torch.Tensor, batch_size: int):
        feat = feat.view(
            batch_size,
            self.input_sequence_length,
            -1,
            self.hr_y_size,
            self.hr_x_size,
        )

        # Interpolate along time
        feat = feat.permute(0, 2, 1, 3, 4)
        feat = F.interpolate(
            feat,
            size=(self.sequence_length, self.hr_y_size, self.hr_x_size),
            mode="trilinear",
            align_corners=True,
        )
        feat = feat.permute(0, 2, 1, 3, 4)

        return feat.reshape(
            batch_size * self.sequence_length,
            -1,
            self.hr_y_size,
            self.hr_x_size,
        )

    def get_obs_feature(
        self, xs_org: torch.Tensor, obs_org: torch.Tensor, encoder_block: int = 0
    ) -> torch.Tensor:
        assert xs_org.shape[1:] == (
            self.input_sequence_length,
            self.in_channels,
            self.lr_y_size,
            self.lr_x_size,
        )
        assert obs_org.shape[1:] == (
            self.sequence_length,
            self.in_channels,
            self.hr_y_size,
            self.hr_x_size,
        )

        # Subsample to make timesteps lr.
        obs = obs_org[:, :: self.input_sampling_interval]

        # Interpolate to hr grid space, while timesteps remain.
        xs = xs_org.permute(0, 2, 1, 3, 4)  # exchange channel and time dim
        size = (self.input_sequence_length, self.hr_y_size, self.hr_x_size)
        xs = F.interpolate(
            xs,
            size=size,
            mode="nearest",
        )
        xs = xs.permute(0, 2, 1, 3, 4)

        # Reshape xs and obs to apply the same encoder at each time step
        xs = xs.view(-1, self.in_channels, self.hr_y_size, self.hr_x_size)
        obs = obs.reshape(-1, self.in_channels, self.hr_y_size, self.hr_x_size)

        feat_o = self.o_feat_extractor(obs)

        latent_o = feat_o
        for i in range(encoder_block):
            latent_o = self.encoder.o_encoder[i](latent_o)

        return feat_o, latent_o

    def forward(self, xs_org: torch.Tensor, obs_org: torch.Tensor) -> torch.Tensor:
        assert xs_org.shape[1:] == (
            self.input_sequence_length,
            self.in_channels,
            self.lr_y_size,
            self.lr_x_size,
        )
        assert obs_org.shape[1:] == (
            self.sequence_length,
            self.in_channels,
            self.hr_y_size,
            self.hr_x_size,
        )
        batch_size = xs_org.shape[0]

        # Subsample to make timesteps lr.
        obs = obs_org[:, :: self.input_sampling_interval]

        # Interpolate to hr grid space, while timesteps remain.
        xs = xs_org.permute(0, 2, 1, 3, 4)  # exchange channel and time dim
        size = (self.input_sequence_length, self.hr_y_size, self.hr_x_size)
        xs = F.interpolate(
            xs,
            size=size,
            mode="nearest",
        )
        xs = xs.permute(0, 2, 1, 3, 4)

        # Reshape xs and obs to apply the same encoder at each time step
        xs = xs.view(-1, self.in_channels, self.hr_y_size, self.hr_x_size)
        obs = obs.reshape(-1, self.in_channels, self.hr_y_size, self.hr_x_size)

        feat_x = self.x_feat_extractor(xs)
        feat_o = self.o_feat_extractor(obs)

        latent_x, latent_o = self.encoder(feat_x, feat_o)

        latent_x = latent_x.view(-1, self.input_sequence_length, self.latent_dim)
        latent_o = latent_o.view(-1, self.input_sequence_length, self.latent_dim)

        # Interpolate along time
        latent_x = latent_x.permute(0, 2, 1)
        latent_o = latent_o.permute(0, 2, 1)
        latent_x = F.interpolate(
            latent_x, size=self.sequence_length, mode="linear", align_corners=True
        )
        latent_o = F.interpolate(
            latent_o, size=self.sequence_length, mode="linear", align_corners=True
        )
        latent_x = latent_x.permute(0, 2, 1)
        latent_o = latent_o.permute(0, 2, 1)

        y = latent_x
        for transformer in self.transformers:
            y = transformer(y, latent_o)
        y = y + latent_x

        y = y.view(
            -1,
            self.latent_channels,
            self.latent_y_size,
            self.latent_x_size,
        )

        y = self.decoder(y)

        if self.use_gsc:
            feat_x = self._interpolate_features_along_time(feat_x, batch_size)
            feat_o = self._interpolate_features_along_time(feat_o, batch_size)
            y = torch.cat([y, feat_x, feat_o], dim=1)  # along channel dim

        y = self.reconstructor(y)

        return y.view(
            -1, self.sequence_length, self.out_channels, self.hr_y_size, self.hr_x_size
        )