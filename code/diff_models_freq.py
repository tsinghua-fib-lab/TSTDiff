import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer
from matplotlib import pyplot as plt
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6'


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_linear_trans(heads=8, layers=1, channels=64, localheads=0, localwindow=0):
    return LinearAttentionTransformer(
        dim=channels,
        depth=layers,
        heads=heads,
        max_seq_len=256,
        n_local_attn_heads=0,
        local_attn_window_size=0,
    )


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI_freq(nn.Module):
    def __init__(self, config, device, inputdim=2, stft_dim=2, L=504, mode=''):
        super().__init__()
        self.thl = 100
        self.device = device
        self.channels = config["channels"]
        self.mode = mode
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps_freq"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 3)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 3)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 3)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, diffusion_step, kge):
        B, inputdim, H, W = x.shape
        x = x.reshape(B, inputdim, H * W)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, H, W)
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_emb, kge)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))

        x = x.reshape(B, self.channels, H * W)  # B, self.channels, H, W

        x = self.output_projection1(x)  # (B, channel,H*W)
        x = F.leaky_relu(x)
        x = self.output_projection2(x)  # (B, 1, H*W)
        x = x.squeeze(1).reshape(B, H, W)  # (B,H,W)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.kge_projection = nn.Linear(32, channels)  # 32
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 3)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 3)
        self.is_linear = is_linear
        if is_linear:
            self.feature_layer = get_linear_trans(heads=nheads, layers=2, channels=channels)
        else:
            self.feature_layer = get_torch_trans(heads=nheads, layers=2, channels=channels)

    def forward_H(self, y, base_shape):
        B, channel, H, W = base_shape
        y = y.reshape(B, channel, H, W).permute(0, 3, 2, 1).reshape(B * W, H, channel)  # (B*W, H, channel)

        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(1, 0, 2)).permute(1, 2, 0)  # (H, B*W, channel)
        y = y.reshape(B, W, channel, H).permute(0, 2, 3, 1).reshape(B, channel, H * W)
        return y

    def forward_W(self, y, base_shape):
        B, channel, H, W = base_shape
        y = y.reshape(B, channel, H, W).permute(0, 2, 3, 1).reshape(B * H, W, channel)  # (B * H, W, channel)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(1, 0, 2)).permute(1, 2, 0)  # (W, B * H, channel)
        y = y.reshape(B, H, channel, W).permute(0, 2, 1, 3).reshape(B, channel, H * W)
        return y

    def forward(self, x, diffusion_emb, kge):
        B, channel, H, W = x.shape
        base_shape = x.shape
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1).unsqueeze(-1)
        kge_emb = self.kge_projection(kge).unsqueeze(-1).unsqueeze(-1)
        y = x + diffusion_emb + kge_emb

        y = self.forward_H(y, base_shape)
        y = self.forward_W(y, base_shape)

        y = y.reshape(B, channel, H * W)
        y = self.mid_projection(y)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
