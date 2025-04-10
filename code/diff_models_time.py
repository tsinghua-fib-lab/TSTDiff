import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer

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
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
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
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI_time(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps_time"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        self.output_projection3 = Conv1d_with_init(3, 3, 1)
        self.output_projection4 = Conv1d_with_init(3, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, diffusion_step, kge, side_info):
        B, inputdim, M, L = x.shape
        x = x.reshape(B, inputdim, M * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, M, L)
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_emb, kge, side_info)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, M * L)
        x = self.output_projection1(x)  # (B, self.channels, M*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B, 1, M*L)
        x = x.squeeze(1).reshape(B, M, L)
        x = self.output_projection4(x)  # (B, 1, L)
        x = x.squeeze(1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.kge_projection = nn.Linear(32, channels)
        self.bn = nn.BatchNorm1d(channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
            self.region_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.region_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, M, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, M, L).permute(0, 2, 1, 3).reshape(B * M, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, M, channel, L).permute(0, 2, 1, 3).reshape(B, channel, M * L)
        return y

    def forward_region(self, y, base_shape):
        B, channel, M, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, M, L).permute(0, 3, 1, 2).reshape(B * L, channel, M)
        if self.is_linear:
            y = self.region_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.region_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, M).permute(0, 2, 3, 1).reshape(B, channel, M * L)
        return y

    def forward(self, x, diffusion_emb, kge, side_info):
        B, channel, M, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, M * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        kge_emb = self.kge_projection(kge).unsqueeze(-1)
        y = x + diffusion_emb + kge_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_region(y, base_shape)
        y = self.mid_projection(y)

        _, cond_dim, _, _ = side_info.shape
        cond_info = side_info.reshape(B, cond_dim, M * L)
        cond_info = self.cond_projection(cond_info)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
