import os
import numpy as np
import torch
import torch.nn as nn
from diff_models_time import diff_CSDI_time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.target_dim = target_dim
        self.device = device
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.emb_total_dim = self.emb_time_dim + 1  # num of regions
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI_time(config_diff, input_dim)

        self.num_steps = config_diff["num_steps_time"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1)

    def get_side_info(self, observed_data):
        B, M, L = observed_data.shape
        time_series = torch.linspace(0, L - 1, L, device=self.device)
        time_series = time_series.unsqueeze(0).expand(B, -1)
        time_embed = self.time_embedding(time_series, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, M, -1)

        # region
        region_indices = torch.tensor([2, 1, 1]).to(self.device)
        region_embed = region_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, L, -1, -1)

        side_info = torch.cat([region_embed, time_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)
        return side_info

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def calc_loss_valid(
            self, bs_kge, observed_data, target_data, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                bs_kge, observed_data, target_data, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
            self, bs_kge, observed_data, target_data, side_info, is_train, set_t=-1
    ):
        B, M, L = observed_data.shape
        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(target_data)  # (B,L)
        noisy_data = (current_alpha ** 0.5) * target_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data)
        predicted = self.diffmodel(total_input, t, bs_kge, side_info)

        residual = noise - predicted
        num_pixel = B * L
        loss = (residual ** 2).sum() / (num_pixel if num_pixel > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        else:
            observed_data = observed_data.unsqueeze(1)

            noisy_target = noisy_data.unsqueeze(1).unsqueeze(1)  # (B,1,1,L)
            noisy_target = noisy_target.expand(-1, -1, 3, -1)  # (B,1,M,L)
            total_input = torch.cat([observed_data, noisy_target], dim=1)  # (B,2,M,L)
        return total_input

    def impute(self, observed_data, bs_kge, output_freq, side_info, n_samples):
        B, M, L = observed_data.shape  # Batch, region_num, Length
        imputed_samples = torch.zeros(B, n_samples, L).to(self.device)

        for i in range(n_samples):
            if self.is_unconditional == True:
                noisy_obs = observed_data  # (B,M,L,N)
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)  # (B,M,L,N)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs)

            current_sample = torch.randn(B, L).to(self.device)  # 输出(B,L)
            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = noisy_cond_history[t] + current_sample
                    diff_input = diff_input.unsqueeze(1)
                else:
                    cond_obs = observed_data.unsqueeze(1)
                    noisy_target = current_sample.unsqueeze(1).unsqueeze(1)
                    noisy_target = noisy_target.expand(-1, -1, M, -1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)

                predicted = self.diffmodel(diff_input, torch.tensor([t]).to(self.device), bs_kge, side_info)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample += sigma * noise  # xt-1 = mean + sigma * z

            output_freq_tensor = torch.from_numpy(output_freq).to(current_sample.device)
            total_output = output_freq_tensor + current_sample  # (B,L)

            imputed_samples[:, i] = total_output.detach()  # (B,n,L)
        return imputed_samples

    def forward(self, batch, output_freq, is_train=1):
        (
            bs_kge,
            observed_data,
            target_data,
            bs_record,
            output_freq,
            observed_tp
        ) = self.process_data(batch, output_freq)

        side_info = self.get_side_info(observed_data)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(bs_kge, observed_data, target_data, side_info, is_train)

    def evaluate(self, batch, output_freq, n_samples):
        (
            bs_kge,
            observed_data,  # X0^(co): 4G
            target_data,
            bs_record,
            output_freq,
            observed_tp
        ) = self.process_data(batch, output_freq)

        with torch.no_grad():
            side_info = self.get_side_info(observed_data)
            samples = self.impute(observed_data, bs_kge, output_freq, side_info, n_samples)
        return samples, observed_data, target_data, bs_record


class CSDI_Time(CSDI_base):
    def __init__(self, config, device, target_dim=3):
        super(CSDI_Time, self).__init__(target_dim, config, device)

    def process_data(self, batch, output_freq):
        bs_kge = batch["bs_kge"].to(self.device).float() / 100.0
        bs_record = batch["tar_5G"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()

        nei_100_4G = batch["nei_100_4G"].to(self.device).float()
        nei_200_4G = batch["nei_200_4G"].to(self.device).float()
        nei_300_4G = batch["nei_300_4G"].to(self.device).float()

        target_data = torch.from_numpy(bs_record.cpu().numpy() - output_freq).to(self.device).float()
        observed_data = torch.cat([nei_100_4G.unsqueeze(1), nei_200_4G.unsqueeze(1), nei_300_4G.unsqueeze(1)], dim=1)
        return (
            bs_kge,
            observed_data,
            target_data,  # residual
            bs_record,  # target
            output_freq,
            observed_tp,
        )


def normalize_max(data):
    B, L = data.shape
    for b in range(B):
        max = np.max(data[b])
        if max == 0:
            max = 1
        data[b] = data[b] / max
    return data
