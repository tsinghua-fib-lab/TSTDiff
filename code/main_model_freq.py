import numpy as np
import torch
import torch.nn as nn
from diff_models_freq import diff_CSDI_freq
import os
# import pywt
from scipy.signal import stft, istft
from matplotlib import pyplot as plt
from dataset import th_h

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
n_fft = 200
hop_length_fft = 12
frame_length = 504


class CSDI_base(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.is_unconditional = config["model"]["is_unconditional"]
        config_diff = config["diffusion"]
        input_dim = 1 if self.is_unconditional == True else 2

        self.diffmodel_mag = diff_CSDI_freq(config_diff, device, input_dim, mode='mag')
        self.diffmodel_ph = diff_CSDI_freq(config_diff, device, input_dim, mode='ph')

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps_freq"]
        if config_diff["schedule"] == "quad":
            self.beta_mag = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end_mag"] ** 0.5, self.num_steps
            ) ** 2
            self.beta_ph = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end_ph"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta_mag = np.linspace(
                config_diff["beta_start"], config_diff["beta_end_mag"], self.num_steps
            )
            self.beta_ph = np.linspace(
                config_diff["beta_start"], config_diff["beta_end_ph"], self.num_steps
            )

        self.alpha_hat_mag = 1 - self.beta_mag
        self.alpha_mag = np.cumprod(self.alpha_hat_mag)
        self.alpha_torch_mag = torch.tensor(self.alpha_mag).float().to(self.device).unsqueeze(1).unsqueeze(1)  # (B,1,1)

        self.alpha_hat_ph = 1 - self.beta_ph
        self.alpha_ph = np.cumprod(self.alpha_hat_ph)
        self.alpha_torch_ph = torch.tensor(self.alpha_ph).float().to(self.device).unsqueeze(1).unsqueeze(1)  # (B,1,1)


    def time_embedding(self, pos, d_model=128):  #
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def calc_loss_valid(
            self, bs_kge, observed_data, target_data, target_traf, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                bs_kge, observed_data, target_data, target_traf, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
            self, bs_kge, observed_data, target_data, target_traf, is_train, set_t=-1
    ):
        B, stft_dim, H, W = observed_data.shape
        L = target_traf.shape[1]  # 504
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha_mag = self.alpha_torch_mag[t]  # (B,1,1)
        current_alpha_ph = self.alpha_torch_ph[t]  # (B,1,1)

        noise_mag = torch.randn(B, H, W, device=self.device)  # (B,2,H,W)  /10
        noise_ph = torch.randn(B, H, W, device=self.device)  # (B,2,H,W)

        noisy_data_mag = (current_alpha_mag ** 0.5) * target_data[:, 0, :, :] + (1.0 - current_alpha_mag) ** 0.5 * noise_mag
        noisy_data_ph = (current_alpha_ph ** 0.5) * target_data[:, 1, :, :] + (1.0 - current_alpha_ph) ** 0.5 * noise_ph

        total_input_mag = self.set_input_to_diffmodel(bs_kge, noisy_data_mag, observed_data[:,0,:,:])
        predicted_mag = self.diffmodel_mag(total_input_mag, t, bs_kge)

        total_input_ph = self.set_input_to_diffmodel(bs_kge, noisy_data_ph, observed_data[:,1,:,:])
        predicted_ph = self.diffmodel_ph(total_input_ph, t, bs_kge)

        residual = (noise_mag - predicted_mag)**2 + (noise_ph - predicted_ph)**2

        num_pixel = B * 2 * H * W
        loss = residual.sum()/ (num_pixel if num_pixel > 0 else 1)
        return loss

    # total input
    def set_input_to_diffmodel(self, bs_kge, noisy_data, observed_data):
        # # noisy data is noised 5G，observed_data is 4G
        B, H, W = observed_data.shape
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)
        else:
            cond_obs = observed_data.unsqueeze(1)
            noisy_target = noisy_data.unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)
        return total_input

    def impute(self, observed_data, bs_kge, n_samples):
        B, dim, H, W = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, 2, H, W).to(self.device)
        Padding_samples = torch.zeros(B, 1, 2, int(n_fft / 2) + 1, W).to(self.device)
        for i in range(n_samples):
            if self.is_unconditional == True:
                pass
            current_sample_mag = torch.randn(B, H, W, device=self.device)
            current_sample_ph = torch.randn(B, H, W, device=self.device)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    pass
                else:
                    cond_obs_mag = observed_data[:,0,:,:].unsqueeze(1).to(self.device)  # (B,H,W)->(B,1,H,W)
                    noisy_target_mag = current_sample_mag.unsqueeze(1).to(self.device)
                    diff_input_mag = torch.cat([cond_obs_mag, noisy_target_mag], dim=1).to(self.device)  # (B,2,H,W)

                    cond_obs_ph = observed_data[:, 1, :, :].unsqueeze(1).to(self.device)  # (B,H,W)->(B,1,H,W)
                    noisy_target_ph = current_sample_ph.unsqueeze(1).to(self.device)
                    diff_input_ph = torch.cat([cond_obs_ph, noisy_target_ph], dim=1).to(self.device)  # (B,2,H,W)

                predicted_mag = self.diffmodel_mag(diff_input_mag, torch.tensor([t]).to(self.device), bs_kge)
                predicted_ph = self.diffmodel_ph(diff_input_ph, torch.tensor([t]).to(self.device), bs_kge)

                coeff1_mag = 1 / self.alpha_hat_mag[t] ** 0.5
                coeff2_mag = (1 - self.alpha_hat_mag[t]) / (1 - self.alpha_mag[t]) ** 0.5

                coeff1_ph = 1 / self.alpha_hat_ph[t] ** 0.5
                coeff2_ph = (1 - self.alpha_hat_ph[t]) / (1 - self.alpha_ph[t]) ** 0.5

                current_sample_mag = coeff1_mag * (current_sample_mag - coeff2_mag * predicted_mag)  # mean
                current_sample_ph = coeff1_ph * (current_sample_ph - coeff2_ph * predicted_ph)  # mean

                if t > 0:
                    noise_mag = torch.randn_like(current_sample_mag).to(self.device)  # (B,H,W)
                    noise_ph = torch.randn_like(current_sample_ph).to(self.device)  # (B,H,W)

                    sigma_mag = (
                                    (1.0 - self.alpha_mag[t - 1]) / (1.0 - self.alpha_mag[t]) * self.beta_mag[t]
                            ) ** 0.5
                    sigma_ph = (
                                    (1.0 - self.alpha_ph[t - 1]) / (1.0 - self.alpha_ph[t]) * self.beta_ph[t]
                            ) ** 0.5

                    current_sample_mag += sigma_mag * noise_mag
                    current_sample_ph += sigma_ph * noise_ph
            current_sample = torch.cat([current_sample_mag.unsqueeze(1), current_sample_ph.unsqueeze(1)],dim=1).to(self.device)
            imputed_samples[:, i, :, :, :] = current_sample

        Padding_samples[:, :, :, 0:th_h, :] = imputed_samples
        return Padding_samples

    def forward(self, batch, is_train=1):
        (
            bs_kge,
            observed_data,  # (B,2,H,W)
            target_data,  # (B,2,H,W)
            target_traf
        ) = self.process_data(batch)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(bs_kge, observed_data, target_data, target_traf, is_train)

    def evaluate(self, batch, n_samples):
        (
            bs_kge,
            observed_data,
            target_data,
            target_traf
        ) = self.process_data(batch)

        with torch.no_grad():
            samples = self.impute(observed_data, bs_kge, n_samples)
        return samples, observed_data, target_data, target_traf


class CSDI_Freq(CSDI_base):
    def __init__(self, config, device):
        super(CSDI_Freq, self).__init__(config, device)

    def process_data(self, batch):
        bs_kge = batch["bs_kge"].to(self.device).float()/40.0

        target_traf = batch["target_traf_dup3"].to(self.device).float()
        stft_tar_5G_mag = batch["stft_tar_5G_mag"].to(self.device).float()
        stft_tar_5G_ph = batch["stft_tar_5G_ph"].to(self.device).float()
        stft_nei_4G_mag = batch["stft_nei_4G_mag"].to(self.device).float()
        stft_nei_4G_ph = batch["stft_nei_4G_ph"].to(self.device).float()

        target_data_tmp = torch.cat([stft_tar_5G_mag.unsqueeze(1), stft_tar_5G_ph.unsqueeze(1)], dim=1)  # [B,2,H,W]
        nei_4G = torch.cat([stft_nei_4G_mag.unsqueeze(1), stft_nei_4G_ph.unsqueeze(1)], dim=1)

        target_data = target_data_tmp[:, :, :th_h, :]
        observed_data = nei_4G[:, :, :th_h, :]
        return (
            bs_kge,
            observed_data,
            target_data,
            target_traf
        )
