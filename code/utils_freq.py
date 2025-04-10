import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import datetime
from matplotlib import pyplot as plt
from scipy.signal import stft, istft
from scipy.spatial import distance
import os
from dataset import th_h


n_fft = 200
hop_length_fft = 12
frame_length = 504


def train_freq(
        model,
        config,
        train_loader,
        valid_loader=None,
        valid_epoch_interval=20,
        foldername="",
        finetune=False,
):
    optimizer = Adam(model.parameters(), lr=config["lr_freq"], weight_decay=1e-6)

    if finetune:
        epochf = config["ftepf"]
    else:
        epochf = config["epochf"]

    p1 = int(0.75 * epochf)
    p2 = int(0.9 * epochf)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(epochf):
        if foldername != "":
            output_path = foldername + "/model_freq_ep{}.pth".format(epoch_no+1)

        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()

                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:  # 进度条
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

        if foldername != "":
            if (epoch_no+1) % 1 == 0:
                torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def evaluate_freq(model, test_loader, device, nsample=10, scaler=1, mean_scaler=0, foldername=""):
    image_counter = 0
    with torch.no_grad():
        model.eval()
        generated_data_list = []
        real_data_list = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, batch_data in enumerate(it, start=1):
                samples, observed_data, target_data, target_traf = model.evaluate(batch_data, nsample)
                # draw_images(samples, image_counter)

                samples_istft = output_istft(samples)
                # draw_istft(samples_istft, image_counter)
                # image_counter += 1

                generated_data_list.append(samples_istft)
                real_data_list.append(target_traf[:, :168])  # (B,168)

                it.set_postfix(
                    ordered_dict={
                        "batch_no": batch_no,
                    },
                    refresh=True
                )

        generated_data_tensors = [torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr for arr in
                                  generated_data_list]
        generated_data = torch.cat(generated_data_tensors, dim=0)

        real_data = torch.cat(real_data_list, dim=0).cpu().numpy()

        metric_results = np.zeros((nsample, 3))
        for n in range(nsample):
            jsd, jsd_diff = compute_jsd(generated_data[:, n, :], real_data, save_dir="./img_gen")
            rmse = pattern_freq_ratio_rmse(generated_data[:, n, :], real_data)
            metric_results[n, :] = [jsd, jsd_diff, rmse]
        metric_result_mean = metric_results.mean(0)
        print("results: ", metric_result_mean)
        save_metrics_to_file(metric_result_mean, 'result_metrics/metrix_freq.txt')


def output_istft(samples):
    B, n_samples, dim, H, W = samples.shape
    L = 168
    samples_istft = np.zeros((B,n_samples, L))
    samples = samples.cpu().numpy()
    for i in range(B):
        for j in range(n_samples):
            stft_samples_mag = samples[i][j][0]
            stft_sample_phase = samples[i][j][1]
            stft_sample = stft_samples_mag * np.exp(1j * stft_sample_phase)
            _, istft_sample = istft(stft_sample, nperseg=n_fft, noverlap=n_fft - hop_length_fft)
            tmp = (istft_sample[:168] +
                   istft_sample[168:336] +
                   istft_sample[336:]) / 3
            samples_istft[i][j] = tmp
    return samples_istft


def draw_images(samples_in, image_counter):
    B, n_samples, dim, H, W = samples_in.shape
    samples_in = samples_in.cpu().numpy()
    samples = samples_in[:,:,:,0:th_h,:]

    save_path = "./img_gen"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(B):
        for j in range(n_samples):
            stft_sample= samples[i][j]
            plt.figure(figsize=(10, 6))
            plt.pcolormesh(np.linspace(0, 43, 43), np.linspace(0,th_h,th_h), stft_sample[0], shading='gouraud')
            plt.colorbar()
            plt.title('Generated Magnitude of Traffic Data')
            plt.savefig(f"{save_path}/img_{image_counter * B + i + 1}_mag.png")
            plt.close()

            stft_sample = samples[i][j]
            plt.figure(figsize=(10, 6))
            plt.pcolormesh(np.linspace(0, 43, 43), np.linspace(0,th_h,th_h), stft_sample[1], shading='gouraud')
            plt.colorbar()
            plt.title('Generated Phase of Traffic Data')
            plt.savefig(f"{save_path}/img_{image_counter * B + i + 1}_ph.png")
            plt.close()


def draw_istft(samples, image_counter):
    B, n_samples, L = samples.shape
    save_path = "./img_gen"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(B):
        for j in range(n_samples):
            istft_sample = samples[i][j]
            istft_sample_norm = normalize_max_sample(istft_sample)
            plt.figure(figsize=(6, 3.5))
            plt.plot(np.linspace(0, L, L), istft_sample_norm, color='steelblue',linewidth=1, label="Generated Traffic")
            plt.ylabel("Normalized Traffic",fontsize=16, labelpad=0.15)
            plt.ylim(0, 1.2)
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            xticks_positions = np.arange(0, 168, 24)
            plt.xticks(xticks_positions, [''] * len(xticks_positions))
            for n, day in enumerate(days):
                plt.text(n * 24 + 12, -0.065, day, ha='center', transform=plt.gca().get_xaxis_transform(),fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{save_path}/img_{image_counter * B + i + 1}_istft.png")
            plt.close()


def compute_jsd(samples, target_traf, save_dir='./', n_bins=100):
    samples[samples < 0] = 0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig, ax = plt.subplots(figsize=(24, 6))
    line_w = 2
    use_cumulative = -1
    use_log = True
    n_real, bins, patches = ax.hist(target_traf.flatten(), n_bins, density=True, histtype='step',
                                    cumulative=use_cumulative, label='real', log=use_log, facecolor='g',
                                    linewidth=line_w)
    n_gene, bins, patches = ax.hist(samples.flatten(), n_bins, density=True, histtype='step',
                                    cumulative=use_cumulative, label='gene', log=use_log, facecolor='b',
                                    linewidth=line_w)
    JSD = distance.jensenshannon(n_real.flatten(), n_gene.flatten(), 2.0)
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title('Cumulative step histograms')
    ax.set_xlabel('Value')
    ax.set_ylabel('Likelihood of occurrence')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'fig_hist_freq.jpg'))
    plt.close()

    fig, ax = plt.subplots(figsize=(24, 6))
    real_diff = target_traf[1:] - target_traf[:-1]
    generated_diff = samples[1:] - samples[:-1]

    n_real, bins, patches = ax.hist(real_diff.flatten(), n_bins, density=True, histtype='step',
                                    cumulative=use_cumulative, label='real', log=use_log, facecolor='g',
                                    linewidth=line_w)
    n_gene, bins, patches = ax.hist(generated_diff.flatten(), n_bins, density=True, histtype='step',
                                    cumulative=use_cumulative, label='gene', log=use_log, facecolor='b',
                                    linewidth=line_w)
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title('Cumulative step histograms')
    ax.set_xlabel('Value')
    ax.set_ylabel('Likelihood of occurrence')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'fig_diff_hist_freq.jpg'))
    plt.close()
    JSD_diff = distance.jensenshannon(n_real.flatten(), n_gene.flatten(), 2.0)

    return JSD, JSD_diff


def pattern_freq_ratio_rmse(samples, target_traf):
    data_f_target = np.abs(np.fft.rfft(target_traf, axis=1))
    daily_target = data_f_target[:, 7] / np.sum(data_f_target, axis=1)
    daily_target = np.nan_to_num(daily_target)

    data_f = np.abs(np.fft.rfft(samples, axis=1))
    daily_real = data_f[:, 7] / np.sum(data_f, axis=1) if data_f.sum(1).all() > 0 else data_f[:, 7]

    daily_real = np.nan_to_num(daily_real)
    rmse_daily = np.sqrt(np.mean((daily_target - daily_real) ** 2))

    return rmse_daily


def save_metrics_to_file(metrics, file_path):
    with open(file_path, 'a') as file:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_metrics = ', '.join(map(str, metrics))
        file.write("********* Current Time: "+current_time+" *********\n")
        file.write(formatted_metrics + '\n\n')


def normalize_max_sample(data):
    max = np.max(data)
    if max == 0:
        max = 1
    return data/max


