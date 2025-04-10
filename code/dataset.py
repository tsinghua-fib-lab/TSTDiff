import os
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.signal import stft, istft
import matplotlib.pyplot as plt

th_h = 50


def get_idlist(data_path):
    traffic_id = np.load(data_path)["bs_id"]
    return traffic_id


def normalize_max(data):
    N = data.shape[0]
    data_new = data
    for n in range(N):
        max = np.max(data[n])
        if max == 0:
            max = 1
        data_new[n] = data[n] / max
    return data_new


def normalize_max_sample(data):
    max = np.max(data)
    if max == 0:
        max = 1
    return data / max


def normalize_z(data):
    return (data - np.mean(data)) / np.std(data)


class Traffic_Dataset(Dataset):
    def __init__(self, eval_length=168, use_index_list=None, missing_ratio=0, seed=1, setting=1):
        np.random.seed(seed)
        self.eval_length = eval_length

        path = (
                "../data/S{}/Train".format(setting) + "_seed" + str(seed) + ".pk"
        )

        data_path = ""
        if setting == 1:  # train
            data_path = "../data/S1/Tar225G_Cond224G_final.npz"
        elif setting == 2:  # ft & test: short-term
            data_path = "../data/S2/Tar235G_Cond234G_23newbs_train_final.npz"
        elif setting == 3:  # test: long-term
            data_path = "../data/S3/Tar235G_Cond224G_23newbs_test_final.npz"

        if not os.path.isfile(path):
            data = np.load(data_path)
            print("NPZ data: ", data.files)
            idlist = get_idlist(data_path)
            print("Length of id list", len(idlist))
            self.bs_id = data["bs_id"].astype(np.float32)
            self.bs_kge = data["bs_kge"].astype(np.float32)
            self.tar_5G = data["bs_record"].astype(np.float32)

            self.nei_100_4G = data["nei_4G_traf_100"].astype(np.float32)
            self.nei_200_4G = data["nei_4G_traf_200"].astype(np.float32)
            self.nei_300_4G = data["nei_4G_traf_300"].astype(np.float32)
            self.tar_5G = normalize_max(self.tar_5G)  # (N,168)
            self.nei_100_4G = normalize_max(self.nei_100_4G)
            self.nei_200_4G = normalize_max(self.nei_200_4G)
            self.nei_300_4G = normalize_max(self.nei_300_4G)

            # STFT
            (self.target_traf_dup3,
             self.stft_tar_5G_mag, self.stft_tar_5G_ph,
             self.stft_nei_4G_mag, self.stft_nei_4G_ph) \
                = stft_data(data["bs_record"], data["nei_4G_traf_100"], data["nei_4G_traf_200"],
                            data["nei_4G_traf_300"])

            with open(path, "wb") as f:
                pickle.dump(
                    [self.bs_id, self.bs_kge,
                     self.tar_5G,
                     self.nei_100_4G, self.nei_200_4G, self.nei_300_4G,
                     self.target_traf_dup3,
                     self.stft_tar_5G_mag, self.stft_tar_5G_ph,
                     self.stft_nei_4G_mag, self.stft_nei_4G_ph,
                     ], f
                )

        else:
            with open(path, "rb") as f:
                (self.bs_id, self.bs_kge,
                 self.tar_5G,
                 self.nei_100_4G, self.nei_200_4G, self.nei_300_4G,
                 self.target_traf_dup3,
                 self.stft_tar_5G_mag, self.stft_tar_5G_ph,
                 self.stft_nei_4G_mag, self.stft_nei_4G_ph) = pickle.load(f)

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.tar_5G))
            print("use_index_list length", len(self.use_index_list))
        else:
            self.use_index_list = use_index_list
            print("use_index_list length", len(self.use_index_list))

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "bs_kge": self.bs_kge[index],
            "tar_5G": self.tar_5G[index],
            "nei_100_4G": self.nei_100_4G[index],
            "nei_200_4G": self.nei_200_4G[index],
            "nei_300_4G": self.nei_300_4G[index],
            "timepoints": np.arange(self.eval_length),

            "target_traf_dup3": self.target_traf_dup3[index],
            "stft_tar_5G_mag": self.stft_tar_5G_mag[index],
            "stft_tar_5G_ph": self.stft_tar_5G_ph[index],
            "stft_nei_4G_mag": self.stft_nei_4G_mag[index],
            "stft_nei_4G_ph": self.stft_nei_4G_ph[index],
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(trainp=0.7, seed=1, nfold=0, batch_size=1, missing_ratio=0, setting=1, finetune=False):
    dataset_tmp = Traffic_Dataset(missing_ratio=missing_ratio, seed=seed, setting=setting)
    print("\n******** load dataset ********")

    # setting 1: All used for training
    if setting == 1:
        indlist = np.arange(len(dataset_tmp))
        np.random.seed(seed)
        np.random.shuffle(indlist)
        num_train = int(len(indlist) * trainp)
        train_index = indlist[:num_train]  # 70% train
        valid_index = indlist[num_train:]  # 30% valid

        dataset = Traffic_Dataset(
            use_index_list=train_index, missing_ratio=missing_ratio, seed=seed, setting=setting
        )
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
        valid_dataset = Traffic_Dataset(
            use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed, setting=setting
        )
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
        test_loader = None
        gen_loader = DataLoader(dataset_tmp, batch_size=batch_size, shuffle=0)

    # setting 2: 20% used for fine-tuning and remained for short term testing
    elif setting == 2:
        indlist = np.arange(len(dataset_tmp))
        np.random.seed(seed)
        np.random.shuffle(indlist)
        start = int(nfold * 0.2 * len(dataset_tmp))
        end = int((nfold + 1) * 0.2 * len(dataset_tmp))
        test_index = indlist[start:end]
        remain_index = np.delete(indlist, np.arange(start, end))

        if finetune:
            finetune_samples = int(0.2 * len(dataset_tmp))
            start_fold = 0  # 0,1,2,3
            start_finetune = start_fold * finetune_samples
            end_finetune = (start_fold + 1) * finetune_samples
            finetune_index = remain_index[start_finetune:end_finetune]

            print("Finetune Length:", len(finetune_index))
            train_dataset = Traffic_Dataset(
                use_index_list=finetune_index, missing_ratio=missing_ratio, seed=seed, setting=setting
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)
            valid_loader = None
            test_dataset = Traffic_Dataset(
                use_index_list=test_index, missing_ratio=missing_ratio, seed=seed, setting=setting
            )
            print("Testing Length:", len(test_dataset), "n-fold:", nfold)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
            gen_loader = DataLoader(dataset_tmp, batch_size=batch_size, shuffle=0)

        else:  # Using 20% to test the fine-tuned model
            train_index = remain_index
            dataset = Traffic_Dataset(
                use_index_list=train_index, missing_ratio=missing_ratio, seed=seed, setting=setting
            )
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
            valid_loader = None
            test_dataset = Traffic_Dataset(
                use_index_list=test_index, missing_ratio=missing_ratio, seed=seed, setting=setting
            )
            print("Testing Length:", len(test_dataset), "n-fold:", nfold)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
            gen_loader = DataLoader(dataset_tmp, batch_size=batch_size, shuffle=0)

    elif setting == 3:  # This setting only used for long-term test
        indlist = np.arange(len(dataset_tmp))
        np.random.seed(seed)
        np.random.shuffle(indlist)
        start = int(nfold * 0.2 * len(dataset_tmp))
        end = int((nfold + 1) * 0.2 * len(dataset_tmp))
        test_index = indlist[start:end]

        train_loader = None
        valid_loader = None
        test_dataset = Traffic_Dataset(
            use_index_list=test_index, missing_ratio=missing_ratio, seed=seed, setting=setting
        )
        print("Testing Length:", len(test_dataset), "n-fold:", nfold)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
        gen_loader = DataLoader(dataset_tmp, batch_size=batch_size, shuffle=0)
    else:
        print("invalid setting. Setting should be 1,2,3")
        train_loader = None
        valid_loader = None
        test_loader = None
        gen_loader = None
    print("Success Loading Data! ")
    return train_loader, valid_loader, test_loader, gen_loader


def stft_data(data_5G, nei_4G_100, nei_4G_200, nei_4G_300):
    target_traf_dup3 = []
    stft_tar_5G_mag = []
    stft_tar_5G_ph = []
    stft_nei_4G_mag = []
    stft_nei_4G_ph = []

    for i in range(len(data_5G)):
        data_5G_newbs = np.concatenate([data_5G[i], data_5G[i], data_5G[i]])
        nei_4G_100[i] = normalize_max_sample(nei_4G_100[i])
        nei_4G_200[i] = normalize_max_sample(nei_4G_200[i])
        nei_4G_300[i] = normalize_max_sample(nei_4G_300[i])
        data_5G_newbs = normalize_max_sample(data_5G_newbs)
        data_4G = np.concatenate([nei_4G_100[i], nei_4G_200[i], nei_4G_300[i]])
        n_fft = 200
        hop_length_fft = 12
        frame_length = 504

        stft5G_mag, stft5G_phase, traffic_rec = stft_traffic(data_5G_newbs, n_fft, hop_length_fft, frame_length)
        stft4G_mag, stft4G_phase, _ = stft_traffic(data_4G, n_fft, hop_length_fft, frame_length)

        target_traf_dup3.append(traffic_rec)
        stft_tar_5G_mag.append(stft5G_mag)
        stft_tar_5G_ph.append(stft5G_phase)
        stft_nei_4G_mag.append(stft4G_mag)
        stft_nei_4G_ph.append(stft4G_phase)

    target_traf_dup3 = np.array(target_traf_dup3)
    stft_tar_5G_mag = np.array(stft_tar_5G_mag)
    stft_tar_5G_ph = np.array(stft_tar_5G_ph)
    stft_nei_4G_mag = np.array(stft_nei_4G_mag)
    stft_nei_4G_ph = np.array(stft_nei_4G_ph)
    return target_traf_dup3, stft_tar_5G_mag, stft_tar_5G_ph, stft_nei_4G_mag, stft_nei_4G_ph


def stft_traffic(data, n_fft, hop_length_fft, frame_length):
    frequencies, times, Zxx = stft(data, fs=1.0, nperseg=n_fft, noverlap=n_fft - hop_length_fft)  # 汉宁窗
    stft_traffic_mag = np.abs(Zxx)
    stft_traffic_phase = np.angle(Zxx)
    stft_traffic_mag[th_h:, :] = 0
    traf_reverse_stft = stft_traffic_mag * np.exp(1j * stft_traffic_phase)
    _, traffic_rec = istft(traf_reverse_stft, nperseg=n_fft, noverlap=n_fft - hop_length_fft)
    return stft_traffic_mag, stft_traffic_phase, traffic_rec
