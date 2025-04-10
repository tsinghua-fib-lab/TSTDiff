import os
import argparse
import torch
import datetime
import json
import yaml
from main_model_time import CSDI_Time
from main_model_freq import CSDI_Freq
from dataset import get_dataloader
from utils_freq import train_freq, evaluate_freq
from utils_time import train_time, evaluate_time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--ftfolder", type=str, default="")  # fine tune folder
parser.add_argument("--nsample", type=int, default=1)
parser.add_argument("--epochf", type=int, default=1)
parser.add_argument("--epocht", type=int, default=1)
parser.add_argument("--setting", type=int, default=2)
parser.add_argument("--ftepf", type=int, default=1)  # fine tune epoch
parser.add_argument("--ftept", type=int, default=1)  # fine tune epoch
parser.add_argument("--trainp", type=float, default=0.7)
args = parser.parse_args()

path = "../config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = 0
config["model"]["is_unconditional"] = False
config["model"]["featureemb"] = 8

# 自定义参数
config["train"]["epochf"] = args.epochf
config["train"]["epocht"] = args.epocht
config["train"]["ftepf"] = args.ftepf
config["train"]["ftept"] = args.ftept
config["train"]["itr_per_epoch"] = 100000
config["train"]["batch_size"] = 8
config["train"]["lr_freq"] = 3e-5
config["train"]["lr_time"] = 3e-5

config["diffusion"]["num_steps_freq"] = 30
config["diffusion"]["num_steps_time"] = 30
config["diffusion"]["diffusion_embedding_dim"] = 32

config["diffusion"]["beta_end_mag"] = 0.2
config["diffusion"]["beta_end_ph"] = 0.2
config["diffusion"]["schedule"] = "quad"
config["diffusion"]["beta_end"] = 0.2


if args.ftfolder == "":
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    finetune_folder = "save_model/Ft_Model_" + str(args.nfold) + "_" + current_time + "/"
    print('【fine tune folder】', finetune_folder)

    os.makedirs(finetune_folder, exist_ok=True)
    with open(finetune_folder + "config.json", "w") as f:
        json.dump(config, f, indent=4)


train_loader, valid_loader, test_loader, gen_loader = get_dataloader(
    trainp=args.trainp,
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
    setting=2,  # args.setting
    finetune=True
)

print("len of train_loader", len(train_loader))
print("len of gen_loader", len(gen_loader))

model_freq = CSDI_Freq(config, args.device).to(args.device)

model_time = CSDI_Time(config, args.device).to(args.device)


TRAIN_FREQ = False
TRAIN_TIME = False  # False  # 如果只有freq model，就设置为True，单独训练时域


# Fine-tune
if args.ftfolder == "":
    print("\n******** load freq model ********")
    model_freq.load_state_dict(torch.load("save_model/" + args.modelfolder + "/model_freq_ep{}.pth".format(args.epochf)))
    print("\n******** load time model ********")
    model_time.load_state_dict(torch.load("save_model/" + args.modelfolder + "/model_time_ep{}.pth".format(args.epocht)))

    for name, param in model_freq.named_parameters():
        param.requires_grad = False
        if 'diffmodel_mag.kge_projection' in name:
            param.requires_grad = True
        if 'diffmodel_ph.kge_projection' in name:
            param.requires_grad = True
        if 'diffmodel_mag.output_projection2' in name:
            param.requires_grad = True
        if 'diffmodel_ph.output_projection2' in name:
            param.requires_grad = True


    for name, param in model_time.named_parameters():
        param.requires_grad = False
        if 'diffmodel.kge_projection' in name:
            param.requires_grad = True
        if 'diffmodel.kge_projection' in name:
            param.requires_grad = True
        if 'diffmodel.output_projection2' in name:
            param.requires_grad = True
        elif 'diffmodel.output_projection4' in name:
            param.requires_grad = True


if args.ftfolder == "":
    print("\n******** fine tuning freq model ********")
    train_freq(
        model_freq,
        config["train"],
        train_loader,  # Except the testing dataset, remain * 20%
        valid_loader=None,
        foldername=finetune_folder,
        finetune=True,
    )
    model_freq.eval()
    model_freq.load_state_dict(torch.load(finetune_folder + "/model_freq_ep{}.pth".format(args.ftepf)))

    print("\n******** fine tuning time model ********")
    train_time(
        model_freq,
        model_time,
        config["train"],
        train_loader,
        valid_loader=None,
        foldername=finetune_folder,
        finetune=True,
    )

else:
    if TRAIN_FREQ==False:
        model_freq.load_state_dict(torch.load("save_model/" + args.ftfolder + "/model_freq_ep{}.pth".format(args.ftepf)))

    if TRAIN_TIME:
        model_freq.load_state_dict(torch.load("save_model/" + args.ftfolder + "/model_freq_ep{}.pth".format(args.ftepf)))
        model_freq.eval()

        for name, param in model_time.named_parameters():
            param.requires_grad = False
            if 'diffmodel.kge_projection' in name:
                param.requires_grad = True
            if 'diffmodel.kge_projection' in name:
                param.requires_grad = True
            if 'diffmodel.output_projection2' in name:
                param.requires_grad = True
            elif 'diffmodel.output_projection4' in name:
                param.requires_grad = True

        foldername = "save_model/" + args.ftfolder
        print("\n******** training time model ********")
        train_time(
            model_freq,
            model_time,
            config["train"],
            train_loader,
            valid_loader=None,
            foldername=foldername,
            finetune=True,
        )
    else:
        print("\n******** load freq model ********")
        model_freq.load_state_dict(torch.load("save_model/" + args.ftfolder + "/model_freq_ep{}.pth".format(args.ftepf)))
        print("\n******** load time model ********")
        model_time.load_state_dict(torch.load("save_model/" + args.ftfolder + "/model_time_ep{}.pth".format(args.ftept)))

    print("\n******** testing ********")
    # Short term test
    print("Test: 23 4G -> 23 5G")
    train_loader, valid_loader, test_loader, gen_loader = get_dataloader(
        seed=args.seed,
        nfold=args.nfold,
        batch_size=config["train"]["batch_size"],
        missing_ratio=config["model"]["test_missing_ratio"],
        setting=2,
        finetune=True
    )
    evaluate_freq(model_freq, test_loader, args.device, nsample=args.nsample, scaler=1,foldername=args.ftfolder)
    evaluate_time(model_freq, model_time, test_loader, args.device, nsample=args.nsample, scaler=1, foldername=args.ftfolder)
