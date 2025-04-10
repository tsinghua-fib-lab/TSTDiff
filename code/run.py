import os
import argparse
import torch
import datetime
import json
import yaml
import os
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
parser.add_argument("--testmissingratio", type=float, default=0)  # 默认为0
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)
parser.add_argument("--epochf", type=int, default=1)
parser.add_argument("--epocht", type=int, default=1)
parser.add_argument("--setting", type=int, default=1)
parser.add_argument("--trainp", type=float, default=0.7)
args = parser.parse_args()

path = "../config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = 0
config["model"]["is_unconditional"] = False
config["model"]["featureemb"] = 8

config["train"]["epochf"] = args.epochf
config["train"]["epocht"] = args.epocht
config["train"]["itr_per_epoch"] = 100000
config["train"]["batch_size"] = 8
config["train"]["lr_freq"] = 3e-4
config["train"]["lr_time"] = 5e-4

config["diffusion"]["num_steps_freq"] = 50
config["diffusion"]["num_steps_time"] = 50
config["diffusion"]["diffusion_embedding_dim"] = 32

config["diffusion"]["beta_end_mag"] = 0.2
config["diffusion"]["beta_end_ph"] = 0.2
config["diffusion"]["schedule"] = "quad"
config["diffusion"]["beta_end"] = 0.2


if args.modelfolder == "":
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = "save_model/Model_" + str(args.nfold) + "_" + current_time + "/"
    print('【model folder】', foldername)

    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, gen_loader = get_dataloader(
    trainp=args.trainp,
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
    setting=args.setting,
)

model_freq = CSDI_Freq(config, args.device).to(args.device)
model_time = CSDI_Time(config, args.device).to(args.device)


TRAIN_FREQ = False
TRAIN_TIME = False  # If the freq model is already trained and time model untrained, then set it to True.

if args.modelfolder == "":
    print("\n******** training freq model ********")
    train_freq(
        model_freq,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
    model_freq.eval()
    model_freq.load_state_dict(torch.load(foldername + "/model_freq_ep{}.pth".format(args.epochf)))

    print("\n******** training time model ********")
    train_time(
        model_freq,
        model_time,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )

else:
    foldername = "save_model/" + args.modelfolder
    if TRAIN_FREQ==False:
        model_freq.load_state_dict(torch.load("save_model/" + args.modelfolder + "/model_freq_ep{}.pth".format(args.epochf)))

    if TRAIN_TIME:
        print("\n******** load freq model ********")
        model_freq.load_state_dict(torch.load("save_model/" + args.modelfolder + "/model_freq_ep{}.pth".format(args.epochf)))
        model_freq.eval()
        print("\n******** training time model ********")
        train_time(
            model_freq,
            model_time,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        print("\n******** load freq model *********")
        model_freq.load_state_dict(torch.load("save_model/" + args.modelfolder + "/model_freq_ep{}.pth".format(args.epochf)))
        print("\n******** load time model ********")
        model_time.load_state_dict(torch.load("save_model/" + args.modelfolder + "/model_time_ep{}.pth".format(args.epocht)))

    print("\n******** testing ********")
    # print("Test w/o FT: 234G -> 235G")
    # train_loader, valid_loader, test_loader, gen_loader = get_dataloader(
    #     seed=args.seed,
    #     nfold=args.nfold,
    #     batch_size=config["train"]["batch_size"],
    #     missing_ratio=config["model"]["test_missing_ratio"],
    #     setting=2,
    # )

    # Long-term test
    print("Test w/o FT: 224G -> 235G")
    train_loader, valid_loader, test_loader, gen_loader = get_dataloader(
        seed=args.seed,
        nfold=args.nfold,
        batch_size=config["train"]["batch_size"],
        missing_ratio=config["model"]["test_missing_ratio"],
        setting=3,
    )
    evaluate_freq(model_freq, test_loader, args.device, nsample=args.nsample, scaler=1, foldername=foldername)
    evaluate_time(model_freq, model_time, test_loader, args.device, nsample=args.nsample, scaler=1, foldername=foldername)
