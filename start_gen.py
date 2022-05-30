from itertools import product
import argparse

parser = argparse.ArgumentParser("US patent model")
# dataset
parser.add_argument("--dataset_name", type=str, default="combined", 
                                        help="split / combined")
# loss
parser.add_argument("--loss_name", type=str, default="mse", 
                                        help="mse / shift_mse / pearson")
### ExpLoss
parser.add_argument("--scale", type=float, default=1.0)
# model
parser.add_argument("--pretrain_name", type=str, default="deberta-v3-base", 
                                        help="bert-for-patents / deberta-v3-large")
parser.add_argument("--model_name", type=str, default="combined_hdc",
                                        help="combined_baseline / split_baseline / split_similarity")
parser.add_argument("--num_layer", type=int, default=1)
### combined_hdc
parser.add_argument("--num_block", type=int, default=1)
parser.add_argument("--update_rate", type=float, default=0.01)
# optimizer
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--wd", type=float, default=0.01)
parser.add_argument("--lr_multi", type=float, default=1)
# scheduler
# training
parser.add_argument("--num_fold", type=int, default=5, 
                                        help="0/1 for training all")
parser.add_argument("--bs", type=int, default=48)
parser.add_argument("--epochs", type=int, default=15)
# general
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--nproc_per_node", type=int, default=1)

opt = parser.parse_args(args=[])
opt_dict = {}
for k in dir(opt):
    if not k.startswith("_") and not k.endswith("_"):
        opt_dict[k] = getattr(opt, k)
opt_dict.pop("debug")
opt_dict.pop("tag")

to_optim_args = {
    "num_layer": [1, 2],
    "num_block": [2, 3, 4],
    "lr_multi": [1, 10],
    "update_rate": [0.01, 0.05]
}

key_list = list(to_optim_args.keys())
value_list = list(to_optim_args.values())

for idx, item in enumerate(product(*value_list)):
    cmd = "- python train.py"
    for idx in range(len(key_list)):
        opt_dict[key_list[idx]] = item[idx]
    for k, v in opt_dict.items():
        cmd += f" --{k} {v}"
    print(cmd)
