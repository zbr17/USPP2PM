import os
import json
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="/home/zbr/Workspace/proj/uspp2pm/out/20220505-PREbert-for-patents-DATcombined-LOSSpearson-FOLD1", type=str)
parser.add_argument("--split", action="store_true")
parser.add_argument("--not_upload", action="store_true")
parser.add_argument("--only-full", action="store_true")

def split_fold(path):
    name = os.path.basename(path)
    full_name = name.replace("--", "-full-")
    full_path = os.path.join(os.path.dirname(path), full_name)
    os.makedirs(full_path, exist_ok=True)
    shutil.copyfile(os.path.join(path, "config.yaml"), os.path.join(full_path, "config.yaml"))
    if not os.path.exists(os.path.join(full_path, "model_all.ckpt")):
        shutil.move(os.path.join(path, "model_all.ckpt"), os.path.join(full_path, "model_all.ckpt"))
    return full_path, path

def upload(path):
    # init
    cmd = f"kaggle datasets init -p {path}"
    os.system(cmd)
    with open(os.path.join(path, "dataset-metadata.json"), encoding="utf-8", mode="r") as f:
        json_file = json.load(f)
    name = os.path.basename(path)
    if len(name) > 50:
        name = name[:50]
    print(name)
    json_file["title"] = name
    json_file["id"] = f"boruizhang/{name}"
    with open(os.path.join(path, "dataset-metadata.json"), "w") as f:
        json.dump(json_file, f)
    
    # create
    cmd = f"kaggle datasets create -r zip -p {path}"
    os.system(cmd)

if __name__ == "__main__":
    opt = parser.parse_args()
    if not opt.split:
        upload(opt.path)
    else:
        full_path, cross_path = split_fold(opt.path)
        if not opt.not_upload:
            upload(full_path)
            if not opt.only_full:
                upload(cross_path)
    
