import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="/home/zbr/Workspace/proj/uspp2pm/out/20220505-PREbert-for-patents-DATcombined-LOSSpearson-FOLD1", type=str)

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
    upload(opt.path)
    
