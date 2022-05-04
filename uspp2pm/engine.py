
import torch.nn as nn
import torch
from typing import Mapping
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from .utils import LogMeter

def give_train_loader(collate_fn, train_set, config):
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.bs,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return train_loader

def give_test_loader(collate_fn, test_set, config):
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=config.bs,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )
    return test_loader

def preprocess_data(data, config):
    if isinstance(data, Mapping):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(config.device)
    else:
        data = data.to(config.device)
    return data

def train_one_epoch(
    model, criterion, collate_fn, train_set, optimizer, scheduler, config
):
    model.train()
    criterion.train()
    loss_meter = LogMeter()
    # get dataloader
    train_loader = give_train_loader(collate_fn, train_set, config)
    train_iter = tqdm(train_loader)

    pred_list = []
    labels_list = []

    for idx, data_info in enumerate(train_iter):
        inputs = preprocess_data(data_info["inputs"], config)
        labels = preprocess_data(data_info["labels"], config)

        # get similarity
        sim = model(**inputs)
        loss = criterion(sim, labels)

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.append(loss.item())

        pred_list.append(process_out(sim.detach(), config))
        labels_list.append(labels)

        if idx % 10 == 0:
            train_iter.set_description(f"Loss={loss_meter.avg()}")
    scheduler.step()
    return torch.cat(pred_list, dim=0).cpu().numpy(), torch.cat(labels_list, dim=0).cpu().numpy()

def predict(
    model, collate_fn, val_set, config
):
    model.eval()
    # get dataloader
    val_loader = give_test_loader(collate_fn, val_set, config)
    val_iter = tqdm(val_loader)
    
    pred_list = []
    labels_list = []
    with torch.no_grad():
        for data_info in val_iter:
            inputs = preprocess_data(data_info["inputs"], config)
            labels = preprocess_data(data_info["labels"], config)
            
            # get similarity
            sim = model(**inputs)
            pred = sim

            pred_list.append(process_out(pred.cpu(), config))
            labels_list.append(labels)
    return torch.cat(pred_list, dim=0).cpu().numpy(), torch.cat(labels_list, dim=0).cpu().numpy()

def process_out(data, config):
    if config.loss_name == "mse":
        return data
    elif config.loss_name == "shift_mse":
        data = torch.floor(data * 5) * 0.25
        return data
