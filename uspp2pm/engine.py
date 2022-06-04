from typing import Mapping
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader

from uspp2pm import tbwriter
from .utils import LogMeter

def give_train_loader(collate_fn, train_set, config):
    sampler = None
    if dist.is_initialized():
        sampler = torch.utils.data.DistributedSampler(
            train_set, 
            num_replicas=dist.get_world_size(), 
            rank=dist.get_rank(), shuffle=True
        )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.bs,
        shuffle=sampler is None,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        sampler=sampler
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
            elif isinstance(v, Mapping):
                data[k] = preprocess_data(v, config)
    else:
        data = data.to(config.device)
    return data

def train_one_epoch(
    model, collate_fn, train_set, optimizer, scheduler, config, return_loss=False
):
    model.train()
    loss_meter = LogMeter()
    # get dataloader
    train_loader = give_train_loader(collate_fn, train_set, config)
    if dist.is_initialized():
        train_loader.sampler.set_epoch(config.epoch)
    train_iter = tqdm(train_loader)

    pred_list = []
    labels_list = []

    for idx, data_info in enumerate(train_iter):
        data_info = preprocess_data(data_info, config)
        labels = data_info.pop("labels")

        # get similarity
        sim, loss = model(data_info, labels)

        # update
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        loss_meter.append(loss.item())

        pred_list.append(process_out(sim.detach(), config))
        labels_list.append(labels)

        if idx % 10 == 0:
            train_iter.set_description(f"Loss={loss_meter.avg()}")
            tbwriter.add_scalar(f"fold{config.fold}/train/loss", loss)
    scheduler.step()
    if return_loss:
        return torch.cat(pred_list, dim=0).cpu().numpy(), torch.cat(labels_list, dim=0).cpu().numpy(), loss_meter.value_list
    else:
        return torch.cat(pred_list, dim=0).cpu().numpy(), torch.cat(labels_list, dim=0).cpu().numpy()

def predict(
    model, collate_fn, val_set, config, is_test=False
):
    model.eval()
    # get dataloader
    val_loader = give_test_loader(collate_fn, val_set, config)
    val_iter = tqdm(val_loader)
    
    pred_list = []
    labels_list = []
    with torch.no_grad():
        for data_info in val_iter:
            data_info = preprocess_data(data_info, config)
            
            # get similarity
            sim = model(data_info)
            pred = sim

            pred_list.append(process_out(pred.cpu(), config))
            if not is_test:
                labels = data_info.pop("labels")
                labels_list.append(labels)
    if not is_test:
        return torch.cat(pred_list, dim=0).cpu().numpy(), torch.cat(labels_list, dim=0).cpu().numpy()
    else:
        return torch.cat(pred_list, dim=0).cpu().numpy()

def process_out(data, config):
    if config.loss_name == "cross_entropy":
        out = torch.argmax(data, dim=-1)
        out = out * 0.25
        return out
    else:
        return data
