
import torch.nn as nn
import torch
from typing import Mapping
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from .collate_fn import DataCollatorWithPadding
from .utils import LogMeter

def give_train_loader(train_set, config):
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.bs,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=DataCollatorWithPadding(config.tokenizer),
        drop_last=True
    )
    return train_loader

def give_test_loader(test_set, config):
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=config.bs,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=DataCollatorWithPadding(config.tokenizer),
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
    model, criterion, train_set, optimizer, scheduler, config
):
    model.train()
    criterion.train()
    loss1_meter = LogMeter()
    loss2_meter = LogMeter()
    # get dataloader
    train_loader = give_train_loader(train_set, config)
    train_iter = tqdm(train_loader)

    pred_list = []
    labels_list = []

    for idx, data_info in enumerate(train_iter):
        data_a = preprocess_data(data_info["anchors"], config)
        data_t = preprocess_data(data_info["targets"], config)
        data_c = preprocess_data(data_info["contexts"], config)
        labels = preprocess_data(data_info["labels"], config).float()

        # get similarity
        sim = model(data_a, data_t, data_c)
        loss_sim, loss_sim_inter = criterion(sim, labels)
        loss = loss_sim + loss_sim_inter

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss1_meter.append(loss_sim.item())
        loss2_meter.append(loss_sim_inter.item())

        pred_list.append(output_mapping(sim[0]).detach())
        labels_list.append(labels)

        if idx % 10 == 0:
            train_iter.set_description(f"Loss-sim={loss1_meter.avg()}, Loss-sim-inter:{loss2_meter.avg()}")
    scheduler.step()
    return torch.cat(pred_list, dim=0).cpu().numpy(), torch.cat(labels_list, dim=0).cpu().numpy()

def predict(
    model, val_set, config
):
    model.eval()
    # get dataloader
    val_loader = give_train_loader(val_set, config)
    val_iter = tqdm(val_loader)
    
    pred_list = []
    labels_list = []
    with torch.no_grad():
        for data_info in val_iter:
            data_a = preprocess_data(data_info["anchors"], config)
            data_t = preprocess_data(data_info["targets"], config)
            data_c = preprocess_data(data_info["contexts"], config)
            labels = data_info["labels"]
            
            # get similarity
            sim = model(data_a, data_t, data_c)
            pred = output_mapping(sim)

            pred_list.append(pred.cpu())
            labels_list.append(labels)
    return torch.cat(pred_list, dim=0).cpu().numpy(), torch.cat(labels_list, dim=0).cpu().numpy()

def output_mapping(output):
    """
    NOTE: Only suitable for ShiftMSE Loss
    """
    output = torch.floor(5 * output - 1e-8) * 0.25
    return output


