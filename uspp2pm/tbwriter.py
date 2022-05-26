
import functools
import sys
import os
from collections import defaultdict
from typing import Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

_step_count_dict: dict = defaultdict(int)

class DefaultLog:
    pass

global log
log = DefaultLog()

_func_list_ = ['add_scalar', 'add_scalars', 'add_histogram', 'add_image', 
               'add_images', 'add_figure', 'add_video', 'add_audio', 'add_text', 
               'add_graph', 'add_embedding', 'add_hparams']

def do_nothing(*args, **kwargs):
    return

@functools.lru_cache()
def config(output_dir, dist_rank=0, only_main_rank=True):
    global log
    if only_main_rank and (dist_rank != 0):
        for _func_name in _func_list_:
            setattr(log, _func_name, do_nothing)
    else:
        log_dir = os.path.join(output_dir, f'Rank{dist_rank}')
        log = SummaryWriter(
            log_dir=log_dir
        )

def count_step(tag: str) -> int:
    global _step_count_dict
    _step_count_dict[tag] += 1
    return _step_count_dict[tag] - 1

def add_scalar(tag: str, value: Union[Tensor, float, int]):
    curr_step = count_step(tag)
    log.add_scalar(tag, value, global_step=curr_step)