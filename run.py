from typing import Tuple, List

import math
import libtmux
import argparse
import yaml
from collections import defaultdict
import torch

def get_config(file_path: str) -> Tuple[str, List[str], List[str]]:
    f = open(file_path, mode='r', encoding='utf-8')
    run_config = yaml.load(f, yaml.FullLoader)
    env_cmd = run_config['env']
    tmux_name = run_config['name']
    cmd_list = run_config['cmd']
    return tmux_name, env_cmd, cmd_list

def allocate(cmd_list: List[str], card_per_exp: int, device: list) -> dict:
    if device == -1:
        device = list(range(torch.cuda.device_count()))
    num_split = math.floor(len(device) / card_per_exp)
    card_env = {}
    for idx in range(num_split):
        sub_env = list(range(idx*card_per_exp, (idx+1)*card_per_exp))
        sub_env = [str(device[int(item)]) for item in sub_env]
        sub_env = ','.join(sub_env)
        card_env[idx] = f'CUDA_VISIBLE_DEVICES={sub_env}'

    cmd_dict = defaultdict(list)
    for idx, cmd in enumerate(cmd_list):
        idx = idx % num_split
        env = card_env[idx]
        cmd_dict[idx].append(f'{env} {cmd}')
    
    return cmd_dict

class TmuxManager:
    def __init__(self, session_name: str):
        self.session_name = session_name
        self.server = libtmux.Server()
        self.session = self.server.find_where({'session_name': self.session_name})
        if self.session is None:
            print('Create new session')
            self.session = self.server.new_session(session_name=self.session_name)
        else:
            print('Session exists!')
        self.window_list = self.session.list_windows()
        self.curr_window = self.window_list[0]
        self.curr_pane = self.curr_window.list_panes()[0]
    
    def get_window(self, window_idx: int = None, *args):
        self.window_list = self.session.list_windows()
        if window_idx >= len(self.window_list):
            self.curr_window = self.session.new_window(
                window_name=str(window_idx),
                attach=False
            )
        else:
            self.curr_window = self.window_list[window_idx]
        self.curr_pane = self.curr_window.list_panes()[0]
    
    def send_cmd(self, cmd: str):
        self.curr_pane.send_keys(cmd)
    
    def close_all(self):
        for w in self.window_list:
            w.kill_window()


def clear_tmux(session_name: str):
    target_session = TmuxManager(session_name=session_name)
    target_session.close_all()
    del target_session

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run code via tmux')
    parser.add_argument('--delete', action='store_true', default=False)
    parser.add_argument('--path', type=str, default='./start.yaml')
    parser.add_argument('--card-per-exp', type=int, default=1)
    parser.add_argument('--device', type=int, nargs='+', default=-1)
    opt = parser.parse_args()
    
    tmux_name, env_cmd, cmd_list = get_config(opt.path)
    if opt.delete:
        clear_tmux(tmux_name)
    else:
        cmd_dict = allocate(cmd_list, opt.card_per_exp, opt.device)
        
        # launch experiments
        tmux_manager = TmuxManager(tmux_name)
        for k, v in cmd_dict.items():
            tmux_manager.get_window(k)
            for cmd in env_cmd:
                tmux_manager.send_cmd(cmd)
            for sub_v in v:
                print(f'{k}: {sub_v}')
                tmux_manager.send_cmd(sub_v)
       
