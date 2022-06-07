#%%
import torch
import torch.nn as nn

_data = torch.load("./data.ckpt")
data = _data["data"]
label = _data["label"]

class PearsonCorr(nn.Module):
    is_regre = True
    def __init__(self):
        super().__init__()
    
    def forward(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        v_data = data - torch.mean(data, dim=-1, keepdim=True)
        v_target = target - torch.mean(target, dim=-1, keepdim=True)

        nomenator = - torch.sum(v_data * v_target, dim=-1)
        denominator = (
            torch.sqrt(torch.sum(v_data ** 2, dim=-1) + 1e-8) *
            torch.sqrt(torch.sum(v_target ** 2, dim=-1) + 1e-8)
        )
        loss = nomenator / denominator
        return loss

# TODO: 检查参数是否被添加了正则化！

#%%
import matplotlib.pyplot as plt

plt.hist(data, bins=30)

#%%
def process(data, m=0.02):
    mark_list = [0., 0.25, 0.5, 0.75, 1.0]
    item_list = []
    for item in data:
        is_adjust = False
        for mark in mark_list:
            if (mark - m) < item <= (mark + m):
                item_list.append(mark)
                is_adjust = True
        if not is_adjust:
            item_list.append(item.item())
    return torch.tensor(item_list)


crite = PearsonCorr()
pdata = process(data)
print(f"pros: {crite(pdata, label).item()}")
print(f"raw: {crite(data, label).item()}")
pass