#%%
import torch
import torch.nn as nn

_data = torch.load("./test/data.ckpt")
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
class Bias(nn.Module):
    """
    Only to contain the parameters with 'bias' string
    """
    def __init__(self, num_classes):
        super().__init__()
        self.tl = nn.Parameter(torch.zeros(num_classes))
        self.tr = nn.Parameter(torch.zeros(num_classes))
    
    def forward(self, x):
        return x

class pros(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.gap = 1 / (self.num_classes - 1)
        self.bias = Bias(num_classes - 1)
        self.marks = [self.gap * item for item in list(range(self.num_classes))]
    
    def group(self, i, data, label, s, e):
        data_list = []
        label_list = []
        tol_mask = label < (e - 1e-2)
        tor_mask = label > (s + 1e-2)
        # to left
        beta = self.bias.tl[i]
        ml = data[tol_mask] - s - beta
        data_list.append(data[tol_mask][ml > 0] - beta + beta.item())
        label_list.append(label[tol_mask][ml > 0])
        data_list.append(data[tol_mask][ml <= 0] * 0 + s)
        label_list.append(label[tol_mask][ml <= 0])
        mr = data[tor_mask] - s - beta
        data_list.append(data[tor_mask][mr <= 0] * 0 + s + beta - beta.item())
        label_list.append(label[tor_mask][mr <= 0])
        data_list.append(data[tor_mask][mr > 0])
        label_list.append(label[tor_mask][mr > 0])
        # to right
        beta = self.bias.tr[i]
        mr = data[tor_mask] - e + beta
        data_list.append(data[tor_mask][mr <= 0] + beta - beta.item())
        label_list.append(label[tor_mask][mr <= 0])
        data_list.append(data[tor_mask][mr > 0] * 0 + e)
        label_list.append(label[tor_mask][mr > 0])
        ml = data[tol_mask] - e + beta
        data_list.append(data[tol_mask][ml > 0] * 0 + e - beta + beta.item())
        label_list.append(label[tol_mask][ml > 0])
        data_list.append(data[tol_mask][ml <= 0] * 0 + s)
        label_list.append(label[tol_mask][ml <= 0])
        
        # beta = self.bias.tr[i]
        # mr = data[tor_mask] - e + beta
        # data[tor_mask][mr < 0] = data[tor_mask] + beta - beta.item()
        # data[tor_mask][mr >= 0] = e + 0 * beta
        # ml = data[tol_mask] - e + beta
        # data[tol_mask][ml >= 0] = e - beta + beta.item()

        data = torch.cat(data_list)
        label = torch.cat(label_list)
        return data, label
    
    def forward(self, data, label):
        data = data.clone()
        # for two extreme range
        data[data < 0] = 0
        data[data > 1] = 1
        data_list = []
        label_list = []
        for i in range(self.num_classes - 1):
            start = self.marks[i]
            end = self.marks[i+1]
            range_mask = (data > start) & (data < end)
            sub_data, sub_label = self.group(i, data[range_mask], label[range_mask], start, end)
            data_list.append(sub_data)
            label_list.append(sub_label)
        return torch.cat(data_list), torch.cat(label_list)

#%%
model = pros()
crite = torch.nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print(f"acc: {crite(data, label).item()}")
for i in range(1000):
    out1, out2 = model(data, label)
    loss = crite(out1, out2)
    print(f"acc: {loss.item()}")
    optim.zero_grad()
    loss.backward()
    optim.step()

            

