import torch.nn as nn

class CrossEntropy(nn.Module):
    is_regre = False
    def __init__(self, num_classes=5):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.gap = float(1 / (num_classes-1))
    
    def forward(self, x, label):
        label = (label / self.gap).long()
        loss = self.criterion(x, label)
        return loss