import numpy as np

class LogMeter:
    def __init__(self):
        self.value_list = []
    
    def append(self, v):
        self.value_list.append(v)
    
    def avg(self):
        return np.mean(self.value_list)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.reshape(len(predictions))
    return {
        "pearson": np.corrcoef(predictions, labels)[0][1]
    }
