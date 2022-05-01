import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.reshape(len(predictions))
    return {
        "pearson": np.corrcoef(predictions, labels)[0][1]
    }