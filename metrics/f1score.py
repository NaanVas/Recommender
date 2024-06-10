from sklearn.metrics import precision_score, recall_score
import numpy as np

def f1_score(actuals, predictions):
    precision = precision_score(np.around(actuals), np.around(predictions), average="macro", zero_division=np.nan)
    recall = recall_score(np.around(actuals), np.around(predictions), average="macro", zero_division=np.nan)
    if precision + recall == 0:
        return 0
    
    return 2* (precision * recall) / (precision + recall)