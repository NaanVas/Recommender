from .precision import precision
from .recall import recall

def f1_score(actuals, prediction, k):
    prec = precision(actuals, prediction, k)
    rec = recall(actuals, prediction, k)
    if prec + rec == 0:
        return 0
    
    return 2* (prec * rec) / (prec + rec)