from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(actuals, predictions):
    return sqrt(mean_squared_error(actuals, predictions))

