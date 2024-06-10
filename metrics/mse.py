from sklearn.metrics import mean_squared_error

def mse(actuals, predictions):
    return mean_squared_error(actuals, predictions)

