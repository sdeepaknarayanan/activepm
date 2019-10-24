import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def rmse(ypred, y):
    mse = mean_squared_error(np.array(ypred), np.array(y))
    return np.sqrt(mse)

def mae(ypred, y):
    return mean_absolute_error(np.array(ypred), np.array(y))
