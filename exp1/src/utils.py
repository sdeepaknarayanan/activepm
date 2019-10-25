import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def rmse(ypred, y):
    mse = mean_squared_error(np.array(ypred), np.array(y))
    return np.sqrt(mse)

def mae(ypred, y):
    return mean_absolute_error(np.array(ypred), np.array(y))

def getfName(srcname):
    leng = len(srcname)
    for ix in range(leng-1, -1, -1):
        if srcname[ix] == '/':
            if '.csv' in srcname[ix + 1:]:
                return srcname[ix + 1:srcname.index('.csv')]
            else:
                return srcname[ix + 1:]
    
    return None
