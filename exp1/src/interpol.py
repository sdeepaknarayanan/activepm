"""
This file would be called with Regressor, stepSize, lastKDays and Datafile of interest
from the main file caller.py
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from sklearn.linear_model import Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.neighbors import KNeighborsRegressor
import xgboost
from sklearn.svm import SVR

from utils import mae, rmse, getfName

# argparse
parser = argparse.ArgumentParser(
    description='Called Interpolator, saves relavant csvs in ')
parser.add_argument(
    '--reg', metavar='xgb|svr|knn|las|gp', dest='reg', default='knn',
    help="Regressors to use", type=str
)
parser.add_argument(
    '--stepSize', metavar='INT', dest='stepSize',
    help="Step size the interpolation takes", type=int
)
parser.add_argument(
    '--lastKDays', metavar='INT', dest='lastKDays',
    help="Last K days only relavent for us.", type=int
)
parser.add_argument(
    '--datafile', metavar='PATH', dest='datafile',
    help="data csv", type=str
)

# utility functions
def rmse_mae_over(
    stepSize,
    lastKDays,
    Regressor,
    hyperparameters,
    datafile
    ):
    ''''''
    # changeable stuff
    stepSize = stepSize # sensor placement in every `stepSize` days
    lastKDays = lastKDays # we forget data before k days
    # hyperparameters - will be defined by the regressor chosen
    Regressor = Regressor
    hyperparameters = hyperparameters
    datafile = datafile

    counter = 0
    splits = 6 # kfold nested cross-validation (Fixed)
    contextDays = 30 # This is a the amount of data (in days) to start with. (Fixed)
    # assert (contextDays <= lastKDays)


    # TODO Windspeedx and Windspeedy need to be changed
    # The file to be passed should contain 'PM2.5' as one of the col


    df = pd.read_csv(datafile)
    df = df[df.columns[1:]]
    times = df['ts'].unique()
    times.sort()
    totalDays = len(times)
    X_cols = list(df.columns)
    X_cols.remove('PM2.5')
    X_cols.remove('station_id')
    y_col = ['PM2.5']

    allStations = df['station_id'].unique()
    allStations.sort()

    kfout = KFold(n_splits=splits, random_state=0)
    kfin = KFold(n_splits=splits - 1, random_state=0)

    store = {
                'reg': [],
                'stepSize': [],
                'lastKDays': [],
                'kout': [],
                'kin': [],
                'time_ix': [],
                'hy_ix': [],
                'rmse': [],
                'mae': [],
            }
    for kout, (sts_ftrain_index, sts_test_index) in enumerate(kfout.split(allStations)):
        for kin, (sts_train_index, sts_val_index) in enumerate(kfin.split(sts_ftrain_index)):
            
            # getting the correct stations
            sts_test = allStations[sts_test_index]
            sts_val = allStations[sts_ftrain_index[sts_val_index]]
            sts_train = allStations[sts_ftrain_index[sts_train_index]]
            
            # plotting for checking if things are correct
            # plt.scatter(sts_test, [1]*len(sts_test), c='r', alpha=.6)
            # plt.scatter(sts_val, [1]*len(sts_val), c='b', alpha=.6)
            # plt.scatter(sts_train, [1]*len(sts_train), c='c', alpha=.6)
            # plt.show()
            
            # getting the train, test, val sets accroding to stations
            test_df = df[df['station_id'].isin(sts_test)]
            val_df = df[df['station_id'].isin(sts_val)]
            train_df = df[df['station_id'].isin(sts_train)]
            
            # checking if there is some intersection, should not be
            # print(np.intersect1d(test_df['station_id'].unique(), val_df['station_id'].unique()))
            # print(np.intersect1d(train_df['station_id'].unique(), val_df['station_id'].unique()))
            # print(np.intersect1d(train_df['station_id'].unique(), test_df['station_id'].unique()))
            
            # getting the temporally relevant data
            for time_ix in range(contextDays-1, totalDays, stepSize): # zero index
                # data before today
                temporal_train_df = train_df[train_df['ts'] <= times[time_ix]]
                temporal_val_df = val_df[val_df['ts'] == times[time_ix]]
                temporal_test_df = test_df[test_df['ts'] == times[time_ix]]
                
                # data after contextDays - lastKDays
                temp = max(0, time_ix - lastKDays + 1)
                temporal_train_df = temporal_train_df[temporal_train_df['ts'] >= times[temp]]
                
                # checking if dfs contain atleast one row, else continue
                trainable = True
                for temp_df in [temporal_train_df, temporal_val_df, temporal_test_df]:
                    if temp_df.shape[0] == 0:
                        trainable = False
                if not trainable:
                    continue

                # for all hyperparameters: depend on regressor chosen
                for hy_ix, hy in enumerate(hyperparameters):
                    counter += 1
                    # initilize the regressor with hyperparams
                    reg = Regressor(**hy)
                    reg.fit(train_df[X_cols], train_df[y_col])
                    predictions = reg.predict(val_df[X_cols])
                    # print (predictions)
                    # print (val_df[y_col].values)

                    rmse0 = rmse(predictions, val_df[y_col].values)
                    mae0 = mae(predictions, val_df[y_col].values)

                    store['reg'].append(Regressor.__name__)
                    store['stepSize'].append(stepSize)
                    store['lastKDays'].append(lastKDays)
                    store['kout'].append(kout)
                    store['kin'].append(kin)
                    store['time_ix'].append(time_ix)
                    store['hy_ix'].append(hy_ix)
                    store['rmse'].append(rmse0)
                    store['mae'].append(mae0)

    return store, counter
def setRegHy(reg):
    '''Sets relevant hyperparameters and regressor, based on the args passed'''
    if reg == 'svr':
        Regressor = SVR
    elif reg == 'knn':
        Regressor = KNeighborsRegressor
    elif reg == 'las':
        Regressor = Lasso
        alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        hyperparameters = []
        for alpha in (alphas):
            hy = { 'alpha': alpha }
            hyperparameters.append(hy)
    elif reg == 'xgb':
        Regressor = xgboost.XGBRFRegressor
        # hyperparameters = 
        # TODO we can have something like, if datapoints > 10000
        # use GPU, obviusly, for best timing, find this critical point
    else:
        raise ValueError("We need a predefined Regressor, for sane hyperparameters")
    return (Regressor, hyperparameters)

if __name__ == "__main__":
    args = parser.parse_args()
    print ("Args parsed. Training Started.")
    # setting relevant regressors and hyperparameters
    Regressor, hyperparameters = setRegHy(args.reg)
    start = time.time()
    store, counter = rmse_mae_over(
        args.stepSize,
        args.lastKDays,
        Regressor, # set by the function setRegHy above.
        hyperparameters, # set by the function setRegHy above.
        args.datafile,
    )
    end = time.time()
    print("Time Taken (s):", end - start)
    print("Trainings performed:", counter)
    results = pd.DataFrame(store)
    print()
    print("RESULTS")
    print(results.head())
    fname = getfName(args.datafile)
    
    # saving the the
    store_path = f"./results/{fname}/{args.reg}/{args.lastKDays}/{args.stepSize}"
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    results.to_csv(store_path + '/results.csv', index=None)
