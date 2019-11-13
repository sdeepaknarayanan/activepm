"""
This file would be called with Regressor, stepSize, lastKDays and Datafile of interest
from the main file caller.py to find the validation errors
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
from sklearn.neighbors import KNeighborsRegressor
import xgboost

from utils import mae, rmse, getfName

# argparse
parser = argparse.ArgumentParser(
    description='Called Interpolator, saves relavant csvs in ')
parser.add_argument(
    '--reg', metavar='xgb|xgbRF|svr|knn|las|gpST|gpFULL', dest='reg', default='knn',
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
parser.add_argument(
    '--gpuid', help="GPU to use", metavar='INT', 
    default=0, type=int
)
parser.add_argument(
    '--totalDays', metavar='INT', default=None,
    help="totalDays", type=int
)
parser.add_argument(
    '--loc', metavar='PATH', default='results',
    help="location to store results", type=str
)
parser.add_argument(
    '-s', help="save results", action='store_true', 
    default=False
)

# utility functions
def rmse_mae_over(
    args,
    Regressor,
    hyperparameters, # only for distinguishing between gpST and gpFULL
    ):

    stepSize = args.stepSize
    lastKDays = args.lastKDays
    datafile = args.datafile
    totalDays = args.totalDays
    fname = getfName(datafile)
    loc = args.loc
    '''Finds the rmse and mae by doing nested cross validation over the dataset'''
    counter = 0
    splits = 6 # kfold nested cross-validation (Fixed)
    contextDays = 30 # This is a the amount of data (in days) to start with. (Fixed)
    contextDays = lastKDays

    df = pd.read_csv(datafile)
    df = df[df.columns[1:]] # assuming the first col is unlabled
    times = df['ts'].unique()
    times.sort()
    if totalDays is None:
        totalDays = len(times)
    X_cols = list(df.columns)
    X_cols.remove('PM2.5')
    X_cols.remove('station_id')
    y_col = ['PM2.5']

    allStations = df['station_id'].unique()
    allStations.sort()

    kfout = KFold(n_splits=splits, shuffle=True, random_state=0)
    kfin = KFold(n_splits=splits - 1, shuffle=True, random_state=0)

    validation_df = pd.read_csv(f"./{args.loc}/{fname}/{args.reg}/{args.lastKDays}/{args.stepSize}/results.csv")
    validation_df = validation_df.groupby("is_val_error").get_group(True)

    outdf = pd.DataFrame()
    for kout, (sts_ftrain_index, sts_test_index) in enumerate(kfout.split(allStations)):
        validation_df0 = validation_df.groupby("kout").get_group(kout)
        assert (validation_df0.shape[0] != 0)

        # getting the correct stations
        sts_test = allStations[sts_test_index]
        # this means train + val in nested cross validation
        sts_train_val = allStations[sts_ftrain_index]
        # getting the train, test, val sets accroding to stations
        test_df = df[df['station_id'].isin(sts_test)]
        train_val_df = df[df['station_id'].isin(sts_train_val)]

        store = { # this is for nest cross validation
                'is_val_error': [],
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

        kin = -1
        times_val = validation_df0["time_ix"].unique()
        times_val.sort()

        for time_ix in times_val: # zero index

            validation_df1 = validation_df0.groupby("time_ix").get_group(time_ix) # kout fixed, time_ix fixed

            d = {}
            for hy_ix, validation_df2 in validation_df1.groupby("hy_ix"):
                mean_rmse = validation_df2["rmse"].values.mean()
                d[hy_ix] = mean_rmse

            # print (d)

            best_hy_ix = min(d, key = d.get)
            # print (best_hy_ix)
            hy = hyperparameters[best_hy_ix]

            # data before today
            temporal_train_val_df = train_val_df[train_val_df['ts'] <= times[time_ix]]
            temporal_test_df = test_df[test_df['ts'] == times[time_ix]]
            temp = max(0, time_ix - lastKDays + 1)
            temporal_train_val_df = temporal_train_val_df[temporal_train_val_df['ts'] >= times[temp]]
            # avoid calcs # we have kouts, that are having test sets == 0
            trainable = True
            for temp_df in [temporal_train_val_df, temporal_test_df]:
                if temp_df.shape[0] == 0:
                    trainable = False
            if not trainable:
                continue
            # initilize the regressor with hyperparams
            counter += 1
            reg = Regressor(**hy)
            reg.fit(temporal_train_val_df[X_cols].values, temporal_train_val_df[y_col].values.ravel())
            predictions = reg.predict(temporal_test_df[X_cols].values)

            rmse0 = rmse(predictions, temporal_test_df[y_col].values)
            mae0 = mae(predictions, temporal_test_df[y_col].values)

            # storing the results for test errors
            store['is_val_error'].append(False)
            store['reg'].append(Regressor.__name__)
            store['stepSize'].append(stepSize)
            store['lastKDays'].append(lastKDays)
            store['kout'].append(kout)
            store['kin'].append(kin) # doesn't make sense for this.
            store['time_ix'].append(time_ix)
            store['hy_ix'].append(best_hy_ix)
            store['rmse'].append(rmse0)
            store['mae'].append(mae0)

        test_err_df = pd.DataFrame(store)
        # intermediate saving of outdf
        outdf = outdf.append(test_err_df, ignore_index=True) # added to final dataframe to return

        print(f"{kout + 1}th Outer KFold done.")
    return outdf, counter

def setRegHy(args):
    '''Sets relevant hyperparameters and regressor, based on the args passed'''
    hyperparameters = [{}] # first use the default hyperparams :)
    reg = args.reg
    lastKDays = args.lastKDays

    if reg == 'svr':
        from sklearn.svm import SVR
        Regressor = SVR
        hyperparameters = [] # SVR cries when we pass empty params
        C = [10**i for i in [-3, -2, 0, 1, 3, 5]]

        for c in C:
            hy = {
               'C': c,
               'gamma': 'auto'
            }
            hyperparameters.append(hy)

    elif reg == 'knn':
        Regressor = KNeighborsRegressor
        n_neighbors = [i for i in range(3, 20, 2)]
        weights=['distance']#, 'uniform'] # essentially idw
        for n in n_neighbors:
            for w in weights:
                hy = {
                    'n_neighbors': n,
                    'weights': w
                }
                hyperparameters.append(hy)

    elif reg == 'las':
        Regressor = Lasso
        alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        for alpha in (alphas):
            hy = { 'alpha': alpha }
            hyperparameters.append(hy)

    elif reg == 'xgb':
        Regressor = xgboost.XGBRegressor
        # hyperparameters given to be searched by Deepak
        depths = [10, 50]
        lrs = [0.01, 0.1, 1]
        estimators = [10, 50]
        for depth in depths:
            for lr in lrs:
                for estimator in estimators:
                    hy = {
                        'max_depth': depth,
                        'learning_rate': lr,
                        'n_estimators': estimator,
            'n_jobs': -1,
                    }
                    hyperparameters.append(hy)
    else:
        raise ValueError("We need a predefined Regressor, for sane hyperparameters")
    return (Regressor, hyperparameters)

if __name__ == "__main__":
    args = parser.parse_args()
    if not args.s: # warn
        print()
        print("NOT SAVING!!!!!!!!!!!!!")
        print()

    print ("Args parsed. Training Started.")
    fname = getfName(args.datafile)
    # setting relevant regressors and hyperparameters
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpuid}"
    Regressor, hyperparameters = setRegHy(args)
    start = time.time()
    results, counter = rmse_mae_over(
        args,
        Regressor, # set by the function setRegHy above.
        hyperparameters, # set by the function setRegHy above.
    )
    end = time.time()
    print("Time Taken (s):", end - start)
    print("Trainings performed:", counter)
    print()
    print("TEST RESULTS")
    print(results.head())

    # saving the results

    if args.s:
        store_path = f"./{args.loc}/{fname}/{args.reg}/{args.lastKDays}/{args.stepSize}"
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        results.to_csv(store_path + '/results_test.csv', index=None)
