"""
This file would be called with Regressor, stepSize, lastKDays and Datafile of interest
from the main file caller.py to find the validation errors
"""
import os
import sys
import time
import gpflow
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
import xgboost

from utils import mae, rmse, getfName

# argparse
parser = argparse.ArgumentParser(
    description='Called Interpolator, saves relavant csvs in loc')
parser.add_argument(
    '--reg', metavar='xgb|xgbRF|svr|knn|las|gpST', dest='reg', default='knn',
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
    hyperparameters,
    ):

    stepSize = args.stepSize
    lastKDays = args.lastKDays
    datafile = args.datafile
    reg_passed = args.reg
    totalDays = args.totalDays
    loc = args.loc
    del args
    fname = getfName(datafile)
    '''Finds the rmse and mae by doing nested cross validation over the dataset'''
    counter = 0
    splits = 6 # kfold nested cross-validation (Fixed)
    contextDays = 30 # This is a the amount of data (in days) to start with. (Fixed)
    contextDays = lastKDays

    df = pd.read_csv(datafile)
    assert(df.columns[0] == 'Unnamed: 0')
    df = df[df.columns[1:]] # assuming the first col is unlabled
    times = df['ts'].unique()
    times.sort()
    if totalDays is None:
        totalDays = len(times)
    X_cols = list(df.columns)
    X_cols.remove('PM2.5')
    X_cols.remove('station_id')
    print ()
    print ('-' * 80)
    print ("Features Used: ", X_cols)
    print ('-' * 80)
    print ()
    y_col = ['PM2.5']

    allStations = df['station_id'].unique()
    allStations.sort()

    kfout = KFold(n_splits=splits, shuffle=True, random_state=0)
    kfin = KFold(n_splits=splits - 1, shuffle=True, random_state=0)

    outdf = pd.DataFrame()
    for kout, (sts_ftrain_index, sts_test_index) in enumerate(kfout.split(allStations)):
        store = { # this is for nest cross validation
            'fname': [],
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

        # Finding the validation error if not gp
        if Regressor.__name__ not in ["GPR"]:
            for kin, (sts_train_index, sts_val_index) in enumerate(kfin.split(sts_ftrain_index)):

                # getting the correct stations
                sts_test = allStations[sts_test_index]
                sts_val = allStations[sts_ftrain_index[sts_val_index]]
                sts_train = allStations[sts_ftrain_index[sts_train_index]]
                sts_train_val = allStations[sts_ftrain_index]

                # plotting for checking if things are correct
                # plt.scatter(sts_test, [1]*len(sts_test), c='r', alpha=.6)
                # plt.scatter(sts_val, [1]*len(sts_val), c='b', alpha=.6)
                # plt.scatter(sts_train, [1]*len(sts_train), c='c', alpha=.6)
                # plt.show()

                # getting the train and val sets accroding to stations
                test_df = df[df['station_id'].isin(sts_test)]
                val_df = df[df['station_id'].isin(sts_val)]
                train_df = df[df['station_id'].isin(sts_train)]
                train_val_df = df[df['station_id'].isin(sts_train_val)]

                # getting the temporally relevant data
                for time_ix in range(contextDays - 1, totalDays, stepSize): # zero index
                    # data before today
                    temporal_train_df = train_df[train_df['ts'] <= times[time_ix]]
                    temporal_val_df = val_df[val_df['ts'] == times[time_ix]]
                    temporal_test_df = test_df[test_df['ts'] == times[time_ix]] # for avoid calcs for empty tests

                    # data after contextDays - lastKDays
                    temp = max(0, time_ix - lastKDays + 1)
                    temporal_train_df = temporal_train_df[temporal_train_df['ts'] >= times[temp]]

                    # plotting the training data -- Debugging
                    # for ix, temp_df in zip("gck", [temporal_train_df, temporal_val_df, temporal_test_df]):
                    #     plt.scatter(temp_df["ts"].values, temp_df["station_id"].values, c=ix, alpha=0.3, s=30)
                    # plt.xlim(-0.03, 1.03)
                    # plt.ylim(1000, 1038)
                    # plt.axvline(x = times[time_ix], alpha=.5, c='r')
                    # plt.legend(["Today's Day", "Train", "Validation", "Test"])
                    # plt.title(f"Data fed for lastKDays={lastKDays}")
                    # plt.xlabel("Day # (Scaled)")
                    # plt.ylabel("Sation IDs")
                    # plt.savefig(f"{time_ix}.png", dpi=120)
                    # plt.show()

                    # checking if dfs contain atleast one, row, else continue
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
                        # TODO if using GPFLOW, take care of Cholesky Decomp Failure
                        try:
                            reg = Regressor(**hy)
                            reg.fit(temporal_train_df[X_cols].values, temporal_train_df[y_col].values.ravel())
                            predictions = reg.predict(temporal_val_df[X_cols].values)
                        except Exception as e:
                            print ()
                            print ('-' * 80)
                            print (e)
                            print ('-' * 80)
                            print ()
                            continue

                        rmse0 = rmse(predictions, temporal_val_df[y_col].values)
                        mae0 = mae(predictions, temporal_val_df[y_col].values)

                        # for getting the best_hy_ix later
                        store['fname'].append(fname)
                        store['is_val_error'].append(True)
                        store['reg'].append(Regressor.__name__)
                        store['stepSize'].append(stepSize)
                        store['lastKDays'].append(lastKDays)
                        store['kout'].append(kout)
                        store['kin'].append(kin)
                        store['time_ix'].append(time_ix)
                        store['hy_ix'].append(hy_ix)
                        store['rmse'].append(rmse0)
                        store['mae'].append(mae0)

            val_err_df = pd.DataFrame(store) # kout is const, time_ix varies
            # added to final dataframe to return + copy added
            outdf = outdf.append(val_err_df, ignore_index=True)
            # temp results being stored
            tempstr = '/'.join([Regressor.__name__, str(lastKDays), str(stepSize)])
            store_path = f"./{loc}/{fname}/temp_results/{tempstr}/{kout}_{kin}/{time_ix}"
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            print (f"storing at {store_path}")
            outdf.to_csv(store_path + '/results.csv', index=None)

            # preparing for finding the test error
            store = { k: [] for k in store.keys() } # reinit store
            for time_ix in times_val: # zero index
                val_err_df1 = val_err_df.groupby("time_ix").get_group(time_ix) # kout fixed, time_ix fixed
                d = {}
                for hy_ix, val_err_df2 in val_err_df1.groupby("hy_ix"):
                    mean_rmse = val_err_df2["rmse"].values.mean()
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
                store['fname'].append(fname)
                store['is_val_error'].append(False)
                store['reg'].append(Regressor.__name__)
                store['stepSize'].append(stepSize)
                store['lastKDays'].append(lastKDays)
                store['kout'].append(kout)
                store['kin'].append(-1) # doesn't make sense for this.
                store['time_ix'].append(time_ix)
                store['hy_ix'].append(best_hy_ix)
                store['rmse'].append(rmse0)
                store['mae'].append(mae0)

            test_err_df = pd.DataFrame(store)
            # intermediate saving of outdf
            outdf = outdf.append(
                test_err_df, 
                ignore_index=True
                )

            # temp results being stored
            tempstr = '/'.join([Regressor.__name__, str(lastKDays), str(stepSize)])
            store_path = f"./{loc}/{fname}/temp_results/{tempstr}/{kout}_{kin}/{time_ix}"
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            print (f"storing at {store_path}")
            outdf.to_csv(store_path + '/results.csv', index=None)

        if Regressor.__name__ in ["GPR"]:
            # reset stuff
            tf.reset_default_graph()
            graph = tf.get_default_graph()
            gpflow.reset_default_session(graph=graph)

            # kernel magik
            if reg_passed == 'gpST': # GP Spatial Temporal
                xy_matern_1 = gpflow.kernels.Matern32(input_dim=2, ARD=True, active_dims=[0, 1])
                xy_matern_2 = gpflow.kernels.Matern32(input_dim=2, ARD=True, active_dims=[0, 1])
                timeKernel = gpflow.kernels.Matern32(input_dim=1, active_dims=[2])
                overall_kernel = (xy_matern_1 + xy_matern_2) * timeKernel

            elif reg_passed == 'gpFULL': # GP Full Data
                raise NotImplemented
                xy_matern_1 = gpflow.kernels.Matern32(input_dim=2, ARD=True, active_dims=[0, 1])
                xy_matern_2 = gpflow.kernels.Matern32(input_dim=2, ARD=True, active_dims=[0, 1])
                t_matern = gpflow.kernels.Matern32(input_dim=1, active_dims=[2])
                t_other = [gpflow.kernels.Matern32(input_dim=1, active_dims=[2])*gpflow.kernels.Periodic(input_dim=1, active_dims=[2]) for i in range(5)]
                timeKernel = t_matern
                for i in t_other:
                    time = time + i
                combined = gpflow.kernels.RBF(input_dim = 1, active_dims = [4])*(gpflow.kernels.Matern52(input_dim = 2, active_dims = [3, 5], ARD=True) + gpflow.kernels.Matern32(input_dim = 2, active_dims = [3,5], ARD=True))
                wsk = gpflow.kernels.RBF(input_dim = 2, active_dims = [6,7], ARD=True)
                weathk = gpflow.kernels.RBF(input_dim = 1, active_dims = [8])
                overall_kernel = (xy_matern_1 + xy_matern_2) * time * combined * wsk * weathk

            # GET DATA
            sts_test = allStations[sts_test_index]
            sts_train_val = allStations[sts_ftrain_index]

            test_df = df[df['station_id'].isin(sts_test)]
            train_val_df = df[df['station_id'].isin(sts_train_val)]

            for time_ix in range(contextDays - 1, totalDays, stepSize): # zero index
                # data before today
                temporal_train_val_df = train_val_df[train_val_df['ts'] <= times[time_ix]]
                temporal_test_df = test_df[test_df['ts'] == times[time_ix]] # for avoid calcs for empty tests

                # data after contextDays - lastKDays
                temp = max(0, time_ix - lastKDays + 1)
                temporal_train_val_df = temporal_train_val_df[temporal_train_val_df['ts'] >= times[temp]]
                del temp

                # checking if dfs contain atleast one, row, else continue
                trainable = True
                for temp_df in [temporal_train_val_df, temporal_test_df]:
                    if temp_df.shape[0] == 0:
                        trainable = False
                if not trainable:
                    continue

                # model init
                # print (temporal_train_val_df[X_cols]) # we can specify cols for safety
                print ()
                print ("Counter: ", counter)
                print ('-' * 80)
                print ("Try to pass data to obj")
                #################################################
                if reg_passed == 'gpST':
                    assert(len(X_cols) == 3)
                else:
                    raise NotImplemented
                X = temporal_train_val_df[X_cols].values
                y = temporal_train_val_df[y_col].values
                try:
                    counter += 1
                    reg = Regressor(
                        X,
                        y,
                        kern = overall_kernel,
                        mean_function = None
                    )
                    print ("At least going in the reg obj")

                    # optimize
                    opt = gpflow.train.ScipyOptimizer()
                    opt.minimize(reg)
                    print ("Optimization Succeeded!")

                    # predict
                    predictions, variance = reg.predict_y(temporal_test_df[X_cols].values)
                    print ("Trained. time_ix, kout", time_ix, kout)

                    rmse0 = rmse(predictions, temporal_test_df[y_col].values)
                    mae0 = mae(predictions, temporal_test_df[y_col].values)
                    # storing the results for test errors
                    store['fname'].append(fname)
                    store['is_val_error'].append(False)
                    store['reg'].append(Regressor.__name__)
                    store['stepSize'].append(stepSize)
                    store['lastKDays'].append(lastKDays)
                    store['kout'].append(kout)
                    store['kin'].append(-1) # doesn't make sense for this.
                    store['time_ix'].append(time_ix)
                    store['hy_ix'].append(-1) # doesn't make sense for this.
                    store['rmse'].append(rmse0)
                    store['mae'].append(mae0)
                    hy_ix = -1

                except Exception as e:
                    print ("not_trained.")
                    print(e)
                    continue

                print ('-' * 80)
                print ()

                gp_results_df = pd.DataFrame(store)
                # added to final dataframe to return + copy added
                # temp results being stored
                tempstr = '/'.join([Regressor.__name__, str(lastKDays), str(stepSize)])
                store_path = f"./{loc}/{fname}/temp_results/{tempstr}/{kout}_{-1}/{time_ix}"
                if not os.path.exists(store_path):
                    os.makedirs(store_path)
                gp_results_df.to_csv(store_path + '/results.csv', index=None)

        test_err_df = pd.DataFrame(store)
        # intermediate saving of outdf
        outdf = outdf.append(test_err_df, ignore_index=True) # added to final dataframe to return
        tempstr = '/'.join([Regressor.__name__, str(lastKDays), str(stepSize)])
        store_path = f"./{loc}/{fname}/temp_results_append/{tempstr}/{kout}_{-1}"
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        outdf.to_csv(store_path + '/results.csv', index=None)

        print ()
        print ('-' * 80)
        print(f"{kout + 1}th Outer KFold done.")
        print ('-' * 80)
        print ()
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

    elif reg == 'xgbRF':
        Regressor = xgboost.XGBRFRegressor
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

    elif reg in ['gpST', 'gpFULL']:
        Regressor = gpflow.models.GPR
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)

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
    print("RESULTS")
    print(results.head())

    # saving the results

    if args.s:
        store_path = f"./{args.loc}/{fname}/{args.reg}/{args.lastKDays}/{args.stepSize}"
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        results.to_csv(store_path + '/results.csv', index=None)
