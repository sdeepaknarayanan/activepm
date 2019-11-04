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
    description='Called Interpolator, saves relavant csvs in ')
parser.add_argument(
    '--reg', metavar='xgb|svr|knn|las|gpST', dest='reg', default='knn',
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
parser.add_argument('-s', help="save results", action='store_true', default=False)

# utility functions
def rmse_mae_over(
    stepSize,
    lastKDays,
    Regressor,
    hyperparameters,
    datafile
    ):
    '''Finds the rmse and mae by doing nested cross validation over the dataset'''
    counter = 0
    splits = 6 # kfold nested cross-validation (Fixed)
    contextDays = 30 # This is a the amount of data (in days) to start with. (Fixed)

    df = pd.read_csv(datafile)
    df = df[df.columns[1:]] # assuming the first col is unlabled
    times = df['ts'].unique()
    times.sort()
    totalDays = len(times)
    X_cols = list(df.columns)
    X_cols.remove('PM2.5')
    X_cols.remove('station_id')
    y_col = ['PM2.5']

    allStations = df['station_id'].unique()
    allStations.sort()

    kfout = KFold(n_splits=splits, shuffle=True, random_state=0)
    kfin = KFold(n_splits=splits - 1, shuffle=True, random_state=0)

    outdf = pd.DataFrame()
    for kout, (sts_ftrain_index, sts_test_index) in enumerate(kfout.split(allStations)):
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
        
        # Finding the validation error if not gp
        if Regressor.__name__ != "GPR":
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
                
                # getting the train and val sets accroding to stations
                test_df = df[df['station_id'].isin(sts_test)]
                val_df = df[df['station_id'].isin(sts_val)]
                train_df = df[df['station_id'].isin(sts_train)]
                
                # checking if there is some intersection, should not be
                # print(np.intersect1d(test_df['station_id'].unique(), val_df['station_id'].unique()))
                # print(np.intersect1d(train_df['station_id'].unique(), val_df['station_id'].unique()))
                # print(np.intersect1d(train_df['station_id'].unique(), test_df['station_id'].unique()))
                
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
                    # # plt.show()
                    # plt.close()
                    
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
                        reg = Regressor(**hy)
                        reg.fit(temporal_train_df[X_cols].values, temporal_train_df[y_col].values.ravel())
                        predictions = reg.predict(temporal_val_df[X_cols].values)

                        rmse0 = rmse(predictions, temporal_val_df[y_col].values)
                        mae0 = mae(predictions, temporal_val_df[y_col].values)

                        # for getting the best_hy_ix later
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

            val_err_df = pd.DataFrame(store)
            # added to final dataframe to return + copy added
            outdf = outdf.append(val_err_df, ignore_index=True)

            # print ("Validation done.")

            # preparing for finding the test error
            # taking the mean of rmse accross the time_ix dim.
            tempstore = val_err_df.loc[0:0].copy()
            tempstore.drop(index=tempstore.index, inplace=True) # getting an empty df with correct dtypes
            assert (tempstore.shape[0] == 0)

            for kInSelect in range(splits - 1):
                tempdf2 = val_err_df[val_err_df['kin'] == kInSelect]
                for hy_ix in range(len(hyperparameters)):
                    tempdf3 = tempdf2[tempdf2["hy_ix"] == hy_ix]
                    rmse_val, mae_val = tempdf3[['rmse', 'mae']].mean().copy()
                    tempstore.loc[tempstore.shape[0]] = tempdf3.loc[tempdf3.index[0]].copy() # randomly add
                    # editing the last row added, 
                    # no side effects as tempdf3 would never be used again
                    # print (tempstore)
                    tempstore.loc[tempstore.shape[0] - 1][['rmse', 'mae', 'time_ix']] = rmse_val, mae_val, -1
                    # print ("SHOULD BE A EMPTY") # it is, append copies
                    # print (outdf[outdf['rmse'] == rmse_val])
                    # time_ix is not relevant
            
            assert tempstore.shape[0] == (splits-1) * len(hyperparameters)

            # print ('Reduced on dimension essentially.')


            # this for nested cross validation. the way we choose the best hyperparameters
            # for the second experiment will be cleared later.
            # TODO Discuss this with deepak.
            # tempstore now contains mean rmse (across time_stamps)
            # we now choose the best_hy_ix for a perticular kout. i.e.
            # we will only have `splits` number of best_hy_ix

            # TODO Need to assure that len(hyperparameters) >= num of splits
            ix = tempstore['rmse'].idxmin()
            best_hy_ix = tempstore.loc[ix]["hy_ix"]
            hy = hyperparameters[best_hy_ix]

        # find the error values on the test set
        store = { k: [] for k in store.keys() } # reinit store
        for time_ix in range(contextDays-1, totalDays, stepSize): # zero index

            # getting the correct stations
            sts_test = allStations[sts_test_index]
            # this means train + val in nested cross validation
            sts_train_val = allStations[sts_ftrain_index]
            
            # plotting for checking if things are correct
            # plt.scatter(sts_test, [1]*len(sts_test), c='r', alpha=.6)
            # plt.scatter(sts_val, [1]*len(sts_val), c='b', alpha=.6)
            # plt.scatter(sts_train, [1]*len(sts_train), c='c', alpha=.6)
            # plt.show()
            
            # getting the train, test, val sets accroding to stations
            test_df = df[df['station_id'].isin(sts_test)]
            train_val_df = df[df['station_id'].isin(sts_train_val)]

            # data before today
            temporal_train_val_df = train_val_df[train_val_df['ts'] <= times[time_ix]]
            # temporal_val_df = val_df[val_df['ts'] == times[time_ix]]
            temporal_test_df = test_df[test_df['ts'] == times[time_ix]]

            # data after contextDays - lastKDays
            temp = max(0, time_ix - lastKDays + 1)
            temporal_train_val_df = temporal_train_val_df[temporal_train_val_df['ts'] >= times[temp]]

            # avoid calcs # we have kouts, that are having test sets == 0
            trainable = True
            for temp_df in [temporal_train_val_df, temporal_test_df]:
                if temp_df.shape[0] == 0:
                    trainable = False
            if not trainable:
                # print ()
                # print ()
                # print ("ONE OF THE TESTSETS FORMED BY THE OUTER KFOLD IS EMPTY!!!!!!")
                # print ()
                # print ()
                continue

            # plotting the training data -- Debugging
            # for ix, temp_df in zip("ck", [temporal_train_val_df, temporal_test_df]):
            #     plt.scatter(temp_df["ts"].values, temp_df["station_id"].values, c=ix, alpha=0.3, s=30)
            # plt.xlim(-0.03, 1.03)
            # plt.ylim(1000, 1038)
            # plt.axvline(x = times[time_ix], alpha=.5, c='r')
            # plt.legend(["Today's Day", "Train_Validation", "Test"])
            # plt.title(f"Data fed for lastKDays={lastKDays}")
            # plt.xlabel("Day # (Scaled)")
            # plt.ylabel("Sation IDs")
            # plt.savefig(f"{time_ix}.png", dpi=120)
            # plt.show()
            # plt.close()

            # initilize the regressor with hyperparams
            counter += 1
            if Regressor.__name__ == "GPR":
                try: # try training
                    # reset stuff
                    tf.reset_default_graph()
                    graph = tf.get_default_graph()
                    gpflow.reset_default_session(graph=graph)

                    # kernel magik
                    xy_matern_1 = gpflow.kernels.Matern32(input_dim=2, ARD=True, active_dims=[0, 1])
                    xy_matern_2 = gpflow.kernels.Matern32(input_dim=2, ARD=True, active_dims=[0, 1])
                    t_matern = gpflow.kernels.Matern32(input_dim=1, active_dims=[2])
                    t_other = [gpflow.kernels.Matern32(input_dim=1, active_dims=[2])*gpflow.kernels.Periodic(input_dim=1, active_dims=[2]) for i in range(5)]
                    time = t_matern
                    for i in t_other:
                        time = time + i
                    overall_kernel = (xy_matern_1 + xy_matern_2) * time
                    
                    # model init
                    # print (temporal_train_val_df[X_cols]) # we can specify cols for safety
                    print ("Try to pass data to obj")
                    reg = Regressor(
                        temporal_train_val_df[["latitude","longitude","ts"]].values,
                        temporal_train_val_df[y_col].values,
                        kern = overall_kernel,
                        mean_function = None
                    )
                    print ("At least going in the reg obj")

                    # optimize
                    opt = gpflow.train.ScipyOptimizer()
                    opt.minimize(reg)
                    print ("Optimization Succeeded")

                    # predict
                    predictions, variance = reg.predict_y(temporal_test_df[["latitude","longitude","ts"]].values)
                    print ("Trained.")
                    hy_ix = -1
                except Exception as e: 
                    print ("not_trained.")
                    print(e)
                    continue

            else: 
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
            store['kin'].append(-1) # doesn't make sense for this.
            store['time_ix'].append(time_ix)
            store['hy_ix'].append(hy_ix)
            store['rmse'].append(rmse0)
            store['mae'].append(mae0)
        
        test_err_df = pd.DataFrame(store)
        outdf = outdf.append(test_err_df, ignore_index=True) # added to final dataframe to return

        print(f"{kout + 1}th Outer KFold done.")
    return outdf, counter

def setRegHy(args):
    '''Sets relevant hyperparameters and regressor, based on the args passed'''
    hyperparameters = [{}] # first use the default hyperparams :)
    reg = args.reg
    lastKDays = args.lastKDays

    if reg == 'svr':
        from thundersvm import SVR
        Regressor = SVR
        hyperparameters = [] # SVR cries when we pass empty params
        C = [10**i for i in [-3, -2, 0, 1, 3, 5]]

        for c in C:
            hy = {
               'C': c,
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

    elif reg == 'gpST':
        if lastKDays <= 30:
            Regressor = gpflow.models.GPR
        else : # sparse GP
            Regressor = gpflow.models.SGPR
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
    # setting relevant regressors and hyperparameters
    Regressor, hyperparameters = setRegHy(arg)
    start = time.time()
    results, counter = rmse_mae_over(
        args.stepSize,
        args.lastKDays, # svr changes this to be within 30
        Regressor, # set by the function setRegHy above.
        hyperparameters, # set by the function setRegHy above.
        args.datafile,
    )
    end = time.time()
    print("Time Taken (s):", end - start)
    print("Trainings performed:", counter)
    print()
    print("RESULTS")
    print(results.head())
    fname = getfName(args.datafile)
    
    # saving the results
    if args.s:
        store_path = f"./results/{fname}/{args.reg}/{args.lastKDays}/{args.stepSize}"
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        results.to_csv(store_path + '/results.csv', index=None)
