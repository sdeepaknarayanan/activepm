import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from utils import mae, rmse

def caller():
    counter = 0
    # hyperparameters
    splits = 6 # kfold nested cross-validation
    stepSize = 5 # sensor placement in every 5 days
    lastKDays = 30 # we forget data before k days
    contextDays = 30 # we start with these many days.
    assert (contextDays <= lastKDays)
    srcname = '../../data/beijingb_scaled.csv'

    # hyperparameters - will be defined by the regressor chosen
    hyperparameters = [{}]
    from sklearn.neighbors import KNeighborsRegressor as Regressor

    # TODO Windspeedx and Windspeedy need to be changed
    # The file to be passed should contain 'PM2.5' as one of the col


    df = pd.read_csv(srcname)
    df = df[df.columns[1:]]
    times = df['ts'].unique()
    times.sort()
    totalDays = len(times)
    X_cols = list(df.columns)
    X_cols.remove('PM2.5')
    y_col = ['PM2.5']

    allStations = df['station_id'].unique()
    allStations.sort()

    kfout = KFold(n_splits=splits, random_state=0)
    kfin = KFold(n_splits=splits - 1, random_state=0)

    store = {
                'reg': [],
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

                    rmse0 = rmse(predictions, val_df[y_col].values)
                    mae0 = mae(predictions, val_df[y_col].values)

                    store['reg'].append(Regressor.__name__)
                    store['kout'].append(kout)
                    store['kin'].append(kin)
                    store['time_ix'].append(time_ix)
                    store['hy_ix'].append(hy_ix)
                    store['rmse0'].append(rmse0)
                    store['mae0'].append(mae0)
                    
    results = pd.DataFrame(store)
    print(results.head())
    return counter

import time

start = time.time()
counter = caller()
end = time.time()
print("Time Taken (s):", end - start)
print("Trainings performed:", counter)

