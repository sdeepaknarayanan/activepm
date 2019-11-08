import numpy as np 
from sklearn.model_selection import KFold
import pandas as pd
import sys
sys.path.append("../")
from qbc_random import ActiveLearning
import xgboost
import multiprocessing
import argparse


# argparse
parser = argparse.ArgumentParser(
    description='Arguments for QBC')
parser.add_argument(
    '--kout',  metavar='INT', dest='kout',
    help="outer fold index", type=int
)
parser.add_argument(
    '--kin', metavar='INT', dest='kin',
    help="inner fold index", type=int
)
parser.add_argument(
    '--train_days', metavar='INT', dest='train_days',
    help="one of 10 20 30", type=int
)
parser.add_argument('--act', metavar='INT', dest='isact', 
    help="active or random", type=int
)



def active(train_days, i, j, isact):
    df = pd.read_csv("../../data/beijingb_scaled.csv", index_col = 0)
    df = df.rename(columns={'ts': 'Time', 'station_id': 'Station'})

    stations = df['Station'].unique()
    stations.sort()

    splits = 6

    kfout = KFold(n_splits=splits, shuffle=True, random_state=0)
    kfin = KFold(n_splits=splits - 1, shuffle=True, random_state=0)

    station_split = {}

    for kout, (sts_ftrain_index, sts_test_index) in enumerate(kfout.split(stations)):
        station_split[kout] = {}
        for kin, (sts_train_index, sts_val_index) in enumerate(kfin.split(sts_ftrain_index)):
            sts_test = stations[sts_test_index]
            sts_val = stations[sts_ftrain_index[sts_val_index]]
            sts_train = stations[sts_ftrain_index[sts_train_index]]
            station_split[kout][kin] = {
                'test': sts_test,
                'train': sts_val,
                'pool': sts_train,
            }


    learners = {
            0:xgboost.XGBRegressor(), 
            1:xgboost.XGBRegressor(max_depth=10, learning_rate=1, n_estimators=10),
            2:xgboost.XGBRegressor(max_depth=10, learning_rate=1, n_estimators=50),
            3:xgboost.XGBRegressor(max_depth=50, learning_rate=1, n_estimators=10),
            4:xgboost.XGBRegressor(max_depth=50, learning_rate=1, n_estimators=50),
           }

    qbc = ActiveLearning(
                        df = df,
                        train_stations = list(station_split[i][j]['train']) ,
                        pool_stations = list(station_split[i][j]['pool']),
                        test_stations = list(station_split[i][j]['test']),
                        learners = learners,
                        context_days = train_days - 1,
                        frequency = 30,
                        test_days = 365 - train_days,
                        train_days = train_days,
                        number_of_seeds = 10,
                        number_to_query = 1,
                        fname = [i, j],
                        gp_choice = True
                        )

    print(isact)
    
    if isact == 1:
        qbc.querybycommittee()
    else:
        qbc.random_sampling()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args.isact)
    active(args.train_days, args.kout, args.kin, args.isact)