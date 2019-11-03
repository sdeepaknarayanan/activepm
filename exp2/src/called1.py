import numpy as np 
from sklearn.model_selection import KFold
import pandas as pd
import sys
sys.path.append("../")
import gpflow
import tensorflow as tf

from gpsampling import GPActive
from qbc_random import ActiveLearning
import xgboost
import multiprocessing




def active(learner, is_random):


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

    if learner == 'gp':


        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)



        gp_objects = {i:{j:{k:0 for k in [10, 20 ,30]} for j in station_split[i]} for i in station_split}

        processes = []
        processes_1 = []
        for i in station_split:
            processes = []
            for j in station_split[i]:
                for train_days in [10, 20, 30]:
                    gp_objects[i][j] = GPActive(
                                        df = df,
                                        train_stations = list(station_split[i][j]['train']),
                                        pool_stations = list(station_split[i][j]['pool']),
                                        test_stations = list(station_split[i][j]['test']),
                                        context_days = train_days - 1 , # train_days - 1
                                        frequency = 30, # 
                                        test_days = 360,  # 
                                        train_days = train_days, # 10, 20, 30
                                        number_to_query = 1,
                                        number_of_seeds= 10,
                                        fname=[i, j]
                                    )

                    processes.append(multiprocessing.Process(target=gp_objects[i][j][train_days].active_gp, args=()))
            for proc in processes:
                proc.start()
            for proc in processes:
                proc.join()




    elif learner == 'qbc':

        processes_1 = []


        learners = {
                    0:xgboost.XGBRegressor(), 
                    1:xgboost.XGBRegressor(max_depth=10, learning_rate=1, n_estimators=10),
                    2:xgboost.XGBRegressor(max_depth=10, learning_rate=1, n_estimators=50),
                    3:xgboost.XGBRegressor(max_depth=50, learning_rate=1, n_estimators=10),
                    4:xgboost.XGBRegressor(max_depth=50, learning_rate=1, n_estimators=50),
                   }

        qbc_objects = {i:{j:{k:0 for k in [10, 20 ,30]} for j in station_split[i]} for i in station_split}

        for i in station_split:
            for j in station_split[i]:

                processes = []


                for train_days in [10, 20, 30]:
                
                    qbc_objects[i][j][train_days] = ActiveLearning(
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
                                        fname = [i, j]
                                    )

                    ## Edit RANDOM SAMPLING FROM PREVIOUS COMMIT
                    processes.append(multiprocessing.Process(target=qbc_objects[i][j][train_days].random_sampling, args=()))
                for proc in processes:
                    proc.start()

                for proc in processes:
                    proc.join()


                    # processes_1.append(multiprocessing.Process(target = qbc_objects[i][j][train_days].random_sampling, args = ()))

        # for proc in processes:
        #     proc.start()

        # for proc in processes:
        #     proc.join()

        print("Query by Committee done")

        for proc in processes_1:
            proc.start()

        for proc in processes_1:
            proc.join()

        print("Random Done")















    else:
        print("Not a correct learner -- enter gp or qbc")



if __name__ == '__main__':
    learner = input()
    active(learner)
