#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:34:29 2019

@author: deepak
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from copy import deepcopy
import matplotlib
import traceback
    


class ActiveLearning():

    def __init__(
            self, df, train_stations, 
            pool_stations, test_stations,
            learners,
            context_days,
            frequency,
            total_days,
            test_days = None
            ):

        self.df = df
        self.train_stations = train_stations
        self.pool_stations = pool_stations
        self.test_stations = test_stations
        self.learners = learners
        self.frequency = frequency
        self.current_day = context_days
        self.context_days = context_days
        self.total_days = total_days

        self.train_columns = list(df.columns)
        self.train_columns.remove('PM2.5')
        self.train_columns.remove('Station')

        ## First I am choosing by stations, it can't be the case that there is no data
        ## for a given station. So there can be no key error here. 
        self.train = pd.concat([self.df.groupby('Station').get_group(station) for station in train_stations])
        self.pool = pd.concat([self.df.groupby('Station').get_group(station) for station in pool_stations])
        self.test = pd.concat([self.df.groupby('Station').get_group(station) for station in test_stations])

        self.timestamps = self.df['Time'].unique()
        self.timestamps.sort()
        initial_timestamps = self.timestamps[:self.context_days]

        # Here also no key error occurs. This is because, The df can be empty but it still gets concatenated
        
        self.train = pd.concat([self.train.groupby('Time').get_group(time) for time in initial_timestamps])
        self.pool = pd.concat([self.pool.groupby('Time').get_group(time) for time in initial_timestamps])

        self.current_test = self.test.groupby('Time').get_group(self.timestamps[self.current_day - 1])
        self.X_test = self.current_test[self.train_columns]
        self.y_test = self.current_test[['PM2.5']]

        self.X_train = self.train[self.train_columns]
        self.y_train = self.train[['PM2.5']]

        self.X_pool = self.pool[self.train_columns]
        self.y_pool = self.pool[['PM2.5']]

        # self.X_test = self.test[self.train_columns]
        # self.y_test = self.test[['PM2.5']]

        self.queried_stations = []

        self.rmse_error = np.zeros(self.total_days - self.context_days) 
        self.mae_error = np.zeros(self.total_days - self.context_days) 

        if test_days is not None:
            pass

        for i in self.learners:
            self.learners[i].fit(self.X_train, self.y_train)
            
        if self.current_test.shape[0]!=0:
            self.to_test = 1
        else:
            self.to_test = 0

        self.reset_train = deepcopy(train_stations)
        self.reset_test = deepcopy(test_stations)
        self.reset_pool = deepcopy(pool_stations)

        self.reset_days = self.context_days

        self.random_rmse = np.zeros((50, self.total_days - self.context_days))
        self.random_mae =  np.zeros((50, self.total_days - self.context_days))
        

    def _next(self):
        self.current_day+=1

    def update_daily(self):

        train_to_add = []
        pool_to_add = []

        # It cannot be the case that there are no stations for a given timestamps in
        # self.timestamps. Else, it can't even exist!

        for station in self.train_stations:

            temp = self.df.groupby('Station').get_group(station)
            
            try:
                temp = temp.groupby('Time').get_group(self.timestamps[self.current_day])
                train_to_add.append(temp)
            except KeyError:
                pass
                # print("Station ", station, " has no data for timestamp ", self.timestamps[self.current_day])
            
        try:
            train_to_add = pd.concat(train_to_add)
            # Value Error comes here
            self.train = pd.concat([self.train, train_to_add])
            self.X_train = self.train[self.train_columns]
            self.y_train = self.train[['PM2.5']]


        except ValueError:
            pass
            # print("Empty Data")

        for station in self.pool_stations:
            temp = self.df.groupby('Station').get_group(station)

            try:
                temp = temp.groupby('Time').get_group(self.timestamps[self.current_day])
                pool_to_add.append(temp)
            except KeyError:
                pass
                # print("Station ", station, " has no data for timestamp ", self.timestamps[self.current_day])
        
        try:
            pool_to_add = pd.concat(pool_to_add)
            self.pool = pd.concat([self.pool, pool_to_add])
            self.X_pool = self.pool[self.train_columns]
            self.y_pool = self.pool[['PM2.5']]


        except ValueError:
            # print("NO POOL DATA")
            pass


        try:
            self.current_test = self.test.groupby('Time').get_group(self.timestamps[self.current_day])
            self.to_test = 1
            self.X_test = self.current_test[self.train_columns]
            self.y_test = self.current_test[['PM2.5']]

        except KeyError:
            # print("No Test Data for ", self.current_day)
            self.to_test = 0

        train_to_add = []
        pool_to_add = []

        # It cannot be the case that there are no stations for a given timestamps in
        # self.timestamps. Else, it can't even exist!

        for station in self.train_stations:

            temp = self.df.groupby('Station').get_group(station)
            
            try:
                temp = temp.groupby('Time').get_group(self.timestamps[self.current_day])
                train_to_add.append(temp)
            except KeyError:
                pass
                # print("Station ", station, " has no data for timestamp ", self.timestamps[self.current_day])
            
        train_to_add = pd.concat(train_to_add)

        for station in self.pool_stations:
            temp = self.df.groupby('Station').get_group(station)

            try:
                temp = temp.groupby('Time').get_group(self.timestamps[self.current_day])
                pool_to_add.append(temp)
            except KeyError:
                pass
                # print("Station ", station, " has no data for timestamp ", self.timestamps[self.current_day])
    
        pool_to_add = pd.concat(pool_to_add)

        self.train = pd.concat([self.train, train_to_add])
        self.X_train = self.train[self.train_columns]
        self.y_train = self.train[['PM2.5']]
        self.pool = pd.concat([self.pool, pool_to_add])
        self.X_pool = self.pool[self.train_columns]
        self.y_pool = self.pool[['PM2.5']]

        try:
            self.current_test = self.test.groupby('Time').get_group(self.timestamps[self.current_day])
            self.to_test = 1
            self.X_test = self.current_test[self.train_columns]
            self.y_test = self.current_test[['PM2.5']]

        except KeyError:
            # print("No Test Data for ", self.current_day)
            self.to_test = 0
        


    def max_variance_sampling(self):

        stddev = pd.DataFrame()
        for station in self.pool_stations:
            pool_data = self.pool.groupby('Station').get_group(station)[self.train_columns]
            temp = dict()
            for i in self.learners:
                temp[i] = self.learners[i].predict(pool_data)
                temp[i] = temp[i].reshape(temp[i].shape[0])
            temp = pd.DataFrame(temp)
            stddev[station] = [(temp.std(axis=1).mean())]
        station_to_add = stddev.loc[0].idxmax()
        self.queried_stations.append(station_to_add)
        return station_to_add


    def query_update(self, station_to_add):
        
        self.pool = self.pool[self.pool['Station']!=station_to_add]
        self.X_pool = self.pool[self.train_columns]
        self.y_pool = self.pool[['PM2.5']]
        self.pool_stations.remove(station_to_add)
        self.train_stations.append(station_to_add)
        self.update_daily()


    def querybycommittee(self):

        # if self.random==False:
        #     self.qbc = True
        #     print("Performing Query by Committee, No random called")
        # else:
        #     print("Random already called. Exiting")
        #     return None


        for itr in range(self.total_days - self.context_days):
            
#             try:
            
            current_timestamp = self.timestamps[self.current_day]
            past_month_timestamps = self.timestamps[self.current_day - 28: self.current_day]
            print(len(past_month_timestamps))
            training_data = []

            for timestamp in past_month_timestamps:
                try:
                    temp = self.train.groupby('Time').get_group(timestamp)
                    training_data.append(temp)
                except KeyError:
                    pass

            training_data = pd.concat(training_data)
            training_data = training_data.drop_duplicates()

            if training_data.shape[0]>0:


                X_train = training_data[self.train_columns]
                y_train = training_data[['PM2.5']]

                X_train = np.array(X_train)
                y_train = np.array(y_train)

                print("TRAINING SIZE - ", X_train.shape)
                self.to_test = 1

            else:
                self.to_test = 0
                    
#             except:
#                 traceback.print_exc()
#                 self.to_test = 1

            
            
            if itr%self.frequency==0:
                print(itr, self.pool.shape)

                station_to_add = self.max_variance_sampling()
                print(station_to_add)
                self.query_update(station_to_add)

            else:
                self.update_daily()
            
            if self.to_test:
                
                mse = np.zeros(len(self.learners))
                mae = np.zeros(len(self.learners))

                for i in self.learners:

                    self.learners[i].fit(X_train, y_train)
                    prediction = self.learners[i].predict(self.X_test)
                    mse[i - 1] = (mean_squared_error(prediction, self.y_test))
                    mae[i - 1] = (mean_absolute_error(prediction, self.y_test))

                self.rmse_error[itr] = np.sqrt(np.mean(mse))
                self.mae_error[itr] = np.mean(mae)
                
            else:
                
                self.rmse_error[itr] = -1 
                self.mae_error[itr] = -1
                
            print(self.rmse_error[itr], self.mae_error[itr])
            self._next()

    def random_sampling(self, seed=0):

        # if self.qbc==False:
        #     self.random = True
        #     print("Random Sampling now.")
        # else:
        #     print("QBC Called, exiting")
        #     return None
            
        # for seed in range(50):

        self.current_day = self.context_days
        
        current_timestamp = self.timestamps[self.current_day]
        past_month_timestamps = self.timestamps[self.current_day - 28: self.current_day]
        print(len(past_month_timestamps))
        training_data = []

        for timestamp in past_month_timestamps:
            try:
                temp = self.train.groupby('Time').get_group(timestamp)
                training_data.append(temp)
            except KeyError:
                pass

        training_data = pd.concat(training_data)
        training_data = training_data.drop_duplicates()

        if training_data.shape[0]>0:


            X_train = training_data[self.train_columns]
            y_train = training_data[['PM2.5']]

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            print("TRAINING SIZE - ", X_train.shape)
            self.to_test = 1

        else:
            self.to_test = 0

        np.random.seed(seed)

        for itr in range(self.total_days - self.context_days):

            if itr%self.frequency==0:
                station_to_add = np.random.choice(self.pool_stations, replace=False)
                self.query_update(station_to_add)

            else:
                self.update_daily()

            if self.to_test:

                mse = np.zeros(len(self.learners))
                mae = np.zeros(len(self.learners))

                for i in self.learners:

                    self.learners[i].fit(X_train, y_train)
                    prediction = self.learners[i].predict(self.X_test)
                    mse[i - 1] = (mean_squared_error(prediction, self.y_test))
                    mae[i - 1] = (mean_absolute_error(prediction, self.y_test))

                self.random_rmse[seed][itr] = np.mean(np.sqrt(mse))
                self.random_mae[seed][itr] = np.mean(mae)

            else:
                self.random_rmse[seed][itr] = -1 
                self.random_mae[seed][itr] = -1

            print(self.random_rmse[seed][itr], self.random_mae[seed][itr])
            self._next()
            
            # self.reset()

    def reset(self):

        print(len(self.reset_train), len(self.reset_pool), len(self.reset_test))

        self.train_stations = deepcopy(self.reset_train)
        self.pool_stations = deepcopy(self.reset_pool)
        self.test_stations = deepcopy(self.reset_test)

        self.current_day = self.context_days

        self.train = pd.concat([self.df.groupby('Station').get_group(station) for station in self.train_stations])
        self.pool = pd.concat([self.df.groupby('Station').get_group(station) for station in self.pool_stations])
        self.test = pd.concat([self.df.groupby('Station').get_group(station) for station in self.test_stations])

        self.timestamps = self.df['Time'].unique()
        self.timestamps.sort()
        initial_timestamps = self.timestamps[:self.context_days]

        self.train = pd.concat([self.train.groupby('Time').get_group(time) for time in initial_timestamps])
        self.pool = pd.concat([self.pool.groupby('Time').get_group(time) for time in initial_timestamps])

        self.current_test = self.test.groupby('Time').get_group(self.timestamps[self.current_day - 1])
        self.X_test = self.current_test[self.train_columns]
        self.y_test = self.current_test[['PM2.5']]

        self.X_train = self.train[self.train_columns]
        self.y_train = self.train[['PM2.5']]

        self.X_pool = self.pool[self.train_columns]
        self.y_pool = self.pool[['PM2.5']]

        print("Train", len(self.train_stations))
        print("Test", len(self.test_stations))
        print("Pool", len(self.pool_stations))




# # # from model import ActiveLearning
# stations = list(df['Station'].unique())
# stations.sort()
# # for station in stations:
# test_station = ['Anand Vihar']
# stations.remove(test_station[0])
# train_station = stations[:4]
# pool_station = stations[4:]
# print(train_station, pool_station)
# context_days = 15
# frequency = 5
# total_days = 48
# learners = {i:KNeighborsRegressor(n_neighbors=i, weights='distance') for i in range(1,6)}
# al = ActiveLearning(df,
#                    train_station, 
#                    pool_station,
#                    test_station,
#                    learners,
#                    context_days,
#                    frequency,
#                    total_days)
# al.querybycommittee()
# # stations.append(station)
#     plt.plot(al.error)