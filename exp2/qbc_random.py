#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Deepak Narayanan Sridharan
Created on Thr 24 Oct 2019 
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
            self, 
            df, 
            train_stations, 
            pool_stations, 
            test_stations,
            learners,
            context_days,
            frequency,
            test_days,
            train_days,
            number_of_seeds,
        ):

        """
        Dataframe
        """

        self.df = df
        self.train_stations = train_stations
        self.pool_stations = pool_stations
        self.test_stations = test_stations
        self.learners = learners
        self.frequency = frequency
        self.current_day = context_days - 1
        self.context_days = context_days
        self.test_days = test_days
        self.train_days = train_days

        self.is_trainable = None
        self.is_testable = None
        self.is_queryable = None


        self.train_columns = list(df.columns)       # Here train columns has all the columns that the dataset has.
        self.train_columns.remove('PM2.5')  # we do not want train to have particulate matter data
        self.train_columns.remove('Station')  # we do not want to have the station data either

        ## First I am choosing by stations, it can't be the case that there is no data
        ## for a given station. So there can be no key error here. 

        self.train = pd.concat([self.df.groupby('Station').get_group(station) for station in train_stations])
        self.pool = pd.concat([self.df.groupby('Station').get_group(station) for station in pool_stations])
        self.test = pd.concat([self.df.groupby('Station').get_group(station) for station in test_stations])

        # get the unique timestamps 
        # need to sort them to index into the timestamp data for the current day s
        self.timestamps = self.df['Time'].unique()
        self.timestamps.sort()
        initial_timestamps = self.timestamps[:self.context_days + 1]


        initial_train_data = []

        for time in initial_timestamps: 
            try:
                temp = self.train.groupby('Time').get_group(time)
                initial_train_data.append(time)
            
            except KeyError:
                continue

        # This If-Else clause avoids a value error
        if len(initial_train_data) != 0:
            self.train = pd.concat( initial_train_data )
            X_train = self.train[self.train_columns]
            y_train = self.train[['PM2.5']]
            self.is_trainable = True

        else:
            print("No initial train data\n")
            self.is_trainable = False

            


        initial_pool_data = []

        for time in initial_timestamps:
            try:
                temp = self.pool.groupby('Time').get_group(time)

            except KeyError:
                continue

        if len(initial_pool_data) > 0:
            self.pool = pd.concat( initial_pool_data )
            self.is_queryable = True

        else:
            print("No initial pool data\n")
            self.is_queryable = False


        try:
            self.current_test = self.test.groupby('Time').get_group(self.timestamps[self.current_day])
            self.X_test = self.current_test[self.train_columns]
            self.y_test = self.current_test[['PM2.5']]
            self.is_testable = True

        except KeyError:
            print("No initial test data\n")
            self.is_testable = False 

        
        self.queried_stations = []
        self.rmse_error = np.zeros(self.test_days + 1) 
        self.mae_error = np.zeros(self.test_days + 1) 
        self.random_rmse = np.zeros((number_of_seeds, self.test_days + 1))
        self.random_mae =  np.zeros((number_of_seeds, self.test_days + 1))


        if self.is_trainable:

            for i in self.learners:
                self.learners[i].fit(X_train, y_train)

            if self.is_testable:

                rmse = np.zeros(len(self.learners))
                mae = np.zeros(len(self.learners))
                for i in range(len(self.learners)):
                    prediction = self.learners[i].predict(self.X_test)
                    rmse[i] = np.sqrt(mean_squared_error(prediction, self.y_test))
                    mae[i] = mean_absolute_error(prediction, self.y_test)
                self.rmse_error[0] = rmse.mean()
                self.mae_error[0] = mae.mean()

                for i in range(number_of_seeds):
                    self.random_rmse[i][0] = rmse.mean()
                    self.random_mae[i][0] = mae.mean()

            else:

                self.rmse_error[0] = None
                self.mae_error[0] = None

                for i in range(number_of_seeds):
                    self.random_rmse[i][0] = None
                    self.random_mae[i][0] = None

        else:

            print("Model not fit since no train data available\n")

            self.rmse_error[0] = None
            self.mae_error[0] = None

            for i in range(number_of_seeds):
                self.random_rmse[i][0] = None
                self.random_mae[i][0] = None

                    

    def _next(self):
        self.current_day = self.current_day + 1

    def data_update_daily(self):

        # Update the Train Set
        train_to_add = []

        for station in self.train_stations:

            # It cannot be the case that there are no stations for a given timestamps in
            # self.timestamps. Else, it can't even exist!
            temp = self.df.groupby('Station').get_group(station)
            
            try:
                temp = temp.groupby('Time').get_group(self.timestamps[self.current_day])
                train_to_add.append(temp)

            except KeyError:
                pass
            
        if len(train_to_add) > 0:
            train_to_add = pd.concat(train_to_add)
            self.train = pd.concat([self.train, train_to_add])
        else:
            pass

        # Update the Pool Set                   
        pool_to_add = []

        for station in self.pool_stations:
            temp = self.df.groupby('Station').get_group(station)

            try:
                temp = temp.groupby('Time').get_group(self.timestamps[self.current_day])
                pool_to_add.append(temp)

            except KeyError:
                pass
        
        if len(pool_to_add) > 0:
            pool_to_add = pd.concat(pool_to_add)
            self.pool = pd.concat([self.pool, pool_to_add])
            self.is_queryable = True


        else:
            pass


        try:
            self.current_test = self.test.groupby('Time').get_group(self.timestamps[self.current_day])
            self.X_test = self.current_test[self.train_columns]
            self.y_test = self.current_test[['PM2.5']]
            self.is_testable = True

        except KeyError:
            self.is_testable = False



    def max_variance_sampling(self):

        if self.is_queryable:
            stddev = pd.DataFrame()
            for station in self.pool_stations:

                try:
                    pool_data = self.pool.groupby('Station').get_group(station)[self.train_columns]
                except KeyError:
                    continue

                temp = dict()
                for i in self.learners:
                    temp[i] = self.learners[i].predict(pool_data)
                    temp[i] = temp[i].reshape(temp[i].shape[0])
                temp = pd.DataFrame(temp)
                stddev[station] = [(temp.std(axis=1).mean())]
            station_to_add = stddev.loc[0].idxmax()
            self.queried_stations.append(station_to_add)
            return station_to_add

        else:
            print("No Data in Pool to Query")
            self.queried_stations.append(None)
            return None






    def query_update(self, station_to_add):
        
        self.pool = self.pool[self.pool['Station']!=station_to_add]
        self.pool_stations.remove(station_to_add)
        self.train_stations.append(station_to_add)
        self.data_update_daily()


    def querybycommittee(self):

        for itr in range(1, self.test_days + 1):

            ##########################################################
            self._next()            # Update the current day
            ##########################################################

            
            current_timestamp = self.timestamps[self.current_day]
            timestamps = self.timestamps[self.current_day + 1 - self.train_days: self.current_day + 1]
            training_data = []

            for time in timestamps:
                try:
                    temp = self.train.groupby('Time').get_group(time)
                    training_data.append(temp)

                except KeyError:
                    pass


            if len(training_data) > 0:

                training_data = pd.concat(training_data)

                X_train = training_data[self.train_columns]
                y_train = training_data[['PM2.5']]

                X_train = np.array(X_train)
                y_train = np.array(y_train)

                self.is_trainable = True

            else:
                self.is_trainable = False



            if itr % self.frequency == 1:

                station_to_add = self.max_variance_sampling()

                if station_to_add is not None:
                    self.query_update(station_to_add)
                else:
                    self.data_update_daily()

            else:
                self.data_update_daily()
            
            if self.is_trainable:

                if self.is_testable:
                
                    mse = np.zeros(len(self.learners))  
                    mae = np.zeros(len(self.learners))

                   for i in range(len(self.learners)):

                        self.learners[i].fit(X_train, y_train)
                        prediction = self.learners[i].predict(self.X_test)
                        mse[i] = (mean_squared_error(prediction, self.y_test))
                        mae[i] = (mean_absolute_error(prediction, self.y_test))

                    self.rmse_error[itr] = np.sqrt(np.mean(mse))
                    self.mae_error[itr] = np.mean(mae)

                else:

                    self.rmse_error[itr] = None
                    self.mae_error[itr] = None

            else:
                
                self.rmse_error[itr] = None
                self.mae_error[itr] = None
                

    def random_sampling(self, seed=0):

        np.random.seed(seed)

        for itr in range(1, self.test_days):

            ##########################################################
            self._next()            # Update the current day
            ##########################################################


            current_timestamp = self.timestamps[self.current_day]
            timestamps = self.timestamps[self.current_day + 1 - self.train_days: self.current_day + 1]
            training_data = []

            for time in timestamps:
                try:
                    temp = self.train.groupby('Time').get_group(time)
                    training_data.append(temp)

                except KeyError:
                    passs


            if len(training_data) > 0:

                training_data = pd.concat(training_data)

                X_train = training_data[self.train_columns]
                y_train = training_data[['PM2.5']]

                X_train = np.array(X_train)
                y_train = np.array(y_train)

                self.is_trainable = True

            else:
                self.is_trainable = False


            if itr % self.frequency == 1:

                station_to_add = np.random.choice(self.pool_stations, replace=False)
                self.query_update(station_to_add)

            else:
                self.update_daily()

            if self.is_trainable:

                if self.is_testable:
                
                    mse = np.zeros(len(self.learners))  
                    mae = np.zeros(len(self.learners))

                   for i in range(len(self.learners)):

                        self.learners[i].fit(X_train, y_train)
                        prediction = self.learners[i].predict(self.X_test)
                        mse[i] = (mean_squared_error(prediction, self.y_test))
                        mae[i] = (mean_absolute_error(prediction, self.y_test))

                    self.rmse_error[itr] = np.sqrt(np.mean(mse))
                    self.mae_error[itr] = np.mean(mae)

                else:

                    self.rmse_error[itr] = None
                    self.mae_error[itr] = None

            else:
                
                self.rmse_error[itr] = None
                self.mae_error[itr] = None