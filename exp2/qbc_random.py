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
import xgboost
from copy import deepcopy
import matplotlib
import traceback
from copy import deepcopy    
import os

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
            number_to_query,
            number_of_seeds,
            fname = [None, None],
            gp_choice = False
        ):

        self.df = df
        self.train_stations = train_stations
        self.pool_stations = pool_stations
        self.test_stations = test_stations
        self.learners = learners
        self.frequency = frequency
        self.number_to_query = number_to_query
        self.number_of_seeds = number_of_seeds

        ########################################
        ########### RESET ATTRIBUTES ###########
        ########################################
        self.reset_df = df
        self.reset_train_stations = train_stations
        self.reset_test_stations = test_stations 
        self.reset_pool_stations = pool_stations 
        self.reset_learners = learners 
        ########################################
        ############# RESET ATTRIBUTES DONE ####
        ########################################

        if fname[0] == None:
            print("PLEASE PROVIDE FILE NAME")
            print("EXITING")
            return
        else:
            self.fname = fname

        self.current_day = context_days
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

        self.initialize_data()

        
        self.queried_stations = []
        self.random_queried_stations = [[] for i in range(self.number_of_seeds)]
        self.qbc_rmse = np.zeros(self.test_days + 1) 
        self.qbc_mae = np.zeros(self.test_days + 1) 
        self.random_rmse = np.zeros((self.number_of_seeds, self.test_days + 1))
        self.random_mae =  np.zeros((self.number_of_seeds, self.test_days + 1))

        self.gp_choice = gp_choice

        if self.gp_choice:
            path  = f"results/{self.train_days}/final_gp/{self.fname[0]}_{self.fname[1]}/stations.npy"
            self.gp_stations = np.load(path)

        else:
            self.gp_stations = None


        if self.is_trainable and self.is_testable:

            rmse = np.zeros(len(self.learners))
            mae = np.zeros(len(self.learners))
            for i in range(len(self.learners)):
                prediction = self.learners[i].predict(self.X_test)
                rmse[i] = np.sqrt(mean_squared_error(prediction, self.y_test))
                mae[i] = mean_absolute_error(prediction, self.y_test)
            self.qbc_rmse[0] = rmse.mean()
            self.qbc_mae[0] = mae.mean()

            for i in range(self.number_of_seeds):
                self.random_rmse[i][0] = rmse.mean()
                self.random_mae[i][0] = mae.mean()


        else:

            print("Model not fit since no train data (or) no test data available\n")

            self.qbc_rmse[0] = np.nan
            self.qbc_mae[0] = np.nan

            for i in range(self.number_of_seeds):
                self.random_rmse[i][0] = np.nan
                self.random_mae[i][0] = np.nan



                    

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


            ######################################################
            ######## POOL what about removal of stations #########
            ######################################################


        else:
            if self.pool.shape[0] == 0:
                self.is_queryable = False
            


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

            ############################################
            ######## Adding Multiple Stations ##########
            ############################################

            stations_to_add = []

            for i in range(self.number_to_query):

                if stddev.shape[1] > 0:
                    current_station = stddev.loc[0].idxmax()
                    stddev.drop(columns = [current_station], inplace = True)
                    stations_to_add.append(current_station)
                else:
                    stations_to_add.append(None)

            self.queried_stations.extend(stations_to_add)

            return stations_to_add

            
        else:
            print("No Data in Pool to Query")
            self.queried_stations.append(None)
            return None





    def query_update(self, stations_to_add):
        
        ##################### NEED TO UPDATE TO HAVE MULTIPLE STATIONS ########################
        for station in stations_to_add:
            self.pool_stations.remove(station)
            self.train_stations.append(station)

        assert(len(self.pool_stations) + len(self.train_stations) == 30)
        self.pool = self.pool[~self.pool['Station'].isin(stations_to_add)]
        self.data_update_daily()


    def querybycommittee(self):

        for itr in range(1, self.test_days + 1):

            print("Current Day before update:", self.current_day)
            ##########################################################
            self._next()            # Update the current day
            ##########################################################
            print("\nCurrent Day after update:", self.current_day)


            try:
                self.timestamps[self.current_day]
            except Exception as e:
                print(e)
                break

            if itr % self.frequency == 1:

                if self.gp_choice:
                    stations_to_add = self.gp_stations[0]
                    self.gp_stations = self.gp_stations[1:]

                    self.queried_stations.append(stations_to_add)
                    if stations_to_add is not None:
                        stations_to_add = [stations_to_add]

                    ### Not completely correct - need to account for pool stations not being present
                else:
                    stations_to_add = self.max_variance_sampling()

                if stations_to_add is not None:
                    self.query_update(stations_to_add)
                    print("Stations added - ", stations_to_add, " on day ",self.current_day)
                else:
                    self.data_update_daily()

            else:
                self.data_update_daily()


            if self.current_day < self.train_days:
                timestamps = self.timestamps[:self.current_day + 1]
            else:
                timestamps = self.timestamps[self.current_day + 1 - self.train_days: self.current_day + 1]

            training_data = []

            ##########################################################
            ########### Last K Days = 100 ############################
            ########### Context days = 30 ###########################
            ##########################################################

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

                self.is_trainable = True

            else:
                self.is_trainable = False



            
            if self.is_trainable and self.is_testable:

                try:

                    assert(self.X_test['Time'].unique()[0] == self.timestamps[self.current_day])

                    print("2nd")

                    assert(X_train['Time'].unique().max() == self.timestamps[self.current_day])

                except AssertionError:
                    print(X_train['Time'].unique().max() , self.timestamps[self.current_day])
                assert(self.X_test['Time'].unique().shape[0] == 1)


                # print("\nTrain DataFrame Shape", X_train.shape)
                # print("\nPool DataFrame Shape", self.pool.shape)
                # print("\nTest DataFrame Shape", self.X_test.shape)

                mse = np.zeros(len(self.learners))  
                mae = np.zeros(len(self.learners))

                for i in range(len(self.learners)):

                    self.learners[i].fit(X_train, y_train)
                    prediction = self.learners[i].predict(self.X_test)
                    mse[i] = (mean_squared_error(prediction, self.y_test))
                    mae[i] = (mean_absolute_error(prediction, self.y_test))

                self.qbc_rmse[itr] = np.sqrt(np.mean(mse))
                self.qbc_mae[itr] = np.mean(mae)

            else:
                
                self.qbc_rmse[itr] = np.nan
                self.qbc_mae[itr] = np.nan


            if itr % 10 == 0:

                if not self.gp_choice:

                    temp_store_path = f"results/{self.train_days}/intermediate_qbc_1/{self.fname[0]}_{self.fname[1]}/{self.current_day}"
                    if not os.path.exists(temp_store_path):
                        os.makedirs(temp_store_path)
                    np.save(temp_store_path + "/rmse", self.qbc_rmse)
                    np.save(temp_store_path + "/mae", self.qbc_mae)
                    np.save(temp_store_path + "/stations", self.queried_stations)

                else:
                    temp_store_path = f"results/{self.train_days}/intermediate_qbc_gps/{self.fname[0]}_{self.fname[1]}/{self.current_day}"
                    if not os.path.exists(temp_store_path):
                        os.makedirs(temp_store_path)
                    np.save(temp_store_path + "/rmse", self.qbc_rmse)
                    np.save(temp_store_path + "/mae", self.qbc_mae)
                    np.save(temp_store_path + "/stations", self.queried_stations)



        if not self.gp_choice:
            store_path = f"results/{self.train_days}/final_qbc_1/{self.fname[0]}_{self.fname[1]}"
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            np.save(store_path + "/final_rmse", self.qbc_rmse)
            np.save(store_path + "/final_mae", self.qbc_mae)
            np.save(store_path + "/stations", self.queried_stations)

        else:
            store_path = f"results/{self.train_days}/final_qbc_gps/{self.fname[0]}_{self.fname[1]}"
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            np.save(store_path + "/final_rmse", self.qbc_rmse)
            np.save(store_path + "/final_mae", self.qbc_mae)
            np.save(store_path + "/stations", self.queried_stations)



                

    def random_sampling(self):

        self.is_trainable = None
        self.is_testable = None
        self.is_queryable = None

        for seed in range(self.number_of_seeds):

            self.initialize_data()



            assert(len(self.train_stations) == 6)
            assert(len(self.pool_stations) == 24)
            assert(len(self.test_stations) == 6)

            print("\n Re - Initialized Dataset")




            print("Seed is", seed)

            rand_state = np.random.RandomState(seed)

            for itr in range(1, self.test_days + 1):

                print("Current Day before update:", self.current_day)
                ##########################################################
                self._next()            # Update the current day
                ##########################################################
                print("\nCurrent Day after update:", self.current_day)
                try:
                    self.timestamps[self.current_day]
                except Exception as e:
                    print(e)
                    break

                if itr % self.frequency == 1:

                    stations_to_add = rand_state.choice(
                        self.pool_stations,
                        self.number_to_query, 
                        replace=False
                        )
                    print("Stations added - ", stations_to_add)
                    self.random_queried_stations[seed].extend(stations_to_add)
                    self.query_update(stations_to_add)

                else:
                    self.data_update_daily()


                if self.current_day < self.train_days:
                    timestamps = self.timestamps[:self.current_day + 1]
                else:
                    timestamps = self.timestamps[self.current_day + 1 - self.train_days: self.current_day + 1]

                training_data = []

                ##########################################################
                ########### Last K Days = 100 ############################
                ########### Context days = 30 ###########################
                ##########################################################

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

                    self.is_trainable = True

                else:
                    self.is_trainable = False


                print(self.is_trainable, self.is_testable)

                if self.is_trainable and self.is_testable:


                    try:

                        assert(self.X_test['Time'].unique()[0] == self.timestamps[self.current_day])

                        print("2nd")

                        assert(X_train['Time'].unique().max() == self.timestamps[self.current_day])

                    except AssertionError:
                        print(X_train['Time'].unique().max() , self.timestamps[self.current_day])
                        print("LOOK ABOVE")



                    assert(self.X_test['Time'].unique().shape[0] == 1)
                    
                    # print("\nTrain DataFrame Shape", X_train.shape)
                    # print("\nPool DataFrame Shape", self.pool.shape)
                    # print("\nTest DataFrame Shape", self.X_test.shape)

                    mse = np.zeros(len(self.learners))  
                    mae = np.zeros(len(self.learners))

                    for i in range(len(self.learners)):

                        self.learners[i].fit(X_train, y_train)
                        prediction = self.learners[i].predict(self.X_test)
                        mse[i] = (mean_squared_error(prediction, self.y_test))
                        mae[i] = (mean_absolute_error(prediction, self.y_test))

                    self.random_rmse[seed][itr] = np.sqrt(np.mean(mse))
                    self.random_mae[seed][itr] = np.mean(mae)

                else:

                    self.random_rmse[seed][itr] = np.nan
                    self.random_mae[seed][itr] = np.nan

                temp_store_path = f"results/{self.train_days}/intermediate_random_qbc_1/{self.fname[0]}_{self.fname[1]}/{seed}/{self.current_day}"
                
                if itr % 10 == 0:
                    if not os.path.exists(temp_store_path):
                        os.makedirs(temp_store_path)
                    np.save(temp_store_path + "/rmse", self.random_rmse[seed])
                    np.save(temp_store_path + "/mae", self.random_mae[seed])
                    np.save(temp_store_path + "/stations", self.random_queried_stations[seed])


        
        store_path = f"results/{self.train_days}/final_random_qbc_1/{self.fname[0]}_{self.fname[1]}"
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        np.save(store_path + "/final_rmse", self.random_rmse)
        np.save(store_path + "/final_mae", self.random_mae)
        np.save(store_path + "/final_stations", self.random_queried_stations)






    def initialize_data(self):


        self.current_day = self.context_days
        self.train = pd.concat([self.df.groupby('Station').get_group(station) for station in self.reset_train_stations])
        self.pool = pd.concat([self.df.groupby('Station').get_group(station) for station in self.reset_pool_stations])
        self.test = pd.concat([self.df.groupby('Station').get_group(station) for station in self.reset_test_stations])

        # get the unique timestamps 
        # need to sort them to index into the timestamp data for the current day s
        self.timestamps = self.df['Time'].unique()
        self.timestamps.sort()
        initial_timestamps = self.timestamps[:self.context_days + 1]


        initial_train_data = []

        for time in initial_timestamps: 
            try:
                temp = self.train.groupby('Time').get_group(time)
                initial_train_data.append(temp)
            
            except KeyError:
                continue


        ##########################################################
        #### The above can be avoided using the isin() method ####
        ##########################################################
        # This If-Else clause avoids a value error

        if len(initial_train_data) > 0:
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
                initial_pool_data.append(temp)
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

        self.train_stations = deepcopy(self.reset_train_stations)
        self.test_stations = deepcopy(self.reset_test_stations)
        self.pool_stations = deepcopy(self.reset_pool_stations)


        if self.is_trainable and self.is_testable:
            for i in self.learners:
                self.learners[i].fit(X_train, y_train)
