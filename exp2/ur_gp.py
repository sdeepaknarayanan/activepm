'''
Author: S Deepak narayanan
IIT Gandhinagar'''


import numpy as np 
import gpflow 
import pandas as pd 
import matplotlib.pyplot as plt
import gpflow
import tensorflow as tf
import operator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import traceback

class GPActive():

    def __init__(
                self, 
                df, 
                train_stations, 
                pool_stations, 
                test_stations, 
                context_days,
                frequency,
                total_days,
                ):

        self.df = df
        self.train_stations = train_stations
        self.pool_stations = pool_stations
        self.test_stations = test_stations
        # self.learners = learners
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

        self.queried_stations = []

        self.rmse_error = np.zeros(self.total_days - self.context_days) 
        self.mae_error = np.zeros(self.total_days - self.context_days) 

        # if test_days is not None:
        #     pass
            
        self.to_test = 1
        self.queried_stations = []
        self.rmse_error = np.zeros(self.total_days - self.context_days)
        self.mae_error = np.zeros(self.total_days - self.context_days)

        self.model_variance = np.zeros(self.total_days - self.context_days)

        self.init_train = 0
        flag = self.gp_train()

        if flag:
            self.trained = 1
        else:
            self.trained = 0




    def _next(self):
        self.current_day = self.current_day + 1

    def gp_sampling(self):

        variance = {i:[0] for i in self.pool_stations}

        for station in self.pool_stations:
            pool_data = self.pool.groupby('Station').get_group(station)
            X_pool = np.array(pool_data[self.train_columns])
            mean, var = self.model.predict_y(X_pool)
            variance[station] = var.mean()

        station_to_add = max(variance.items(), key=operator.itemgetter(1))[0]
        self.queried_stations.append(station_to_add)
        return station_to_add 

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

    def query_update(self, station_to_add):

        self.pool = self.pool[self.pool['Station']!=station_to_add]
        self.X_pool = self.pool[self.train_columns]
        self.y_pool = self.pool[['PM2.5']]
        self.pool_stations.remove(station_to_add)
        self.train_stations.append(station_to_add)
        self.update_daily()


    def gp_train(self):


        try:
            
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
                
            else:
                return 0 
            
            tf.reset_default_graph()
            graph = tf.get_default_graph()
            gpflow.reset_default_session(graph=graph)

            xy_matern_1 = gpflow.kernels.Matern52(input_dim=2, ARD=True, active_dims=[0, 1])
            xy_matern_2 = gpflow.kernels.Matern52(input_dim=2, ARD=True, active_dims=[0, 1])
            
            t_matern = gpflow.kernels.Matern52(input_dim=1, active_dims = [2])
            
            t_1 = gpflow.kernels.Matern52(input_dim=1, active_dims=[2])*gpflow.kernels.Periodic(input_dim=1, active_dims=[2]) 
            t_2 = gpflow.kernels.Matern52(input_dim=1, active_dims=[2])*gpflow.kernels.Periodic(input_dim=1, active_dims=[2]) 
            t_3 = gpflow.kernels.Matern52(input_dim=1, active_dims=[2])*gpflow.kernels.Periodic(input_dim=1, active_dims=[2]) 
            t_4 = gpflow.kernels.Matern52(input_dim=1, active_dims=[2])*gpflow.kernels.Periodic(input_dim=1, active_dims=[2]) 
            t_5 = gpflow.kernels.Matern52(input_dim=1, active_dims=[2])*gpflow.kernels.Periodic(input_dim=1, active_dims=[2]) 
            
            time = t_matern + t_1 + t_2 + t_3 + t_4 + t_5 

            combined = gpflow.kernels.Constant(input_dim = 1, active_dims = [4])*(gpflow.kernels.Matern52(input_dim = 2, active_dims = [3, 5], ARD=True) + gpflow.kernels.Matern32(input_dim = 2, active_dims = [3,5], ARD=True))
            
            wsk = gpflow.kernels.RBF(input_dim = 2, active_dims = [6,7], ARD=True)

            weathk = gpflow.kernels.RBF(input_dim = 1, active_dims = [8])
            
            overall_kernel = (xy_matern_1 + xy_matern_2) * time * combined * wsk * weathk
            
            self.model = gpflow.models.GPR(X_train, y_train, kern=overall_kernel, mean_function=None)
            opt = gpflow.train.ScipyOptimizer()
            opt.minimize(self.model)
            self.trained = 1
            
            filename = "models/ScipyOpt1/"+str(self.current_day)+ "_1_1_gpr.gpflow"
            path = Path(filename)
            if path.exists():
                path.unlink()
            saver = gpflow.saver.Saver()
            saver.save(filename, self.model)
            return 1
       

        except Exception:
            traceback.print_exc()
            self.trained = 0
            return 0


    def rmse(self, arr1, arr2):
        return np.sqrt(mean_squared_error(arr1, arr2))

    def mae(self, arr1, arr2):
        return mean_absolute_error(arr1, arr2)




    def active_gp(self):

        for curr_day in range( self.total_days - self.context_days ):
            
            flag = self.gp_train()


            print("Today is ", curr_day)

            if curr_day % self.frequency == 0:

                if self.trained:
                    station_to_add = self.gp_sampling()
                    self.query_update(station_to_add)

            else:
                
                self.update_daily()



            if flag:

                if self.to_test:

                    X_test = np.array(self.X_test)
                    y_test = np.array(self.y_test)

                    mean, variance = self.model.predict_y(self.X_test)

                    self.rmse_error[ curr_day ] = self.rmse(mean, y_test)
                    self.mae_error[ curr_day ] = self.mae(mean, y_test)

                    self.model_variance[ curr_day ] = variance.sum()

                else:
                    self.rmse_error[curr_day] = None
                    self.mae_error[curr_day] = None
            else:
                self.rmse_error[curr_day] = None
                self.mae_error[curr_day] = None
            
            print(self.rmse_error[curr_day], self.mae_error[curr_day])


            self._next()










        











































