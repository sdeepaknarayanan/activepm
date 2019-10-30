# from polire.interpolate import  Kriging, Idw
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from polire.interpolate import  Kriging, Idw
import sys
import gpflow
import tensorflow as tf





def interpolation(regressor):


    df = pd.read_csv("../../data/beijinga_scaled.csv")
    df = df.loc[:, ~df.columns.str.match('Unnamed')]
    df = df.rename(columns={'ts': 'Time', 'station_id': 'Station'})
    timestamps = df['Time'].unique()
    timestamps.sort()
    stations = df['Station'].unique()
    stations.sort()
    splits = 6

    kf = KFold(n_splits=splits, random_state=0, shuffle=True)

    if regressor == "gp":
        print("SPATIAL GP")

        rmse = {'GP':{timestamp:{i:np.nan for i in range(splits)} for timestamp in timestamps}}
        mae = {'GP':{timestamp:{i:np.nan for i in range(splits)} for timestamp in timestamps}}
        
        for timestamp in timestamps:
            
            i = -1 

            for train_index, test_index in kf.split(stations):
                i = i + 1
                train_stations = stations[train_index]
                test_stations = stations[test_index]
                try:
                    train_df = pd.concat([df.groupby('Station').get_group(stn) for stn in train_stations])
                    train_df = train_df.groupby('Time').get_group(timestamp)
                    train_df.drop(columns = ['Station', 'Time'])
                    test_df = pd.concat([df.groupby('Station').get_group(stn) for stn in test_stations])
                    test_df = test_df.groupby('Time').get_group(timestamp)
                    test_df.drop(columns = ['Station', 'Time'])

                except KeyError:
                    print("No train or test data for,", timestamp)
                    continue


                X_train = np.array(train_df[['longitude', 'latitude']])
                y_train = np.array(train_df[['PM2.5']])
                X_test = np.array(test_df[['longitude', 'latitude']])
                y_test = np.array(test_df[['PM2.5']])


                tf.reset_default_graph()
                graph = tf.get_default_graph()
                gpflow.reset_default_session(graph=graph)

                xy_matern_1 = gpflow.kernels.Matern52(input_dim = 2, ARD = True, active_dims = [0, 1])
                xy_matern_2 = gpflow.kernels.Matern52(input_dim = 2, ARD = True, active_dims = [0, 1])
                kernel = xy_matern_2 + xy_matern_1

                model = gpflow.models.GPR(X_train, y_train, kern = kernel, mean_function = None)
                opt = gpflow.train.ScipyOptimizer()
                opt.minimize(model)

                mean, variance = model.predict_y(X_test)


                rmse_error = np.sqrt(mean_squared_error(mean, y_test))
                mae_error = mean_absolute_error(mean, y_test)

                rmse['GP'][timestamp][i] = rmse_error
                mae['GP'][timestamp][i] = mae_error

        pd.DataFrame(rmse['GP']).to_csv("spatial_gp_rmse.csv", index = None)
        pd.DataFrame(mae['GP']).to_csv("spatial_gp_mae.csv", index = None)


    elif regressor == "krig":
        print("Kriging")

        rmse = {'Krig':{timestamp:{i:np.nan for i in range(splits)} for timestamp in timestamps}}
        mae = {'Krig':{timestamp:{i:np.nan for i in range(splits)} for timestamp in timestamps}}
        
        for timestamp in timestamps:
            
            i = -1 

            for train_index, test_index in kf.split(stations):
                i = i + 1
                train_stations = stations[train_index]
                test_stations = stations[test_index]
                try:
                    train_df = pd.concat([df.groupby('Station').get_group(stn) for stn in train_stations])
                    train_df = train_df.groupby('Time').get_group(timestamp)
                    train_df.drop(columns = ['Station', 'Time'])
                    test_df = pd.concat([df.groupby('Station').get_group(stn) for stn in test_stations])
                    test_df = test_df.groupby('Time').get_group(timestamp)
                    test_df.drop(columns = ['Station', 'Time'])

                except KeyError:
                    print("No train or test data for,", timestamp)
                    continue


                X_train = np.array(train_df[['longitude', 'latitude']])
                y_train = np.array(train_df[['PM2.5']])
                X_test = np.array(test_df[['longitude', 'latitude']])
                y_test = np.array(test_df[['PM2.5']])

                krig = Kriging()
                krig.fit(X_train, y_train)
                predicted = krig.predict(X_test)

                rmse_error = np.sqrt(mean_squared_error(predicted, y_test))
                mae_error = mean_absolute_error(predicted, y_test)

                rmse['Krig'][timestamp][i] = rmse_error
                mae['Krig'][timestamp][i] = mae_error

        pd.DataFrame(rmse['Krig']).to_csv("krig_rmse.csv", index = None)
        pd.DataFrame(mae['Krig']).to_csv("krig_mae.csv", index = None)



    elif regressor == "idw":
        print("idw")

        rmse = {'idw':{timestamp:{i:np.nan for i in range(splits)} for timestamp in timestamps}}
        mae = {'idw':{timestamp:{i:np.nan for i in range(splits)} for timestamp in timestamps}}
        
        for timestamp in timestamps:
            
            i = -1 

            for train_index, test_index in kf.split(stations):
                i = i + 1
                train_stations = stations[train_index]
                test_stations = stations[test_index]
                try:
                    train_df = pd.concat([df.groupby('Station').get_group(stn) for stn in train_stations])
                    train_df = train_df.groupby('Time').get_group(timestamp)
                    train_df.drop(columns = ['Station', 'Time'])
                    test_df = pd.concat([df.groupby('Station').get_group(stn) for stn in test_stations])
                    test_df = test_df.groupby('Time').get_group(timestamp)
                    test_df.drop(columns = ['Station', 'Time'])

                except KeyError:
                    print("No train or test data for,", timestamp)
                    continue


                X_train = np.array(train_df[['longitude', 'latitude']])
                y_train = np.array(train_df[['PM2.5']])
                X_test = np.array(test_df[['longitude', 'latitude']])
                y_test = np.array(test_df[['PM2.5']])

                idw = Idw()
                idw.fit(X_train, y_train)
                predicted = idw.predict(X_test)

                rmse_error = np.sqrt(mean_squared_error(predicted, y_test))
                mae_error = mean_absolute_error(predicted, y_test)
                rmse['idw'][timestamp][i] = rmse_error
                mae['idw'][timestamp][i] = mae_error

        pd.DataFrame(rmse['idw']).to_csv("idw_rmse.csv", index = None)
        pd.DataFrame(mae['idw']).to_csv("idw_mae.csv", index = None)


    else:
        print("Invalid Argument")


if __name__ == '__main__':
    interpolation(sys.argv[1])
