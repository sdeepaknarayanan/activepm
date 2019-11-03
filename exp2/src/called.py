import numpy as np 
from sklearn.model_selection import KFold
import pandas as pd
import sys
sys.path.append("../")
import gpflow
import tensorflow as tf

from gpsampling import GPActive
from qbc_ranom import ActiveLearning
import xgboost
import multiprocessing



"""



def print_cube(num): 
    """ 
    function to print cube of given num 
    """
    print("Cube: {}".format(num * num * num)) 
  
def print_square(num): 
    """ 
    function to print square of given num 
    """
    print("Square: {}".format(num * num)) 
  
if __name__ == "__main__": 
    # creating processes 
    p1 = multiprocessing.Process(target=print_square, args=(10, )) 
    p2 = multiprocessing.Process(target=print_cube, args=(10, )) 
  
    # starting process 1 
    p1.start() 
    # starting process 2 
    p2.start() 
  
    # wait until process 1 is finished 
    p1.join() 
    # wait until process 2 is finished 
    p2.join() 
  
    # both processes finished 
    print("Done!") 

"""

def active(learner):


	df = pd.read_csv("../data/beijingb_scaled.csv", index_col = 0)
	df = df.rename(columns={'ts': 'Time', 'station_id': 'Station'})

	stations = df['Station'].unique()
	stations.sort()

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)

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

    	gp_objects = {i:{j:0 for j in station_split[i]} for i in station_split}

    	processes = []

    	for i in station_split:
    		for j in station_split[i]:
		    	gp_objects[i][j] = GPActive(
								    df = df,
								    train_stations = list(station_split[i][j]['train']),
								    pool_stations = list(station_split[i][j]['pool']),
								    test_stations = list(station_split[i][j]['test']),
								    context_days = 9 , # train_days - 1
								    frequency = 30, # 
								    test_days = 360,  # 
								    train_days = 20, # 10, 20, 30
								    number_to_query = 1,
								    number_of_seeds= 8,
								    fname=[i, j]
								)
	    		processes.append()


	elif learner == 'qbc':

		learners = {
					0:xgboost.XGBRegressor(), 
		            1:xgboost.XGBRegressor(max_depth=10, learning_rate=1, n_estimators=10),
		            2:xgboost.XGBRegressor(max_depth=10, learning_rate=1, n_estimators=50),
		            3:xgboost.XGBRegressor(max_depth=50, learning_rate=1, n_estimators=10),
		            4:xgboost.XGBRegressor(max_depth=50, learning_rate=1, n_estimators=50),
		           }

		qbc_objects = {i:{j:0 for j in station_split[i]} for i in station_split}

		for i in station_split:
			for j in station_split[i]:
				
				qbc_objects[i][j] = al = ActiveLearning(
								    df = df,
								    train_stations = list(station_split[i][j]['train']) ,
								    pool_stations = list(station_split[i][j]['pool']),
								    test_stations = list(station_split[i][j]['test']),
								    learners = learners,
								    context_days = 9,
								    frequency = 5,
								    test_days = 50,
								    train_days = 20,
								    number_of_seeds = 5,
								    number_to_query = 1,
								    fname = [i, j]
					            )





	else:
		print("Not a correct learner -- enter gp or qbc")
		return 