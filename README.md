# activepm
Codes and results for active learning for station deployment
Run all experiments from the master branch

Instructions for running the code:
* For the interpolation experiment,
    * Go to exp1/src/. Then run called.py with the arguments or caller.py with the corresponding arguments.
* For the active learning experiment,
    * Go to exp2/src/. Then run called.py with the arguments or caller.py with the corresponding arguments.
    
For experiment 2 - Active Learning - Use 

Scripts in the following format:

```python called1.py --kout 3 --kin 0 --train_days 10 --act 0 --id 0```

For experiment 1 - interpolation - use

```python called.py --reg <reg> --stepSize 2 -lastKDays 10 --datafile <path> --totalDays 350 -s True --loc <location to store, no need to change>```
Choose regressors by using help for argparse, path as datasets (beijingscaleda and beijingscaledb for feature sets a and b).

before running experiments, kindly remove all the folders named results so that results can be regenerated. 
To obtain the mean errors, kindly take the mean of the rmse's of the generated csv's.



